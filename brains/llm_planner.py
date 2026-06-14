"""llm_planner.py — Tier A: the LLM planner (spec §3).

build_prompt is a pure function Observation -> (system, user) with the exact
section layout from the spec: STATUS / YOU KNOW / YOU BELIEVE / NOTES / NEARBY
/ RECENT EVENTS / ACTIONS. Parsing is strict: one retry with the error quoted
back, then fall back to [wait] and flag parse_error (the sim logs the event).
Every provider call is appended to prompts.jsonl.
"""
from __future__ import annotations
import json
import re
import threading
import time

from .base import Observation, PlanResult
from .providers import Provider
from .telemetry import (CompletionError, CompletionMetrics, CompletionResult,
                        RequestRecorder)

# Survival basics are COMMON knowledge (spec §1: "Eating berries: +25 energy");
# only the tech tree is hidden. Withholding them produced agents that starved
# while mining stone (m3 runs), which measures confusion, not discovery.
SYSTEM_TEMPLATE = (
    "You are {name}, a survivor in an unknown world. Basic facts you know: "
    "your energy drains every tick (more when moving) and at 0 energy you "
    "die, permanently. Berries are food — eat(berries) restores 25 energy. "
    "Wood and stone are materials, not food. You must stand on a tile to "
    "gather from it; give gather a target x, y and you will walk there "
    "first. Beyond that you do NOT know how this world works. You can "
    "discover how to create things by combining items (action: combine), "
    "but failures may destroy ingredients. You can talk to nearby agents — "
    "your voice only carries a few tiles. "
    "Your goals: survive, and understand this world."
)

# embodied common sense, like the survival basics: an agent that has walked
# its world knows its shape. Stated only when the sim provides world_size.
WORLD_SHAPE_TEMPLATE = (
    " The world is a {w}x{h} grid whose edges wrap around (stepping off one "
    "side brings you out the opposite side, and the shortest path may cross "
    "an edge)."
)

NOTES_CAP = 1500          # spec: ~1500 chars (open question 3 — revisit with transcripts)
MAX_PLAN_STEPS = 20
MAX_BELIEFS = 12
BELIEF_CAP = 250
MESSAGE_CAP = 300
BOOK_NOTE_CAP = 400       # mirror engine/actions.py caps
BOOK_ANSWER_CAP = 300
RESOURCES = ("berries", "wood", "stone")

# the --combine-note experiment knob. When on, EXACTLY this sentence is added
# to the system prompt (and nothing equivalent when off). It hints at the shape
# of the self-combine recipe (1 stone -> 1 brick) without naming any recipe.
COMBINE_NOTE = "An item can sometimes be combined by itself."

# book conditions: one goal sentence appended to the system prompt
BOOK_SYSTEM = {
    "shared": ("This world contains a single Book listing open questions "
               "about how the world works. Filling it with correct answers "
               "is a shared goal of your society."),
    "personal": ("You carry a private Book listing open questions about how "
                 "this world works. Recording correct answers in it is one "
                 "of your goals."),
    # empty-book variants (book_questions off): no seeded questions to answer
    "shared_empty": ("This world contains a single, empty Book your society can "
                     "fill with whatever you learn about how the world works."),
    "personal_empty": ("You carry a private, empty Book you can fill with "
                       "whatever you learn about how this world works."),
}

ACTIONS_DOC = (
    "move_to(x, y) | gather(resource: berries/wood/stone, x, y — walks to "
    "(x, y) then gathers; omit x/y to gather where you stand) | eat(item) | "
    'combine(items: {"item": qty, ...}) | say(target, message) | '
    "give(target, item, qty) | drop(item, qty) | wait"
)

_PLAN_DOC = (
    "Plan several steps ahead (up to 20): you will not be consulted again "
    "until the plan finishes, a step fails, or many ticks pass, and each "
    "consultation costs you time. One gather collects from one tile; to "
    "collect several, gather at several positions.\n"
)
_SCHEMA_DOC = (
    # the combine example must NOT be a real techtree recipe: models copy
    # examples verbatim, which would hand them a free "discovery"
    # (playtest 2026-06-13: the old wood 2 + stone 1 example WAS the axe)
    '{"reasoning": "<one short sentence: what you are doing and why>",\n'
    ' "updated_beliefs": ["<your current hypotheses about how this world works>", ...],\n'
    ' "updated_notes": "<your private scratchpad, rewritten in full, max 1500 chars>",\n'
    ' "plan": [{"action": "gather", "resource": "wood", "x": 3, "y": 4},\n'
    '          {"action": "gather", "resource": "wood", "x": 4, "y": 4},\n'
    '          {"action": "move_to", "x": 6, "y": 5},\n'
    '          {"action": "combine", "items": {"berries": 1, "wood": 1}}]}'
)
FORMAT_DOC = (_PLAN_DOC
              + "Respond ONLY with a JSON object, no other text:\n" + _SCHEMA_DOC)
# the freeform_thoughts knob (notes page 2: "is this helpful?"): same schema,
# but the model may reason in prose around it
FORMAT_DOC_FREEFORM = (
    _PLAN_DOC
    + "You may think out loud in plain text first if that helps you. Your "
    "reply MUST END with one JSON object in exactly this format (use curly "
    "braces nowhere else):\n" + _SCHEMA_DOC)


# ---------------------------------------------------------------- prompt
def _render_book(book: dict) -> str:
    """The == BOOK == section. Far from a central book: just where it is.
    Readable: every question with its current answer (author + tick — answers
    are editable) and the latest immutable note entries."""
    if not book["readable"]:
        bx, by = book["where"]
        return (f"The Book stands at ({bx}, {by}). Stand on or next to it "
                "to read it and write in it.")
    lines = []
    for q in book["questions"]:
        a = q.get("answer")
        ans = (f'answered by {a["author"]} at t{a["tick"]}: "{a["text"]}"'
               if a else "(unanswered)")
        lines.append(f"- [{q['id']}] {q['text']}\n    {ans}")
    n = book.get("n_entries", 0)
    shown = book.get("entries", [])
    if not book["questions"] and not shown:
        lines.append("The Book is empty.")
    if n > len(shown):
        lines.append(f"Notes (latest {len(shown)} of {n}; entries are "
                     "permanent and cannot be edited):")
    elif shown:
        lines.append("Notes (permanent, cannot be edited):")
    for e in shown:
        lines.append(f'  [t{e["tick"]}] {e["author"]}: "{e["text"]}"')
    if book["writable"]:
        lines.append("You may write: book_answer overwrites a question's "
                     "answer, book_note adds a permanent note."
                     if book["questions"] else
                     "You may write: book_note adds a permanent note.")
    elif book.get("reason"):
        lines.append(book["reason"].capitalize() + ".")
    return "\n".join(lines)


def build_prompt(obs: Observation, persona: str | None = None,
                 freeform: bool = False,
                 combine_note: bool = False) -> tuple[str, str]:
    me = obs.me
    system = SYSTEM_TEMPLATE.format(name=me["id"])
    if obs.world_size:
        system += WORLD_SHAPE_TEMPLATE.format(w=obs.world_size[0],
                                              h=obs.world_size[1])
    if combine_note:
        system += " " + COMBINE_NOTE
    if obs.book:
        cond = obs.book["condition"]
        scope = "personal" if cond == "personal" else "shared"
        if not obs.book.get("has_questions", True):
            scope += "_empty"
        system += " " + BOOK_SYSTEM[scope]
        if cond == "librarian":
            mgr = obs.book["manager"]
            system += (" You are the keeper of the Book: only you can write "
                       "in it, so gather what others learn."
                       if mgr == me["id"] else
                       f" Only {mgr} can write in the Book; share what you "
                       "learn so it gets recorded.")
    if persona:
        system += " " + persona

    inv = ", ".join(f"{i} x{q}" for i, q in sorted(me["inventory"].items())) or "empty"
    tools = ", ".join(sorted(me["tools"])) or "none"
    x, y = me["pos"]
    status = (f"energy {me['energy']}/100 | position ({x}, {y}) | "
              f"standing on {me['standing_on']} | inventory: {inv} "
              f"({me['slots_used']}/{me['slot_capacity']} slots) | tools: {tools}")

    near = []
    for res in RESOURCES:
        spots = obs.nearby_resources.get(res, [])[:3]
        if spots:
            near.append(f"{res} at " + ", ".join(f"({px}, {py})" for px, py in spots))
    for s in obs.nearby_structures:
        sx, sy = s["pos"]
        if s["owner"] == "world":   # world-placed (e.g. the Book): nobody built it
            near.append(f"a {s['type']} at ({sx}, {sy})")
        else:
            near.append(f"a {s['type']} at ({sx}, {sy}), built by {s['owner']}")
    for a in obs.nearby_agents:
        ax, ay = a["pos"]
        near.append(f"{a['id']} at ({ax}, {ay})")

    actions_doc = ACTIONS_DOC
    if obs.book and obs.book["writable"]:
        actions_doc += (" | book_answer(question, answer — overwrite the "
                        "Book's answer to one [question id]) | "
                        "book_note(text — permanent timestamped Book entry)")

    lines = [
        f"== STATUS == {status}",
    ]
    if obs.replan_reason:
        lines.append(f"== WHY NOW == You are planning because {obs.replan_reason}")
    lines += [
        "== YOU KNOW == " + ("; ".join(obs.knowledge) or "no confirmed recipes yet"),
        "== YOU BELIEVE == " + ("; ".join(obs.beliefs) or "(nothing yet)"),
        "== NOTES == " + (obs.notes or "(empty)"),
        "== NEARBY == " + ("; ".join(near) or "nothing of interest in sight"),
    ]
    if obs.book:
        lines.append("== BOOK ==\n" + _render_book(obs.book))
    lines += [
        "== RECENT EVENTS ==\n" + ("\n".join(obs.journal_tail) or "(nothing yet)"),
        "== ACTIONS == " + actions_doc,
        FORMAT_DOC_FREEFORM if freeform else FORMAT_DOC,
    ]
    return system, "\n".join(lines)


# ---------------------------------------------------------------- parsing
def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _int(v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise ValueError(f"expected integer, got {v!r}")
    return int(v)


def _validate_step(step) -> dict:
    if not isinstance(step, dict):
        raise ValueError(f"plan step is not an object: {step!r}")
    kind = step.get("action")
    if kind == "move_to":
        return {"action": "move_to", "x": _int(step["x"]), "y": _int(step["y"])}
    if kind == "gather":
        res = step.get("resource")
        if res not in RESOURCES:
            raise ValueError(f"gather: unknown resource {res!r}")
        out = {"action": "gather", "resource": res}
        # optional walk-to target (engine treats targeted gather as one
        # multi-tick primitive) — dropping these was the v0.32 smoke-run bug:
        # 795/795 model gathers carried x/y and all died "no wood here"
        if step.get("x") is not None and step.get("y") is not None:
            out["x"], out["y"] = _int(step["x"]), _int(step["y"])
        return out
    if kind == "eat":
        return {"action": "eat", "item": str(step.get("item", "berries"))}
    if kind == "combine":
        items = step.get("items")
        if not isinstance(items, dict) or not items:
            raise ValueError("combine: items must be a non-empty object")
        return {"action": "combine",
                "items": {str(k): _int(v) for k, v in items.items()}}
    if kind == "say":
        msg = str(step.get("message", ""))[:MESSAGE_CAP]
        if not msg:
            raise ValueError("say: empty message")
        return {"action": "say", "target": str(step.get("target", "")), "message": msg}
    if kind == "give":
        return {"action": "give", "target": str(step["target"]),
                "item": str(step["item"]), "qty": _int(step.get("qty", 1))}
    if kind in ("drop", "deposit", "withdraw"):
        return {"action": kind, "item": str(step["item"]),
                "qty": _int(step.get("qty", 1))}
    if kind == "book_answer":
        qid = str(step.get("question", "")).strip()
        ans = str(step.get("answer", "")).strip()[:BOOK_ANSWER_CAP]
        if not qid or not ans:
            raise ValueError("book_answer: question and answer required")
        return {"action": "book_answer", "question": qid, "answer": ans}
    if kind == "book_note":
        text = str(step.get("text", "")).strip()[:BOOK_NOTE_CAP]
        if not text:
            raise ValueError("book_note: empty text")
        return {"action": "book_note", "text": text}
    if kind == "wait":
        return {"action": "wait"}
    raise ValueError(f"unknown action {kind!r}")


def parse_response(raw: str) -> PlanResult:
    """Strict schema check; raises ValueError/KeyError/json errors on garbage."""
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("top level is not an object")
    raw_plan = data.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError("plan missing or empty")
    plan = [_validate_step(s) for s in raw_plan[:MAX_PLAN_STEPS]]
    intention = str(data.get("reasoning", ""))[:120]
    beliefs = data.get("updated_beliefs")
    if beliefs is not None:
        if not isinstance(beliefs, list):
            raise ValueError("updated_beliefs is not a list")
        beliefs = [str(b)[:BELIEF_CAP] for b in beliefs[:MAX_BELIEFS]]
    notes = data.get("updated_notes")
    if notes is not None:
        notes = str(notes)[:NOTES_CAP]
    return PlanResult(plan, intention, beliefs, notes)


# ---------------------------------------------------------------- brain
# brains share one prompts.jsonl per run; serialize appends across planner threads
_telemetry_lock = threading.Lock()


class LLMPlannerBrain:
    """Tier A planner. Tier B execution (pathfinding, reflexes) stays in the
    engine; this class only turns an Observation into a PlanResult."""

    def __init__(self, provider: Provider, telemetry_path: str | None = None,
                 persona: str | None = None, freeform: bool = False,
                 combine_note: bool = False, request_path: str | None = None):
        self.provider = provider
        self.telemetry_path = telemetry_path
        self.persona = persona
        self.freeform = freeform   # allow prose around the JSON (ablation knob)
        self.combine_note = combine_note   # --combine-note experiment knob
        self.request_recorder = RequestRecorder(request_path) if request_path else None
        # sim.py fans planning out to threads only when every brain opts in
        self.parallel_safe = getattr(provider, "parallel_safe", False)

    def plan(self, obs: Observation) -> PlanResult:
        system, user = build_prompt(obs, self.persona, self.freeform,
                                    self.combine_note)
        last_err = None
        for attempt in (1, 2):
            prompt = user if attempt == 1 else (
                user + "\n\nYour previous reply could not be used "
                f"({last_err}). Reply again with ONLY the JSON object, "
                "exactly in the format requested.")
            raw, err, result, completion = None, None, None, None
            provider_ok = False
            started_at = time.time()
            t0 = time.monotonic()
            try:
                if hasattr(self.provider, "complete_measured"):
                    completion = self.provider.complete_measured(system, prompt)
                else:
                    raw = self.provider.complete(system, prompt)
                    finished_at = time.time()
                    completion = CompletionResult(raw, "", CompletionMetrics(
                        requested_model=self.provider.name,
                        actual_model=self.provider.name,
                        started_at=started_at,
                        first_token_at=None,
                        finished_at=finished_at))
                provider_ok = True
                raw = completion.text
                result = parse_response(raw)
            except CompletionError as e:
                completion = e
                err = f"{type(e).__name__}: {e}"
            except Exception as e:  # provider/network errors count as failures too
                err = f"{type(e).__name__}: {e}"
                if completion is None:
                    completion = CompletionError(str(e), CompletionMetrics(
                        requested_model=self.provider.name,
                        actual_model=self.provider.name,
                        started_at=started_at,
                        first_token_at=None,
                        finished_at=time.time()))
            ms = int((time.monotonic() - t0) * 1000)
            self._log(obs, attempt, system, prompt, raw, ms, err)
            if self.request_recorder:
                self.request_recorder.write(
                    completion, role="planner", tick=obs.tick,
                    subject=obs.me["id"], attempt=attempt,
                    provider_ok=provider_ok, parse_ok=result is not None,
                    error=err)
            if result is not None:
                result.planning_attempts = attempt
                return result
            last_err = err[:200]
        return PlanResult([{"action": "wait"}], "(planner output unusable)",
                          parse_error=True, planning_attempts=2)

    def _log(self, obs: Observation, attempt: int, system: str, user: str,
             raw: str | None, ms: int, err: str | None):
        if not self.telemetry_path:
            return
        rec = {"tick": obs.tick, "agent": obs.me["id"], "attempt": attempt,
               "provider": self.provider.name, "ms": ms, "ok": err is None,
               "error": err, "system": system, "user": user, "response": raw}
        with _telemetry_lock:
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class BatchLLMPlannerBrain:
    """A planner that plans every agent of a tick in ONE batched model call.

    Shared by all agents (the factory hands the same instance to each), so the
    sim's planning phase can hand it the whole tick's observations at once via
    plan_batch(). The provider must implement complete_batch((system, user)...)
    -> (texts, metrics). Per-agent telemetry still lands in prompts.jsonl
    (build_prompt is unchanged, so a batched run is prompt-identical to the
    serial one); batch-level inference metrics land in model_metrics.jsonl.

    Retry policy mirrors LLMPlannerBrain: one re-ask, but batched — every agent
    whose first reply did not parse is re-prompted together with the error
    quoted back. Whatever still fails falls back to [wait] + parse_error.
    """

    parallel_safe = False   # the batch IS the parallelism; sim must not thread it

    def __init__(self, provider, telemetry_path: str | None = None,
                 metrics_path: str | None = None, persona: str | None = None,
                 freeform: bool = False, combine_note: bool = False,
                 request_path: str | None = None):
        if not hasattr(provider, "complete_batch"):
            raise TypeError(f"{type(provider).__name__} has no complete_batch; "
                            "BatchLLMPlannerBrain needs a batching provider")
        self.provider = provider
        self.telemetry_path = telemetry_path
        self.metrics_path = metrics_path
        self.persona = persona
        self.freeform = freeform
        self.combine_note = combine_note
        self.request_recorder = RequestRecorder(request_path) if request_path else None

    def plan(self, obs: Observation) -> PlanResult:
        """Single-agent path (e.g. a reflex-only tick): just a batch of one."""
        return self.plan_batch([obs])[0]

    def plan_batch(self, observations: list[Observation]) -> list[PlanResult]:
        n = len(observations)
        prompts = [build_prompt(o, self.persona, self.freeform, self.combine_note)
                   for o in observations]
        results: list[PlanResult | None] = [None] * n
        last_err: dict[int, str] = {}
        pending = list(range(n))
        tick = observations[0].tick if observations else 0
        for attempt in (1, 2):
            if not pending:
                break
            batch_inputs = []
            for i in pending:
                system, user = prompts[i]
                if attempt == 2:
                    user = (user + "\n\nYour previous reply could not be used "
                            f"({last_err.get(i)}). Reply again with ONLY the "
                            "JSON object, exactly in the format requested.")
                batch_inputs.append((system, user))
            texts, metrics = self.provider.complete_batch(batch_inputs)
            self._log_metrics(tick, attempt, len(pending), metrics)
            ms = int(round(1000 * (metrics.get("duration_s") or 0.0)))
            still = []
            for k, i in enumerate(pending):
                system, user = batch_inputs[k]
                raw = texts[k]
                err = None
                try:
                    results[i] = parse_response(raw)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    last_err[i] = err[:200]
                self._log(observations[i], attempt, system, user, raw, ms, err)
                if results[i] is None:
                    still.append(i)
            pending = still
        for i in pending:
            results[i] = PlanResult([{"action": "wait"}],
                                    "(planner output unusable)", parse_error=True)
        return results  # type: ignore[return-value]

    def _log(self, obs: Observation, attempt: int, system: str, user: str,
             raw: str | None, ms: int, err: str | None):
        if not self.telemetry_path:
            return
        rec = {"tick": obs.tick, "agent": obs.me["id"], "attempt": attempt,
               "provider": self.provider.name, "ms": ms, "ok": err is None,
               "error": err, "system": system, "user": user, "response": raw}
        with _telemetry_lock:
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _log_metrics(self, tick: int, attempt: int, batch_size: int,
                     metrics: dict):
        if not self.metrics_path:
            return
        rec = {"tick": tick, "attempt": attempt, "batch_size": batch_size,
               "provider": self.provider.name, **metrics}
        with _telemetry_lock:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
