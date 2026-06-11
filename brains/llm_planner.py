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
    "but failures may destroy ingredients. You can talk to nearby agents. "
    "Your goals: survive, and understand this world."
)

NOTES_CAP = 1500          # spec: ~1500 chars (open question 3 — revisit with transcripts)
MAX_PLAN_STEPS = 20
MAX_BELIEFS = 12
BELIEF_CAP = 250
MESSAGE_CAP = 300
RESOURCES = ("berries", "wood", "stone")

ACTIONS_DOC = (
    "move_to(x, y) | gather(resource: berries/wood/stone, x, y — walks to "
    "(x, y) then gathers; omit x/y to gather where you stand) | eat(item) | "
    'combine(items: {"item": qty, ...}) | say(target, message) | '
    "give(target, item, qty) | drop(item, qty) | wait"
)

FORMAT_DOC = (
    "Plan several steps ahead (up to 20): you will not be consulted again "
    "until the plan finishes, a step fails, or many ticks pass, and each "
    "consultation costs you time. One gather collects from one tile; to "
    "collect several, gather at several positions.\n"
    "Respond ONLY with a JSON object, no other text:\n"
    '{"reasoning": "<one short sentence: what you are doing and why>",\n'
    ' "updated_beliefs": ["<your current hypotheses about how this world works>", ...],\n'
    ' "updated_notes": "<your private scratchpad, rewritten in full, max 1500 chars>",\n'
    ' "plan": [{"action": "gather", "resource": "wood", "x": 3, "y": 4},\n'
    '          {"action": "gather", "resource": "wood", "x": 4, "y": 4},\n'
    '          {"action": "move_to", "x": 6, "y": 5},\n'
    '          {"action": "combine", "items": {"wood": 2, "stone": 1}}]}'
)


# ---------------------------------------------------------------- prompt
def build_prompt(obs: Observation, persona: str | None = None) -> tuple[str, str]:
    me = obs.me
    system = SYSTEM_TEMPLATE.format(name=me["id"])
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
        near.append(f"a {s['type']} at ({sx}, {sy}), built by {s['owner']}")
    for a in obs.nearby_agents:
        ax, ay = a["pos"]
        near.append(f"{a['id']} at ({ax}, {ay})")

    lines = [
        f"== STATUS == {status}",
        "== YOU KNOW == " + ("; ".join(obs.knowledge) or "no confirmed recipes yet"),
        "== YOU BELIEVE == " + ("; ".join(obs.beliefs) or "(nothing yet)"),
        "== NOTES == " + (obs.notes or "(empty)"),
        "== NEARBY == " + ("; ".join(near) or "nothing of interest in sight"),
        "== RECENT EVENTS ==\n" + ("\n".join(obs.journal_tail) or "(nothing yet)"),
        "== ACTIONS == " + ACTIONS_DOC,
        FORMAT_DOC,
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
        return {"action": "gather", "resource": res}
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
                 persona: str | None = None):
        self.provider = provider
        self.telemetry_path = telemetry_path
        self.persona = persona
        # sim.py fans planning out to threads only when every brain opts in
        self.parallel_safe = getattr(provider, "parallel_safe", False)

    def plan(self, obs: Observation) -> PlanResult:
        system, user = build_prompt(obs, self.persona)
        last_err = None
        for attempt in (1, 2):
            prompt = user if attempt == 1 else (
                user + "\n\nYour previous reply could not be used "
                f"({last_err}). Reply again with ONLY the JSON object, "
                "exactly in the format requested.")
            raw, err, result = None, None, None
            t0 = time.monotonic()
            try:
                raw = self.provider.complete(system, prompt)
                result = parse_response(raw)
            except Exception as e:  # provider/network errors count as failures too
                err = f"{type(e).__name__}: {e}"
            ms = int((time.monotonic() - t0) * 1000)
            self._log(obs, attempt, system, prompt, raw, ms, err)
            if result is not None:
                return result
            last_err = err[:200]
        return PlanResult([{"action": "wait"}], "(planner output unusable)",
                          parse_error=True)

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
