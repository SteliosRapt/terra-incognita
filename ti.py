"""ti.py — the Terra Incognita experiment CLI.

Built to be driven from anywhere a shell reaches this machine (SSH from a
phone agent included): plain-text output, no curses, and a `screenshot`
subcommand that renders any tick of any run to a PNG (stdlib only) and prints
its path — so a remote agent can attach the image to a chat.

  py ti.py run --provider openrouter --preset owl --book central --ticks 500
  py ti.py run --provider mock --ticks 120 --book cloud --survey off
  py ti.py list
  py ti.py status                      # latest run
  py ti.py screenshot                  # latest run, last tick -> PNG path
  py ti.py screenshot myrun --tick 200
  py ti.py summary myrun               # discoveries, book, survey curve
  py ti.py survey myrun                # who answered what, when
  py ti.py models                      # chains/presets from configs/openrouter.json
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.sim import Simulation
from engine.world import World
from engine.questions import load_questions, ground_truth
from engine.defaults import BASE, ALL_RECIPES
from run_m3 import summarize

RUNS = ROOT / "runs"
OPENROUTER_CONFIG = ROOT / "configs" / "openrouter.json"


# ---------------------------------------------------------------- run files
def read_events(run_dir: pathlib.Path) -> list[dict]:
    out = []
    with open(run_dir / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break    # truncated tail of a live run
    return out


def latest_run() -> pathlib.Path:
    cands = [d for d in RUNS.iterdir()
             if d.is_dir() and (d / "config.json").exists()
             and (d / "events.jsonl").exists()] if RUNS.exists() else []
    if not cands:
        sys.exit("no runs found under runs/")
    return max(cands, key=lambda d: (d / "events.jsonl").stat().st_mtime)


def resolve_run(name: str | None) -> pathlib.Path:
    if not name:
        return latest_run()
    d = RUNS / name
    if not (d / "config.json").exists():
        sys.exit(f"no run named {name!r} under runs/")
    return d


# ---------------------------------------------------------------- PNG output
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data)))


class Canvas:
    def __init__(self, w: int, h: int, bg=(24, 24, 28)):
        self.w, self.h = w, h
        self.px = bytearray(bg * (w * h))

    def rect(self, x: int, y: int, rw: int, rh: int, c: tuple):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + rw), min(self.h, y + rh)
        row = bytes(c) * (x1 - x0)
        for yy in range(y0, y1):
            i = (yy * self.w + x0) * 3
            self.px[i:i + (x1 - x0) * 3] = row

    def text(self, x: int, y: int, s: str, c: tuple, scale: int = 2):
        for ch in s.upper():
            glyph = FONT.get(ch, FONT["?"])
            for gy, rowbits in enumerate(glyph):
                for gx, bit in enumerate(rowbits):
                    if bit == "1":
                        self.rect(x + gx * scale, y + gy * scale, scale, scale, c)
            x += 4 * scale

    def save(self, path: pathlib.Path):
        raw = b"".join(b"\x00" + bytes(self.px[y * self.w * 3:(y + 1) * self.w * 3])
                       for y in range(self.h))
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h,
                                                8, 2, 0, 0, 0)))
            f.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
            f.write(_chunk(b"IEND", b""))


FONT = {  # 3x5 bitmap font: digits, A-Z and a little punctuation
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"), "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "110", "100", "110", "101"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "101", "101", "101"), "N": ("110", "101", "101", "101", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "011", "001"), "R": ("110", "101", "110", "110", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"), "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "101", "111", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"),
    ":": ("000", "010", "000", "010", "000"), "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"), "/": ("001", "001", "010", "100", "100"),
    " ": ("000", "000", "000", "000", "000"), "?": ("110", "001", "010", "000", "010"),
}

TILE = 16
HEADER = 22
TERRAIN_RGB = {
    "grass": (96, 132, 72), "water": (52, 96, 148),
    "forest": (38, 84, 48), "forest_empty": (70, 104, 66),
    "mountain": (138, 138, 142), "mountain_empty": (92, 92, 96),
}
STRUCT_RGB = {"fire": (224, 120, 32), "farm": (150, 116, 52),
              "storehouse": (104, 76, 40), "book": (208, 172, 92),
              "shelter": (96, 200, 104)}   # a distinctly greener square
AGENT_RGB = [(230, 70, 70), (70, 130, 230), (240, 200, 60), (170, 90, 220),
             (70, 200, 200), (240, 130, 200), (130, 220, 80), (240, 150, 60),
             (110, 110, 240), (90, 210, 140), (220, 90, 130), (160, 200, 230),
             (200, 170, 110), (120, 160, 90), (210, 110, 240), (250, 240, 200)]


def render_png(cfg: dict, events: list[dict], tick: int,
               out_path: pathlib.Path) -> pathlib.Path:
    """The world as it stood after `tick`, reconstructed exactly like the
    viewer's replay tab: regenerate the initial world from config+seed (same
    rng sequence as Simulation.__init__), then apply the event log."""
    rng = random.Random(cfg["seed"])
    w = World(cfg, rng)
    names = list(cfg.get("agent_names") or
                 (f"A{i:02d}" for i in range(cfg["n_agents"])))
    agents: dict[str, list] = {}
    for nm in names:
        while True:
            x, y = rng.randrange(w.W), rng.randrange(w.H)
            if w.tile(x, y).type == "grass":
                break
        agents[nm] = [x, y, True]
    structures: dict[tuple, str] = {}
    for ev in events:
        if ev["tick"] > tick:
            break
        t, p = ev["type"], ev.get("payload") or {}
        pos = tuple(ev["position"]) if ev.get("position") else None
        actor = ev.get("actor")
        if actor in agents and pos:
            agents[actor][0:2] = pos
        if t == "agent_died" and actor in agents:
            agents[actor][2] = False
        elif t == "structure_built" and pos:
            structures[pos] = p["structure"]
        elif t == "structure_expired" and pos:
            structures.pop(pos, None)
        elif t == "gathered" and pos:
            tl = w.tile(*pos)
            if p["resource"] == "berries":
                tl.berries = max(0, tl.berries - p["qty"])
            elif p["resource"] == "wood":
                tl.wood = 0
            elif p["resource"] == "stone":
                tl.stone = max(0, tl.stone - p["qty"])
        elif t == "resource_regrown" and pos:
            tl = w.tile(*pos)
            if p["resource"] == "berries":
                tl.berries = max(tl.berries, 1)
            else:
                tl.wood = 1
        elif t == "resource_depleted" and pos:
            tl = w.tile(*pos)
            if p["resource"] == "wood":
                tl.wood = 0
            else:
                tl.stone = 0
        elif t == "farm_yielded" and pos:
            w.tile(*pos).berries += p["amount"]

    cv = Canvas(w.W * TILE, h := w.H * TILE + HEADER)
    for y in range(w.H):
        for x in range(w.W):
            tl = w.grid[y][x]
            key = tl.type
            if tl.type == "forest":
                key = "forest" if tl.wood else "forest_empty"
            elif tl.type == "mountain":
                key = "mountain" if tl.stone else "mountain_empty"
            cv.rect(x * TILE, HEADER + y * TILE, TILE, TILE,
                    TERRAIN_RGB.get(key, TERRAIN_RGB["grass"]))
            if tl.berries > 0:
                cv.rect(x * TILE + 5, HEADER + y * TILE + 5, 6, 6, (196, 40, 64))
    for (x, y), st in structures.items():
        cv.rect(x * TILE + 1, HEADER + y * TILE + 1, TILE - 2, TILE - 2,
                STRUCT_RGB.get(st, (200, 200, 200)))
        cv.text(x * TILE + 5, HEADER + y * TILE + 3, st[0], (20, 20, 20), 2)
    for i, nm in enumerate(names):           # dead first, living on top
        x, y, alive = agents[nm]
        if not alive:
            cv.text(x * TILE + 5, HEADER + y * TILE + 3, "X", (40, 40, 40), 2)
    for i, nm in enumerate(names):
        x, y, alive = agents[nm]
        if alive:
            cv.rect(x * TILE + 2, HEADER + y * TILE + 2, TILE - 4, TILE - 4,
                    AGENT_RGB[i % len(AGENT_RGB)])
            cv.text(x * TILE + 5, HEADER + y * TILE + 3, nm[0], (16, 16, 16), 2)
    n_alive = sum(1 for a in agents.values() if a[2])
    cv.text(4, 6, f"T{tick}  ALIVE {n_alive}/{len(names)}  "
            f"{cfg.get('knowledge_condition', '?')}/"
            f"{cfg.get('book_condition', 'none')}", (235, 235, 225), 2)
    cv.save(out_path)
    return out_path


# ---------------------------------------------------------------- survey glue
def survey_subjects(sim: Simulation) -> dict[str, str]:
    """What the judge reads: each living agent's beliefs+notes (+ its own
    answers in the personal condition), and the shared book's answers."""
    subs = {}
    cond = sim.cfg.get("book_condition", "none")
    for name, ag in sim.agents.items():
        if not ag.alive:
            continue
        parts = []
        if ag.beliefs:
            parts.append("BELIEFS:\n" + "\n".join(f"- {b}" for b in ag.beliefs))
        if ag.notes.strip():
            parts.append("NOTES:\n" + ag.notes)
        if cond == "personal":
            bk = sim.world.books.get(name)
            if bk and bk.answers:
                parts.append("BOOK ANSWERS:\n" + "\n".join(
                    f"- [{q}] {a['text']}" for q, a in sorted(bk.answers.items())))
        if parts:
            subs[name] = "\n\n".join(parts)
    if cond in ("central", "librarian", "cloud"):
        bk = sim.world.books["shared"]
        if bk.answers:
            subs["book"] = "THE BOOK'S ANSWERS:\n" + "\n".join(
                f"- [{q}] {a['text']} (by {a['author']}, t{a['tick']})"
                for q, a in sorted(bk.answers.items()))
    return subs


def read_survey(run_dir: pathlib.Path) -> list[dict]:
    p = run_dir / "survey.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return out


def first_correct(records: list[dict]) -> dict[str, dict[str, int]]:
    """qid -> {subject: first tick the judge marked it true}."""
    out: dict[str, dict[str, int]] = {}
    for rec in records:
        if not rec.get("ok"):
            continue
        for qid, verdict in (rec.get("results") or {}).items():
            if verdict and rec["subject"] not in out.setdefault(qid, {}):
                out[qid][rec["subject"]] = rec["tick"]
    return out


# ---------------------------------------------------------------- commands
def cmd_run(args):
    cfg = dict(BASE, seed=args.seed, n_agents=args.agents,
               knowledge_condition=args.condition,
               replan_interval=args.replan,
               book_condition=args.book,
               freeform_thoughts=bool(args.freeform),
               combine_note=(args.combine_note == "on"))
    if args.book_manager:
        cfg["book_manager"] = args.book_manager

    name = args.name or f"ti_{args.provider}_{args.condition}_book-{args.book}_s{args.seed}"
    run_dir = RUNS / name
    if run_dir.exists() and not args.name:
        i = 2
        while (RUNS / f"{name}_{i}").exists():
            i += 1
        run_dir = RUNS / f"{name}_{i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = run_dir / "prompts.jsonl"
    telemetry.write_text("", encoding="utf-8")

    spec = {"backend": args.provider}
    if args.model:
        spec["model"] = args.model
    if args.preset:
        spec["preset"] = args.preset
    if args.host and args.provider == "ollama":
        spec["host"] = args.host
    if args.provider == "openrouter":
        spec["cost_path"] = str(run_dir / "costs.jsonl")
    if args.provider == "ollama" and args.freeform:
        spec["json_format"] = False   # freeform prose around the JSON object

    from brains.llm_planner import LLMPlannerBrain
    from brains.providers import make_provider
    shared = {}

    def factory(rng):
        if "provider" not in shared:
            shared["provider"] = make_provider(spec, rng)
        return LLMPlannerBrain(shared["provider"], telemetry_path=str(telemetry),
                               freeform=cfg["freeform_thoughts"],
                               combine_note=cfg["combine_note"])

    sim = Simulation(cfg, factory, str(run_dir))
    # config on disk from tick 0: live `status`/`screenshot` need it
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # ground truth for the survey judge + the end-of-run analysis
    judge = None
    questions = []
    if sim.world.techtree:
        questions = load_questions(cfg.get("questions_path"), sim.world.techtree)
        truth = ground_truth(questions, cfg, sim.world, sim.world.techtree)
        with open(run_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump({"questions": questions, "truth": truth}, f, indent=2,
                      ensure_ascii=False)
        survey_on = (args.survey == "on"
                     or (args.survey == "auto" and args.provider != "mock"))
        if survey_on:
            if not os.environ.get("OPENROUTER_API_KEY"):
                sys.exit("survey needs OPENROUTER_API_KEY (judge runs on the "
                         "free openrouter chain); use --survey off to skip")
            from brains.survey import SurveyJudge
            with open(OPENROUTER_CONFIG, encoding="utf-8") as f:
                judge_models = json.load(f)["judge_models"]
            (run_dir / "survey.jsonl").write_text("", encoding="utf-8")
            judge = SurveyJudge(make_provider({"backend": "openrouter",
                                               "model": ",".join(judge_models)}),
                                questions, truth, str(run_dir / "survey.jsonl"))

    print(f"run {run_dir.name}: {args.agents} agents x {args.ticks} ticks | "
          f"{args.provider}"
          + (f"/preset:{args.preset}" if args.preset else "")
          + (f"/{args.model}" if args.model else "")
          + f" | knowledge={args.condition} book={args.book} seed={args.seed}"
          + (" | freeform" if args.freeform else "")
          + (f" | survey every {args.survey_every}t" if judge else " | survey off"),
          flush=True)

    judge_calls = 0
    for _ in range(args.ticks):
        sim.step()
        sim.logf.flush()             # live status/screenshot read this file
        if args.render_every and sim.tick % args.render_every == 0:
            print(sim.render_ascii())
        if judge and sim.tick % args.survey_every == 0:
            judge_calls += judge.grade(sim.tick, survey_subjects(sim))
        if sim.tick % 50 == 0:
            alive = sum(a.alive for a in sim.agents.values())
            print(f"  t{sim.tick}: alive {alive}/{args.agents}", flush=True)
        if not any(a.alive for a in sim.agents.values()):
            print(f"  population extinct at t{sim.tick}")
            break
    sim.logf.close()
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + summarize(run_dir))
    if judge:
        fc = first_correct(read_survey(run_dir))
        answered = sorted((qid, min(t.values())) for qid, t in fc.items() if t)
        line = " ".join(f"{q}@t{t}" for q, t in answered) or "(none judged correct)"
        print(f"survey: {judge_calls} judge calls | first-correct: {line}")
    totals = getattr(shared.get("provider"), "totals", None)
    if totals:
        print(f"spend: {totals['calls']} calls | {totals['prompt_tokens']} in / "
              f"{totals['completion_tokens']} out tokens | "
              f"${totals['cost_usd']:.4f} | rate-limit hits {totals['rate_limited']}")
    print(f"\nrun dir: {run_dir.resolve()}")


def cmd_list(args):
    if not RUNS.exists():
        sys.exit("no runs/ directory yet")
    rows = []
    for d in sorted(RUNS.iterdir(),
                    key=lambda d: -(d / "events.jsonl").stat().st_mtime
                    if (d / "events.jsonl").exists() else 0):
        if not (d / "config.json").exists() or not (d / "events.jsonl").exists():
            continue
        with open(d / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        # last tick: read the tail of the log without parsing all of it
        with open(d / "events.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace").strip().splitlines()
        tick = "?"
        for line in reversed(tail):
            try:
                tick = json.loads(line)["tick"]
                break
            except (json.JSONDecodeError, KeyError):
                continue
        rows.append((d.name, tick, cfg.get("n_agents", "?"),
                     cfg.get("knowledge_condition", "?"),
                     cfg.get("book_condition", "none")))
    if not rows:
        sys.exit("no runs found")
    wname = max(len(r[0]) for r in rows)
    print(f"{'run':<{wname}}  {'tick':>5}  {'ag':>3}  knowledge/book")
    for n, t, a, k, b in rows[:args.n]:
        print(f"{n:<{wname}}  {t:>5}  {a:>3}  {k}/{b}")


def cmd_status(args):
    d = resolve_run(args.run)
    evs = read_events(d)
    with open(d / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    deaths = [e for e in evs if e["type"] == "agent_died"]
    disc, taught = {}, 0
    calls = errors = 0
    for e in evs:
        if e["type"] == "recipe_learned":
            disc.setdefault(e["payload"]["recipe"], e["tick"])
            taught += e["payload"]["source"] == "taught"
        elif e["type"] == "planner_called":
            calls += 1
        elif e["type"] == "planner_parse_error":
            errors += 1
    tick = evs[-1]["tick"] if evs else 0
    n = cfg.get("n_agents", 0)
    print(f"{d.name}  t{tick} | alive {n - len(deaths)}/{n} | planner {calls} "
          f"calls, {errors} parse errors | taught {taught}")
    print("discovered: " + (" ".join(f"{r}@t{t}" for r, t in
                                     sorted(disc.items(), key=lambda kv: kv[1]))
                            or "(nothing yet)"))
    answers = {}
    for e in evs:
        if e["type"] == "book_answer_written":
            answers[e["payload"]["question"]] = e
    if answers:
        print(f"book: {len(answers)} questions answered, "
              f"{sum(e['type'] == 'book_note_written' for e in evs)} notes")
    sv = read_survey(d)
    if sv:
        last_ok = [r for r in sv if r.get("ok")]
        if last_ok:
            latest = {}
            for r in last_ok:
                latest[r["subject"]] = r
            qids = set()
            correct = 0
            for r in latest.values():
                for q, v in r["results"].items():
                    qids.add(q)
                    correct += bool(v)
            print(f"survey: latest verdicts — {correct} correct answers "
                  f"across {len(latest)} subjects x {len(qids)} questions")
    costs = d / "costs.jsonl"
    if costs.exists():
        last = None
        with open(costs, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if last:
            rec = json.loads(last)
            print(f"spend: ${rec['total_cost_usd']:.4f} over {rec['total_calls']} calls")
    for e in evs[-6:]:
        print(f"  [t{e['tick']}] {e['type']} {e.get('actor') or ''} "
              f"{json.dumps(e.get('payload') or {}, ensure_ascii=False)[:90]}")


def cmd_summary(args):
    d = resolve_run(args.run)
    print(d.name)
    print(summarize(d))
    evs = read_events(d)
    answers = {}
    for e in evs:
        if e["type"] == "book_answer_written":
            answers[e["payload"]["question"]] = e
    if answers:
        print("\nbook answers (final):")
        for q, e in sorted(answers.items()):
            print(f"  [{q}] (by {e['actor']}, t{e['tick']}) {e['payload']['text']}")
    fc = first_correct(read_survey(d))
    if fc:
        print("\nsurvey first-correct (any subject):")
        for qid, subs in sorted(fc.items(), key=lambda kv: min(kv[1].values())):
            t = min(subs.values())
            who = min(subs, key=subs.get)
            print(f"  t{t:>4}  {qid}  ({who}, {len(subs)} subjects total)")


def cmd_survey(args):
    d = resolve_run(args.run)
    recs = read_survey(d)
    if not recs:
        sys.exit(f"no survey.jsonl in {d.name}")
    ok = [r for r in recs if r.get("ok")]
    fails = len(recs) - len(ok)
    print(f"{d.name}: {len(recs)} judge calls ({fails} failed)")
    fc = first_correct(recs)
    qids = sorted({q for r in ok for q in r["results"]})
    for qid in qids:
        subs = fc.get(qid, {})
        if subs:
            order = ", ".join(f"{s}@t{t}" for s, t in
                              sorted(subs.items(), key=lambda kv: kv[1]))
            print(f"  {qid}: {order}")
        else:
            print(f"  {qid}: never answered correctly")


def cmd_screenshot(args):
    d = resolve_run(args.run)
    with open(d / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    evs = read_events(d)
    tick = args.tick if args.tick is not None else (evs[-1]["tick"] if evs else 0)
    out = pathlib.Path(args.out) if args.out else d / f"shot_t{tick}.png"
    render_png(cfg, evs, tick, out)
    print(out.resolve())


def cmd_models(args):
    with open(OPENROUTER_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    print("default chain: " + " -> ".join(cfg["models"]))
    print("paid (tier=paid): " + " -> ".join(cfg["paid_models"]))
    print("judge: " + " -> ".join(cfg["judge_models"]))
    notes = cfg.get("preset_notes", {})
    print("presets (--preset NAME):")
    for name, chain in cfg.get("presets", {}).items():
        note = f"   # {notes[name]}" if name in notes else ""
        print(f"  {name:<16} {' -> '.join(chain)}{note}")


def main():
    p = argparse.ArgumentParser(prog="ti", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("--provider", choices=["mock", "ollama", "anthropic", "openrouter"],
                   default="openrouter")
    r.add_argument("--model", default=None,
                   help="model id (openrouter: comma list = fallback chain)")
    r.add_argument("--preset", default=None,
                   help="openrouter chain preset from configs/openrouter.json")
    r.add_argument("--host", default=None, help="ollama host override")
    r.add_argument("--condition", default="blind",
                   choices=["blind", "oracle", "fragmented", "one_expert"])
    r.add_argument("--book", default="none",
                   choices=["none", "central", "librarian", "personal", "cloud"])
    r.add_argument("--book-manager", default=None,
                   help="librarian condition: which agent keeps the book")
    r.add_argument("--agents", type=int, default=10)
    r.add_argument("--ticks", type=int, default=500)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--replan", type=int, default=15)
    r.add_argument("--freeform", action="store_true",
                   help="let models think in prose around the JSON (ablation)")
    r.add_argument("--combine-note", choices=["on", "off"], default="off",
                   help="add the exact hint 'An item can sometimes be combined "
                        "by itself.' to the system prompt")
    r.add_argument("--survey", choices=["auto", "on", "off"], default="auto",
                   help="free-LLM judge grades agents' world-knowledge "
                        "(auto: on for real providers)")
    r.add_argument("--survey-every", type=int, default=10)
    r.add_argument("--name", default=None, help="run directory name under runs/")
    r.add_argument("--render-every", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    l = sub.add_parser("list", help="list runs, newest first")
    l.add_argument("-n", type=int, default=20)
    l.set_defaults(fn=cmd_list)

    s = sub.add_parser("status", help="quick status of a run (default: latest)")
    s.add_argument("run", nargs="?")
    s.set_defaults(fn=cmd_status)

    m = sub.add_parser("summary", help="full summary: discoveries, book, survey")
    m.add_argument("run", nargs="?")
    m.set_defaults(fn=cmd_summary)

    v = sub.add_parser("survey", help="who answered which question, when")
    v.add_argument("run", nargs="?")
    v.set_defaults(fn=cmd_survey)

    c = sub.add_parser("screenshot", help="render a tick to PNG, print its path")
    c.add_argument("run", nargs="?")
    c.add_argument("--tick", type=int, default=None)
    c.add_argument("-o", "--out", default=None)
    c.set_defaults(fn=cmd_screenshot)

    d = sub.add_parser("models", help="show model chains and presets")
    d.set_defaults(fn=cmd_models)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
