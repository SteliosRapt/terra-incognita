"""analyze_diffusiongemma.py — read the 2x2 shelter runs and report.

Per cell: survival, death ticks + evidence, combine attempts, brick discoveries,
shelters built, shelter occupancy, energy saved by shelters, and fire-
understanding evidence (construction / cooking / speech / book answers). Then
the Note and Book main effects and their interaction.

  py experiments/analyze_diffusiongemma.py                 # all cells under runs/
  py experiments/analyze_diffusiongemma.py --runs-dir runs --json out.json

ONE seed per cell: every effect below is exploratory, not statistically
conclusive.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.world import World
from experiments.diffusiongemma_matrix import load_matrix, build_configs


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
                break
    return out


def _spawn_positions(cfg: dict) -> dict[str, tuple[int, int]]:
    """Re-derive the agents' spawn tiles exactly as Simulation.__init__ does
    (same World + seeded rng order), so occupancy is correct before the first
    'moved' event of each agent."""
    rng = random.Random(cfg["seed"])
    w = World(cfg, rng)
    names = list(cfg.get("agent_names") or
                 (f"A{i:02d}" for i in range(cfg["n_agents"])))
    pos = {}
    for nm in names:
        while True:
            x, y = rng.randrange(w.W), rng.randrange(w.H)
            if w.tile(x, y).type == "grass":
                break
        pos[nm] = (x, y)
    return pos


def analyze_cell(run_dir: pathlib.Path) -> dict:
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    evs = read_events(run_dir)
    names = list(cfg.get("agent_names") or
                 (f"A{i:02d}" for i in range(cfg["n_agents"])))
    n = len(names)
    last_tick = evs[-1]["tick"] if evs else 0

    deaths = {e["actor"]: e["tick"] for e in evs if e["type"] == "agent_died"}
    combine_ok = [e for e in evs if e["type"] == "combine_succeeded"]
    combine_fail = [e for e in evs if e["type"] == "combine_failed"]
    shelter_limit_rejections = sum(
        1 for e in combine_fail if (e.get("payload") or {}).get("outcome") == "shelter_limit")

    # discoveries (first learn per recipe per agent)
    bricks = [e for e in evs if e["type"] == "recipe_learned"
              and e["payload"]["recipe"] == "brick"]
    shelter_recipe = [e for e in evs if e["type"] == "recipe_learned"
                      and e["payload"]["recipe"] == "shelter"]
    shelters_built = [e for e in evs if e["type"] == "structure_built"
                      and (e.get("payload") or {}).get("structure") == "shelter"]

    # ---- shelter occupancy + energy saved (exact: 1 energy per agent-tick on
    # a shelter at tick start, whether the agent then rests or leaves) ----
    pos = _spawn_positions(cfg)
    alive = {nm: True for nm in names}
    shelters: set[tuple[int, int]] = set()
    occupancy = 0          # agent-ticks spent standing on a shelter
    by_tick: dict[int, list[dict]] = {}
    for e in evs:
        by_tick.setdefault(e["tick"], []).append(e)
    for t in range(last_tick + 1):
        # count BEFORE applying this tick's events: positions are start-of-tick
        for nm in names:
            if alive[nm] and pos[nm] in shelters:
                occupancy += 1
        for e in by_tick.get(t, []):
            p = e.get("payload") or {}
            ep = tuple(e["position"]) if e.get("position") else None
            actor = e.get("actor")
            if actor in pos and ep:
                pos[actor] = ep
            if e["type"] == "structure_built" and p.get("structure") == "shelter" and ep:
                shelters.add(ep)
            elif e["type"] == "agent_died" and actor in alive:
                alive[actor] = False
    energy_saved = occupancy   # 1 energy saved per sheltered agent-tick

    # ---- fire understanding evidence ----
    fires_built = sum(1 for e in evs if e["type"] == "structure_built"
                      and (e.get("payload") or {}).get("structure") == "fire")
    cooked = sum(1 for e in combine_ok if e["payload"].get("recipe") == "cooked_food")
    fire_speech = sum(1 for e in evs if e["type"] == "said"
                      and "fire" in (e.get("payload") or {}).get("message", "").lower())
    fire_book_answers = [e for e in evs if e["type"] == "book_answer_written"
                         and (e.get("payload") or {}).get("question") == "recipe_fire"]

    return {
        "name": run_dir.name,
        "note": "on" if cfg.get("combine_note") else "off",
        "book": cfg.get("book_condition", "none"),
        "ticks": last_tick + 1,
        "n_agents": n,
        "survivors": n - len(deaths),
        "death_ticks": dict(sorted(deaths.items())),
        "combine_attempts": len(combine_ok) + len(combine_fail),
        "combine_succeeded": len(combine_ok),
        "combine_failed": len(combine_fail),
        "shelter_limit_rejections": shelter_limit_rejections,
        "brick_discoveries": len(bricks),
        "first_brick_tick": min((e["tick"] for e in bricks), default=None),
        "shelter_discoveries": len(shelter_recipe),
        "shelters_built": len(shelters_built),
        "shelter_owners": sorted({e["actor"] for e in shelters_built}),
        "shelter_occupancy_agentticks": occupancy,
        "energy_saved_by_shelters": energy_saved,
        "fire": {
            "built": fires_built,
            "cooked_food": cooked,
            "speech_mentions": fire_speech,
            "book_answers": len(fire_book_answers),
            "understood": bool(fires_built or cooked or fire_book_answers),
        },
    }


def _cell(cells: dict, note: str, book: str, metric) -> float:
    c = cells.get(f"dg_note-{note}_book-{book}")
    return float(metric(c)) if c else float("nan")


def main_effects(cells: dict, metric, label: str) -> str:
    off_none = _cell(cells, "off", "none", metric)
    off_lib = _cell(cells, "off", "librarian", metric)
    on_none = _cell(cells, "on", "none", metric)
    on_lib = _cell(cells, "on", "librarian", metric)
    note_eff = ((on_none + on_lib) - (off_none + off_lib)) / 2
    book_eff = ((off_lib + on_lib) - (off_none + on_none)) / 2
    interaction = (on_lib - off_lib) - (on_none - off_none)
    out = [f"  {label}:",
           f"            book=none  book=librarian",
           f"    note=off  {off_none:>8.2f}  {off_lib:>13.2f}",
           f"    note=on   {on_none:>8.2f}  {on_lib:>13.2f}",
           f"    Note main effect:  {note_eff:+.2f}    "
           f"Book main effect: {book_eff:+.2f}    "
           f"Interaction: {interaction:+.2f}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--json", default=None, help="also write the metrics as JSON")
    args = ap.parse_args()

    runs_dir = pathlib.Path(args.runs_dir)
    names = [c["name"] for c in build_configs(load_matrix())]
    cells = {}
    for nm in names:
        d = runs_dir / nm
        if (d / "config.json").exists() and (d / "events.jsonl").exists():
            cells[nm] = analyze_cell(d)

    if not cells:
        sys.exit(f"no matrix cells found under {runs_dir} "
                 f"(expected: {', '.join(names)})")

    for nm in names:
        c = cells.get(nm)
        if not c:
            print(f"== {nm} ==  (not run yet)\n")
            continue
        print(f"== {nm} == note={c['note']} book={c['book']} | "
              f"{c['ticks']} ticks")
        print(f"  survivors: {c['survivors']}/{c['n_agents']}  "
              f"deaths: {c['death_ticks'] or 'none'}")
        print(f"  combine: {c['combine_attempts']} attempts "
              f"({c['combine_succeeded']} ok, {c['combine_failed']} failed, "
              f"{c['shelter_limit_rejections']} shelter-cap rejections)")
        print(f"  bricks discovered: {c['brick_discoveries']} "
              f"(first t{c['first_brick_tick']})   "
              f"shelters built: {c['shelters_built']} by {c['shelter_owners'] or '-'}")
        print(f"  shelter occupancy: {c['shelter_occupancy_agentticks']} agent-ticks "
              f"(~{c['energy_saved_by_shelters']} energy saved)")
        f = c["fire"]
        print(f"  fire understanding: {'YES' if f['understood'] else 'no'} "
              f"(built {f['built']}, cooked {f['cooked_food']}, "
              f"said {f['speech_mentions']}, book {f['book_answers']})")
        print()

    print("== 2x2 main effects & interaction (one seed/cell — exploratory) ==")
    print(main_effects(cells, lambda c: c["survivors"], "survivors"))
    print(main_effects(cells, lambda c: c["shelters_built"], "shelters built"))
    print(main_effects(cells, lambda c: c["brick_discoveries"], "brick discoveries"))
    print(main_effects(cells, lambda c: c["energy_saved_by_shelters"],
                       "energy saved by shelters"))
    print("\nNOTE: a single seed per condition makes these effects exploratory, "
          "not statistically conclusive.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(cells, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
