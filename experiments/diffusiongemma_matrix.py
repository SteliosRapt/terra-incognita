"""diffusiongemma_matrix.py — the Note x Book 2x2 shelter experiment.

Reads configs/experiments/diffusiongemma_2x2.json and builds EXACTLY four run
configs (Note off/on x Book none/librarian), each: 7 agents, 1000 ticks, seed
42, blind knowledge, replan 15, surveys off. Every cell is planned by ONE
shared DiffusionGemma batching brain (so each tick is a single batched
generate); per-cell outputs are the normal config.json / events.jsonl /
prompts.jsonl / ground_truth.json plus model_metrics.jsonl.

One model load serves the whole matrix. Completed cells are skipped on rerun.

  py experiments/diffusiongemma_matrix.py --dry-run        # print the 4 configs
  py experiments/diffusiongemma_matrix.py --only dg_note-on_book-librarian
  py experiments/diffusiongemma_matrix.py                  # full matrix (needs GPU)
  py experiments/diffusiongemma_matrix.py --ticks 20 --provider mock   # local smoke
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.defaults import BASE
from engine.sim import Simulation
from engine.questions import load_questions, ground_truth

MATRIX_CONFIG = ROOT / "configs" / "experiments" / "diffusiongemma_2x2.json"
RUNS = ROOT / "runs"


# ---------------------------------------------------------------- config build
def load_matrix(path: pathlib.Path | str = MATRIX_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_configs(matrix: dict) -> list[dict]:
    """The 2x2 expanded to exactly four cells. Each entry:
    {"name", "cfg", "ticks", "note", "book"}. Pure (no model, no I/O) so the
    config count is testable without weights."""
    shared = matrix["shared"]
    out = []
    for note in matrix["axes"]["combine_note"]:          # ["off", "on"]
        for book in matrix["axes"]["book_condition"]:    # ["none", "librarian"]
            cfg = dict(
                BASE,
                seed=shared["seed"],
                n_agents=shared["n_agents"],
                agent_names=list(shared["agent_names"]),
                knowledge_condition=shared["knowledge_condition"],
                replan_interval=shared["replan_interval"],
                book_condition=book,
                combine_note=(note == "on"),
                freeform_thoughts=False,
            )
            if book == "librarian":
                cfg["book_manager"] = matrix["book_manager"]
            out.append({"name": f"dg_note-{note}_book-{book}", "cfg": cfg,
                        "ticks": shared["ticks"], "note": note, "book": book})
    return out


# ---------------------------------------------------------------- run helpers
def _last_tick(run_dir: pathlib.Path) -> int | None:
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        return None
    with open(ev, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))
        tail = f.read().decode("utf-8", "replace").strip().splitlines()
    for line in reversed(tail):
        try:
            return json.loads(line)["tick"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def is_complete(run_dir: pathlib.Path, ticks: int) -> bool:
    last = _last_tick(run_dir)
    return last is not None and last >= ticks - 1


def run_condition(cond: dict, provider, runs_dir: pathlib.Path,
                  ticks: int | None = None) -> pathlib.Path:
    """Run one cell with the shared `provider` (one batching brain for all
    agents). Returns the run dir. Skips if already complete."""
    from brains.llm_planner import BatchLLMPlannerBrain

    cfg = dict(cond["cfg"])
    ticks = ticks or cond["ticks"]
    run_dir = runs_dir / cond["name"]
    if is_complete(run_dir, ticks):
        print(f"  skip {cond['name']}: already complete at t{_last_tick(run_dir)}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = run_dir / "prompts.jsonl"
    metrics = run_dir / "model_metrics.jsonl"
    telemetry.write_text("", encoding="utf-8")
    metrics.write_text("", encoding="utf-8")

    # one brain instance for every agent -> sim plans the tick in one batch
    brain = BatchLLMPlannerBrain(provider, telemetry_path=str(telemetry),
                                 metrics_path=str(metrics),
                                 combine_note=cfg["combine_note"])
    sim = Simulation(cfg, lambda rng: brain, str(run_dir))

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if sim.world.techtree:
        questions = load_questions(cfg.get("questions_path"), sim.world.techtree)
        truth = ground_truth(questions, cfg, sim.world, sim.world.techtree)
        with open(run_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump({"questions": questions, "truth": truth}, f, indent=2,
                      ensure_ascii=False)

    print(f"  run {cond['name']}: {cfg['n_agents']} agents x {ticks} ticks "
          f"(note={cond['note']}, book={cond['book']})", flush=True)
    for _ in range(ticks):
        sim.step()
        sim.logf.flush()
        if sim.tick % 100 == 0:
            alive = sum(a.alive for a in sim.agents.values())
            print(f"    t{sim.tick}: alive {alive}/{cfg['n_agents']}", flush=True)
        if not any(a.alive for a in sim.agents.values()):
            print(f"    population extinct at t{sim.tick}", flush=True)
            break
    sim.logf.close()
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return run_dir


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(MATRIX_CONFIG))
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--only", default=None, help="run only this cell name")
    ap.add_argument("--ticks", type=int, default=None, help="override tick count")
    ap.add_argument("--provider", default=None,
                    help="override backend (e.g. 'mock' for a local smoke run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the four configs and exit (no model load)")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    conds = build_configs(matrix)
    if args.only:
        conds = [c for c in conds if c["name"] == args.only]
        if not conds:
            sys.exit(f"no cell named {args.only!r}")

    if args.dry_run:
        print(f"{matrix['experiment']}: {len(build_configs(matrix))} cells")
        for c in conds:
            print(f"  {c['name']}: note={c['note']} book={c['book']} "
                  f"ticks={args.ticks or c['ticks']} "
                  f"manager={c['cfg'].get('book_manager', '-')}")
        return

    from brains.providers import make_provider
    spec = dict(matrix["provider"])
    if args.provider:
        spec["backend"] = args.provider
    provider = make_provider(spec)     # one load for the whole matrix
    runs_dir = pathlib.Path(args.runs_dir)
    print(f"{matrix['experiment']}: {len(conds)} cell(s) | provider {provider.name}")
    for c in conds:
        run_condition(c, provider, runs_dir, ticks=args.ticks)
    print("matrix complete.")


if __name__ == "__main__":
    main()
