"""run_m3.py — M3 acceptance: an LLM society survives a long run.

Examples:
  python run_m3.py --provider mock --ticks 300                 # pipeline smoke, $0
  python run_m3.py --provider ollama --model qwen2.5:7b \
                   --condition blind --agents 10 --ticks 1000  # the overnight run
  python run_m3.py --provider anthropic --model claude-haiku-4-5 --ticks 500
"""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.sim import Simulation
from brains.llm_planner import LLMPlannerBrain
from brains.providers import make_provider

BASE = {
    "grid_size": 32,
    "forest_fraction": 0.15, "mountain_fraction": 0.08, "stone_per_tile": 12,
    "wood_regrow_ticks": 40, "berry_patches": 4, "berry_patch_size": 14,
    "berry_regrow_prob": 0.05, "berry_spoil_prob": 0.02,
    "water_fraction": 0.10, "lakes": 3, "forest_regions": 4, "mountain_ranges": 3,
    "techtree_path": str(ROOT / "configs" / "techtree_v0.json"),
    "replan_interval": 15,
    "vis_radius": 6, "say_radius": 3,
}
ALL_RECIPES = ["axe", "pickaxe", "basket", "fire", "farm", "cooked_food", "storehouse"]


def summarize(run_dir: pathlib.Path) -> str:
    disc, taught, deaths, parse_errors, calls = {}, 0, 0, 0, 0
    for line in open(run_dir / "events.jsonl", encoding="utf-8"):
        ev = json.loads(line)
        if ev["type"] == "recipe_learned":
            disc.setdefault(ev["payload"]["recipe"], ev["tick"])
            if ev["payload"]["source"] == "taught":
                taught += 1
        elif ev["type"] == "agent_died":
            deaths += 1
        elif ev["type"] == "planner_parse_error":
            parse_errors += 1
        elif ev["type"] == "planner_called":
            calls += 1
    rate = (parse_errors / calls * 100) if calls else 0.0
    line = " ".join(f"{r}@t{disc[r]}" if r in disc else f"{r}:—" for r in ALL_RECIPES)
    return (f"planner calls {calls} | parse failures {parse_errors} ({rate:.1f}%) "
            f"| deaths {deaths} | taught recipes {taught}\n{line}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["mock", "ollama", "anthropic"], default="mock")
    p.add_argument("--model", default=None, help="backend model id")
    p.add_argument("--host", default=None, help="ollama host override")
    p.add_argument("--condition", default="blind",
                   choices=["blind", "oracle", "fragmented", "one_expert"])
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replan", type=int, default=15)
    p.add_argument("--name", default=None, help="run directory name under runs/")
    p.add_argument("--render-every", type=int, default=0)
    args = p.parse_args()

    cfg = dict(BASE, seed=args.seed, n_agents=args.agents,
               knowledge_condition=args.condition, replan_interval=args.replan)
    name = args.name or f"m3_{args.provider}_{args.condition}_s{args.seed}"
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = run_dir / "prompts.jsonl"
    telemetry.write_text("")   # fresh file per run

    spec = {"backend": args.provider}
    if args.model:
        spec["model"] = args.model
    if args.host and args.provider == "ollama":
        spec["host"] = args.host

    def factory(rng):
        return LLMPlannerBrain(make_provider(spec, rng),
                               telemetry_path=str(telemetry))

    sim = Simulation(cfg, factory, str(run_dir))
    print(f"run {name}: {args.agents} agents x {args.ticks} ticks, "
          f"{args.provider}{'/' + args.model if args.model else ''}, "
          f"condition={args.condition}, seed={args.seed}")
    for t in range(args.ticks):
        sim.step()
        if args.render_every and sim.tick % args.render_every == 0:
            print(sim.render_ascii())
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


if __name__ == "__main__":
    main()
