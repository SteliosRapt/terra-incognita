"""run_m3.py — M3 acceptance: an LLM society survives a long run.

Examples:
  python run_m3.py --provider mock --ticks 300                 # pipeline smoke, $0
  python run_m3.py --provider ollama --model qwen2.5:7b \
                   --condition blind --agents 10 --ticks 1000  # the overnight run
  python run_m3.py --provider anthropic --model claude-haiku-4-5 --ticks 500
  python run_m3.py --provider openrouter --ticks 500   # chain from configs/openrouter.json
  python run_m3.py --provider openrouter \
                   --model "meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-120b:free"
"""
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.sim import Simulation
from engine.defaults import BASE, ALL_RECIPES
from brains.llm_planner import LLMPlannerBrain
from brains.providers import make_provider


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
    p.add_argument("--provider", choices=["mock", "ollama", "anthropic", "openrouter"],
                   default="mock")
    p.add_argument("--model", default=None,
                   help="backend model id (openrouter: comma list = fallback chain)")
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
    if args.provider == "openrouter":
        spec["cost_path"] = str(run_dir / "costs.jsonl")

    # ONE provider shared by all brains: rate-limit cooldowns and the spend
    # ledger must be global, not per-agent.
    shared = {}

    def factory(rng):
        if "provider" not in shared:
            shared["provider"] = make_provider(spec, rng)
        return LLMPlannerBrain(shared["provider"], telemetry_path=str(telemetry))

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
    totals = getattr(shared.get("provider"), "totals", None)
    if totals:
        print(f"spend: {totals['calls']} calls | "
              f"{totals['prompt_tokens']} in / {totals['completion_tokens']} out tokens | "
              f"${totals['cost_usd']:.4f} | rate-limit hits {totals['rate_limited']}")


if __name__ == "__main__":
    main()
