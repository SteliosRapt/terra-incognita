"""run_m2.py — M2 acceptance: random-scientist baseline + determinism."""
import hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.sim import Simulation
from brains.scripted import RandomScientistBrain

BASE = {
    "grid_size": 32, "n_agents": 12,
    "forest_fraction": 0.15, "mountain_fraction": 0.08, "stone_per_tile": 12,
    "wood_regrow_ticks": 40, "berry_patches": 4, "berry_patch_size": 14,
    "berry_regrow_prob": 0.05, "berry_spoil_prob": 0.02,
    "water_fraction": 0.10, "lakes": 3, "forest_regions": 4, "mountain_ranges": 3,
    "techtree_path": str(ROOT / "configs" / "techtree_v0.json"),
}
TICKS = 1000
SEEDS = [42, 43, 44]


def run(seed, run_dir):
    cfg = dict(BASE, seed=seed)
    sim = Simulation(cfg, lambda rng: RandomScientistBrain(rng), run_dir)
    sim.run(TICKS)
    return sim


def stats(run_dir):
    disc, attempts, succ = {}, 0, 0
    deaths = 0
    for line in open(f"{run_dir}/events.jsonl"):
        ev = json.loads(line)
        if ev["type"] == "recipe_learned":
            r = ev["payload"]["recipe"]
            disc.setdefault(r, ev["tick"])
        elif ev["type"] == "combine_failed":
            attempts += 1
        elif ev["type"] == "combine_succeeded":
            attempts += 1; succ += 1
        elif ev["type"] == "agent_died":
            deaths += 1
    return disc, attempts, succ, deaths


if __name__ == "__main__":
    print(f"Random-scientist baseline: 12 agents x {TICKS} ticks x {len(SEEDS)} seeds\n")
    all_recipes = ["axe", "pickaxe", "basket", "fire", "farm", "cooked_food", "storehouse"]
    for seed in SEEDS:
        d = str(ROOT / "runs" / f"m2_s{seed}")
        run(seed, d)
        disc, att, suc, deaths = stats(d)
        line = " ".join(f"{r}@t{disc[r]}" if r in disc else f"{r}:—"
                        for r in all_recipes)
        print(f"seed {seed}: deaths {deaths:2d} | combines {att:4d} (success {suc}) | {line}")

    # determinism on seed 42
    run(42, str(ROOT / "runs" / "m2_det"))
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    same = (h(ROOT / "runs" / "m2_s42" / "events.jsonl")
            == h(ROOT / "runs" / "m2_det" / "events.jsonl"))
    print(f"\nDeterminism: {'PASS' if same else 'FAIL'}")
