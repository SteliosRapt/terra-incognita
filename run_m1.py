"""run_m1.py — M1 acceptance: 12 foragers, 500 ticks, determinism check."""
import hashlib, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine.sim import Simulation
from brains.scripted import ForagerBrain

CFG = {
    "seed": 42,
    "grid_size": 32,
    "n_agents": 12,
    "forest_fraction": 0.15,
    "mountain_fraction": 0.08,
    "stone_per_tile": 12,
    "wood_regrow_ticks": 40,
    "berry_patches": 4,
    "berry_patch_size": 14,
    "berry_regrow_prob": 0.05,
    "berry_spoil_prob": 0.02,
    "water_fraction": 0.10,
    "lakes": 3,
    "forest_regions": 4,
    "mountain_ranges": 3,
}


def run(run_dir):
    sim = Simulation(CFG, lambda rng: ForagerBrain(rng), run_dir)
    sim.run(500)
    return sim


def log_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    sim1 = run(str(ROOT / "runs" / "m1_a"))
    print(sim1.render_ascii())
    alive = sum(a.alive for a in sim1.agents.values())
    print(f"\nSurvivors after 500 ticks: {alive}/12")
    for a in sorted(sim1.agents.values(), key=lambda a: a.id):
        status = f"energy {a.energy}" if a.alive else "DEAD"
        print(f"  {a.id}: {status:12s} inv={dict(a.inventory)}")

    # determinism acceptance test
    run(str(ROOT / "runs" / "m1_b"))
    h1 = log_hash(str(ROOT / "runs" / "m1_a" / "events.jsonl"))
    h2 = log_hash(str(ROOT / "runs" / "m1_b" / "events.jsonl"))
    print(f"\nDeterminism: {'PASS — identical logs' if h1 == h2 else 'FAIL'}")
    print(f"  sha256 {h1[:16]}…")
