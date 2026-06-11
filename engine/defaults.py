"""engine/defaults.py — the canonical run configuration, shared by the
milestone scripts and the viewer (one source of truth instead of four
copy-pasted dicts)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TECHTREE_PATH = ROOT / "configs" / "techtree_v0.json"

# world-generation parameters (every milestone uses these values)
WORLD = {
    "grid_size": 32,
    "forest_fraction": 0.15, "mountain_fraction": 0.08, "stone_per_tile": 12,
    "wood_regrow_ticks": 40, "berry_patches": 4, "berry_patch_size": 14,
    "berry_regrow_prob": 0.05, "berry_spoil_prob": 0.02,
    "water_fraction": 0.10, "lakes": 3, "forest_regions": 4, "mountain_ranges": 3,
}

# M3 base config: the LLM runs and the viewer start from this
BASE = dict(WORLD,
            techtree_path=str(TECHTREE_PATH),
            replan_interval=15,
            vis_radius=6, say_radius=3)

# recipe ids in techtree order — derived, so adding a recipe to the
# techtree automatically reaches the viewer and the run summaries
with open(TECHTREE_PATH, encoding="utf-8") as f:
    ALL_RECIPES = [r["id"] for r in json.load(f)["recipes"]]
