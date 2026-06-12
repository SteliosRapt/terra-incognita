"""questions.py — the epistemics questionnaire (notes: 'will they successfully
identify how the world works?').

Two halves, both engine-clean (no LLM imports):
- load_questions(): the question list agents may see (book conditions write
  these into the book; the survey judge grades against them in every condition).
- ground_truth(): the true answers, derived from the run config, the GENERATED
  map (coverage is measured, not nominal) and the techtree. Written to the run
  dir for the judge and the analysis — never into any prompt an agent sees.
"""
from __future__ import annotations
import json
import pathlib

from .actions import ENERGY_IDLE, ENERGY_MOVE
from .world import World, FOREST, MOUNTAIN, WATER
from .techtree import TechTree

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "questions_v0.json"


def load_questions(path: str | None = None,
                   techtree: TechTree | None = None) -> list[dict]:
    """[{id, text}, ...]: the static questions plus one recipe question per
    techtree node (so swapping the tree swaps the questionnaire with it)."""
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    out = [{"id": q["id"], "text": q["text"]} for q in cfg["static"]]
    if techtree is not None:
        tmpl = cfg["recipe_template"]
        for r in techtree.recipes:
            out.append({"id": f"recipe_{r['id']}",
                        "text": tmpl.format(item=r["produces"]["item"])})
    return out


def _measured_coverage(world: World) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in world.grid:
        for t in row:
            counts[t.type] = counts.get(t.type, 0) + 1
    area = world.W * world.H
    return {k: v / area for k, v in counts.items()}


def ground_truth(questions: list[dict], cfg: dict, world: World,
                 techtree: TechTree | None) -> dict[str, str]:
    """qid -> true answer, rendered as text for the survey judge."""
    cov = _measured_coverage(world)
    p = cfg.get("berry_regrow_prob", 0.0)
    truths: dict[str, str] = {
        "time_energy": (
            "Time passes in discrete ticks; every agent acts once per tick. "
            f"Energy drains exactly {ENERGY_IDLE} per tick when idle and "
            f"{ENERGY_MOVE} on ticks spent moving; at 0 energy you die "
            "permanently."),
        "regrow_berries": (
            f"Yes, but only on berry-patch tiles: each empty patch tile "
            f"regrows with {p:.0%} probability per tick "
            f"(about one berry every {round(1 / p)} ticks)."
            if p > 0 else "No - berries never grow back."),
        "regrow_wood": (
            f"Yes - a harvested forest tile regrows its tree after exactly "
            f"{cfg['wood_regrow_ticks']} ticks."),
        "regrow_stone": (
            f"No - stone is finite. Each mountain tile starts with "
            f"{cfg['stone_per_tile']} stone and never replenishes."),
        "coverage": (
            f"Measured on this map: forest {cov.get(FOREST, 0):.0%}, "
            f"mountain {cov.get(MOUNTAIN, 0):.0%}, water {cov.get(WATER, 0):.0%} "
            "(the rest is grass). Answers within 5 percentage points per "
            "terrain count as correct."),
    }
    if techtree is not None:
        cost = techtree.failure_cost
        truths["combine_loss"] = (
            f"{cost:.0%} of the offered ingredients are destroyed on a failed "
            "combine attempt." + (" Nothing is ever refunded."
                                  if cost >= 1.0 else ""))
        for r in techtree.recipes:
            truths[f"recipe_{r['id']}"] = techtree.describe(r["id"])
    return {q["id"]: truths[q["id"]] for q in questions if q["id"] in truths}
