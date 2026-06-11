"""techtree.py — reference implementation of recipe resolution.

The single trickiest contract in the engine: given (items the agent threw in,
the agent's context), decide what happens. Rules:

1. EXACT ingredient match required (quantities must equal the recipe exactly;
   throwing in 3 wood + 1 stone does NOT make an axe — v0 keeps this strict so
   experimentation is informative; relaxing to ">= with leftovers" is a knob).
2. Among matching ingredient sets, condition-specific recipes match FIRST
   (fire [1w+2s on grass] beats pickaxe [1w+2s anywhere] when on grass).
3. NEAR MISS = right ingredients, failed condition (drives 'medium' feedback).
4. Failure destroys ingredients * failure_cost (config knob).

Zero LLM dependencies. Fully testable headless.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field


@dataclass
class CombineContext:
    """Everything about the agent's situation that recipes may condition on."""
    tile_type: str                      # "grass" | "forest" | "mountain" | "dirt"
    tools: set[str] = field(default_factory=set)
    adjacent_structures: set[str] = field(default_factory=set)


@dataclass
class CombineResult:
    outcome: str                        # "success" | "near_miss" | "failure"
    recipe_id: str | None
    produced: dict | None               # the recipe's `produces` block
    consumed: dict                      # items actually destroyed
    feedback: str                       # message that enters the agent's journal


class TechTree:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.recipes = self.cfg["recipes"]
        self.failure_cost = self.cfg.get("failure_cost", 1.0)
        self.verbosity = self.cfg["failure_feedback"]["verbosity"]
        self.msgs = self.cfg["failure_feedback"]["messages"]

    # ---------- public API ----------

    def resolve(self, offered: dict[str, int], ctx: CombineContext) -> CombineResult:
        """offered: e.g. {"wood": 1, "stone": 2}. Engine guarantees agent owns these."""
        ingredient_matches = [r for r in self.recipes
                              if r["ingredients"] == offered]
        # Rule 2: most-conditional recipe first
        ingredient_matches.sort(key=lambda r: -len(r.get("conditions", {})))

        for r in ingredient_matches:
            ok, _missing = self._check_conditions(r.get("conditions", {}), ctx)
            if ok:
                return CombineResult("success", r["id"], r["produces"],
                                     consumed=offered,
                                     feedback=r["feedback_success"])

        consumed = {k: round(v * self.failure_cost) for k, v in offered.items()}
        if ingredient_matches:  # Rule 3: near miss
            r = ingredient_matches[0]
            _ok, missing = self._check_conditions(r.get("conditions", {}), ctx)
            return CombineResult("near_miss", None, None, consumed,
                                 feedback=self._fail_msg(missing))
        return CombineResult("failure", None, None, consumed,
                             feedback=self.msgs["low"])

    def describe(self, recipe_id: str) -> str:
        """One-line English rendering of a recipe, for prompts. Only ever shown
        for recipes the agent has CONFIRMED (or been granted by the knowledge
        condition) — this is the agent's own knowledge, not a leak."""
        r = next((r for r in self.recipes if r["id"] == recipe_id), None)
        if r is None:
            return recipe_id
        ing = " + ".join(f"{q} {i}" for i, q in sorted(r["ingredients"].items()))
        conds = r.get("conditions", {})
        extra = []
        if "tile_type" in conds:
            extra.append(f"while standing on {conds['tile_type']}")
        if "requires_tool" in conds:
            extra.append(f"requires {conds['requires_tool']}")
        if "adjacent_structure" in conds:
            extra.append(f"next to a {conds['adjacent_structure']}")
        s = f"{r['id']} = {ing}"
        if extra:
            s += " (" + ", ".join(extra) + ")"
        p = r["produces"]
        return s + f" -> {p['type']}: {p['item']}"

    # ---------- internals ----------

    def _check_conditions(self, conds: dict, ctx: CombineContext):
        missing = []
        if "tile_type" in conds and ctx.tile_type != conds["tile_type"]:
            missing.append(f"tile:{conds['tile_type']}")
        if "requires_tool" in conds and conds["requires_tool"] not in ctx.tools:
            missing.append(f"tool:{conds['requires_tool']}")
        if ("adjacent_structure" in conds
                and conds["adjacent_structure"] not in ctx.adjacent_structures):
            missing.append(f"near:{conds['adjacent_structure']}")
        return (len(missing) == 0), missing

    def _fail_msg(self, missing: list[str]) -> str:
        if self.verbosity == "high":
            return self.msgs["high"].format(missing_condition=", ".join(missing))
        if self.verbosity == "medium":
            return self.msgs["medium_near_miss"].format(
                hint="the materials seemed right, but something about this place was wrong")
        return self.msgs["low"]  # low: near miss indistinguishable from failure


# ---------------- smoke tests ----------------
if __name__ == "__main__":
    import os
    tt = TechTree(os.path.join(os.path.dirname(__file__), "..",
                               "configs", "techtree_v0.json"))

    cases = [
        # (description, offered, ctx, expected_outcome, expected_recipe)
        ("axe anywhere",        {"wood": 2, "stone": 1}, CombineContext("forest"), "success", "axe"),
        ("pickaxe on mountain", {"wood": 1, "stone": 2}, CombineContext("mountain"), "success", "pickaxe"),
        ("FIRE beats pickaxe on grass", {"wood": 1, "stone": 2}, CombineContext("grass"), "success", "fire"),
        ("farm w/o axe = near miss", {"wood": 4, "berries": 2}, CombineContext("grass"), "near_miss", None),
        ("farm with axe",       {"wood": 4, "berries": 2}, CombineContext("grass", tools={"axe"}), "success", "farm"),
        ("cook needs fire",     {"berries": 1}, CombineContext("dirt"), "near_miss", None),
        ("cook next to fire",   {"berries": 1}, CombineContext("dirt", adjacent_structures={"fire"}), "success", "cooked_food"),
        ("garbage in",          {"wood": 5, "berries": 5}, CombineContext("grass"), "failure", None),
        ("wrong quantities (3w+1s is NOT an axe)", {"wood": 3, "stone": 1}, CombineContext("grass"), "failure", None),
    ]

    all_ok = True
    for desc, offered, ctx, exp_out, exp_recipe in cases:
        res = tt.resolve(offered, ctx)
        ok = res.outcome == exp_out and res.recipe_id == exp_recipe
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {desc:45s} -> {res.outcome:9s} "
              f"{res.recipe_id or '-':12s} | {res.feedback}")
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
