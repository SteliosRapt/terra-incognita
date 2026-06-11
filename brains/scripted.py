"""scripted.py — rule-based brains. Zero LLM. Brains see ONLY the Observation."""
from __future__ import annotations
import random

from .base import Observation, PlanResult  # noqa: F401  (Observation re-exported)


class ForagerBrain:
    """Survive: eat when hungry, stock berries, wander otherwise."""
    def __init__(self, rng: random.Random):
        self.rng = rng

    def plan(self, obs: Observation) -> PlanResult:
        me = obs.me
        hungry = me["energy"] < 60
        has_food = me["inventory"].get("berries", 0) > 0

        if me["energy"] < 40 and has_food:
            return PlanResult([{"action": "eat", "item": "berries"}], "eat now")

        if hungry or me["inventory"].get("berries", 0) < 3:
            spots = obs.nearby_resources.get("berries", [])
            if spots:
                x, y = spots[0]
                return PlanResult([{"action": "move_to", "x": x, "y": y},
                                   {"action": "gather", "resource": "berries"}],
                                  "forage berries")

        # wander: deterministic via shared rng
        dx, dy = self.rng.randint(-4, 4), self.rng.randint(-4, 4)
        return PlanResult([{"action": "move_to", "x": me["pos"][0] + dx, "y": me["pos"][1] + dy},
                           {"action": "wait"}],
                          "wander")


class RandomScientistBrain(ForagerBrain):
    """Null-hypothesis baseline: survives like a forager, but when safe it
    stockpiles materials and tries RANDOM combinations. Quantities are weighted
    small (most recipes use 1-4 of a thing), else uniform sampling over
    quantities would make the baseline uselessly weak."""
    QTY_WEIGHTS = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]

    def plan(self, obs: Observation) -> PlanResult:
        me = obs.me
        has_food = me["inventory"].get("berries", 0) > 0

        # survival first (same as forager)
        if me["energy"] < 40 and has_food:
            return PlanResult([{"action": "eat", "item": "berries"}], "eat now")
        if me["energy"] < 60:
            spots = obs.nearby_resources.get("berries", [])
            if spots:
                x, y = spots[0]
                return PlanResult([{"action": "move_to", "x": x, "y": y},
                                   {"action": "gather", "resource": "berries"}],
                                  "forage berries")

        # safe (energy >= 60): science time
        inv = {k: v for k, v in me["inventory"].items() if v > 0 and k != "cooked_food"}
        materials = sum(inv.get(m, 0) for m in ("wood", "stone"))
        if inv and materials >= 2 and self.rng.random() < 0.5:
            n_types = 1 if (len(inv) == 1 or self.rng.random() < 0.4) else 2
            kinds = self.rng.sample(sorted(inv.keys()), n_types)
            items = {}
            for k in kinds:
                q = min(inv[k], self.rng.choice(self.QTY_WEIGHTS))
                items[k] = q
            return PlanResult([{"action": "combine", "items": items}],
                              f"experiment: {items}")

        # otherwise stockpile a random material
        want = self.rng.choice(["wood", "stone", "berries"])
        spots = obs.nearby_resources.get(want, [])
        if spots:
            x, y = spots[0]
            return PlanResult([{"action": "move_to", "x": x, "y": y},
                               {"action": "gather", "resource": want}],
                              f"stockpile {want}")
        dx, dy = self.rng.randint(-5, 5), self.rng.randint(-5, 5)
        return PlanResult([{"action": "move_to", "x": me["pos"][0] + dx, "y": me["pos"][1] + dy},
                           {"action": "wait"}], "explore")
