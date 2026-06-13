"""agent.py — agent state. The engine owns STATUS fields; the mind owns
beliefs/notes. Journal is engine-written, mind-readable."""
from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class AgentState:
    id: str
    x: int
    y: int
    energy: int = 100
    inventory: dict = field(default_factory=dict)   # item -> qty
    tools: set = field(default_factory=set)
    knowledge: set = field(default_factory=set)     # confirmed recipe ids
    beliefs: list = field(default_factory=list)     # mind-owned
    notes: str = ""                                  # mind-owned
    journal: list = field(default_factory=list)     # engine-written strings
    alive: bool = True
    # executor state
    plan: list = field(default_factory=list)        # queue of primitive actions
    intention: str = ""                              # last plan's stated purpose
    last_plan_tick: int = -1                         # when the planner last ran
    replan_flag: bool = False                        # set when spoken to (directed say)
    interrupted: bool = False                        # last plan died on a failed step
    reflex_fired: bool = False                       # survival reflex replaced the plan
    # measurement metadata (engine-written, never used for world logic):
    # recipe ids mentioned in speech this agent has heard — lets the engine tag
    # recipe_learned with source=taught vs discovered without reading the mind
    heard_recipes: set = field(default_factory=set)

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    # inventory: capacity in SLOTS; 1 slot holds up to 5 units of one item type
    BASE_SLOTS = 8
    UNITS_PER_SLOT = 5

    def slots_used(self) -> int:
        return sum(math.ceil(q / self.UNITS_PER_SLOT)
                   for q in self.inventory.values() if q > 0)

    def slot_capacity(self) -> int:
        bonus = 8 if "basket" in self.tools else 0   # basket effect (M2 wires from tree)
        return self.BASE_SLOTS + bonus

    def can_add(self, item: str, qty: int) -> bool:
        inv = dict(self.inventory)
        inv[item] = inv.get(item, 0) + qty
        used = sum(math.ceil(q / self.UNITS_PER_SLOT) for q in inv.values() if q > 0)
        return used <= self.slot_capacity()

    def add(self, item: str, qty: int) -> int:
        """Add up to qty respecting capacity; returns qty actually added."""
        added = 0
        for _ in range(qty):
            if self.can_add(item, 1):
                self.inventory[item] = self.inventory.get(item, 0) + 1
                added += 1
            else:
                break
        return added

    def remove(self, item: str, qty: int) -> bool:
        if self.inventory.get(item, 0) < qty:
            return False
        self.inventory[item] -= qty
        if self.inventory[item] == 0:
            del self.inventory[item]
        return True

    def journal_add(self, line: str, cap: int = 40):
        self.journal.append(line)
        if len(self.journal) > cap:
            self.journal[:] = self.journal[-cap:]
