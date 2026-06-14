"""base.py — the Brain contract. A brain sees an Observation and returns a
PlanResult; nothing else crosses the mind/world boundary in either direction."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Observation:
    """Everything a mind may know. Built by sim.build_observation()."""
    tick: int
    me: dict                 # id, pos, energy, inventory, tools, standing_on, slots
    nearby_resources: dict   # resource -> list of positions (sorted near -> far)
    nearby_agents: list      # [{id, pos}]
    journal_tail: list       # last journal lines (engine-written English)
    nearby_structures: list = field(default_factory=list)  # [{type, pos, owner}]
    knowledge: list = field(default_factory=list)   # confirmed recipes, rendered to text
    beliefs: list = field(default_factory=list)     # mind-owned hypotheses
    notes: str = ""                                  # mind-owned scratchpad
    book: dict | None = None                         # book view (None = no book condition)
    world_size: tuple | None = None                  # (W, H); the map wraps at the edges
    replan_reason: str = ""                          # why the planner is consulted right now


@dataclass
class PlanResult:
    """What a brain hands back. The engine writes beliefs/notes back verbatim
    (it never reads them for logic) and queues the plan; every action is
    validated at execution time."""
    plan: list                              # queue of primitive action dicts
    intention: str = ""                     # short stated purpose (telemetry)
    updated_beliefs: list | None = None     # None = leave unchanged
    updated_notes: str | None = None        # None = leave unchanged
    parse_error: bool = False               # LLM output unusable; plan is the fallback
    planning_attempts: int = 1              # provider requests used for this plan
