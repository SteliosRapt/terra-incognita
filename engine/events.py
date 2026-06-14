"""events.py — Event records: the ground truth of every run."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Event:
    tick: int
    type: str
    actor: str | None = None          # agent id or None for world events
    target: str | None = None         # other agent / structure id
    payload: dict = field(default_factory=dict)
    position: tuple[int, int] | None = None
    visibility: list[str] = field(default_factory=list)  # agent ids; filled by sim

    def to_json(self) -> str:
        d = asdict(self)
        d["position"] = list(self.position) if self.position else None
        return json.dumps(d, ensure_ascii=False)


def _an(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def render_for_journal(ev: Event, viewer_id: str) -> str:
    """One-line English rendering of an event for an agent's journal."""
    you = ev.actor == viewer_id
    who = "You" if you else ev.actor
    p = ev.payload
    t = ev.type
    if t == "gathered":
        return f"{who} gathered {p['qty']} {p['resource']} at {ev.position}."
    if t == "ate":
        return f"{who} ate {p['item']} (+{p['energy']} energy)."
    if t == "moved":
        return f"{who} arrived at {ev.position}."
    if t == "dropped":
        return f"{who} dropped {p['qty']} {p['item']} at {ev.position}."
    if t == "picked_up":
        return f"{who} picked up {p['qty']} {p['item']}."
    if t == "combine_succeeded":
        if you:
            ing = ", ".join(f"{q} {i}" for i, q in p["ingredients"].items())
            return f"You combined {ing}. {p['feedback']}"
        return f"{who} made something: {_an(p['produced'])}."
    if t == "combine_failed":
        if you:
            return f"Your experiment failed. {p['feedback']}"
        return f"{who} fiddled with some materials, but nothing came of it."
    if t == "recipe_learned":
        return f"You now know how to make: {p['recipe']}."
    if t == "said":
        msg = p["message"]
        if ev.target is None:
            return f'{who} said: "{msg}"'
        to = "you" if ev.target == viewer_id else ev.target
        return f'{who} said to {to}: "{msg}"'
    if t == "gave":
        if ev.target == viewer_id:
            return f"{who} gave you {p['qty']} {p['item']}."
        to = ev.target
        return f"{who} gave {p['qty']} {p['item']} to {to}."
    if t == "tool_equipped":
        return f"{who} {'are' if you else 'is'} now carrying {_an(p['tool'])}."
    if t == "structure_built":
        return f"{who} built a {p['structure']} at {ev.position}."
    if t == "structure_expired":
        return f"The {p['structure']} at {ev.position} is gone."
    if t == "farm_yielded":
        return f"The farm at {ev.position} produced {p['amount']} {p['item']}."
    if t == "storehouse_deposit":
        return f"{who} stored {p['qty']} {p['item']} in the storehouse."
    if t == "storehouse_withdraw":
        return f"{who} took {p['qty']} {p['item']} from the storehouse."
    if t == "book_note_written":
        if you:
            return f'You wrote in the book: "{p["text"]}"'
        return f"{who} wrote a note in the book."
    if t == "book_answer_written":
        if you:
            return f"You recorded the book's answer to '{p['question']}'."
        return f"{who} updated the book's answer to '{p['question']}'."
    if t == "spoiled":
        return f"{p['qty']} of your berries rotted away."
    if t == "plan_interrupted":
        return f"Your plan stopped: {p.get('reason', 'unknown')}."
    if t == "energy_critical":
        return f"{who} look{'s' if not you else ''} exhausted and weak."
    if t == "agent_died":
        return f"{who} collapsed and died at {ev.position}."
    if t == "resource_depleted":
        return f"The {p['resource']} at {ev.position} is exhausted."
    if t == "resource_regrown":
        return f"New {p['resource']} appeared at {ev.position}."
    return f"{who}: {t} {p}"
