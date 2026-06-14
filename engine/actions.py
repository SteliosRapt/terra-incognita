"""actions.py — THE only place World/Agent state mutates during the agent phase.
Every primitive returns (done, events). done=False means the primitive stays at
the head of the plan (multi-tick actions like move_to)."""
from __future__ import annotations
from .events import Event
from .agent import AgentState
from .world import World, Structure, FOREST, MOUNTAIN
from .techtree import CombineContext

ENERGY_MOVE = 2
ENERGY_IDLE = 1
FOOD_ENERGY = {"berries": 25, "cooked_food": 50}
ENERGY_CAP = 100


def apply(action: dict, ag: AgentState, world: World, tick: int,
          agents: dict[str, AgentState] | None = None):
    kind = action.get("action")
    fn = _HANDLERS.get(kind)
    if fn is None:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"unknown action {kind}"})]
    if kind in ("say", "give"):
        return fn(action, ag, world, tick, agents or {})
    return fn(action, ag, world, tick)


def _move_to(a, ag, world, tick):
    tx, ty = a["x"] % world.W, a["y"] % world.H
    if ag.pos == (tx, ty):
        return True, [Event(tick, "moved", actor=ag.id, position=ag.pos)]
    nxt = world.next_step(ag.pos, (tx, ty))
    if nxt is None:
        return True, [Event(tick, "plan_interrupted", actor=ag.id, position=ag.pos,
                            payload={"reason": "destination unreachable"})]
    ag.x, ag.y = nxt
    ag.energy -= (ENERGY_MOVE - ENERGY_IDLE)  # idle cost applied globally; top-up here
    if ag.pos == (tx, ty):
        return True, [Event(tick, "moved", actor=ag.id, position=ag.pos)]
    return False, []  # still traveling; no event spam


def _tool_mult(ag, world, resource: int) -> int:
    """Yield multiplier from equipped tools, read from the tech tree effects."""
    m = 1
    if world.techtree:
        for r in world.techtree.recipes:
            item = r["produces"].get("item")
            if item in ag.tools:
                gm = r.get("effects", {}).get("gather_multiplier", {})
                m = max(m, gm.get(resource, 1))
    return m


def _gather(a, ag, world, tick):
    res = a["resource"]
    # optional target: walk there first (Tier B pathfinding, one step per tick).
    # This makes "gather wood from (7, 27)" a single multi-tick primitive —
    # the shape every model assumes gather has (measured in m3 runs: 430/480
    # plans started with an untargeted gather on a bare tile and died).
    if a.get("x") is not None and a.get("y") is not None:
        tx, ty = a["x"] % world.W, a["y"] % world.H
        if ag.pos != (tx, ty):
            nxt = world.next_step(ag.pos, (tx, ty))
            if nxt is None:
                return True, [Event(tick, "plan_interrupted", actor=ag.id,
                                    position=ag.pos,
                                    payload={"reason": f"gather: ({tx}, {ty}) unreachable"})]
            ag.x, ag.y = nxt
            ag.energy -= (ENERGY_MOVE - ENERGY_IDLE)
            return False, []  # still traveling
    t = world.tile(ag.x, ag.y)
    evs = []
    got = 0
    if res == "berries" and t.berries > 0:
        take = min(t.berries, 1)          # berries: 1 per action regardless of tools
        got = ag.add("berries", take)
        if got:
            t.berries -= got
    elif res == "wood" and t.type == FOREST and t.wood > 0:
        got = ag.add("wood", _tool_mult(ag, world, "wood"))   # axe: 3 per tree
        if got:
            t.wood = 0
            t.wood_regrow_at = tick + world.cfg["wood_regrow_ticks"]
            evs.append(Event(tick, "resource_depleted", position=ag.pos,
                             payload={"resource": "wood"}))
    elif res == "stone" and t.type == MOUNTAIN and t.stone > 0:
        take = min(t.stone, _tool_mult(ag, world, "stone"))
        got = ag.add("stone", take)
        if got:
            t.stone -= got
            if t.stone == 0:
                evs.append(Event(tick, "resource_depleted", position=ag.pos,
                                 payload={"resource": "stone"}))
    if got:
        evs.insert(0, Event(tick, "gathered", actor=ag.id, position=ag.pos,
                            payload={"resource": res, "qty": got}))
        return True, evs
    return True, [Event(tick, "plan_interrupted", actor=ag.id, position=ag.pos,
                        payload={"reason": f"no {res} here or inventory full"})]


def _eat(a, ag, world, tick):
    item = a.get("item", "berries")
    if item in FOOD_ENERGY and ag.remove(item, 1):
        gain = FOOD_ENERGY[item]
        ag.energy = min(ENERGY_CAP, ag.energy + gain)
        return True, [Event(tick, "ate", actor=ag.id, position=ag.pos,
                            payload={"item": item, "energy": gain})]
    return True, [Event(tick, "plan_interrupted", actor=ag.id,
                        payload={"reason": f"no {item} to eat"})]


def _drop(a, ag, world, tick):
    item, qty = a["item"], a.get("qty", 1)
    if ag.remove(item, qty):
        # M1: dropped items vanish (ground storage arrives with structures in M2)
        return True, [Event(tick, "dropped", actor=ag.id, position=ag.pos,
                            payload={"item": item, "qty": qty})]
    return True, [Event(tick, "plan_interrupted", actor=ag.id,
                        payload={"reason": f"not enough {item}"})]


def _combine(a, ag, world, tick):
    offered = {k: int(v) for k, v in (a.get("items") or {}).items() if int(v) > 0}
    if not offered:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "combine: no items offered"})]
    if not all(ag.inventory.get(i, 0) >= q for i, q in offered.items()):
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "combine: you lack those materials"})]
    ctx = CombineContext(tile_type=world.tile(ag.x, ag.y).type,
                         tools=set(ag.tools),
                         adjacent_structures=world.adjacent_structures(ag.pos))
    res = world.techtree.resolve(offered, ctx)
    # one shelter per builder: reject the excess attempt BEFORE consuming
    # anything — the bricks stay in the pack (spec: retained materials).
    if (res.outcome == "success" and res.produced
            and res.produced.get("type") == "structure"
            and res.produced.get("item") == "shelter" and ag.shelter_built):
        return True, [Event(tick, "combine_failed", actor=ag.id, position=ag.pos,
                            payload={"ingredients": offered, "outcome": "shelter_limit",
                                     "feedback": "You have already built your shelter; "
                                                 "the bricks stay in your pack."})]
    for i, q in res.consumed.items():
        ag.remove(i, min(q, ag.inventory.get(i, 0)))
    if res.outcome != "success":
        return True, [Event(tick, "combine_failed", actor=ag.id, position=ag.pos,
                            payload={"ingredients": offered, "outcome": res.outcome,
                                     "feedback": res.feedback})]
    evs = []
    p = res.produced
    placed = True
    if p["type"] == "tool":
        ag.tools.add(p["item"])
        evs.append(Event(tick, "tool_equipped", actor=ag.id,
                         payload={"tool": p["item"]}))
    elif p["type"] == "item":
        ag.add(p["item"], 1)
    elif p["type"] == "structure":
        if ag.pos in world.structures:
            placed = False
        else:
            exp = tick + p["lifetime_ticks"] if "lifetime_ticks" in p else -1
            world.structures[ag.pos] = Structure(p["item"], ag.id, tick, exp)
            if p["item"] == "shelter":
                ag.shelter_built = True
            evs.append(Event(tick, "structure_built", actor=ag.id, position=ag.pos,
                             payload={"structure": p["item"]}))
    if not placed:
        return True, [Event(tick, "combine_failed", actor=ag.id, position=ag.pos,
                            payload={"ingredients": offered, "outcome": "no_room",
                                     "feedback": "It would have worked, but there is no room here."})] + evs
    if res.recipe_id not in ag.knowledge:
        ag.knowledge.add(res.recipe_id)
        # taught = someone told this agent about the recipe before it first
        # crafted it (heard_recipes is engine-tracked from said events)
        source = "taught" if res.recipe_id in ag.heard_recipes else "discovered"
        evs.append(Event(tick, "recipe_learned", actor=ag.id,
                         payload={"recipe": res.recipe_id, "source": source}))
    evs.insert(0, Event(tick, "combine_succeeded", actor=ag.id, position=ag.pos,
                        payload={"recipe": res.recipe_id, "ingredients": offered,
                                 "produced": p["item"], "feedback": res.feedback}))
    return True, evs


def _deposit(a, ag, world, tick):
    st = world.structures.get(ag.pos)
    item, qty = a["item"], int(a.get("qty", 1))
    if not st or st.type != "storehouse":
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "no storehouse here"})]
    qty = min(qty, ag.inventory.get(item, 0), Structure.STORE_UNITS - st.units())
    if qty <= 0:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "nothing to deposit or storehouse full"})]
    ag.remove(item, qty)
    st.storage[item] = st.storage.get(item, 0) + qty
    return True, [Event(tick, "storehouse_deposit", actor=ag.id, position=ag.pos,
                        payload={"item": item, "qty": qty})]


def _withdraw(a, ag, world, tick):
    st = world.structures.get(ag.pos)
    item, qty = a["item"], int(a.get("qty", 1))
    if not st or st.type != "storehouse":
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "no storehouse here"})]
    qty = min(qty, st.storage.get(item, 0))
    if qty <= 0:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"no {item} stored here"})]
    added = ag.add(item, qty)
    st.storage[item] -= added
    if st.storage[item] == 0:
        del st.storage[item]
    return True, [Event(tick, "storehouse_withdraw", actor=ag.id, position=ag.pos,
                        payload={"item": item, "qty": added})]


def _say(a, ag, world, tick, agents):
    """Free-form speech. The engine NEVER validates content (lying is allowed
    and measurable). Heard within radius 3 (sim handles fan-out). Speech is a
    normal primitive: it occupies the agent's action for the tick (decided —
    costed speech, spec open question 4)."""
    msg = str(a.get("message", ""))[:300]
    if not msg:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "say: nothing to say"})]
    target = a.get("target") or None
    if target is not None:
        target = str(target)
        if target not in agents or not agents[target].alive:
            target = None   # talking to nobody in particular still makes sound
    return True, [Event(tick, "said", actor=ag.id, target=target,
                        position=ag.pos, payload={"message": msg})]


def _give(a, ag, world, tick, agents):
    target_id, item = a.get("target"), a.get("item")
    qty = int(a.get("qty", 1))
    tgt = agents.get(target_id)
    if tgt is None or not tgt.alive:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"give: no agent {target_id} here"})]
    if world.torus_dist(ag.pos, tgt.pos) > 1:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"give: {target_id} is too far away"})]
    qty = min(qty, ag.inventory.get(item, 0))
    if qty <= 0:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"give: no {item} to give"})]
    given = tgt.add(item, qty)   # capacity-limited; leftovers stay with giver
    if given == 0:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"give: {target_id} cannot carry more"})]
    ag.remove(item, given)
    return True, [Event(tick, "gave", actor=ag.id, target=target_id,
                        position=ag.pos, payload={"item": item, "qty": given})]


BOOK_NOTE_CAP = 400
BOOK_ANSWER_CAP = 300


def _book_note(a, ag, world, tick):
    """Append a permanent, timestamped entry to the book (never editable)."""
    book, writable, reason = world.book_access(ag.id, ag.pos)
    if book is None or not writable:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": reason or "you may not write in the book"})]
    text = str(a.get("text", "")).strip()[:BOOK_NOTE_CAP]
    if not text:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "book_note: nothing to write"})]
    book.entries.append({"tick": tick, "author": ag.id, "text": text})
    return True, [Event(tick, "book_note_written", actor=ag.id, position=ag.pos,
                        payload={"text": text})]


def _book_answer(a, ag, world, tick):
    """Set/overwrite the book's answer to one question (editable, last write
    wins; author and tick recorded)."""
    book, writable, reason = world.book_access(ag.id, ag.pos)
    if book is None or not writable:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": reason or "you may not write in the book"})]
    qid = str(a.get("question", "")).strip()
    if not any(q["id"] == qid for q in book.questions):
        ids = ", ".join(q["id"] for q in book.questions)
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": f"book_answer: no question '{qid}' "
                                               f"(questions: {ids})"})]
    text = str(a.get("answer", "")).strip()[:BOOK_ANSWER_CAP]
    if not text:
        return True, [Event(tick, "plan_interrupted", actor=ag.id,
                            payload={"reason": "book_answer: empty answer"})]
    prev = book.answers.get(qid)
    book.answers[qid] = {"text": text, "author": ag.id, "tick": tick}
    return True, [Event(tick, "book_answer_written", actor=ag.id, position=ag.pos,
                        payload={"question": qid, "text": text,
                                 "replaced": (prev or {}).get("author")})]


def _wait(a, ag, world, tick):
    return True, []


_HANDLERS = {"move_to": _move_to, "gather": _gather, "eat": _eat,
             "drop": _drop, "wait": _wait, "combine": _combine,
             "deposit": _deposit, "withdraw": _withdraw,
             "say": _say, "give": _give,
             "book_note": _book_note, "book_answer": _book_answer}
