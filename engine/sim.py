"""sim.py — tick loop per spec §3c: WORLD PHASE → AGENT PHASE → COMMIT."""
from __future__ import annotations
import json, os, random, re
from .world import World, Structure, Book, FOREST, MOUNTAIN
from .techtree import TechTree
from .questions import load_questions
from .agent import AgentState
from .events import Event, render_for_journal
from . import actions
from brains.base import Observation, PlanResult

REFLEX_ENERGY = 30
CRITICAL_ENERGY = 20


class Simulation:
    def __init__(self, cfg: dict, brain_factory, run_dir: str):
        self.cfg = cfg
        self.vis_radius = cfg.get("vis_radius", 6)
        self.say_radius = cfg.get("say_radius", 3)     # speech is local (spec §3b)
        self.replan_interval = cfg.get("replan_interval", 15)
        self.rng = random.Random(cfg["seed"])
        self.world = World(cfg, self.rng)
        if "techtree_path" in cfg:
            self.world.techtree = TechTree(cfg["techtree_path"])
        self.tick = 0
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.logf = open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8")
        # spawn agents on grass. Names ARE the ids: they flow into prompts,
        # speech targets and the event log, so named agents genuinely know
        # themselves and each other (cfg["agent_names"], optional).
        self.agents: dict[str, AgentState] = {}
        names = list(cfg.get("agent_names") or
                     (f"A{i:02d}" for i in range(cfg["n_agents"])))
        if len(names) != cfg["n_agents"] or len(set(names)) != len(names):
            raise ValueError("agent_names must be unique and match n_agents")
        for name in names:
            while True:
                x, y = self.rng.randrange(self.world.W), self.rng.randrange(self.world.H)
                if self.world.tile(x, y).type == "grass":
                    break
            self.agents[name] = AgentState(id=name, x=x, y=y)
        self._inject_knowledge(cfg.get("knowledge_condition", "blind"), names)
        self._init_book(names)
        self.brains = {name: brain_factory(self.rng) for name in names}
        self._was_critical: set[str] = set()

    def _init_book(self, names: list[str]):
        """Book conditions (notes page 1). Uses no rng: condition=none runs
        stay byte-identical to pre-book builds (M1/M3 determinism)."""
        cond = self.cfg.get("book_condition", "none")
        if cond == "none":
            return
        if cond not in ("central", "librarian", "personal", "cloud"):
            raise ValueError(f"unknown book_condition: {cond}")
        qs = load_questions(self.cfg.get("questions_path"), self.world.techtree)
        if cond == "personal":
            for name in names:
                self.world.books[name] = Book(qs)
        else:
            self.world.books["shared"] = Book(qs)
        if cond in ("central", "librarian"):
            # nearest grass tile to the map centre, deterministic
            c = (self.world.W // 2, self.world.H // 2)
            pos = self.world.find_tiles(lambda t: t.type == "grass", c,
                                        self.world.W // 2)[0]
            self.world.book_pos = pos
            self.world.structures[pos] = Structure("book", "world", 0)
            # replay/screenshot reconstruction learns of the book from the log
            self.logf.write(Event(0, "structure_built", position=pos,
                                  payload={"structure": "book"}).to_json() + "\n")
        if cond == "librarian":
            self.cfg.setdefault("book_manager", names[0])
            if self.cfg["book_manager"] not in names:
                raise ValueError(f"book_manager {self.cfg['book_manager']!r} "
                                 "is not an agent")

    def _inject_knowledge(self, condition: str, names: list[str]):
        """Knowledge conditions (spec §4): seed agent.knowledge at init. Oracle
        is implemented as full knowledge for everyone — YOU KNOW then carries
        the whole tree into every prompt."""
        if condition == "blind" or self.world.techtree is None:
            return
        rids = [r["id"] for r in self.world.techtree.recipes]
        if condition == "oracle":
            for ag in self.agents.values():
                ag.knowledge.update(rids)
        elif condition == "fragmented":
            for name in names:
                self.agents[name].knowledge.add(self.rng.choice(rids))
        elif condition == "one_expert":
            self.agents[names[0]].knowledge.update(rids)
        else:
            raise ValueError(f"unknown knowledge_condition: {condition}")

    def _recipe_mentions(self, text: str) -> set:
        """Recipe ids name-dropped in a piece of speech (word-boundary match,
        underscores match spaces too). Measurement only — drives the
        source=taught tag on recipe_learned."""
        out = set()
        for r in self.world.techtree.recipes if self.world.techtree else []:
            pat = r"\b" + r["id"].replace("_", "[ _]") + r"\b"
            if re.search(pat, text, re.IGNORECASE):
                out.add(r["id"])
        return out

    # ---------- observation (the ONLY thing brains see) ----------
    def _replan_reason(self, ag: AgentState, trigger: str, steps_left: int) -> str:
        """One plain-English line for the prompt: why the planner runs now.
        Without it, models can't tell a finished plan from a discarded one
        (playtest 2026-06-13: the timer silently ate experiments mid-plan)."""
        steps = f"{steps_left} step{'s' if steps_left != 1 else ''}"
        if trigger == "spoken_to":
            return ("someone spoke to you" +
                    (f" (your unfinished plan, {steps}, was set aside — "
                     "re-issue what you still want)" if steps_left else "."))
        if trigger == "timer":
            return (f"of the routine {self.replan_interval}-tick check-in; "
                    f"your unfinished plan ({steps} left) was set aside — "
                    "re-issue what you still want.")
        if trigger == "step_failed":
            return ("a step of your plan failed (see RECENT EVENTS for the "
                    "reason); the rest of that plan was dropped.")
        if trigger == "reflex":
            return ("your survival reflex took over and ate food (your energy "
                    "was critically low); whatever you were doing before was "
                    "dropped.")
        if ag.last_plan_tick < 0:
            return "this is your first decision."
        return "your previous plan finished."

    def build_observation(self, ag: AgentState,
                          replan_reason: str = "") -> Observation:
        w = self.world
        R = self.vis_radius
        res = {
            "berries": w.find_tiles(lambda t: t.berries > 0, ag.pos, R),
            "wood":    w.find_tiles(lambda t: t.type == FOREST and t.wood > 0, ag.pos, R),
            "stone":   w.find_tiles(lambda t: t.type == MOUNTAIN and t.stone > 0, ag.pos, R),
        }
        others = sorted([{"id": o.id, "pos": o.pos} for o in self.agents.values()
                         if o.alive and o.id != ag.id
                         and w.torus_dist(ag.pos, o.pos) <= R],
                        key=lambda d: d["id"])
        structs = []
        for pos in sorted(w.structures):
            if w.torus_dist(ag.pos, pos) <= R:
                st = w.structures[pos]
                structs.append({"type": st.type, "pos": pos, "owner": st.owner})
        knowledge = ([w.techtree.describe(rid) for rid in sorted(ag.knowledge)]
                     if w.techtree else [])
        book_view = None
        cond = self.cfg.get("book_condition", "none")
        if cond != "none":
            book, writable, reason = w.book_access(ag.id, ag.pos)
            book_view = {"condition": cond, "where": w.book_pos,
                         "manager": self.cfg.get("book_manager"),
                         "readable": book is not None, "writable": writable,
                         "reason": reason}
            if book is not None:
                book_view["questions"] = [
                    dict(q, answer=book.answers.get(q["id"]))
                    for q in book.questions]
                book_view["entries"] = list(book.entries[-8:])
                book_view["n_entries"] = len(book.entries)
        return Observation(
            tick=self.tick,
            me={"id": ag.id, "pos": ag.pos, "energy": ag.energy,
                "inventory": dict(ag.inventory), "tools": set(ag.tools),
                "standing_on": w.tile(ag.x, ag.y).type,
                "slots_used": ag.slots_used(), "slot_capacity": ag.slot_capacity()},
            nearby_resources=res, nearby_agents=others,
            nearby_structures=structs,
            journal_tail=ag.journal[-12:],
            knowledge=knowledge,
            beliefs=list(ag.beliefs), notes=ag.notes,
            book=book_view,
            world_size=(w.W, w.H),
            replan_reason=replan_reason)

    # ---------- one tick ----------
    def step(self):
        evs: list[Event] = []
        # 1. WORLD PHASE
        evs += self.world.world_phase(self.tick)

        # 2a. UPKEEP — randomized but seeded order: metabolism, spoilage,
        # replan triggers, reflexes. Decides who needs the planner this tick.
        order = sorted(self.agents.keys())
        self.rng.shuffle(order)
        to_plan: list[tuple[str, str, int]] = []   # (agent, trigger, steps_left)
        for name in order:
            ag = self.agents[name]
            if not ag.alive:
                continue
            # idle metabolism — but standing on ANY shelter (anyone's) is free.
            # Combined with the per-step move surcharge in actions.py this gives
            # the four shelter energy cases: on-shelter idle 0, on-shelter move 1
            # (0 base + 1 step), normal idle 1, normal move 2 (1 base + 1 step).
            on_shelter = (st := self.world.structures.get(ag.pos)) is not None \
                and st.type == "shelter"
            ag.energy -= 0 if on_shelter else 1
            # berry spoilage (each unit rolls independently; seeded rng)
            sp = self.cfg.get("berry_spoil_prob", 0.0)
            if sp and ag.inventory.get("berries", 0):
                spoiled = sum(1 for _ in range(ag.inventory["berries"])
                              if self.rng.random() < sp)
                if spoiled:
                    ag.remove("berries", spoiled)
                    evs.append(Event(self.tick, "spoiled", actor=name,
                                     payload={"item": "berries", "qty": spoiled}))
            # forced replan triggers (spec §3): directed speech, or K-tick timer
            steps_left = len(ag.plan)
            trigger = "step_failed" if ag.interrupted else "plan_done"
            if ag.plan and ag.replan_flag:
                ag.plan = []
                trigger = "spoken_to"
            elif (ag.plan and ag.last_plan_tick >= 0
                  and self.tick - ag.last_plan_tick >= self.replan_interval):
                ag.plan = []
                trigger = "timer"
            ag.replan_flag = False
            # reflex: survival override (prefers cooked food: bigger gain)
            if ag.energy < REFLEX_ENERGY:
                if ag.inventory.get("cooked_food", 0) > 0:
                    ag.plan = [{"action": "eat", "item": "cooked_food"}]
                    ag.intention = "reflex: eat"
                    ag.reflex_fired = True
                elif ag.inventory.get("berries", 0) > 0:
                    ag.plan = [{"action": "eat", "item": "berries"}]
                    ag.intention = "reflex: eat"
                    ag.reflex_fired = True
            if not ag.plan:
                if trigger == "plan_done" and ag.reflex_fired:
                    trigger = "reflex"
                to_plan.append((name, trigger, steps_left))
                ag.interrupted = False
                ag.reflex_fired = False

        # 2b. PLANNING — all observations snapshot the same post-upkeep world
        # (simultaneous decisions). Brains that declare parallel_safe (network
        # LLM providers) are called concurrently: with 10 agents on a ~10s
        # remote model this is the difference between ~100s and ~10s per tick.
        if to_plan:
            obs = {name: self.build_observation(
                       self.agents[name],
                       self._replan_reason(self.agents[name], trig, left))
                   for name, trig, left in to_plan}
            names = [name for name, _, _ in to_plan]
            batch_brain = self.brains[names[0]]
            if (hasattr(batch_brain, "plan_batch")
                    and all(self.brains[n] is batch_brain for n in names)):
                # one shared batching brain (e.g. DiffusionGemma): plan the
                # whole tick in a single batched model call.
                plans = batch_brain.plan_batch([obs[n] for n in names])
                results = dict(zip(names, plans))
            elif (len(names) > 1
                    and all(getattr(self.brains[n], "parallel_safe", False)
                            for n in names)):
                from concurrent.futures import ThreadPoolExecutor
                workers = min(len(names), self.cfg.get("llm_workers", 8))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    results = dict(zip(names, ex.map(
                        lambda n: self.brains[n].plan(obs[n]), names)))
            else:
                results = {n: self.brains[n].plan(obs[n]) for n in names}
            for name, trigger, _ in to_plan:
                ag = self.agents[name]
                res = results[name]
                ag.plan, ag.intention = res.plan, res.intention
                # mind-owned fields written back verbatim (engine never reads them)
                if res.updated_beliefs is not None:
                    ag.beliefs = res.updated_beliefs
                if res.updated_notes is not None:
                    ag.notes = res.updated_notes
                ag.last_plan_tick = self.tick
                if res.parse_error:
                    evs.append(Event(self.tick, "planner_parse_error", actor=name))
                evs.append(Event(self.tick, "planner_called", actor=name,
                                 payload={"intention": ag.intention,
                                          "trigger": trigger}))

        # 2c. EXECUTION — same seeded order
        for name in order:
            ag = self.agents[name]
            if not ag.alive:
                continue
            if ag.plan:
                done, aevs = actions.apply(ag.plan[0], ag, self.world, self.tick,
                                           self.agents)
                if any(e.type == "plan_interrupted" for e in aevs):
                    ag.plan = []   # failed step invalidates the rest → replan next tick
                    ag.interrupted = True   # next consultation says WHY NOW = step failed
                elif done:
                    ag.plan.pop(0)
                    if not ag.plan:
                        evs.append(Event(self.tick, "plan_completed", actor=name,
                                         payload={"intention": ag.intention}))
                evs += aevs
            # life checks
            if ag.energy < CRITICAL_ENERGY and name not in self._was_critical:
                self._was_critical.add(name)
                evs.append(Event(self.tick, "energy_critical", actor=name,
                                 position=ag.pos))
            if ag.energy >= CRITICAL_ENERGY:
                self._was_critical.discard(name)
            if ag.energy <= 0:
                ag.alive = False
                evs.append(Event(self.tick, "agent_died", actor=name, position=ag.pos))

        # 3. COMMIT: visibility fan-out + journals + log
        system_types = {"planner_called", "plan_completed", "plan_interrupted",
                        "planner_parse_error", "spoiled", "recipe_learned"}
        for ev in evs:
            if ev.type in system_types:
                ev.visibility = [ev.actor] if ev.actor else []
                # plan_interrupted IS journaled (v0.32, deviating from spec
                # §3b): without the failure reason in context, models repeat
                # the same broken step forever — measured 459/475 plans dying
                # on identical "no wood here" mistakes in m3 runs.
                if ev.type in ("spoiled", "recipe_learned",
                               "plan_interrupted") and ev.actor:
                    self.agents[ev.actor].journal_add(
                        f"[t{ev.tick}] " + render_for_journal(ev, ev.actor))
            else:
                radius = self.say_radius if ev.type == "said" else self.vis_radius
                vis = []
                for o in self.agents.values():
                    if not o.alive and o.id != ev.actor:
                        continue
                    anchor = ev.position or (self.agents[ev.actor].pos if ev.actor else None)
                    if anchor is None or self.world.torus_dist(o.pos, anchor) <= radius:
                        vis.append(o.id)
                ev.visibility = sorted(vis)
                for vid in ev.visibility:
                    self.agents[vid].journal_add(
                        f"[t{ev.tick}] " + render_for_journal(ev, vid))
                if ev.type == "said":
                    # being spoken to directly triggers a replan (spec §3)
                    if ev.target and ev.target in ev.visibility:
                        self.agents[ev.target].replan_flag = True
                    elif ev.target and ev.actor:
                        # earshot feedback: without it the speaker's journal
                        # reads "You said to X" and they believe X heard
                        # (playtest 2026-06-13: false-confidence coordination)
                        self.agents[ev.actor].journal_add(
                            f"[t{ev.tick}] ({ev.target} was too far away "
                            "to hear you.)")
                    # measurement: who has heard which recipes mentioned
                    mentions = self._recipe_mentions(ev.payload["message"])
                    if mentions:
                        for vid in ev.visibility:
                            if vid != ev.actor:
                                self.agents[vid].heard_recipes |= mentions
            self.logf.write(ev.to_json() + "\n")
        self.tick += 1

    def run(self, ticks: int, render_every: int = 0):
        for _ in range(ticks):
            self.step()
            if render_every and self.tick % render_every == 0:
                print(self.render_ascii())
        self.logf.close()
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

    # ---------- ASCII renderer ----------
    def render_ascii(self) -> str:
        w = self.world
        chars = {("grass", 0): ".", ("forest", 1): "T", ("forest", 0): "t",
                 ("mountain", 1): "^", ("mountain", 0): "_"}
        rows = []
        agent_at = {a.pos: a.id[-1] for a in self.agents.values() if a.alive}
        struct_char = {"fire": "F", "farm": "f", "storehouse": "#", "book": "B",
                       "shelter": "H"}
        for y in range(w.H):
            row = []
            for x in range(w.W):
                if (x, y) in agent_at:
                    row.append(agent_at[(x, y)])
                    continue
                if (x, y) in w.structures:
                    row.append(struct_char[w.structures[(x, y)].type])
                    continue
                t = w.tile(x, y)
                if t.berries > 0:
                    row.append("*")
                elif t.type == "forest":
                    row.append("T" if t.wood else "t")
                elif t.type == "mountain":
                    row.append("^" if t.stone else "_")
                elif t.type == "water":
                    row.append("~")
                else:
                    row.append(".")
            rows.append("".join(row))
        alive = sum(a.alive for a in self.agents.values())
        avg_e = (sum(a.energy for a in self.agents.values() if a.alive) / alive
                 if alive else 0)
        head = f"== tick {self.tick} | alive {alive}/{len(self.agents)} | avg energy {avg_e:.0f} =="
        return head + "\n" + "\n".join(rows)
