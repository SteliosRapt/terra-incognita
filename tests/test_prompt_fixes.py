"""test_prompt_fixes.py — the v0.39 playtest fixes (2026-06-13): no recipe
leak in the schema example, torus + voice range documented, WHY NOW replan
reasons, say earshot feedback, reflex/step-failure triggers."""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import Simulation
from engine.techtree import TechTree, CombineContext
from brains.base import Observation, PlanResult
from brains import llm_planner
from brains.llm_planner import build_prompt

from test_m3 import CFG, ScriptBrain, make_sim, events_of, obs_fixture


# ------------------------------------------------- the recipe leak (the axe)
def test_schema_example_combine_is_not_a_real_recipe():
    """Models copy the example plan verbatim; if the example combine matches
    any techtree recipe (under ANY conditions) discovery metrics get a
    freebie. The old example was literally the axe."""
    m = re.search(r'"combine",\s*"items":\s*({[^}]*})', llm_planner._SCHEMA_DOC)
    assert m, "schema example no longer shows a combine step"
    items = json.loads(m.group(1))
    tree = TechTree(str(ROOT / "configs" / "techtree_v0.json"))
    for tile in ("grass", "forest", "mountain"):
        for tools in (set(), {"axe", "pickaxe", "basket"}):
            ctx = CombineContext(tile_type=tile, tools=tools,
                                 adjacent_structures={"fire", "storehouse"})
            assert tree.resolve(items, ctx).outcome != "success", (
                f"example combine {items} IS a recipe (tile={tile}, tools={tools})")


# ------------------------------------------------- documented world physics
def test_system_prompt_states_wraparound_and_voice_range():
    system, _ = build_prompt(obs_fixture(world_size=(32, 32)))
    assert "32x32" in system and "wrap" in system
    assert "voice only carries a few tiles" in system


def test_world_shape_omitted_when_size_unknown():
    system, _ = build_prompt(obs_fixture())
    assert "wrap" not in system


def test_world_placed_structures_have_no_builder():
    _, user = build_prompt(obs_fixture(
        nearby_structures=[{"type": "book", "pos": (8, 8), "owner": "world"}]))
    assert "a book at (8, 8)" in user
    assert "built by world" not in user


# ------------------------------------------------- WHY NOW
def test_why_now_line_present_only_with_reason():
    _, user = build_prompt(obs_fixture(replan_reason="your previous plan finished."))
    assert "== WHY NOW == You are planning because your previous plan finished." in user
    _, user = build_prompt(obs_fixture())
    assert "== WHY NOW ==" not in user


def test_observation_carries_replan_reason(tmp_path):
    seen = {}

    class SpyBrain:
        def plan(self, obs):
            seen[obs.me["id"]] = obs.replan_reason
            return PlanResult([{"action": "wait"}], "spy")

    cfg = dict(CFG, n_agents=1)
    sim = Simulation(cfg, lambda rng: SpyBrain(), str(tmp_path / "run"))
    sim.step()
    assert seen["A00"] == "this is your first decision."
    sim.step()
    assert seen["A00"] == "your previous plan finished."


def test_step_failure_reported_as_reason(tmp_path):
    reasons = []

    class SpyBrain:
        def plan(self, obs):
            reasons.append(obs.replan_reason)
            # eating without food fails -> plan_interrupted -> step_failed
            return PlanResult([{"action": "eat", "item": "cooked_food"}], "spy")

    cfg = dict(CFG, n_agents=1)
    sim = Simulation(cfg, lambda rng: SpyBrain(), str(tmp_path / "run"))
    sim.step()
    sim.step()
    assert "a step of your plan failed" in reasons[1]
    calls = [e for e in events_of(sim) if e["type"] == "planner_called"]
    assert calls[1]["payload"]["trigger"] == "step_failed"


def test_reflex_override_reported_as_reason(tmp_path):
    reasons = []

    class SpyBrain:
        def plan(self, obs):
            reasons.append(obs.replan_reason)
            return PlanResult([{"action": "wait"}] * 5, "spy")

    cfg = dict(CFG, n_agents=1)
    sim = Simulation(cfg, lambda rng: SpyBrain(), str(tmp_path / "run"))
    ag = sim.agents["A00"]
    ag.energy = 31                       # next upkeep drops below REFLEX_ENERGY
    ag.inventory["berries"] = 2
    sim.step()                           # plans 5 waits
    sim.step()                           # energy 29 -> reflex eats, plan dropped
    sim.step()                           # replans: reason = reflex
    assert any("survival reflex" in r for r in reasons), reasons


def test_timer_reason_counts_set_aside_steps(tmp_path):
    reasons = []

    class SpyBrain:
        def plan(self, obs):
            reasons.append(obs.replan_reason)
            return PlanResult([{"action": "wait"}] * 20, "spy")

    cfg = dict(CFG, n_agents=1, replan_interval=3)
    sim = Simulation(cfg, lambda rng: SpyBrain(), str(tmp_path / "run"))
    for _ in range(5):
        sim.step()
    assert any("3-tick check-in" in r and "17 steps left" in r for r in reasons), reasons


# ------------------------------------------------- say earshot feedback
def _place(sim, name, x, y):
    sim.agents[name].x, sim.agents[name].y = x, y


def test_say_out_of_earshot_tells_the_speaker(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain([{"action": "say", "target": "A01", "message": "hello?"}])})
    _place(sim, "A00", 0, 0)
    _place(sim, "A01", 8, 8)
    _place(sim, "A02", 8, 0)
    sim.step()
    assert any("A01 was too far away to hear you" in line
               for line in sim.agents["A00"].journal), sim.agents["A00"].journal
    assert not sim.agents["A01"].replan_flag


def test_say_in_earshot_no_false_warning(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain([{"action": "say", "target": "A01", "message": "hi"}])})
    _place(sim, "A00", 0, 0)
    _place(sim, "A01", 1, 1)
    _place(sim, "A02", 8, 8)
    sim.step()
    assert not any("too far away to hear" in line
                   for line in sim.agents["A00"].journal)
    assert sim.agents["A01"].replan_flag or any(
        "said to you" in line for line in sim.agents["A01"].journal)
