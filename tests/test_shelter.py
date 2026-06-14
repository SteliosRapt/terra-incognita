"""test_shelter.py — brick/shelter recipes, the one-per-builder cap, shared
shelter use, the four energy cases, and the exact --combine-note string."""
import pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.world import World, Structure, GRASS
from engine.agent import AgentState
from engine.techtree import TechTree
from engine.sim import Simulation
from engine import actions
from engine.defaults import BASE
from brains.base import Observation, PlanResult
from brains.llm_planner import build_prompt, COMBINE_NOTE

TT = str(ROOT / "configs" / "techtree_v0.json")
CFG = {"grid_size": 16, "seed": 7, "forest_fraction": 0.1, "mountain_fraction": 0.05,
       "stone_per_tile": 12, "wood_regrow_ticks": 40, "berry_patches": 2,
       "berry_patch_size": 8, "berry_regrow_prob": 0.05, "water_fraction": 0.05,
       "lakes": 1, "forest_regions": 2, "mountain_ranges": 1}


def _world_agent():
    w = World(CFG, random.Random(7))
    w.techtree = TechTree(TT)
    gx, gy = next((x, y) for y in range(16) for x in range(16)
                  if w.tile(x, y).type == GRASS and (x, y) not in w.structures)
    return w, AgentState(id="T", x=gx, y=gy)


# ---------------------------------------------------------------- recipes
def test_brick_recipe():
    w, ag = _world_agent()
    ag.add("stone", 1)
    done, evs = actions.apply({"action": "combine", "items": {"stone": 1}}, ag, w, 0)
    types = [e.type for e in evs]
    assert "combine_succeeded" in types and "recipe_learned" in types
    assert ag.inventory.get("brick") == 1
    assert ag.inventory.get("stone", 0) == 0


def test_shelter_recipe_is_permanent():
    w, ag = _world_agent()
    ag.add("brick", 4)
    done, evs = actions.apply({"action": "combine", "items": {"brick": 4}}, ag, w, 5)
    assert any(e.type == "structure_built" for e in evs)
    st = w.structures[ag.pos]
    assert st.type == "shelter" and st.expires_at == -1   # permanent
    assert ag.shelter_built is True
    assert ag.inventory.get("brick", 0) == 0


def test_one_shelter_per_builder_keeps_bricks():
    w, ag = _world_agent()
    ag.add("brick", 8)
    actions.apply({"action": "combine", "items": {"brick": 4}}, ag, w, 0)  # first OK
    assert ag.shelter_built and ag.inventory.get("brick") == 4
    # move to an empty grass tile so it isn't a "no room" rejection
    ag.x, ag.y = next((x, y) for y in range(16) for x in range(16)
                      if w.tile(x, y).type == GRASS and (x, y) not in w.structures)
    done, evs = actions.apply({"action": "combine", "items": {"brick": 4}}, ag, w, 1)
    assert [e.type for e in evs] == ["combine_failed"]
    assert (evs[0].payload["outcome"] == "shelter_limit")
    assert ag.inventory.get("brick") == 4         # bricks RETAINED, not consumed
    assert (ag.x, ag.y) not in w.structures       # nothing built


# ---------------------------------------------------------------- energy cases
class _Dummy:
    parallel_safe = False
    def plan(self, obs):                # never actually called in these tests
        return PlanResult([{"action": "wait"}], "")


def _sim_one_agent():
    cfg = dict(BASE, grid_size=16, seed=3, n_agents=1,
               knowledge_condition="blind", book_condition="none",
               replan_interval=10_000, combine_note=False)
    sim = Simulation(cfg, lambda rng: _Dummy(), str(pathlib.Path(
        __import__("tempfile").mkdtemp())))
    return sim


def _empty_grass_neighbor(w, pos):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p = ((pos[0] + dx) % w.W, (pos[1] + dy) % w.H)
        if w.tile(*p).passable:
            return p
    raise AssertionError("no passable neighbor")


def _run_one_tick(sim, ag, plan):
    """Pin a plan so the agent never replans, run one tick, return energy delta."""
    ag.plan = list(plan)
    ag.last_plan_tick = sim.tick
    ag.interrupted = False
    e0 = ag.energy
    sim.step()
    return ag.energy - e0


def test_four_energy_cases():
    # on-shelter idle = 0 | on-shelter move = -1 | off idle = -1 | off move = -2
    sim = _sim_one_agent()
    ag = next(iter(sim.agents.values()))
    w = sim.world
    # ensure a clean grass tile with a passable neighbor, no structures
    gx, gy = next((x, y) for y in range(w.H) for x in range(w.W)
                  if w.tile(x, y).type == GRASS and (x, y) not in w.structures)
    ag.x, ag.y = gx, gy

    # 1. off-shelter idle
    assert _run_one_tick(sim, ag, [{"action": "wait"}]) == -1
    # 2. off-shelter move
    nb = _empty_grass_neighbor(w, ag.pos)
    assert _run_one_tick(sim, ag, [{"action": "move_to", "x": nb[0], "y": nb[1]}]) == -2

    # build a shelter under the agent (owner is SOMEONE ELSE — shared benefit)
    w.structures[ag.pos] = Structure("shelter", "other_agent", sim.tick)
    # 3. on-shelter idle
    assert _run_one_tick(sim, ag, [{"action": "wait"}]) == 0
    # 4. on-shelter move (off it)
    nb = _empty_grass_neighbor(w, ag.pos)
    assert _run_one_tick(sim, ag, [{"action": "move_to", "x": nb[0], "y": nb[1]}]) == -1


# ---------------------------------------------------------------- the Note
def _obs():
    return Observation(tick=0,
                       me={"id": "A00", "pos": (1, 1), "energy": 100,
                           "inventory": {}, "tools": set(), "standing_on": "grass",
                           "slots_used": 0, "slot_capacity": 8},
                       nearby_resources={}, nearby_agents=[], journal_tail=[],
                       world_size=(16, 16))


def test_combine_note_exact_inclusion_and_exclusion():
    sys_off, _ = build_prompt(_obs(), combine_note=False)
    sys_on, _ = build_prompt(_obs(), combine_note=True)
    assert COMBINE_NOTE == "An item can sometimes be combined by itself."
    assert COMBINE_NOTE not in sys_off       # nothing equivalent when off
    assert sys_on == sys_off + " " + COMBINE_NOTE
