"""test_m3.py — M3 acceptance tests. No LLM is ever called: brains are fakes,
providers are canned or the deterministic MockProvider."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import Simulation
from brains.base import Observation, PlanResult
from brains.llm_planner import LLMPlannerBrain, build_prompt, parse_response, NOTES_CAP
from brains.providers import MockProvider

CFG = {
    "seed": 7, "grid_size": 16, "n_agents": 3,
    "forest_fraction": 0.10, "mountain_fraction": 0.06, "stone_per_tile": 12,
    "wood_regrow_ticks": 40, "berry_patches": 2, "berry_patch_size": 10,
    "berry_regrow_prob": 0.05, "berry_spoil_prob": 0.0,
    "water_fraction": 0.05, "lakes": 1, "forest_regions": 2, "mountain_ranges": 1,
    "techtree_path": str(ROOT / "configs" / "techtree_v0.json"),
}


class ScriptBrain:
    """Returns queued plans in order, then waits forever."""
    def __init__(self, *plans):
        self.plans = list(plans)

    def plan(self, obs):
        if self.plans:
            return PlanResult(self.plans.pop(0), "scripted")
        return PlanResult([{"action": "wait"}], "idle")


class CannedProvider:
    """Provider returning a fixed sequence of raw responses (last one repeats)."""
    name = "canned"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def make_sim(tmp_path, brains=None, **over):
    cfg = dict(CFG, **over)
    sim = Simulation(cfg, lambda rng: ScriptBrain(), str(tmp_path / "run"))
    if brains:
        sim.brains.update(brains)
    return sim


def events_of(sim):
    sim.logf.flush()
    with open(pathlib.Path(sim.run_dir) / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def obs_fixture(**over):
    base = dict(
        tick=42,
        me={"id": "A00", "pos": (12, 7), "energy": 64,
            "inventory": {"wood": 3, "stone": 1}, "tools": {"axe"},
            "standing_on": "grass", "slots_used": 1, "slot_capacity": 8},
        nearby_resources={"berries": [(3, 4)], "wood": [(5, 5), (5, 6)], "stone": []},
        nearby_agents=[{"id": "A01", "pos": (13, 7)}],
        nearby_structures=[{"type": "fire", "pos": (10, 5), "owner": "A02"}],
        journal_tail=["[t40] You gathered 1 wood at (12, 7)."],
        knowledge=["axe = 1 stone + 2 wood -> tool: axe"],
        beliefs=["maybe stone needs fire"],
        notes="trust A01",
    )
    base.update(over)
    return Observation(**base)


# ---------------------------------------------------------------- prompt
def test_build_prompt_has_all_sections_and_facts():
    system, user = build_prompt(obs_fixture())
    assert "You are A00" in system
    for section in ("== STATUS ==", "== YOU KNOW ==", "== YOU BELIEVE ==",
                    "== NOTES ==", "== NEARBY ==", "== RECENT EVENTS ==",
                    "== ACTIONS =="):
        assert section in user, f"missing {section}"
    assert "energy 64/100" in user
    assert "position (12, 7)" in user
    assert "standing on grass" in user
    assert "wood x3" in user and "stone x1" in user
    assert "axe = 1 stone + 2 wood" in user
    assert "maybe stone needs fire" in user
    assert "trust A01" in user
    assert "berries at (3, 4)" in user
    assert "a fire at (10, 5), built by A02" in user
    assert "A01 at (13, 7)" in user
    assert "You gathered 1 wood" in user


def test_prompt_renders_empty_mind_gracefully():
    _, user = build_prompt(obs_fixture(knowledge=[], beliefs=[], notes="",
                                       journal_tail=[]))
    assert "no confirmed recipes yet" in user
    assert "(nothing yet)" in user
    assert "(empty)" in user


# ---------------------------------------------------------------- parsing
def test_parse_valid_response():
    raw = json.dumps({"reasoning": "go get wood", "updated_beliefs": ["b1"],
                      "updated_notes": "n",
                      "plan": [{"action": "move_to", "x": 3, "y": 4},
                               {"action": "gather", "resource": "wood"},
                               {"action": "combine", "items": {"wood": 2, "stone": 1}},
                               {"action": "say", "target": "A01", "message": "hi"},
                               {"action": "give", "target": "A01", "item": "wood", "qty": 2},
                               {"action": "wait"}]})
    res = parse_response(raw)
    assert res.plan[0] == {"action": "move_to", "x": 3, "y": 4}
    assert res.plan[2]["items"] == {"wood": 2, "stone": 1}
    assert res.intention == "go get wood"
    assert res.updated_beliefs == ["b1"]
    assert res.updated_notes == "n"


def test_parse_keeps_gather_target():
    """Regression (v0.32 smoke run): the validator stripped x/y from gather
    steps, turning every targeted gather into 'no wood here'."""
    raw = '{"reasoning": "r", "plan": [{"action": "gather", "resource": "wood", "x": 5, "y": "7"}]}'
    assert parse_response(raw).plan == [
        {"action": "gather", "resource": "wood", "x": 5, "y": 7}]
    raw = '{"reasoning": "r", "plan": [{"action": "gather", "resource": "wood"}]}'
    assert parse_response(raw).plan == [{"action": "gather", "resource": "wood"}]


def test_parse_accepts_fenced_json_and_coerces_string_ints():
    raw = '```json\n{"reasoning": "r", "plan": [{"action": "move_to", "x": "3", "y": "4"}]}\n```'
    res = parse_response(raw)
    assert res.plan == [{"action": "move_to", "x": 3, "y": 4}]


@pytest.mark.parametrize("bad", [
    "I think I should gather wood.",
    '{"reasoning": "r"}',                                       # no plan
    '{"plan": []}',                                             # empty plan
    '{"plan": [{"action": "teleport", "x": 1, "y": 1}]}',       # unknown action
    '{"plan": [{"action": "combine", "items": {}}]}',           # empty combine
    '{"plan": "gather wood"}',                                  # plan not a list
])
def test_parse_rejects_garbage(bad):
    with pytest.raises(Exception):
        parse_response(bad)


def test_caps_on_notes_and_beliefs():
    raw = json.dumps({"reasoning": "r", "updated_notes": "x" * 5000,
                      "updated_beliefs": [str(i) * 600 for i in range(50)],
                      "plan": [{"action": "wait"}] * 99})
    res = parse_response(raw)
    assert len(res.updated_notes) == NOTES_CAP
    assert len(res.updated_beliefs) == 12
    assert all(len(b) <= 250 for b in res.updated_beliefs)
    assert len(res.plan) == 20


def test_retry_recovers_then_fallback():
    good = '{"reasoning": "ok", "plan": [{"action": "wait"}]}'
    brain = LLMPlannerBrain(CannedProvider("garbage", good))
    res = brain.plan(obs_fixture())
    assert not res.parse_error and res.intention == "ok"

    brain = LLMPlannerBrain(CannedProvider("garbage"))
    res = brain.plan(obs_fixture())
    assert res.parse_error
    assert res.plan == [{"action": "wait"}]
    assert brain.provider.calls == 2          # exactly one retry


def test_provider_exception_is_a_failure_not_a_crash():
    class ExplodingProvider:
        name = "boom"
        def complete(self, system, user):
            raise ConnectionError("ollama is down")
    res = LLMPlannerBrain(ExplodingProvider()).plan(obs_fixture())
    assert res.parse_error and res.plan == [{"action": "wait"}]


def test_telemetry_written(tmp_path):
    path = tmp_path / "prompts.jsonl"
    brain = LLMPlannerBrain(
        CannedProvider('{"reasoning": "ok", "plan": [{"action": "wait"}]}'),
        telemetry_path=str(path))
    brain.plan(obs_fixture())
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["agent"] == "A00" and rec["tick"] == 42 and rec["ok"]
    assert "== STATUS ==" in rec["user"] and rec["response"]


# ---------------------------------------------------------------- say / give
def place(sim, **positions):
    for name, (x, y) in positions.items():
        sim.agents[name].x, sim.agents[name].y = x, y


def test_say_visibility_radius_3(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain([{"action": "say", "target": "A01", "message": "hello there"}]),
    })
    place(sim, A00=(0, 0), A01=(2, 0), A02=(8, 8))
    sim.step()
    said = [e for e in events_of(sim) if e["type"] == "said"][0]
    assert said["visibility"] == ["A00", "A01"]          # A02 out of earshot
    assert any('said to you: "hello there"' in l for l in sim.agents["A01"].journal)
    assert not any("hello there" in l for l in sim.agents["A02"].journal)
    assert sim.agents["A01"].replan_flag                  # directed speech → replan


def test_spoken_to_triggers_replan(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain([{"action": "say", "target": "A01", "message": "oi"}]),
        "A01": ScriptBrain([{"action": "wait"}] * 10),
    })
    place(sim, A00=(0, 0), A01=(1, 0), A02=(8, 8))
    sim.step()
    sim.step()
    calls = [e for e in events_of(sim)
             if e["type"] == "planner_called" and e["actor"] == "A01"]
    assert calls[-1]["payload"]["trigger"] == "spoken_to"
    assert calls[-1]["tick"] == 1


def test_give_transfers_and_requires_adjacency(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain([{"action": "give", "target": "A01", "item": "wood", "qty": 2}],
                           [{"action": "give", "target": "A02", "item": "wood", "qty": 1}]),
    })
    place(sim, A00=(0, 0), A01=(1, 1), A02=(8, 8))
    sim.agents["A00"].add("wood", 3)
    sim.step()   # adjacent give succeeds
    sim.step()   # far give interrupted
    evs = events_of(sim)
    gave = [e for e in evs if e["type"] == "gave"]
    assert len(gave) == 1 and gave[0]["payload"] == {"item": "wood", "qty": 2}
    assert sim.agents["A00"].inventory.get("wood") == 1
    assert sim.agents["A01"].inventory.get("wood") == 2
    assert any(e["type"] == "plan_interrupted" and "too far" in e["payload"]["reason"]
               for e in evs)
    assert any("gave you 2 wood" in l for l in sim.agents["A01"].journal)


# ---------------------------------------------------------------- teaching
def test_recipe_learned_source_taught_vs_discovered(tmp_path):
    sim = make_sim(tmp_path, brains={
        "A00": ScriptBrain(
            [{"action": "say", "target": "A01",
              "message": "an axe is 2 wood and 1 stone"}],
            [{"action": "combine", "items": {"wood": 2, "stone": 1}}]),
        "A01": ScriptBrain(
            [{"action": "wait"}],
            [{"action": "combine", "items": {"wood": 2, "stone": 1}}]),
    })
    place(sim, A00=(0, 0), A01=(1, 0), A02=(8, 8))
    for name in ("A00", "A01"):
        sim.agents[name].add("wood", 2)
        sim.agents[name].add("stone", 1)
    sim.step()                                   # A00 teaches
    assert "axe" in sim.agents["A01"].heard_recipes
    assert "axe" not in sim.agents["A00"].heard_recipes   # speakers don't teach themselves
    sim.step()                                   # both craft
    learned = {e["actor"]: e["payload"]["source"]
               for e in events_of(sim) if e["type"] == "recipe_learned"}
    assert learned["A01"] == "taught"
    assert learned["A00"] == "discovered"


# ---------------------------------------------------------------- conditions
def test_knowledge_conditions(tmp_path):
    # derive from the techtree so adding recipes (brick, shelter, ...) doesn't
    # require touching this test
    sim = make_sim(tmp_path / "o", knowledge_condition="oracle")
    rids = {r["id"] for r in sim.world.techtree.recipes}
    assert {"axe", "fire", "shelter", "brick"} <= rids   # sanity: tree has them
    assert all(ag.knowledge == rids for ag in sim.agents.values())

    sim = make_sim(tmp_path / "f", knowledge_condition="fragmented")
    assert all(len(ag.knowledge) == 1 and ag.knowledge <= rids
               for ag in sim.agents.values())

    sim = make_sim(tmp_path / "e", knowledge_condition="one_expert")
    assert sim.agents["A00"].knowledge == rids
    assert all(not sim.agents[n].knowledge for n in ("A01", "A02"))

    sim = make_sim(tmp_path / "b")               # default blind
    assert all(not ag.knowledge for ag in sim.agents.values())

    obs = sim.build_observation(sim.agents["A00"])
    assert obs.knowledge == [] and obs.me["standing_on"]
    oracle = make_sim(tmp_path / "o2", knowledge_condition="oracle")
    obs = oracle.build_observation(oracle.agents["A00"])
    assert any("fire = 2 stone + 1 wood (while standing on grass)" in k
               for k in obs.knowledge)


# ---------------------------------------------------------------- replanning
def test_forced_replan_every_k_ticks(tmp_path):
    class WaitForever:
        def plan(self, obs):
            return PlanResult([{"action": "wait"}] * 100, "zzz")
    sim = make_sim(tmp_path, n_agents=1, replan_interval=15,
                   brains={"A00": WaitForever()})
    for _ in range(40):
        sim.step()
    calls = [e for e in events_of(sim) if e["type"] == "planner_called"]
    assert [c["tick"] for c in calls] == [0, 15, 30]
    assert calls[1]["payload"]["trigger"] == "timer"


def test_plan_interrupted_clears_rest_of_plan(tmp_path):
    sim = make_sim(tmp_path, n_agents=1, brains={
        "A00": ScriptBrain([{"action": "eat", "item": "cooked_food"},  # has none
                            {"action": "drop", "item": "wood", "qty": 1}]),
    })
    sim.step()
    sim.logf.flush()
    assert sim.agents["A00"].plan == []          # rest of plan abandoned
    assert not any(e["type"] == "dropped" for e in events_of(sim))


# ---------------------------------------------------------------- end to end
def test_mock_society_is_deterministic_and_survives(tmp_path):
    def run(d):
        cfg = dict(CFG, n_agents=4, seed=11)
        sim = Simulation(cfg, lambda rng: LLMPlannerBrain(MockProvider(rng)), str(d))
        for _ in range(120):
            sim.step()
        sim.logf.close()
        return sim

    s1 = run(tmp_path / "a")
    s2 = run(tmp_path / "b")
    log1 = (tmp_path / "a" / "events.jsonl").read_bytes()
    log2 = (tmp_path / "b" / "events.jsonl").read_bytes()
    assert log1 == log2, "same seed + mock provider must be byte-identical"
    assert any(a.alive for a in s1.agents.values())
    types = {json.loads(l)["type"] for l in log1.splitlines()}
    assert "planner_called" in types and "gathered" in types
    assert "planner_parse_error" not in types


def test_parse_error_emits_event_and_agent_waits(tmp_path):
    cfg = dict(CFG, n_agents=1)
    sim = Simulation(cfg, lambda rng: LLMPlannerBrain(CannedProvider("nonsense")),
                     str(tmp_path / "run"))
    sim.step()
    evs = events_of(sim)
    assert any(e["type"] == "planner_parse_error" for e in evs)
    assert sim.agents["A00"].alive


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
