"""test_epistemics.py — the book (all five conditions), the world-questions
ground truth, the survey judge, and the CLI's PNG reconstruction.
No LLM/network calls (FakeBrain / fake judge provider)."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import Simulation
from engine.defaults import BASE
from engine.questions import load_questions, ground_truth
from brains.base import Observation, PlanResult
from brains.llm_planner import build_prompt, parse_response
from brains.survey import SurveyJudge


class FakeBrain:
    """Returns canned plans, then waits forever."""
    parallel_safe = False

    def __init__(self, plans=None):
        self.plans = list(plans or [])

    def plan(self, obs):
        if self.plans:
            return PlanResult(self.plans.pop(0))
        return PlanResult([{"action": "wait"}])


def make_sim(tmp_path, book="none", agents=3, plans_by_agent=None, **over):
    cfg = dict(BASE, seed=7, n_agents=agents, knowledge_condition="blind",
               book_condition=book, **over)
    made = {"i": 0}

    def factory(rng):
        plans = (plans_by_agent or {}).get(made["i"])
        made["i"] += 1
        return FakeBrain(plans)

    return Simulation(cfg, factory, str(tmp_path / f"run_{book}"))


# ------------------------------------------------------------------ questions
def test_questions_cover_statics_and_recipes(tmp_path):
    sim = make_sim(tmp_path)
    qs = load_questions(None, sim.world.techtree)
    ids = [q["id"] for q in qs]
    assert "regrow_wood" in ids and "coverage" in ids and "combine_loss" in ids
    assert "recipe_axe" in ids and "recipe_cooked_food" in ids
    truth = ground_truth(qs, sim.cfg, sim.world, sim.world.techtree)
    assert set(truth) == set(ids)
    assert "40 ticks" in truth["regrow_wood"]
    assert "5%" in truth["regrow_berries"]
    assert "100%" in truth["combine_loss"]
    # coverage is measured on the actual map and stated in percent
    assert "forest" in truth["coverage"] and "%" in truth["coverage"]
    # recipe truths include the non-obvious conditions
    assert "grass" in truth["recipe_fire"]
    assert "axe" in truth["recipe_farm"]


# ------------------------------------------------------------------ the book
def test_central_book_placed_near_centre_and_gated_by_distance(tmp_path):
    sim = make_sim(tmp_path, book="central")
    w = sim.world
    assert w.book_pos is not None
    assert w.structures[w.book_pos].type == "book"
    assert w.torus_dist(w.book_pos, (w.W // 2, w.H // 2)) <= 6
    book, writable, reason = w.book_access("anyone", w.book_pos)
    assert book is w.books["shared"] and writable
    far = ((w.book_pos[0] + 10) % w.W, w.book_pos[1])
    book, writable, reason = w.book_access("anyone", far)
    assert book is None and not writable and "stand on or next to" in reason


def test_book_actions_write_answer_and_immutable_notes(tmp_path):
    sim = make_sim(tmp_path, book="cloud")
    ag = next(iter(sim.agents.values()))
    from engine import actions
    done, evs = actions.apply({"action": "book_note", "text": "fire needs grass?"},
                              ag, sim.world, 5, sim.agents)
    assert done and evs[0].type == "book_note_written"
    done, evs = actions.apply({"action": "book_answer", "question": "regrow_wood",
                               "answer": "trees return after 40 ticks"},
                              ag, sim.world, 6, sim.agents)
    assert evs[0].type == "book_answer_written"
    book = sim.world.books["shared"]
    assert book.entries[0]["text"] == "fire needs grass?"
    assert book.answers["regrow_wood"]["author"] == ag.id
    # answers are editable (last write wins, author tracked)...
    other = list(sim.agents.values())[1]
    actions.apply({"action": "book_answer", "question": "regrow_wood",
                   "answer": "60 ticks"}, other, sim.world, 9, sim.agents)
    assert book.answers["regrow_wood"] == {"text": "60 ticks",
                                           "author": other.id, "tick": 9}
    # ...while unknown questions are rejected
    done, evs = actions.apply({"action": "book_answer", "question": "nope",
                               "answer": "x"}, ag, sim.world, 10, sim.agents)
    assert evs[0].type == "plan_interrupted"


def test_librarian_only_manager_writes(tmp_path):
    sim = make_sim(tmp_path, book="librarian")
    names = list(sim.agents)
    assert sim.cfg["book_manager"] == names[0]
    w = sim.world
    mgr, other = sim.agents[names[0]], sim.agents[names[1]]
    mgr.x, mgr.y = w.book_pos
    other.x, other.y = w.book_pos
    book, writable, _ = w.book_access(mgr.id, mgr.pos)
    assert book is not None and writable
    book, writable, reason = w.book_access(other.id, other.pos)
    assert book is not None and not writable          # may read, not write
    assert names[0] in reason
    from engine import actions
    done, evs = actions.apply({"action": "book_note", "text": "hi"},
                              other, sim.world, 3, sim.agents)
    assert evs[0].type == "plan_interrupted"


def test_personal_books_are_private_and_everywhere(tmp_path):
    sim = make_sim(tmp_path, book="personal")
    names = list(sim.agents)
    assert set(sim.world.books) == set(names)
    a, b = sim.agents[names[0]], sim.agents[names[1]]
    book_a, writable, _ = sim.world.book_access(a.id, a.pos)
    book_b, _, _ = sim.world.book_access(b.id, b.pos)
    assert writable and book_a is not book_b


def test_observation_and_prompt_render_book(tmp_path):
    sim = make_sim(tmp_path, book="cloud")
    ag = next(iter(sim.agents.values()))
    from engine import actions
    actions.apply({"action": "book_answer", "question": "regrow_stone",
                   "answer": "stone is finite"}, ag, sim.world, 2, sim.agents)
    obs = sim.build_observation(ag)
    assert obs.book["readable"] and obs.book["writable"]
    system, user = build_prompt(obs)
    assert "== BOOK ==" in user
    assert "[regrow_stone]" in user and "stone is finite" in user
    assert "(unanswered)" in user
    assert "book_answer" in user            # writable -> actions documented
    assert "Book" in system                 # shared-goal line
    # none condition: no book anywhere
    sim2 = make_sim(tmp_path, book="none")
    obs2 = sim2.build_observation(next(iter(sim2.agents.values())))
    assert obs2.book is None
    _, user2 = build_prompt(obs2)
    assert "== BOOK ==" not in user2


def test_central_far_prompt_shows_location_only(tmp_path):
    sim = make_sim(tmp_path, book="central")
    ag = next(iter(sim.agents.values()))
    bx, by = sim.world.book_pos
    ag.x, ag.y = (bx + 9) % sim.world.W, by
    obs = sim.build_observation(ag)
    assert obs.book and not obs.book["readable"]
    _, user = build_prompt(obs)
    assert f"({bx}, {by})" in user
    assert "[regrow_wood]" not in user      # contents hidden until you travel


def test_freeform_changes_format_instructions_only(tmp_path):
    sim = make_sim(tmp_path)
    obs = sim.build_observation(next(iter(sim.agents.values())))
    _, strict = build_prompt(obs)
    _, free = build_prompt(obs, freeform=True)
    assert "Respond ONLY with a JSON object" in strict
    assert "think out loud" in free and "MUST END with one JSON object" in free
    assert strict.split("Respond ONLY")[0] == free.split("You may think")[0]
    # parser already tolerates prose around the object
    res = parse_response('I wonder about fire...\n{"reasoning": "r", '
                         '"plan": [{"action": "wait"}]}\nDone thinking.')
    assert res.plan == [{"action": "wait"}]


def test_validator_accepts_book_steps():
    raw = json.dumps({"reasoning": "r", "plan": [
        {"action": "book_answer", "question": "coverage", "answer": "lots of grass"},
        {"action": "book_note", "text": "axe = 2 wood + 1 stone"}]})
    res = parse_response(raw)
    assert res.plan[0]["question"] == "coverage"
    assert res.plan[1]["action"] == "book_note"


def test_none_condition_logs_identical_to_pre_book_build(tmp_path):
    """book_condition=none must not perturb rng or events (M1/M3 determinism)."""
    s1 = make_sim(tmp_path / "a", book="none")
    s2 = make_sim(tmp_path / "b")     # default: none
    for _ in range(30):
        s1.step()
        s2.step()
    s1.logf.close()
    s2.logf.close()
    e1 = (tmp_path / "a" / "run_none" / "events.jsonl").read_text(encoding="utf-8")
    e2 = (tmp_path / "b" / "run_none" / "events.jsonl").read_text(encoding="utf-8")
    assert e1 == e2 and e1


# ------------------------------------------------------------------ survey
class FakeJudgeProvider:
    parallel_safe = False
    name = "fake-judge"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_survey_judge_grades_and_is_change_gated(tmp_path):
    questions = [{"id": "q1", "text": "Q one?"}, {"id": "q2", "text": "Q two?"}]
    truth = {"q1": "truth one", "q2": "truth two"}
    out = tmp_path / "survey.jsonl"
    prov = FakeJudgeProvider([json.dumps({"q1": True, "q2": False}),
                              "garbage not json",
                              json.dumps({"q1": True, "q2": True})])
    judge = SurveyJudge(prov, questions, truth, str(out))
    assert judge.grade(10, {"Asha": "wood regrows"}) == 1
    # unchanged text -> no second call
    assert judge.grade(20, {"Asha": "wood regrows"}) == 0
    # failed parse is recorded and retried next wave (text not marked graded)
    assert judge.grade(30, {"Asha": "wood regrows in 40"}) == 1
    assert judge.grade(40, {"Asha": "wood regrows in 40"}) == 1
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["ok"] for r in recs] == [True, False, True]
    assert recs[0]["results"] == {"q1": True, "q2": False}
    assert recs[2]["tick"] == 40
    # ground truth reaches the judge prompt; subject text too
    system, user = prov.calls[0]
    assert "GROUND TRUTH: truth one" in user and "wood regrows" in user


def test_survey_judge_empty_subjects_cost_nothing(tmp_path):
    judge = SurveyJudge(FakeJudgeProvider([]), [{"id": "q", "text": "?"}],
                        {"q": "t"}, str(tmp_path / "s.jsonl"))
    assert judge.grade(5, {"Asha": "   "}) == 0


# ------------------------------------------------------------------ CLI
def test_cli_screenshot_reconstruction(tmp_path, monkeypatch):
    import ti
    sim = make_sim(tmp_path, book="central")
    for _ in range(10):
        sim.step()
    sim.logf.close()
    run_dir = tmp_path / "run_central"
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(sim.cfg, f)
    evs = ti.read_events(run_dir)
    assert any(e["type"] == "structure_built" and e["payload"]["structure"] == "book"
               for e in evs)
    out = ti.render_png(sim.cfg, evs, 9, run_dir / "shot.png")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data == ti.render_png(sim.cfg, evs, 9, run_dir / "shot2.png").read_bytes()


def test_cli_first_correct():
    import ti
    recs = [
        {"tick": 10, "subject": "A", "ok": True, "results": {"q1": False, "q2": True}},
        {"tick": 20, "subject": "A", "ok": True, "results": {"q1": True, "q2": True}},
        {"tick": 20, "subject": "B", "ok": False, "results": None},
        {"tick": 30, "subject": "B", "ok": True, "results": {"q1": True, "q2": False}},
    ]
    fc = ti.first_correct(recs)
    assert fc["q2"] == {"A": 10}
    assert fc["q1"] == {"A": 20, "B": 30}
