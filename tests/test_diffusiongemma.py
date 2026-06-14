"""test_diffusiongemma.py — the batched provider/brain path with FAKE model and
processor objects (no weights downloaded, local DiffusionGemma never touched),
and the 'exactly four configs' guarantee of the experiment matrix."""
import json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brains.providers import DiffusionGemmaProvider, make_provider
from brains.llm_planner import BatchLLMPlannerBrain
from brains.base import Observation
from experiments.diffusiongemma_matrix import load_matrix, build_configs

VALID = json.dumps({"reasoning": "ok", "plan": [{"action": "wait"}]})


class _Tok:
    padding_side = "right"
    pad_token_id = 0


class _FakeProcessor:
    """Mimics AutoProcessor's chat path with tiny torch tensors."""
    def __init__(self, scripted_replies):
        self.scripted = scripted_replies   # list of reply-lists, one per call
        self.tokenizer = _Tok()
        self.calls = 0
        self.batch_sizes = []
        self.seen_systems = []

    def apply_chat_template(self, messages, **kw):
        n = len(messages)
        self.batch_sizes.append(n)
        self.seen_systems += [m[0]["content"] for m in messages]
        L = 5
        return {"input_ids": torch.ones((n, L), dtype=torch.long),
                "attention_mask": torch.ones((n, L), dtype=torch.long)}

    def batch_decode(self, new, skip_special_tokens=True):
        replies = self.scripted[self.calls]
        self.calls += 1
        assert new.shape[0] == len(replies), "decoded count must match batch"
        return list(replies)


class _FakeModel:
    """Mimics DiffusionGemmaForBlockDiffusion.generate (no device -> no .to())."""
    def __init__(self):
        self.generate_calls = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
        n, _ = input_ids.shape
        self.generate_calls.append(n)
        gen = torch.full((n, 4), 7, dtype=input_ids.dtype)   # nonzero -> counted
        return torch.cat([input_ids, gen], dim=1)


def _obs(i):
    return Observation(tick=0,
                       me={"id": f"A{i:02d}", "pos": (i, i), "energy": 100,
                           "inventory": {}, "tools": set(), "standing_on": "grass",
                           "slots_used": 0, "slot_capacity": 8},
                       nearby_resources={}, nearby_agents=[], journal_tail=[],
                       world_size=(16, 16))


def test_batched_plan_one_call_for_whole_tick(tmp_path):
    proc = _FakeProcessor([[VALID, VALID, VALID]])
    model = _FakeModel()
    provider = DiffusionGemmaProvider(model=model, processor=proc)
    brain = BatchLLMPlannerBrain(provider,
                                 telemetry_path=str(tmp_path / "prompts.jsonl"),
                                 metrics_path=str(tmp_path / "metrics.jsonl"))
    results = brain.plan_batch([_obs(0), _obs(1), _obs(2)])

    assert len(results) == 3
    assert all(r.plan == [{"action": "wait"}] and not r.parse_error for r in results)
    # the whole tick was ONE generate of batch 3 (not three separate calls)
    assert model.generate_calls == [3]
    assert proc.batch_sizes == [3]

    metrics = [json.loads(l) for l in
               (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    m = metrics[0]
    assert m["batch_size"] == 3
    assert m["prompt_tokens"] == 15 and m["completion_tokens"] == 12   # 3x5, 3x4
    for k in ("duration_s", "tps", "device", "peak_gpu_mem_bytes", "total_tokens"):
        assert k in m

    prompts = (tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prompts) == 3 and all(json.loads(p)["ok"] for p in prompts)


def test_batched_retry_only_reprompts_failures(tmp_path):
    # agent 0 returns garbage first, valid on retry; agents 1,2 valid immediately
    proc = _FakeProcessor([["not json at all", VALID, VALID], [VALID]])
    model = _FakeModel()
    provider = DiffusionGemmaProvider(model=model, processor=proc)
    brain = BatchLLMPlannerBrain(provider,
                                 telemetry_path=str(tmp_path / "prompts.jsonl"),
                                 metrics_path=str(tmp_path / "metrics.jsonl"))
    results = brain.plan_batch([_obs(0), _obs(1), _obs(2)])

    assert all(not r.parse_error for r in results)
    assert model.generate_calls == [3, 1]      # retry batch holds ONLY the failure
    # agent A00 has two telemetry rows (fail, then ok); A01/A02 one each
    rows = [json.loads(p) for p in
            (tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    a00 = [r for r in rows if r["agent"] == "A00"]
    assert len(a00) == 2 and a00[0]["ok"] is False and a00[1]["ok"] is True
    assert len([r for r in rows if r["agent"] == "A01"]) == 1


def test_make_provider_is_lazy_no_weights():
    p = make_provider({"backend": "diffusiongemma"})
    assert isinstance(p, DiffusionGemmaProvider)
    assert "diffusiongemma" in p.name
    assert p.model is None and p.processor is None       # nothing loaded yet
    assert p.thinking is True                            # thinking ON by default
    assert p.think_token == "<|think|>"


def test_thinking_strips_thought_channel_keeps_json(tmp_path):
    # the model emits reasoning in a thought channel, then the JSON answer
    reply = ("<|channel>thought\nI should rest this tick to save energy."
             "<channel|> " + VALID)
    proc = _FakeProcessor([[reply]])
    model = _FakeModel()
    provider = DiffusionGemmaProvider(model=model, processor=proc)  # thinking=True
    assert provider.thinking is True
    # the system prompt sent to the model must start with the think token
    out_texts, _ = provider.complete_batch([("SYSTEM", "USER")])
    assert proc.seen_systems[0].startswith("<|think|>")
    # only the post-delimiter answer comes back, and it parses
    assert "thought" not in out_texts[0]
    from brains.llm_planner import parse_response
    assert parse_response(out_texts[0]).plan == [{"action": "wait"}]


def test_quantization_param_validates_and_is_lazy():
    import pytest
    p = make_provider({"backend": "diffusiongemma", "quantization": "4bit"})
    assert p.quantization == "4bit"
    assert p.model is None and p.processor is None      # still nothing loaded
    assert DiffusionGemmaProvider().quantization is None  # default: full precision
    with pytest.raises(ValueError):
        DiffusionGemmaProvider(quantization="3bit")


def test_analyze_cell_occupancy_and_energy_saved(tmp_path):
    from experiments.analyze_diffusiongemma import analyze_cell
    cfg = {"grid_size": 16, "seed": 1, "n_agents": 1, "agent_names": ["A00"],
           "forest_fraction": 0.1, "mountain_fraction": 0.05, "stone_per_tile": 12,
           "wood_regrow_ticks": 40, "berry_patches": 2, "berry_patch_size": 8,
           "berry_regrow_prob": 0.05, "water_fraction": 0.05, "lakes": 1,
           "forest_regions": 2, "mountain_ranges": 1, "book_condition": "none",
           "combine_note": False}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def ev(tick, typ, **kw):
        return json.dumps({"tick": tick, "type": typ, **kw})
    lines = [
        ev(0, "moved", actor="A00", position=[5, 5]),
        ev(1, "structure_built", actor="A00", position=[5, 5],
           payload={"structure": "shelter"}),
        ev(1, "recipe_learned", actor="A00", payload={"recipe": "shelter",
                                                      "source": "discovered"}),
        ev(5, "moved", actor="A00", position=[5, 5]),   # marks last tick = 5
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    c = analyze_cell(tmp_path)
    assert c["shelters_built"] == 1 and c["shelter_owners"] == ["A00"]
    # on the shelter at the start of ticks 2,3,4,5 -> 4 sheltered agent-ticks,
    # 4 energy saved (built during t1, so t1 doesn't count)
    assert c["shelter_occupancy_agentticks"] == 4
    assert c["energy_saved_by_shelters"] == 4
    assert c["survivors"] == 1


def test_matrix_builds_exactly_four_configs():
    cfgs = build_configs(load_matrix())
    assert len(cfgs) == 4
    assert {c["name"] for c in cfgs} == {
        "dg_note-off_book-none", "dg_note-off_book-librarian",
        "dg_note-on_book-none", "dg_note-on_book-librarian"}
    for c in cfgs:
        assert c["cfg"]["combine_note"] == (c["note"] == "on")
        assert c["cfg"]["n_agents"] == 7
        assert c["cfg"]["knowledge_condition"] == "blind"
        assert c["ticks"] == 1000
        if c["book"] == "librarian":
            assert c["cfg"]["book_manager"] == "A00"
        else:
            assert "book_manager" not in c["cfg"]
