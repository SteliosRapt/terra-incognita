"""test_openrouter.py — OpenRouterProvider: fallback chain, cooldowns, spend
ledger. No network: requests is replaced with a scripted fake."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brains.providers import OpenRouterProvider, make_provider
from ti import cost_breakdown


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body

    def iter_lines(self):
        if "error" in self._body:
            lines = ["data: " + json.dumps(self._body), "data: [DONE]"]
        elif self._body:
            message = self._body["choices"][0]["message"]
            chunk = {
                "model": "routed/test",
                "choices": [{"delta": {"content": message["content"]}}],
                "usage": self._body.get("usage") or {},
            }
            lines = ["data: " + json.dumps(chunk), "data: [DONE]"]
        else:
            lines = []
        return iter(line.encode("utf-8") for line in lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def ok_body(text, cost=0.0, pt=100, ct=50):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost": cost}}


class FakeRequests:
    """Stands in for the requests module: pops scripted responses, records
    which model each POST asked for."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.models_called = []
        self.bodies = []

    def post(self, url, headers=None, json=None, timeout=None, stream=None):
        self.models_called.append(json["model"])
        self.bodies.append(json)
        return self.responses.pop(0)


def make(fake, models=("m/a:free", "m/b:free"), **kw):
    p = OpenRouterProvider(list(models), api_key="test-key", **kw)
    p._requests = fake
    return p


def test_happy_path_uses_primary_and_tracks_usage(tmp_path):
    cost_file = tmp_path / "costs.jsonl"
    fake = FakeRequests(FakeResponse(200, ok_body("hello", cost=0.0012)))
    p = make(fake, cost_path=str(cost_file))
    assert p.complete("sys", "user") == "hello"
    assert fake.models_called == ["m/a:free"]
    assert p.totals["calls"] == 1
    assert p.totals["prompt_tokens"] == 100
    assert p.totals["completion_tokens"] == 50
    assert p.totals["cost_usd"] == pytest.approx(0.0012)
    rec = json.loads(cost_file.read_text(encoding="utf-8").strip())
    assert rec["model"] == "routed/test"
    assert rec["cost_usd"] == pytest.approx(0.0012)
    assert rec["total_calls"] == 1


def test_reasoning_effort_is_sent_and_detailed_usage_is_recorded(tmp_path):
    cost_file = tmp_path / "costs.jsonl"
    body = ok_body("hello", cost=0.0042, pt=120, ct=80)
    body["usage"]["prompt_tokens_details"] = {"cached_tokens": 70}
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 55}
    fake = FakeRequests(FakeResponse(200, body))
    p = make(fake, reasoning_effort="high", max_tokens=8192,
             cost_path=str(cost_file))

    assert p.complete("sys", "user") == "hello"
    assert fake.bodies[0]["reasoning"] == {"effort": "high"}
    assert fake.bodies[0]["max_tokens"] == 8192
    assert p.totals["cached_tokens"] == 70
    assert p.totals["reasoning_tokens"] == 55
    rec = json.loads(cost_file.read_text(encoding="utf-8").strip())
    assert rec["cached_tokens"] == 70
    assert rec["reasoning_tokens"] == 55


def test_cost_breakdown_groups_planner_and_survey_models(tmp_path):
    planner = tmp_path / "costs.jsonl"
    survey = tmp_path / "survey_costs.jsonl"
    planner.write_text(
        json.dumps({"model": "deepseek/v4", "prompt_tokens": 100,
                    "cached_tokens": 40, "completion_tokens": 80,
                    "reasoning_tokens": 50, "cost_usd": 0.002}) + "\n" +
        json.dumps({"model": "deepseek/v4", "prompt_tokens": 120,
                    "cached_tokens": 0, "completion_tokens": 60,
                    "reasoning_tokens": 20, "cost_usd": 0.003}) + "\n",
        encoding="utf-8")
    survey.write_text(
        json.dumps({"model": "judge/free", "prompt_tokens": 200,
                    "completion_tokens": 30, "cost_usd": 0}) + "\n",
        encoding="utf-8")

    rows, total = cost_breakdown({"planner": planner, "survey": survey})

    assert rows == [
        {"role": "planner", "model": "deepseek/v4", "calls": 2,
         "prompt_tokens": 220, "cached_tokens": 40,
         "completion_tokens": 140, "reasoning_tokens": 70,
         "visible_output_tokens": 70, "cost_usd": pytest.approx(0.005)},
        {"role": "survey", "model": "judge/free", "calls": 1,
         "prompt_tokens": 200, "cached_tokens": 0,
         "completion_tokens": 30, "reasoning_tokens": 0,
         "visible_output_tokens": 30, "cost_usd": 0},
    ]
    assert total["calls"] == 3
    assert total["prompt_tokens"] == 420
    assert total["cost_usd"] == pytest.approx(0.005)


def test_429_falls_back_to_next_model_and_cooldown_sticks():
    fake = FakeRequests(FakeResponse(429),
                        FakeResponse(200, ok_body("from b")),
                        FakeResponse(200, ok_body("from b again")))
    p = make(fake)
    assert p.complete("s", "u") == "from b"
    assert fake.models_called == ["m/a:free", "m/b:free"]
    assert p.totals["rate_limited"] == 1
    # primary still cooling: the next call goes straight to b
    assert p.complete("s", "u") == "from b again"
    assert fake.models_called[-1] == "m/b:free"


def test_primary_retried_after_cooldown_expires():
    fake = FakeRequests(FakeResponse(429),
                        FakeResponse(200, ok_body("from b")),
                        FakeResponse(200, ok_body("from a")))
    p = make(fake)
    assert p.complete("s", "u") == "from b"
    p._cooldown_until["m/a:free"] = 0.0   # cooldown over; preference order wins again
    assert p.complete("s", "u") == "from a"
    assert fake.models_called == ["m/a:free", "m/b:free", "m/a:free"]


def test_all_models_limited_waits_then_recovers():
    fake = FakeRequests(FakeResponse(429, headers={"Retry-After": "0"}),
                        FakeResponse(429, headers={"Retry-After": "0"}),
                        FakeResponse(200, ok_body("eventually")))
    p = make(fake)
    assert p.complete("s", "u") == "eventually"
    assert p.totals["rate_limited"] == 2


def test_402_uses_long_credit_cooldown():
    fake = FakeRequests(FakeResponse(402),
                        FakeResponse(200, ok_body("from b")))
    p = make(fake, credit_cooldown_s=3600.0)
    assert p.complete("s", "u") == "from b"
    import time
    assert p._cooldown_until["m/a:free"] > time.time() + 3000


def test_provider_error_in_body_raises():
    fake = FakeRequests(FakeResponse(200, {"error": {"message": "upstream down"}}))
    p = make(fake)
    with pytest.raises(RuntimeError, match="upstream down"):
        p.complete("s", "u")


def test_http_500_raises_no_silent_fallback():
    fake = FakeRequests(FakeResponse(500))
    p = make(fake)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        p.complete("s", "u")
    assert len(fake.models_called) == 1   # did NOT try the next model


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(["m/a:free"])


def test_make_provider_reads_config_and_model_overrides_chain(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    p = make_provider({"backend": "openrouter"})
    cfg = json.loads((ROOT / "configs" / "openrouter.json").read_text(encoding="utf-8"))
    assert p.models == cfg["models"]
    assert p.models[0].startswith("google/gemma-4")    # gemma 4 is the main model
    p2 = make_provider({"backend": "openrouter", "model": "x/y:free, z/w"})
    assert p2.models == ["x/y:free", "z/w"]


def test_make_provider_paid_tier_uses_config_paid_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = json.loads((ROOT / "configs" / "openrouter.json").read_text(encoding="utf-8"))
    p = make_provider({"backend": "openrouter", "tier": "paid"})
    assert p.models == cfg["paid_models"] == ["openai/gpt-5-nano"]


def test_make_provider_preset_selects_chain(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = json.loads((ROOT / "configs" / "openrouter.json").read_text(encoding="utf-8"))
    p = make_provider({"backend": "openrouter", "preset": "owl"})
    assert p.models == cfg["presets"]["owl"]
    assert p.models[0] == "openrouter/owl-alpha"
    with pytest.raises(ValueError, match="unknown preset"):
        make_provider({"backend": "openrouter", "preset": "nope"})
    # explicit --model still beats a preset
    p2 = make_provider({"backend": "openrouter", "preset": "owl", "model": "x/y"})
    assert p2.models == ["x/y"]


def _write_run(d, events, with_prompts=None):
    d.mkdir()
    with open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    if with_prompts is not None:
        with open(d / "prompts.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(with_prompts) + "\n")


def test_compute_baseline_includes_scripted_and_mock_excludes_llm(tmp_path, monkeypatch):
    import viewer.server as vs
    ev = lambda t, ty, **pl: {"tick": t, "type": ty, "actor": "A00", "payload": pl}
    # scripted m2-style run: no prompts.jsonl — 100 ticks, 1 death, 4/5 plans done
    _write_run(tmp_path / "m2_x",
               [ev(0, "planner_called"), ev(1, "plan_completed"),
                ev(2, "planner_called"), ev(3, "plan_completed"),
                ev(4, "planner_called"), ev(5, "plan_completed"),
                ev(6, "planner_called"), ev(7, "plan_completed"),
                ev(8, "planner_called"),
                ev(10, "gathered", resource="wood", qty=1),
                ev(50, "agent_died"),
                ev(99, "recipe_learned", recipe="axe", source="discovered")])
    # mock run: prompts.jsonl says provider=mock — counted
    _write_run(tmp_path / "ui_mock", [ev(0, "gathered", resource="berries", qty=1),
                                      ev(99, "moved")],
               with_prompts={"provider": "mock"})
    # LLM run: provider=ollama — excluded
    _write_run(tmp_path / "m3_llm", [ev(0, "agent_died"), ev(99, "agent_died")],
               with_prompts={"provider": "ollama:gemma4:e2b"})
    monkeypatch.setattr(vs, "RUNS", tmp_path)
    b = vs.compute_baseline()
    assert sorted(b["runs"]) == ["m2_x", "ui_mock"]
    assert b["ticks"] == 200
    assert b["deaths_per_1k"] == 5.0          # 1 death / 200 ticks
    assert b["completion_pct"] == 80.0        # 4 of 5 plans
    assert b["gathers_per_100"] == 1.0        # 2 gathers / 200 ticks
    assert b["recipes_discovered"] == 1
