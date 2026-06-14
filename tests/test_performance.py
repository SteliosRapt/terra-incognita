import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brains.base import PlanResult
from engine.defaults import BASE
from engine.sim import Simulation
from experiments.performance import percentile, summarize_run


class RetryingBrain:
    def plan(self, obs):
        return PlanResult([{"action": "wait"}], "wait",
                          planning_attempts=2)


def test_simulation_exposes_tick_and_planning_metrics(tmp_path):
    cfg = dict(BASE, seed=42, n_agents=1, knowledge_condition="blind",
               book_condition="none")
    sim = Simulation(cfg, lambda rng: RetryingBrain(), str(tmp_path / "run"))

    sim.step()

    metrics = sim.last_step_metrics
    assert metrics["tick"] == 0
    assert metrics["duration_ms"] >= metrics["planning_ms"] >= 0
    assert metrics["planner_calls"] == 1
    assert metrics["retries"] == 1
    assert metrics["final_parse_failures"] == 0
    sim.logf.close()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def request(*, tick, subject, attempt, parse_ok, prompt, cached,
            completion, reasoning, ttft, duration, tps, cost):
    return {
        "role": "planner", "tick": tick, "subject": subject,
        "attempt": attempt, "provider_ok": True, "parse_ok": parse_ok,
        "requested_model": "m/test", "actual_model": "m/test",
        "prompt_tokens": prompt, "cached_tokens": cached,
        "completion_tokens": completion, "reasoning_tokens": reasoning,
        "visible_response_tokens": completion - reasoning,
        "ttft_ms": ttft, "duration_ms": duration,
        "generation_tps": tps, "cost_usd": cost,
    }


def test_summarize_performance_splits_tokens_cost_and_parse_rates(tmp_path):
    write_jsonl(tmp_path / "requests.jsonl", [
        request(tick=0, subject="A00", attempt=1, parse_ok=False,
                prompt=100, cached=20, completion=80, reasoning=50,
                ttft=200, duration=1000, tps=100, cost=0.01),
        request(tick=0, subject="A00", attempt=2, parse_ok=True,
                prompt=110, cached=0, completion=40, reasoning=10,
                ttft=100, duration=500, tps=100, cost=0.005),
    ])
    write_jsonl(tmp_path / "ticks.jsonl", [
        {"tick": 0, "duration_ms": 1200, "planning_ms": 1100,
         "planner_calls": 1, "retries": 1, "final_parse_failures": 0},
    ])

    summary = summarize_run(tmp_path, expected_ticks=1)
    planner = summary["groups"]["planner"]

    assert summary["complete"] is True
    assert summary["tick_time_ms"] == {"average": 1200.0,
                                        "p50": 1200, "p95": 1200}
    assert planner["average_tokens_per_plan"]["reasoning"] == 60
    assert planner["average_tokens_per_plan"]["visible_response"] == 60
    assert planner["cost_per_request"] == pytest.approx(0.0075)
    assert planner["cost_per_plan"] == pytest.approx(0.015)
    assert planner["cost_per_tick"] == pytest.approx(0.015)
    assert planner["first_attempt_parse_failure_pct"] == 100.0
    assert planner["final_parse_failure_pct"] == 0.0
    assert planner["ttft_ms"] == {"average": 150.0, "p50": 100, "p95": 200}
    assert planner["generation_tps"]["average"] == 100.0
    assert planner["total"]["cost_usd"] == pytest.approx(0.015)


def test_percentile_uses_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert percentile([], 0.5) is None
