import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.benchmark_gemma4 import (build_run_command, gemma4_matrix, should_run,
                                          write_reports)


def test_gemma4_matrix_has_ten_exact_entries():
    rows = gemma4_matrix()

    assert len(rows) == 10
    assert {(row.provider, row.model, row.thinking) for row in rows} == {
        ("openrouter", "google/gemma-4-31b-it:free", True),
        ("openrouter", "google/gemma-4-31b-it:free", False),
        ("openrouter", "google/gemma-4-31b-it", True),
        ("openrouter", "google/gemma-4-31b-it", False),
        ("openrouter", "google/gemma-4-26b-a4b-it:free", True),
        ("openrouter", "google/gemma-4-26b-a4b-it:free", False),
        ("openrouter", "google/gemma-4-26b-a4b-it", True),
        ("openrouter", "google/gemma-4-26b-a4b-it", False),
        ("ollama", "gemma4:e2b", True),
        ("ollama", "gemma4:e2b", False),
    }


def test_completed_entry_is_skipped(tmp_path):
    entry = gemma4_matrix()[0]
    run_dir = tmp_path / entry.run_name
    run_dir.mkdir()
    (run_dir / "performance.json").write_text(json.dumps({
        "complete": True, "completed_ticks": 100,
    }), encoding="utf-8")

    assert should_run(entry, tmp_path) is False


def test_matrix_run_command_uses_shared_8192_token_ceiling():
    command = build_run_command(gemma4_matrix()[0])

    index = command.index("--max-tokens")
    assert command[index + 1] == "8192"


def completed_performance():
    dist = {"average": 10.0, "p50": 9, "p95": 15}
    group = {
        "request_count": 20, "plan_count": 18, "successful_plans": 17,
        "retry_count": 2, "failed_request_count": 1,
        "first_attempt_parse_failure_pct": 11.1,
        "final_parse_failure_pct": 5.5,
        "ttft_ms": dist, "request_duration_ms": dist,
        "generation_tps": {"average": 50.0, "p50": 48, "p95": 60},
        "average_tokens_per_request": {},
        "average_tokens_per_plan": {
            "prompt": 100, "cached": 10, "completion": 50,
            "reasoning": 20, "visible_response": 30, "total": 150,
        },
        "cost_per_request": 0.001, "cost_per_plan": 0.0011,
        "cost_per_tick": 0.0002, "actual_models": ["routed/model"],
        "total": {"prompt_tokens": 2000, "cached_tokens": 200,
                  "completion_tokens": 1000, "reasoning_tokens": 400,
                  "visible_response_tokens": 600, "tokens": 3000,
                  "cost_usd": 0.02},
    }
    return {
        "complete": True, "completed_ticks": 100, "expected_ticks": 100,
        "tick_time_ms": dist, "planning_time_ms": dist,
        "total_time_s": 1.0, "groups": {"planner": group, "combined": group},
    }


def test_reports_include_required_comparison_columns(tmp_path):
    entries = gemma4_matrix()[:2]
    for entry in entries:
        run_dir = tmp_path / entry.run_name
        run_dir.mkdir()
        (run_dir / "performance.json").write_text(
            json.dumps(completed_performance()), encoding="utf-8")

    paths = write_reports(tmp_path, entries)

    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "average_tick_ms" in header
    assert "average_ttft_ms" in header
    assert "average_generation_tps" in header
    assert "cost_per_plan" in header
    assert "first_attempt_parse_failure_pct" in header
    assert len(json.loads(paths["json"].read_text(encoding="utf-8"))) == 2
