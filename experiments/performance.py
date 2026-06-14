"""Aggregate ordinary-run request and tick telemetry."""
from __future__ import annotations

import json
import math
import pathlib
import statistics


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[index]


def distribution(values: list[float]) -> dict:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"average": None, "p50": None, "p95": None}
    return {
        "average": statistics.fmean(clean),
        "p50": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
    }


TOKEN_FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens",
                "reasoning_tokens", "visible_response_tokens")


def _token_averages(total: dict, denominator: int) -> dict:
    labels = {
        "prompt_tokens": "prompt",
        "cached_tokens": "cached",
        "completion_tokens": "completion",
        "reasoning_tokens": "reasoning",
        "visible_response_tokens": "visible_response",
    }
    out = {labels[field]: (total[field] / denominator if denominator else 0)
           for field in TOKEN_FIELDS}
    out["total"] = ((total["prompt_tokens"] + total["completion_tokens"])
                    / denominator if denominator else 0)
    return out


def summarize_requests(rows: list[dict], ticks: list[dict]) -> dict:
    plan_groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("role"), row.get("tick"), row.get("subject"))
        plan_groups.setdefault(key, []).append(row)

    total = {field: sum((row.get(field, 0) or 0) for row in rows)
             for field in TOKEN_FIELDS}
    total["tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    total["cost_usd"] = sum((row.get("cost_usd", 0.0) or 0.0)
                            for row in rows)
    request_count = len(rows)
    plan_count = len(plan_groups)
    first_failures = 0
    final_failures = 0
    for attempts in plan_groups.values():
        first = min(attempts, key=lambda row: row.get("attempt", 1))
        first_failures += not bool(first.get("parse_ok"))
        final_failures += not any(row.get("parse_ok") for row in attempts)

    tick_count = len(ticks)
    return {
        "request_count": request_count,
        "plan_count": plan_count,
        "successful_plans": plan_count - final_failures,
        "retry_count": sum(1 for row in rows if (row.get("attempt", 1) or 1) > 1),
        "failed_request_count": sum(not bool(row.get("provider_ok")) for row in rows),
        "first_attempt_parse_failure_pct": (100 * first_failures / plan_count
                                             if plan_count else 0.0),
        "final_parse_failure_pct": (100 * final_failures / plan_count
                                    if plan_count else 0.0),
        "ttft_ms": distribution([row.get("ttft_ms") for row in rows]),
        "request_duration_ms": distribution(
            [row.get("duration_ms") for row in rows]),
        "generation_tps": distribution(
            [row.get("generation_tps") for row in rows]),
        "average_tokens_per_request": _token_averages(total, request_count),
        "average_tokens_per_plan": _token_averages(total, plan_count),
        "cost_per_request": total["cost_usd"] / request_count if request_count else 0.0,
        "cost_per_plan": total["cost_usd"] / plan_count if plan_count else 0.0,
        "cost_per_tick": total["cost_usd"] / tick_count if tick_count else 0.0,
        "actual_models": sorted({row.get("actual_model") for row in rows
                                 if row.get("actual_model")}),
        "total": total,
    }


def summarize_run(run_dir: str | pathlib.Path,
                  expected_ticks: int | None = None) -> dict:
    run_dir = pathlib.Path(run_dir)
    requests = read_jsonl(run_dir / "requests.jsonl")
    ticks = read_jsonl(run_dir / "ticks.jsonl")
    roles = sorted({row.get("role", "unknown") for row in requests})
    groups = {role: summarize_requests(
        [row for row in requests if row.get("role", "unknown") == role], ticks)
        for role in roles}
    groups["combined"] = summarize_requests(requests, ticks)
    return {
        "completed_ticks": len(ticks),
        "expected_ticks": expected_ticks,
        "complete": expected_ticks is None or len(ticks) == expected_ticks,
        "tick_time_ms": distribution([row.get("duration_ms") for row in ticks]),
        "planning_time_ms": distribution([row.get("planning_ms") for row in ticks]),
        "total_time_s": sum((row.get("duration_ms", 0) or 0) for row in ticks) / 1000,
        "groups": groups,
    }
