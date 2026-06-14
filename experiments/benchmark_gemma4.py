"""Reproducible Gemma 4 benchmark orchestration over ordinary runs."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORT_DIR = RUNS / "benchmark_gemma4"


@dataclass(frozen=True)
class BenchmarkEntry:
    slug: str
    provider: str
    model: str
    thinking: bool

    @property
    def run_name(self) -> str:
        mode = "thinking" if self.thinking else "no-thinking"
        return f"benchmark_gemma4_{self.slug}_{mode}_5a_100t_s42"


def gemma4_matrix() -> list[BenchmarkEntry]:
    entries = []
    remote = [
        ("31b-free", "google/gemma-4-31b-it:free"),
        ("31b-paid", "google/gemma-4-31b-it"),
        ("26b-free", "google/gemma-4-26b-a4b-it:free"),
        ("26b-paid", "google/gemma-4-26b-a4b-it"),
    ]
    for slug, model in remote:
        for thinking in (False, True):
            entries.append(BenchmarkEntry(slug, "openrouter", model, thinking))
    for thinking in (False, True):
        entries.append(BenchmarkEntry("local-e2b", "ollama", "gemma4:e2b",
                                      thinking))
    return entries


def _read_json(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def should_run(entry: BenchmarkEntry, runs_root: pathlib.Path = RUNS) -> bool:
    performance = _read_json(runs_root / entry.run_name / "performance.json")
    return not (performance and performance.get("complete") is True
                and performance.get("completed_ticks") == 100)


def build_run_command(entry: BenchmarkEntry) -> list[str]:
    return [
        sys.executable, str(ROOT / "ti.py"), "run",
        "--provider", entry.provider, "--model", entry.model,
        "--thinking", "on" if entry.thinking else "off",
        "--agents", "5", "--ticks", "100", "--seed", "42",
        "--condition", "blind", "--book", "none",
        "--survey", "off", "--warmup", "--max-tokens", "8192",
        "--name", entry.run_name,
    ]


REPORT_FIELDS = [
    "slug", "provider", "model", "thinking", "run_name", "complete",
    "completed_ticks", "total_time_s", "average_tick_ms", "p50_tick_ms",
    "p95_tick_ms", "average_planning_ms", "p50_planning_ms",
    "p95_planning_ms", "average_ttft_ms", "p50_ttft_ms", "p95_ttft_ms",
    "average_generation_tps", "median_generation_tps", "request_count",
    "plan_count", "retry_count", "failed_request_count",
    "average_prompt_tokens_per_plan", "average_cached_tokens_per_plan",
    "average_reasoning_tokens_per_plan", "average_visible_tokens_per_plan",
    "average_completion_tokens_per_plan", "average_total_tokens_per_plan",
    "cost_per_request", "cost_per_plan", "cost_per_tick",
    "total_prompt_tokens", "total_cached_tokens", "total_reasoning_tokens",
    "total_visible_tokens", "total_completion_tokens", "total_tokens",
    "total_cost_usd", "first_attempt_parse_failure_pct",
    "final_parse_failure_pct", "actual_models",
]


def _flatten(entry: BenchmarkEntry, performance: dict | None) -> dict:
    row = {field: None for field in REPORT_FIELDS}
    row.update({"slug": entry.slug, "provider": entry.provider,
                "model": entry.model, "thinking": entry.thinking,
                "run_name": entry.run_name})
    if not performance:
        row["complete"] = False
        return row
    planner = performance.get("groups", {}).get("planner", {})
    tick = performance.get("tick_time_ms", {})
    planning = performance.get("planning_time_ms", {})
    ttft = planner.get("ttft_ms", {})
    tps = planner.get("generation_tps", {})
    per_plan = planner.get("average_tokens_per_plan", {})
    total = planner.get("total", {})
    row.update({
        "complete": performance.get("complete", False),
        "completed_ticks": performance.get("completed_ticks", 0),
        "total_time_s": performance.get("total_time_s"),
        "average_tick_ms": tick.get("average"),
        "p50_tick_ms": tick.get("p50"), "p95_tick_ms": tick.get("p95"),
        "average_planning_ms": planning.get("average"),
        "p50_planning_ms": planning.get("p50"),
        "p95_planning_ms": planning.get("p95"),
        "average_ttft_ms": ttft.get("average"),
        "p50_ttft_ms": ttft.get("p50"), "p95_ttft_ms": ttft.get("p95"),
        "average_generation_tps": tps.get("average"),
        "median_generation_tps": tps.get("p50"),
        "request_count": planner.get("request_count"),
        "plan_count": planner.get("plan_count"),
        "retry_count": planner.get("retry_count"),
        "failed_request_count": planner.get("failed_request_count"),
        "average_prompt_tokens_per_plan": per_plan.get("prompt"),
        "average_cached_tokens_per_plan": per_plan.get("cached"),
        "average_reasoning_tokens_per_plan": per_plan.get("reasoning"),
        "average_visible_tokens_per_plan": per_plan.get("visible_response"),
        "average_completion_tokens_per_plan": per_plan.get("completion"),
        "average_total_tokens_per_plan": per_plan.get("total"),
        "cost_per_request": planner.get("cost_per_request"),
        "cost_per_plan": planner.get("cost_per_plan"),
        "cost_per_tick": planner.get("cost_per_tick"),
        "total_prompt_tokens": total.get("prompt_tokens"),
        "total_cached_tokens": total.get("cached_tokens"),
        "total_reasoning_tokens": total.get("reasoning_tokens"),
        "total_visible_tokens": total.get("visible_response_tokens"),
        "total_completion_tokens": total.get("completion_tokens"),
        "total_tokens": total.get("tokens"),
        "total_cost_usd": total.get("cost_usd"),
        "first_attempt_parse_failure_pct": planner.get(
            "first_attempt_parse_failure_pct"),
        "final_parse_failure_pct": planner.get("final_parse_failure_pct"),
        "actual_models": ",".join(planner.get("actual_models", [])),
    })
    return row


def write_reports(runs_root: pathlib.Path = RUNS,
                  entries: list[BenchmarkEntry] | None = None,
                  report_dir: pathlib.Path | None = None) -> dict[str, pathlib.Path]:
    entries = entries or gemma4_matrix()
    report_dir = report_dir or runs_root
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [_flatten(entry, _read_json(
        runs_root / entry.run_name / "performance.json")) for entry in entries]
    json_path = report_dir / "benchmark_results.json"
    csv_path = report_dir / "benchmark_results.csv"
    md_path = report_dir / "benchmark_results.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Gemma 4 Benchmark Results",
        "",
        "| Model | Mode | Complete | Avg tick ms | TTFT ms | TPS | Tokens/plan | Cost | Parse fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mode = "thinking" if row["thinking"] else "no-thinking"
        lines.append(
            f"| {row['model']} | {mode} | {row['complete']} | "
            f"{row['average_tick_ms']} | {row['average_ttft_ms']} | "
            f"{row['average_generation_tps']} | "
            f"{row['average_total_tokens_per_plan']} | {row['total_cost_usd']} | "
            f"{row['first_attempt_parse_failure_pct']}% |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def run_gemma4_benchmark(runs_root: pathlib.Path = RUNS,
                         report_dir: pathlib.Path = REPORT_DIR,
                         report_only: bool = False) -> dict[str, pathlib.Path]:
    entries = gemma4_matrix()
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "benchmark_status.json"
    statuses = []
    if not report_only:
        for entry in entries:
            if not should_run(entry, runs_root):
                statuses.append({"run_name": entry.run_name, "status": "skipped"})
                continue
            started = time.time()
            cmd = build_run_command(entry)
            completed = subprocess.run(cmd, cwd=ROOT, check=False)
            statuses.append({"run_name": entry.run_name,
                             "status": "completed" if completed.returncode == 0
                             else "failed",
                             "returncode": completed.returncode,
                             "started_at": started, "finished_at": time.time()})
            status_path.write_text(json.dumps(statuses, indent=2),
                                   encoding="utf-8")
    return write_reports(runs_root, entries, report_dir)
