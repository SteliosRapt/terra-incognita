# Run Performance Telemetry and Gemma 4 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ordinary runs record streaming request and tick performance telemetry, then benchmark the ten approved Gemma 4 model/mode combinations through that same run path.

**Architecture:** Providers return a structured completion object containing assembled text and timing/token/cost metrics while preserving the existing `complete()` string API. Planners and survey judges add run context and parse outcomes to `requests.jsonl`; `Simulation` exposes per-tick planning metrics; a focused `experiments/performance.py` module aggregates normal run artifacts and a `experiments/benchmark_gemma4.py` module orchestrates and compares deterministic ordinary runs.

**Tech Stack:** Python 3.11+, stdlib dataclasses/JSON/CSV/statistics/subprocess, `requests` streaming HTTP, pytest, existing Terra Incognita simulation and CLI.

---

## Starting State and File Responsibilities

The worktree already contains uncommitted OpenRouter reasoning and detailed-cost changes in `brains/providers.py`, `tests/test_openrouter.py`, and `ti.py`. Treat them as prior work: preserve their behavior, adapt them into the structured telemetry design, and do not revert them.

- Create `brains/telemetry.py`: completion metrics, provider error metrics, and thread-safe JSONL request recorder.
- Modify `brains/providers.py`: streaming OpenRouter/Ollama implementations and thinking controls.
- Modify `brains/base.py`: expose planning-attempt count on `PlanResult`.
- Modify `brains/llm_planner.py`: attach tick/agent/attempt/parse results to provider metrics.
- Modify `brains/survey.py`: emit survey-role request records.
- Modify `engine/sim.py`: measure planning and full tick wall time.
- Create `experiments/performance.py`: aggregate `requests.jsonl` and `ticks.jsonl` into `performance.json`.
- Create `experiments/benchmark_gemma4.py`: matrix construction, resume logic, subprocess orchestration, and JSON/CSV/Markdown comparison reports.
- Modify `ti.py`: ordinary-run telemetry wiring, thinking CLI, warm-up support, performance generation, and benchmark subcommand.
- Create `tests/test_provider_streaming.py`: OpenRouter/Ollama streaming and thinking tests.
- Create `tests/test_performance.py`: aggregation and tick-timing tests.
- Create `tests/test_benchmark_gemma4.py`: matrix, resume, and report tests.
- Modify `tests/test_openrouter.py` and `tests/test_m3.py`: migrate existing expectations and cover backward compatibility.

### Task 1: Structured Completion Metrics and Request Recorder

**Files:**
- Create: `brains/telemetry.py`
- Modify: `brains/providers.py`
- Test: `tests/test_provider_streaming.py`

- [ ] **Step 1: Write failing tests for completion metrics and JSONL recording**

Create `tests/test_provider_streaming.py` with the core API expectations:

```python
import json

from brains.telemetry import CompletionMetrics, CompletionResult, RequestRecorder


def test_completion_metrics_derive_visible_tokens_and_tps():
    metrics = CompletionMetrics(
        requested_model="google/gemma-4-31b-it",
        actual_model="google/gemma-4-31b-it",
        started_at=100.0,
        first_token_at=100.25,
        finished_at=101.25,
        prompt_tokens=600,
        cached_tokens=100,
        completion_tokens=80,
        reasoning_tokens=50,
        cost_usd=0.001,
    )
    assert metrics.ttft_ms == 250
    assert metrics.duration_ms == 1250
    assert metrics.visible_response_tokens == 30
    assert metrics.generation_tps == 80.0


def test_request_recorder_merges_context_and_metrics(tmp_path):
    path = tmp_path / "requests.jsonl"
    recorder = RequestRecorder(path)
    result = CompletionResult(
        text='{"plan":[{"action":"wait"}]}',
        reasoning_text="consider options",
        metrics=CompletionMetrics(
            requested_model="m/requested", actual_model="m/routed",
            started_at=10.0, first_token_at=10.1, finished_at=10.5,
            prompt_tokens=20, cached_tokens=5, completion_tokens=12,
            reasoning_tokens=7, cost_usd=0.002,
        ),
    )
    recorder.write(result, role="planner", tick=4, subject="A02",
                   attempt=1, provider_ok=True, parse_ok=False,
                   error="JSONDecodeError: bad response")
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["role"] == "planner"
    assert row["tick"] == 4
    assert row["subject"] == "A02"
    assert row["actual_model"] == "m/routed"
    assert row["visible_response_tokens"] == 5
    assert row["parse_ok"] is False
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_provider_streaming.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'brains.telemetry'`.

- [ ] **Step 3: Implement the structured telemetry types**

Create `brains/telemetry.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import pathlib
import threading


@dataclass
class CompletionMetrics:
    requested_model: str
    actual_model: str
    started_at: float
    first_token_at: float | None
    finished_at: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    token_split_method: str = "provider_reported"

    @property
    def ttft_ms(self) -> int | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000)

    @property
    def duration_ms(self) -> int:
        return round((self.finished_at - self.started_at) * 1000)

    @property
    def visible_response_tokens(self) -> int:
        return max(0, self.completion_tokens - self.reasoning_tokens)

    @property
    def generation_tps(self) -> float | None:
        if self.first_token_at is None or self.finished_at <= self.first_token_at:
            return None
        return self.completion_tokens / (self.finished_at - self.first_token_at)

    def as_record(self) -> dict:
        return {**asdict(self), "ttft_ms": self.ttft_ms,
                "duration_ms": self.duration_ms,
                "visible_response_tokens": self.visible_response_tokens,
                "generation_tps": self.generation_tps}


@dataclass
class CompletionResult:
    text: str
    reasoning_text: str
    metrics: CompletionMetrics


class CompletionError(RuntimeError):
    def __init__(self, message: str, metrics: CompletionMetrics):
        super().__init__(message)
        self.metrics = metrics


class RequestRecorder:
    _lock = threading.Lock()

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)

    def write(self, result: CompletionResult | CompletionError, **context):
        metrics = result.metrics
        rec = {**context, **metrics.as_record()}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

Modify `Provider` in `brains/providers.py` to preserve compatibility:

```python
def complete_measured(self, system: str, user: str) -> CompletionResult:
    started = time.time()
    text = self.complete(system, user)
    finished = time.time()
    return CompletionResult(text, "", CompletionMetrics(
        requested_model=self.name, actual_model=self.name,
        started_at=started, first_token_at=None, finished_at=finished))
```

Providers with real metrics override `complete_measured`; their `complete`
method becomes `return self.complete_measured(system, user).text`.

- [ ] **Step 4: Run the focused tests**

Run the Step 2 command. Expected: both tests pass.

- [ ] **Step 5: Commit the telemetry foundation**

```powershell
git add brains/telemetry.py brains/providers.py tests/test_provider_streaming.py
git commit -m "feat: add structured provider telemetry"
```

### Task 2: Stream OpenRouter and Ollama with Thinking Controls

**Files:**
- Modify: `brains/providers.py`
- Modify: `tests/test_provider_streaming.py`
- Modify: `tests/test_openrouter.py`

- [ ] **Step 1: Add failing streaming tests**

Add scripted streaming response helpers that implement `iter_lines()` and
assert these behaviors:

```python
def test_openrouter_stream_records_first_token_usage_and_reasoning():
    chunks = [
        'data: {"model":"routed/gemma","choices":[{"delta":{"reasoning":"think"}}]}',
        'data: {"model":"routed/gemma","choices":[{"delta":{"content":"{\\"plan\\":"}}]}',
        'data: {"model":"routed/gemma","choices":[{"delta":{"content":"[]}"}}],"usage":{"prompt_tokens":40,"completion_tokens":9,"cost":0.0003,"prompt_tokens_details":{"cached_tokens":10},"completion_tokens_details":{"reasoning_tokens":4}}}',
        'data: [DONE]',
    ]
    provider = make_streaming_openrouter(chunks, thinking=True)
    result = provider.complete_measured("system", "user")
    assert result.text == '{"plan":[]}'
    assert result.reasoning_text == "think"
    assert result.metrics.actual_model == "routed/gemma"
    assert result.metrics.cached_tokens == 10
    assert result.metrics.reasoning_tokens == 4
    assert provider._requests.bodies[0]["stream"] is True
    assert provider._requests.bodies[0]["reasoning"] == {"enabled": True}


def test_openrouter_nonthinking_sends_explicit_disable():
    provider = make_streaming_openrouter(valid_chunks(), thinking=False)
    provider.complete_measured("system", "user")
    assert provider._requests.bodies[0]["reasoning"] == {"enabled": False}


def test_ollama_stream_assembles_thinking_content_and_native_counts():
    chunks = [
        {"message": {"thinking": "consider ", "content": ""}, "done": False},
        {"message": {"thinking": "", "content": '{"plan":[]}'}, "done": False},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 30, "eval_count": 8,
         "eval_duration": 400_000_000},
    ]
    provider = make_streaming_ollama(chunks, thinking=True)
    result = provider.complete_measured("system", "user")
    assert result.text == '{"plan":[]}'
    assert result.reasoning_text == "consider "
    assert result.metrics.prompt_tokens == 30
    assert result.metrics.completion_tokens == 8
    assert provider._requests.bodies[0]["think"] is True
```

Also test Ollama `think=False`, HTTP errors producing `CompletionError`, and
OpenRouter 429/402 fallback preserving the existing cooldown behavior.

- [ ] **Step 2: Verify the streaming tests fail**

Run:

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_provider_streaming.py tests/test_openrouter.py -q
```

Expected: failures show missing `thinking`, streaming request arguments, and
`complete_measured` overrides.

- [ ] **Step 3: Implement OpenRouter SSE streaming**

Update `OpenRouterProvider.__init__` to accept `thinking: bool | None = None`
alongside the existing `reasoning_effort`. Build the request with:

```python
body = {
    "model": model,
    "messages": messages,
    "temperature": self.temperature,
    "max_tokens": self.max_tokens,
    "stream": True,
    "usage": {"include": True},
}
if self.thinking is not None:
    body["reasoning"] = {"enabled": self.thinking}
elif self.reasoning_effort:
    body["reasoning"] = {"effort": self.reasoning_effort}
```

Use `requests.post(..., stream=True)`, parse only `data:` lines, set
`first_token_at` on the first non-empty reasoning or content delta, assemble
the two text channels separately, and take token/cost fields from the final
usage block. Continue writing the compatibility `costs.jsonl` record from the
same metrics object.

- [ ] **Step 4: Implement Ollama NDJSON streaming**

Update `OllamaProvider.__init__` with `thinking: bool | None = None` and
`max_tokens: int = 2048`. Send `stream=True`, `think` when explicitly set,
and `options.num_predict=max_tokens`. Parse one JSON object per line, assemble
`message.thinking` separately from `message.content`, capture TTFT from the
first non-empty delta, and use `prompt_eval_count`, `eval_count`, and
`eval_duration` from the done record.

Ollama does not expose separate reasoning token counts. Count non-empty
thinking and content stream deltas, then divide `eval_count` proportionally
between the two channels using delta counts; when only one channel is present,
assign all completion tokens to it. Add `token_split_method="stream_delta_ratio"`
to its metrics record so the estimate is explicit.

- [ ] **Step 5: Run provider tests and the full existing suite**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_provider_streaming.py tests/test_openrouter.py -q
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
```

Expected: provider tests pass; the full suite remains green.

- [ ] **Step 6: Commit streaming providers**

```powershell
git add brains/providers.py tests/test_provider_streaming.py tests/test_openrouter.py
git commit -m "feat: stream OpenRouter and Ollama completions"
```

### Task 3: Attribute Requests to Planner and Survey Outcomes

**Files:**
- Modify: `brains/base.py`
- Modify: `brains/llm_planner.py`
- Modify: `brains/survey.py`
- Modify: `tests/test_m3.py`
- Modify: `tests/test_provider_streaming.py`

- [ ] **Step 1: Add failing planner recorder tests**

Add tests using a provider whose `complete_measured` returns known metrics:

```python
def test_planner_records_each_attempt_and_parse_outcome(tmp_path):
    provider = MeasuredCannedProvider(["not json", VALID_WAIT_JSON])
    requests_path = tmp_path / "requests.jsonl"
    brain = LLMPlannerBrain(provider, request_path=str(requests_path))
    result = brain.plan(observation())
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    assert result.planning_attempts == 2
    assert [row["attempt"] for row in rows] == [1, 2]
    assert [row["parse_ok"] for row in rows] == [False, True]
    assert all(row["role"] == "planner" for row in rows)


def test_survey_records_role_subject_and_parse_result(tmp_path):
    provider = MeasuredCannedProvider([VALID_SURVEY_JSON])
    judge = SurveyJudge(provider, QUESTIONS, TRUTH,
                        str(tmp_path / "survey.jsonl"),
                        request_path=str(tmp_path / "requests.jsonl"))
    assert judge.grade(10, {"A00": "a changed belief"}) == 1
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["role"] == "survey"
    assert row["subject"] == "A00"
    assert row["tick"] == 10
```

- [ ] **Step 2: Run tests and confirm missing constructor fields**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_m3.py tests/test_provider_streaming.py -q
```

Expected: failures mention `request_path` and `planning_attempts`.

- [ ] **Step 3: Add attempt metadata to `PlanResult`**

Modify `brains/base.py`:

```python
@dataclass
class PlanResult:
    plan: list
    intention: str = ""
    updated_beliefs: list | None = None
    updated_notes: str | None = None
    parse_error: bool = False
    planning_attempts: int = 1
```

- [ ] **Step 4: Record planner requests after parsing**

Add `request_path` to both planner constructors. For each attempt, call
`complete_measured`, parse `result.text`, then write one request record with
provider and parse success separated. If `CompletionError` is raised, write
its metrics with `provider_ok=False` and `parse_ok=False`. Return
`planning_attempts=attempt` on success and `planning_attempts=2` on fallback.

Keep `prompts.jsonl` unchanged so existing replay and debugging tools continue
to work.

- [ ] **Step 5: Record survey requests**

Add `request_path` to `SurveyJudge`. Use `complete_measured`, write role
`survey`, subject ID, tick, attempt 1, and parse status around `_extract_json`.
The existing `survey.jsonl` verdict record remains unchanged.

- [ ] **Step 6: Run focused and full tests**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_m3.py tests/test_provider_streaming.py -q
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit request attribution**

```powershell
git add brains/base.py brains/llm_planner.py brains/survey.py tests/test_m3.py tests/test_provider_streaming.py
git commit -m "feat: record planner and survey request outcomes"
```

### Task 4: Measure Tick Timing and Aggregate Performance

**Files:**
- Modify: `engine/sim.py`
- Create: `experiments/performance.py`
- Create: `tests/test_performance.py`

- [ ] **Step 1: Add failing tick and aggregation tests**

Create `tests/test_performance.py` with deterministic records:

```python
def test_simulation_exposes_tick_and_planning_metrics(tmp_path):
    sim = make_one_agent_sim(tmp_path, brain=RetryingBrain())
    sim.step()
    metrics = sim.last_step_metrics
    assert metrics["tick"] == 0
    assert metrics["duration_ms"] >= metrics["planning_ms"] >= 0
    assert metrics["planner_calls"] == 1
    assert metrics["retries"] == 1
    assert metrics["final_parse_failures"] == 0


def test_summarize_performance_splits_tokens_cost_and_parse_rates(tmp_path):
    write_requests(tmp_path / "requests.jsonl", [
        request(tick=0, subject="A00", attempt=1, parse_ok=False,
                prompt=100, cached=20, completion=80, reasoning=50,
                ttft=200, duration=1000, tps=100, cost=0.01),
        request(tick=0, subject="A00", attempt=2, parse_ok=True,
                prompt=110, cached=0, completion=40, reasoning=10,
                ttft=100, duration=500, tps=100, cost=0.005),
    ])
    write_ticks(tmp_path / "ticks.jsonl", [
        {"tick": 0, "duration_ms": 1200, "planning_ms": 1100,
         "planner_calls": 1, "retries": 1, "final_parse_failures": 0}
    ])
    summary = summarize_run(tmp_path, expected_ticks=1)
    planner = summary["groups"]["planner"]
    assert planner["average_tokens_per_plan"]["reasoning"] == 60
    assert planner["average_tokens_per_plan"]["visible_response"] == 60
    assert planner["cost_per_request"] == pytest.approx(0.0075)
    assert planner["cost_per_plan"] == pytest.approx(0.015)
    assert planner["cost_per_tick"] == pytest.approx(0.015)
    assert planner["first_attempt_parse_failure_pct"] == 100.0
    assert planner["final_parse_failure_pct"] == 0.0
```

Also assert nearest-rank p50/p95 values and that charged provider failures are
included in token/cost totals.

- [ ] **Step 2: Verify tests fail**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_performance.py -q
```

Expected: missing `last_step_metrics` and `experiments.performance`.

- [ ] **Step 3: Instrument `Simulation.step`**

Import `time`. At step entry capture `step_started = time.monotonic()`. Around
the existing planning block capture `planning_started`, then calculate:

```python
attempts = sum(result.planning_attempts for result in results.values())
self.last_step_metrics = {
    "tick": self.tick,
    "duration_ms": round((time.monotonic() - step_started) * 1000),
    "planning_ms": round(planning_elapsed * 1000),
    "planner_calls": len(results),
    "retries": attempts - len(results),
    "final_parse_failures": sum(r.parse_error for r in results.values()),
}
```

For ticks without planning, use zero for planning fields. Assign the record
before incrementing `self.tick`.

- [ ] **Step 4: Implement performance aggregation**

Create `experiments/performance.py` with:

```python
def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[index]


def summarize_run(run_dir: pathlib.Path, expected_ticks: int | None = None) -> dict:
    requests = read_jsonl(run_dir / "requests.jsonl")
    ticks = read_jsonl(run_dir / "ticks.jsonl")
    groups = {role: summarize_requests(
        [row for row in requests if row["role"] == role], ticks)
        for role in sorted({row["role"] for row in requests})}
    groups["combined"] = summarize_requests(requests, ticks)
    return {
        "completed_ticks": len(ticks),
        "expected_ticks": expected_ticks,
        "complete": expected_ticks is None or len(ticks) == expected_ticks,
        "tick_time_ms": distribution([row["duration_ms"] for row in ticks]),
        "planning_time_ms": distribution([row["planning_ms"] for row in ticks]),
        "total_time_s": sum(row["duration_ms"] for row in ticks) / 1000,
        "groups": groups,
    }
```

`summarize_requests` must group logical plans by `(tick, subject)`, count first
attempt failures from `attempt == 1`, count final failures when no attempt for
that logical plan has `parse_ok=True`, report both per-request averages and
per-logical-plan sums averaged across plans, and report totals over every
charged request.

- [ ] **Step 5: Run aggregation and full tests**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_performance.py -q
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit timing and aggregation**

```powershell
git add engine/sim.py experiments/performance.py tests/test_performance.py
git commit -m "feat: aggregate ordinary run performance"
```

### Task 5: Wire Telemetry into Every Ordinary Run

**Files:**
- Modify: `ti.py`
- Modify: `tests/test_performance.py`
- Modify: `tests/test_openrouter.py`

- [ ] **Step 1: Add failing ordinary-run integration tests**

Add a mock-provider CLI-level test that calls a factored `run_experiment(args)`
for three ticks and asserts:

```python
assert (run_dir / "requests.jsonl").exists()
assert len(read_jsonl(run_dir / "ticks.jsonl")) == 3
performance = json.loads((run_dir / "performance.json").read_text())
assert performance["completed_ticks"] == 3
assert performance["complete"] is True
assert performance["groups"]["planner"]["request_count"] > 0
```

Add argument parsing tests for `--thinking default|on|off` and `--warmup`.

- [ ] **Step 2: Verify integration tests fail**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_performance.py tests/test_openrouter.py -q
```

Expected: missing telemetry artifacts and CLI arguments.

- [ ] **Step 3: Factor and wire the ordinary run path**

Refactor `cmd_run` into `run_experiment(args) -> pathlib.Path` plus a thin
printing wrapper. At run creation:

```python
requests_path = run_dir / "requests.jsonl"
ticks_path = run_dir / "ticks.jsonl"
requests_path.write_text("", encoding="utf-8")
ticks_path.write_text("", encoding="utf-8")
```

Pass `request_path` to planner brains and survey judges. After each
`sim.step()`, append `sim.last_step_metrics` to `ticks.jsonl`. In a `finally`
block, close the simulation log and write:

```python
performance = summarize_run(run_dir, expected_ticks=args.ticks)
(run_dir / "performance.json").write_text(
    json.dumps(performance, indent=2), encoding="utf-8")
```

This guarantees a partial performance file after interruption or provider
failure.

- [ ] **Step 4: Add explicit thinking and warm-up options**

Add:

```python
r.add_argument("--thinking", choices=["default", "on", "off"],
               default="default")
r.add_argument("--warmup", action="store_true",
               help="perform one excluded provider warm-up request")
```

Map `on` and `off` to provider `thinking=True` and `False`; omit the provider
argument for `default`. Warm-up uses a separate provider instance with
`warmup_requests.jsonl` and `warmup_costs.jsonl`, calls a small valid planning
prompt once, and never appends to ordinary `requests.jsonl` or `ticks.jsonl`.

- [ ] **Step 5: Replace the temporary cost-only summary**

Keep `costs.jsonl` compatibility, but print the canonical planner/survey/
combined values from `performance.json`: total time, requests, prompt/cache/
reasoning/visible tokens, average TTFT/TPS, parse failure rates, and cost.

- [ ] **Step 6: Run focused tests and a mock smoke run**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_performance.py tests/test_openrouter.py -q
$env:PYTHONPATH='scratch/python-deps'; py ti.py run --provider mock --agents 5 --ticks 5 --seed 42 --survey off --name telemetry_mock_smoke
```

Expected: tests pass and `runs/telemetry_mock_smoke/performance.json` reports
five completed ticks.

- [ ] **Step 7: Run the full suite and commit ordinary-run telemetry**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
git add ti.py tests/test_performance.py tests/test_openrouter.py
git commit -m "feat: emit performance data for ordinary runs"
```

### Task 6: Add the Gemma 4 Benchmark Orchestrator and Reports

**Files:**
- Create: `experiments/benchmark_gemma4.py`
- Create: `tests/test_benchmark_gemma4.py`
- Modify: `ti.py`

- [ ] **Step 1: Add failing matrix and reporting tests**

Create `tests/test_benchmark_gemma4.py`:

```python
def test_gemma4_matrix_has_ten_exact_entries():
    rows = gemma4_matrix()
    assert len(rows) == 10
    assert {(r.provider, r.model, r.thinking) for r in rows} == {
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
    (run_dir / "performance.json").write_text(
        json.dumps({"complete": True, "completed_ticks": 100}))
    assert should_run(entry, tmp_path) is False


def test_reports_include_required_comparison_columns(tmp_path):
    write_completed_performance(tmp_path, gemma4_matrix()[:2])
    paths = write_reports(tmp_path, gemma4_matrix()[:2])
    header = paths["csv"].read_text().splitlines()[0]
    assert "average_tick_ms" in header
    assert "average_ttft_ms" in header
    assert "average_generation_tps" in header
    assert "cost_per_plan" in header
    assert "first_attempt_parse_failure_pct" in header
```

- [ ] **Step 2: Verify benchmark tests fail**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_benchmark_gemma4.py -q
```

Expected: missing `experiments.benchmark_gemma4`.

- [ ] **Step 3: Implement the matrix and deterministic names**

Create an immutable entry type:

```python
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
```

Return the exact ten entries asserted in Step 1.

- [ ] **Step 4: Implement subprocess orchestration and resume behavior**

For each incomplete entry, run the ordinary CLI sequentially:

```python
cmd = [
    sys.executable, str(ROOT / "ti.py"), "run",
    "--provider", entry.provider,
    "--model", entry.model,
    "--thinking", "on" if entry.thinking else "off",
    "--agents", "5", "--ticks", "100", "--seed", "42",
    "--condition", "blind", "--book", "none",
    "--survey", "off", "--warmup", "--name", entry.run_name,
]
completed = subprocess.run(cmd, cwd=ROOT, check=False)
```

Record exit code and timestamps in `benchmark_status.json`. Continue to the
next entry after a failure. A completed performance file is never overwritten.

- [ ] **Step 5: Implement JSON, CSV, and Markdown reports**

Flatten each completed `performance.json` into columns for model, provider,
thinking, completion status, total time/tokens/cost, average and p50/p95 tick,
planning, and TTFT values, average/median TPS, token categories per plan,
cost per plan/tick, request/retry counts, and both parse failure percentages.

Write:

- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_results.md`

Sort report rows by model size, free before paid, then thinking off before on.

- [ ] **Step 6: Add the CLI subcommand**

Add:

```python
b = sub.add_parser("benchmark", help="run a reproducible model benchmark")
b.add_argument("suite", choices=["gemma4"])
b.add_argument("--report-only", action="store_true")
b.set_defaults(fn=cmd_benchmark)
```

`cmd_benchmark` delegates to `run_gemma4_benchmark` and prints report paths.

- [ ] **Step 7: Run benchmark tests and CLI help**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest tests/test_benchmark_gemma4.py -q
$env:PYTHONPATH='scratch/python-deps'; py ti.py benchmark --help
```

Expected: tests pass and help lists `gemma4` plus `--report-only`.

- [ ] **Step 8: Run the full suite and commit the orchestrator**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
git add experiments/benchmark_gemma4.py tests/test_benchmark_gemma4.py ti.py
git commit -m "feat: add Gemma 4 benchmark matrix"
```

### Task 7: Validate Locally, Run the Paid Matrix, and Publish Results

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Generated, ignored: `runs/benchmark_gemma4_*`
- Generated, ignored: benchmark JSON/CSV/Markdown reports under `runs/`

- [ ] **Step 1: Verify the local model and Ollama thinking modes**

```powershell
& 'C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe' list
$env:PYTHONPATH='scratch/python-deps'; py ti.py run --provider ollama --model gemma4:e2b --thinking off --agents 1 --ticks 2 --survey off --name benchmark_local_off_smoke
$env:PYTHONPATH='scratch/python-deps'; py ti.py run --provider ollama --model gemma4:e2b --thinking on --agents 1 --ticks 2 --survey off --name benchmark_local_on_smoke
```

Expected: `gemma4:e2b` is listed and both smoke runs produce complete
`performance.json` files with two ticks.

- [ ] **Step 2: Verify OpenRouter thinking modes with one paid smoke call each**

Run two one-tick, one-agent paid 26B smoke runs, one thinking on and one off.
Confirm `requests.jsonl` shows the requested mode, non-null TTFT, and actual
cost. Stop if either backend rejects the explicit reasoning control.

- [ ] **Step 3: Run the ten-entry benchmark**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py ti.py benchmark gemma4
```

Expected: all ten entries complete 100 ticks, and combined reports are written.
This step requires explicit approval because it makes paid OpenRouter calls.

- [ ] **Step 4: Validate report arithmetic against raw ledgers**

For every completed row, independently sum `requests.jsonl` token and cost
fields and compare them to `performance.json` and the combined CSV. Assert:

```python
assert completion_tokens == reasoning_tokens + visible_response_tokens
assert reported_total_cost == pytest.approx(sum(request_costs))
assert completed_ticks == 100
assert survey_request_count == 0
```

- [ ] **Step 5: Document usage and measured results**

Add to `README.md`:

```markdown
## Performance telemetry

Every `ti.py run` writes `requests.jsonl`, `ticks.jsonl`, and
`performance.json`. Run the fixed Gemma 4 comparison with:

    py ti.py benchmark gemma4

Use `--report-only` to rebuild comparison reports without making model calls.
```

Add a dated `CHANGELOG.md` entry listing the benchmark command, all ten rows,
report locations, and any failed or unsupported mode explicitly.

- [ ] **Step 6: Run final verification**

```powershell
$env:PYTHONPATH='scratch/python-deps'; py -m pytest -q
$env:PYTHONPATH='scratch/python-deps'; py ti.py benchmark gemma4 --report-only
git diff --check
```

Expected: all tests pass, reports regenerate without network calls, and Git
reports no whitespace errors.

- [ ] **Step 7: Commit documentation and result summary**

```powershell
git add README.md CHANGELOG.md
git commit -m "docs: publish Gemma 4 benchmark results"
```
