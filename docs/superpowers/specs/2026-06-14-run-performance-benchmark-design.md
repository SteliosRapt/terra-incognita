# Ordinary Run Performance Telemetry and Gemma 4 Benchmark Design

## Goal

Make every ordinary Terra Incognita run emit enough timing, token, parse, and
cost telemetry to compare model performance. Add a benchmark orchestrator that
runs a fixed Gemma 4 matrix using the same ordinary run path.

## Scope

Every ordinary run records provider request metrics and per-tick wall-clock
timing. Benchmarking is orchestration and reporting over those normal run
artifacts, not a separate simulation implementation.

The initial benchmark matrix contains ten runs:

- OpenRouter Gemma 4 31B free, thinking on and off.
- OpenRouter Gemma 4 31B paid, thinking on and off.
- OpenRouter Gemma 4 26B A4B free, thinking on and off.
- OpenRouter Gemma 4 26B A4B paid, thinking on and off.
- Local Ollama `gemma4:e2b`, thinking on and off.

Each benchmark run uses five agents, 100 ticks, seed 42, blind knowledge, no
book, and no survey. OpenRouter's actual 31B label is used rather than calling
the model 30B. Gemma 4 12B is omitted because OpenRouter does not expose it.

## Ordinary Run Artifacts

Each ordinary run writes these additional files:

- `requests.jsonl`: one record for every provider request, including retries
  and failed calls.
- `ticks.jsonl`: one record for every simulation tick.
- `performance.json`: aggregate metrics derived from the raw records.

Existing `events.jsonl`, `prompts.jsonl`, `costs.jsonl`, survey files, and run
configuration remain supported. New telemetry must not change simulation
behavior.

### Request Records

Each request record contains:

- Run role: planner or survey.
- Provider and requested model.
- Actual routed model where the provider reports it.
- Tick, agent or survey subject, and retry attempt.
- Request start timestamp.
- Time to first generated token in milliseconds.
- Total request duration in milliseconds.
- Prompt tokens.
- Cached prompt tokens.
- Completion tokens.
- Reasoning tokens.
- Visible response tokens, calculated as completion minus reasoning tokens.
- Generation TPS, calculated as completion tokens divided by elapsed time from
  first generated token through completion.
- Actual provider cost in USD.
- HTTP/provider success, response parse success, and error text.

Streaming is required for OpenRouter and Ollama so time to first token is
measured rather than estimated. The provider assembles streamed text before
returning it to the existing planner parser.

### Tick Records

Each tick record contains:

- Tick number.
- Tick start timestamp.
- Total wall-clock duration.
- Time spent in the planning phase.
- Number of planner calls triggered during the tick.
- Number of retries and parse failures.

Tick time includes all work performed by that ordinary simulation tick.
Planning time is measured separately around planning execution in the engine.

## Thinking Modes

Thinking is an explicit boolean capability, independent of reasoning effort.

- OpenRouter receives reasoning enabled or disabled through its reasoning
  request control.
- Ollama receives its supported `think` boolean control.
- Reasoning content remains excluded from the JSON plan passed to the parser.
- The raw provider metrics retain separate reasoning and visible response token
  counts where the API exposes them.

If a backend cannot honor the requested thinking mode, the matrix entry fails
clearly instead of silently running a different mode.

## Aggregation

`performance.json` reports planner, survey, and combined groups when surveys
are enabled. Benchmark runs disable surveys and therefore compare planner work
only.

Per group, report:

- Average, p50, and p95 tick duration.
- Average, p50, and p95 planning duration.
- Average, p50, and p95 first-token latency.
- Average and median generation TPS.
- Average prompt, cached, reasoning, visible response, completion, and total
  tokens per plan attempt.
- Cost per plan attempt and cost per tick.
- Total wall-clock time, prompt tokens, cached tokens, reasoning tokens,
  visible response tokens, completion tokens, and cost.
- First-attempt parse failure percentage.
- Final parse failure percentage after the one retry.
- Request count, successful plan count, retry count, and failed request count.

Cost totals include charged failed requests. Free and local calls report zero
provider cost.

## Benchmark Orchestration

`ti.py benchmark gemma4` runs the ten matrix entries through the ordinary run
command implementation. It writes each result under a deterministic benchmark
directory and generates combined JSON, CSV, and Markdown reports.

The orchestrator:

- Skips entries whose `performance.json` confirms 100 completed ticks.
- Preserves partial or failed entries for diagnosis.
- Regenerates combined reports from all available completed entries.
- Runs entries sequentially so local and remote resource contention does not
  distort comparisons.
- Performs one unmeasured warm-up request before each entry. Warm-up cost and
  telemetry are stored separately and excluded from benchmark aggregates.

## Concurrency and Comparability

OpenRouter retains ordinary run concurrency, allowing planner calls for the
five agents to execute in parallel. Ollama follows its provider's ordinary
execution behavior. Reports expose both request-level TPS and wall-clock tick
time so concurrency does not get confused with single-request generation
speed.

All matrix entries use the same world seed, agent count, prompt construction,
retry policy, token ceiling, and simulation configuration. Model and thinking
mode are the only intended differences.

## Error Handling

- Network and provider failures generate request records with timing and any
  provider-reported token or cost data.
- A malformed streamed response is retried by the existing planner policy and
  counted as a first-attempt parse failure.
- A second parse failure produces the existing wait fallback and counts toward
  final parse failures.
- Benchmark orchestration continues to the next entry after a failed entry and
  includes the failure in the combined report.
- Interrupted runs keep valid line-delimited telemetry and can be diagnosed or
  replaced without deleting other benchmark entries.

## Testing

Tests cover:

- Streaming chunk assembly and true first-token timestamp capture.
- OpenRouter and Ollama thinking controls.
- Reasoning versus visible token accounting.
- Charged failed-call accounting.
- Tick and planning timers.
- Percentile, TPS, parse-rate, per-plan, and per-tick aggregation.
- Planner/survey role separation.
- Matrix construction, deterministic naming, resume behavior, and report
  generation.
- Backward compatibility for ordinary runs and existing provider tests.

