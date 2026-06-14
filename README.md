# Terra Incognita
Multi-agent simulation framework for studying collective discovery in LLM agent
societies: a survival grid world whose technology rules are hidden from the
agents, who must discover them by experimentation.

- SPEC.md — full design specification (world, tech tree, agent architecture,
  experimental design, metrics)
- IMPLEMENTATION.md — ground rules, milestones M1-M5, open questions
- CHANGELOG.md — version notes incl. measured behavioral effects of each change
- engine/ — world, agents, actions, events, tick loop (zero LLM dependencies);
  defaults.py holds the canonical run config shared by scripts and viewer
- brains/ — ForagerBrain, RandomScientistBrain, LLMPlannerBrain + providers
  (mock / ollama / anthropic / openrouter)
- configs/techtree_v0.json — the hidden ground-truth technology tree;
  configs/experiments/ — experiment matrix definitions
- viewer/ — live dashboard (`py viewer/server.py`): run configurator (incl. the
  combine-note knob), agents, events, stats, replay of past runs
- ti.py — experiment CLI (`py ti.py run …`, list/status/summary/screenshot)
- experiments/ — the DiffusionGemma 2×2 shelter study (matrix runner + analyzer,
  see DIFFUSIONGEMMA_EXPERIMENT_PLAN.md), the Gemma 4 planner benchmark
  (benchmark_gemma4.py), and run-performance aggregation (performance.py)
- notebooks/diffusiongemma_colab_2x2.ipynb — Colab GPU runner for that study
- run_m1.py / run_m2.py / run_m3.py — milestone acceptance scripts
- tests/ — pytest suite (`py -m pytest`), no LLM or network calls

Status: M2 complete; M3 (LLM planner) in progress — acceptance run pending.
Quick start: `py run_m3.py --provider mock --ticks 300` or `py viewer/server.py`

## Performance telemetry

Every `py ti.py run` writes raw request telemetry to `requests.jsonl`, per-tick
timings to `ticks.jsonl`, and aggregate speed, token, parse, and cost metrics to
`performance.json` in the run directory. Provider-reported prompt, cache,
completion, and reasoning tokens remain separate; visible response tokens and
generation TPS are also reported.

Run the fixed Gemma 4 comparison with:

```powershell
py ti.py benchmark gemma4
```

Each cell runs 5 agents, 100 ticks, seed 42, surveys off, no-thinking, a warm-up
request, and an 8192-token output ceiling. The matrix compares the two paid
OpenRouter routes, `google/gemma-4-31b-it` and `google/gemma-4-26b-a4b-it`.
Results are written as JSON, CSV, and Markdown under `runs/benchmark_gemma4/`;
completed entries are skipped when the command is resumed. Use
`py ti.py benchmark gemma4 --report-only` to rebuild reports.

The matrix was scoped down to the paid no-thinking routes after testing: the
`:free` routes throttle to per-request 429s once a day's free-token budget is
spent (the warm-up request fails first), and thinking-on roughly triples
wall-time and cost by emitting growing reasoning traces. Both remain reachable
through `py ti.py run --thinking on` for one-off measurements.

Some providers do not expose hidden reasoning as a distinct token stream. In
that case `reasoning_tokens` remains zero rather than estimating unavailable
hidden tokens. In particular, the tested local `gemma4:e2b` Ollama endpoint
returns no separate `message.thinking` stream even when thinking is requested.

## First-pass experiment planner

The first experimental pass — the Note × Book 2×2 (note off/on × book
none/librarian, the design in DIFFUSIONGEMMA_EXPERIMENT_PLAN.md) — uses
**`google/gemma-4-26b-a4b-it` on the paid OpenRouter route with thinking off** as
the agent planner. That choice is the conclusion of the `py ti.py benchmark
gemma4` matrix (full numbers in CHANGELOG v0.41); the decision process:

- **JSON reliability first.** Over a 5-agent × 100-tick run the 26B route had 0%
  first-attempt parse failures; the larger `gemma-4-31b-it` had 37.3% (3.0% after
  retries). For an unattended four-cell matrix, parse reliability dominates — a
  planner that needs constant retries is both slower and noisier to analyse.
- **Faster and cheaper anyway.** 26B also ran ~2.2× faster (62.8 vs 43.9 gen
  tps; 371s vs 815s for the 100-tick run) at ~half the cost ($0.011 vs $0.022),
  so reliability cost nothing here.
- **Paid, not `:free`.** The `:free` routes drop to per-request 429s once the
  day's free-token budget is spent — the warm-up request fails first — so they
  can't anchor an unattended run. The paid route is stable.
- **Thinking off.** Thinking-on roughly triples wall-time and cost (growing
  reasoning traces) with no parse benefit at this task, so the first pass omits
  it. It stays available via `py ti.py run --thinking on`.
- **No GPU.** An API-hosted model lets the first pass run from the CLI; the Colab
  DiffusionGemma path (GPU, batched) is reserved for a later, heavier pass.

This model is now the project default: the OpenRouter default chain in
`configs/openrouter.json` is the single paid `google/gemma-4-26b-a4b-it`, and the
viewer's model selector defaults to the **gemma26** preset (no thinking), with a
**gemma26-thinking** variant alongside it. The old free gemma-4 chain still lives
under the `gemma` preset (`py ti.py run --preset gemma`).

This first pass is one seed per cell — exploratory, not statistically conclusive.
