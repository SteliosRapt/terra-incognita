# DiffusionGemma Shelter Experiment Plan

> **Status (2026-06-14): aborted, Colab path parked.** DiffusionGemma was too
> slow to run locally, and the first attempts at the Colab implementation did not
> fit in the available Colab GPU/RAM. The experiment was therefore dropped as the
> first experimental pass — that pass now runs on the benchmarked
> `google/gemma-4-26b-a4b-it` (paid OpenRouter, no thinking) instead (see
> README "First-pass experiment planner" and CHANGELOG v0.41). The Colab
> implementation may still be worth finishing for a later, heavier pass: the
> design and code below are kept intact for that. The local DiffusionGemma install
> was never used and remains untouched.

## Directory Changes

- Save this plan as `DIFFUSIONGEMMA_EXPERIMENT_PLAN.md` in the repository root during implementation.
- Create:
  - `notebooks/diffusiongemma_colab_2x2.ipynb`
  - `experiments/diffusiongemma_matrix.py`
  - `experiments/analyze_diffusiongemma.py`
  - `configs/experiments/diffusiongemma_2x2.json`
- Keep model caches, checkpoints, and generated runs untracked.

## Mechanics

- Begin from `v0.39-prompt-fixes`.
- Add the recipe `1 stone -> 1 brick`.
- Add the recipe `4 bricks -> 1 permanent shelter`.
- Permit each agent to build at most one shelter.
- Any agent may benefit from any shelter.
- Reject excess shelter attempts without consuming bricks.
- Energy costs:
  - Remaining on a shelter: zero energy.
  - Moving from a shelter: one energy.
  - Normal idle: one energy.
  - Normal movement: two energy.
- Render shelters as greener squares in ASCII, PNG, and browser views.
- Add `--combine-note {on,off}`, default `off`.
- Note-on prompts contain exactly: "An item can sometimes be combined by itself."
- Note-off prompts contain no equivalent hint.

## DiffusionGemma

- The locally installed DiffusionGemma model and runner will not be used, inspected, modified, moved, or included in the experiment.
- All experiment inference occurs inside Google Colab.
- Colab downloads the official `google/diffusiongemma-26B-A4B-it` Transformers checkpoint into its own Hugging Face cache or optional Google Drive cache.
- Add a lazy Transformers provider that loads `AutoProcessor` and `DiffusionGemmaForBlockDiffusion` once.
- Require GPU execution and fail clearly when no CUDA device is available.
- Use `dtype="auto"` and `device_map="auto"`.
- Disable thinking to reduce output length while retaining the model's standard diffusion sampler.
- Batch all agents that require planning during the same tick.
- Preserve individual planner telemetry while also recording batch size, tokens, duration, TPS, device allocation, and peak GPU memory.

## Experiment Matrix

Run four headless conditions with seven agents, 1,000 ticks, seed 42, blind knowledge, replan interval 15, and surveys disabled:

1. Note off, no book.
2. Note on, no book.
3. Note off, librarian book.
4. Note on, librarian book.

For librarian conditions, `A00` is the sole writer.

Each run produces the normal `config.json`, `events.jsonl`, `prompts.jsonl`, and `ground_truth.json`, plus `model_metrics.jsonl`.

## Analysis

- Report survival count, death tick, and evidence-based causes of death.
- Measure combine attempts, brick discoveries, shelters built, shelter occupancy, and estimated energy saved.
- Assess fire understanding using construction, cooking, speech, and book-answer evidence.
- Compare Note and Book main effects and their interaction.
- State clearly that one seed per condition provides exploratory, not statistically conclusive, results.

## Testing

- Test brick and shelter recipes.
- Test the one-shelter-per-builder limit and retained materials after rejection.
- Test shared shelter use and all four energy cases.
- Test exact Note inclusion and exclusion.
- Test batched DiffusionGemma integration using fake model objects, without downloading weights.
- Test generation of exactly four experiment configurations.
- Run the complete local test suite without invoking local DiffusionGemma.

## Notebook Specification

- Clone the repository at a pinned commit.
- Accept parameters for repository ref, Hugging Face token, cache location, output location, seed, and `RUN_FULL_MATRIX`.
- Optionally mount Google Drive for model caching and durable results.
- Install pinned Transformers, Accelerate, PyTorch, and analysis dependencies.
- Display GPU, CUDA, RAM, and disk information before downloading.
- Download and load the official Transformers checkpoint entirely within Colab.
- Benchmark batch sizes one and seven and report end-to-end TPS.
- Run a short seven-agent smoke test.
- Present projected matrix runtime, storage, parse-success rate, and GPU-memory usage before the full-run cell.
- Run the complete 2x2 matrix only when `RUN_FULL_MATRIX=True`.
- Skip completed conditions when the notebook is rerun.
- Generate summary tables and plots.
- Package all logs, metrics, analysis, configurations, and commit metadata into one downloadable ZIP.

---

## Implementation notes (as built)

- **Base branch:** `v0.39-prompt-fixes` already contains the v0.38 epistemics
  instruments (Book, survey judge) — it is a superset of `main` — so the Book
  conditions the matrix needs are present. Work is on branch
  `diffusiongemma-shelter`.
- **Recipes** live in `configs/techtree_v0.json`: `brick` (`1 stone`, a
  self-combine) and `shelter` (`4 brick`, permanent). They flow automatically
  into the viewer, run summaries, the questionnaire and ground truth.
- **One-shelter cap:** `engine/agent.py` adds `shelter_built`;
  `engine/actions.py` rejects a second shelter *before* consuming anything
  (`outcome: "shelter_limit"`).
- **Energy model:** `engine/sim.py` charges `0` idle energy when an agent starts
  a tick on any shelter tile (`1` otherwise); the per-step move surcharge in
  `engine/actions.py` is unchanged. The four cases fall out exactly, so each
  sheltered agent-tick saves exactly one energy (used by the analyzer).
- **Rendering:** ASCII `H` (`engine/sim.py`), greener PNG square
  (`ti.py STRUCT_RGB`), greener browser sprite (`viewer/static/sprites.js` +
  `app.js`).
- **The Note:** `brains/llm_planner.COMBINE_NOTE` is the exact string; `ti.py`
  exposes `--combine-note {on,off}` (default off).
- **Batched inference:** `brains/providers.DiffusionGemmaProvider` (lazy,
  GPU-only, `complete_batch`) + `brains/llm_planner.BatchLLMPlannerBrain`
  (shared by every agent). `engine/sim.py` plans the whole tick in one batched
  call when one shared batch-capable brain is in use. Batch telemetry →
  `model_metrics.jsonl`; per-agent telemetry → `prompts.jsonl` (unchanged).
- **Thinking is ON** (revised from the plan's "disable thinking" at the user's
  request). The provider prepends `<|think|>` to the system prompt (model card),
  the model reasons in a thought channel, and the provider keeps only the answer
  after the `<channel|>` delimiter so the JSON plan parses; the full reasoning is
  preserved in `prompts.jsonl`. `max_new_tokens=1024` for headroom. The Colab
  notebook installs `-U transformers torch accelerate` (the card pins nothing)
  and exposes the card's diffusion sampler settings (48 steps, temp 0.8→0.4,
  entropy 0.1 / 0.005) as commented `GEN_KWARGS` to map onto `generate()`.
- **Run/analyze:** `experiments/diffusiongemma_matrix.py` (builds exactly four
  cells, skips completed ones, `--provider mock` for local smoke) and
  `experiments/analyze_diffusiongemma.py`.
- **Tests:** `tests/test_shelter.py`, `tests/test_diffusiongemma.py`. The full
  suite (`py -m pytest tests/`) passes without any weights.
