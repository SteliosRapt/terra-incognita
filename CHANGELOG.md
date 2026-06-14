# CHANGELOG / Design Notes

## v0.41 — Run performance telemetry + Gemma 4 benchmark (2026-06-14)
Branch `codex/run-performance-benchmark`. Ordinary runs now emit streaming
request telemetry (`requests.jsonl`), per-tick timings (`ticks.jsonl`), and an
aggregate `performance.json` (speed/token/parse/cost, planner vs survey vs
combined). Providers return structured `CompletionResult`/`CompletionMetrics`
while keeping the old `complete()` string API; OpenRouter and Ollama stream with
explicit thinking controls. New `experiments/performance.py` aggregates,
`experiments/benchmark_gemma4.py` orchestrates, `py ti.py benchmark gemma4` runs
the matrix (`--report-only` rebuilds reports without calls).

**Benchmark scope.** Each cell: 5 agents × 100 ticks, seed 42, blind, no book,
survey off, no-thinking, warm-up, `--max-tokens 8192`. The matrix was reduced to
the two **paid** OpenRouter routes after live testing (below). Reports under
`runs/benchmark_gemma4/benchmark_results.{json,csv,md}`; per-cell status in
`benchmark_status.json`.

**Results (both cells complete, 100/100 ticks; arithmetic reconciled against raw
`requests.jsonl` — cost to the cent, `completion == reasoning + visible`, 0
survey requests):**

| Model (resolved snapshot) | Total time | Avg tick | TTFT | TPS | Tok/plan | Total cost | First-attempt parse fail | Final parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma-4-31b-it` (…-20260402) | 815.0s | 8150ms | 1632ms | 43.9 | 1712 | $0.022413 | 37.3% | 3.0% |
| `gemma-4-26b-a4b-it` (…-20260403) | 371.4s | 3714ms | 1431ms | 62.8 | 1476 | $0.011397 | 0.0% | 0.0% |

The smaller 26B route was ~2.2× faster, ~half the cost, and far more JSON-reliable
(0% vs 37.3% first-attempt parse failures) than 31B; 31B recovered to 3.0% final
failures via retry. (31B discovered axe@t69, 0 deaths; 26B taught nothing, 1
death — single seed, not significant.) p50 tick is ~0–1ms because planning only
fires periodically; the cost lives in the p95 planning ticks.

**Dropped/unsupported modes (explicit):**
- `:free` OpenRouter routes — once a day's free-token budget is spent they throttle
  to per-request 429s; the warm-up request fails first and aborts the run. A free
  run *can* succeed when the budget is fresh (a 31B-free thinking run reached
  ~tick 54 earlier) but it's unreliable for an unattended matrix.
- thinking-on — works on both backends (reasoning control honored, reasoning
  tokens reported distinctly), but roughly triples wall-time/cost via growing
  reasoning traces; dropped for throughput-focused benchmarking. Still available
  via `py ti.py run --thinking on`.
- local Ollama `gemma4:e2b` — runs fine but exposes no separate `message.thinking`
  stream even with thinking requested, so `reasoning_tokens` stays 0; dropped from
  the matrix per request.

## v0.40 — DiffusionGemma 2×2 shelter experiment (2026-06-14)
Branch `diffusiongemma-shelter`, off `v0.39-prompt-fixes` (verified to be a
superset of `main`: it already carries the v0.38 Book/epistemics work the Book
conditions need). Verified the checkpoint exists:
`google/diffusiongemma-26B-A4B-it` (Google, public, `diffusion_gemma` arch);
inference API confirmed from the model card (`DiffusionGemmaForBlockDiffusion`
+ `AutoProcessor`, `generate(**inputs, max_new_tokens=...)`, thinking off).

**Shelter mechanics.** Two new recipes in `configs/techtree_v0.json`: `brick`
(`1 stone`, a *self-combine* — one item type, no partner) and `shelter`
(`4 brick`, a **permanent** structure). Each agent may build at most one
(`AgentState.shelter_built`); a second attempt is rejected *before* consuming
anything (`combine_failed outcome="shelter_limit"`, bricks retained). Any agent
benefits from any shelter: `engine/sim.py` charges 0 idle energy when an agent
starts a tick on a shelter tile (1 otherwise); the per-step move surcharge is
unchanged, so the four cases (on-shelter idle 0 / on-shelter move 1 / idle 1 /
move 2) fall out exactly — hence each sheltered agent-tick saves exactly 1
energy (the analyzer's energy estimate is therefore exact). Greener rendering
in ASCII (`H`), PNG (`ti.py`), and the browser viewer (new sprite + legend).

**The Note.** `brains/llm_planner.COMBINE_NOTE` = the exact string
"An item can sometimes be combined by itself."; `ti.py --combine-note {on,off}`
(default off) appends it to the system prompt (nothing equivalent when off). It
hints at the *shape* of the brick self-combine without naming any recipe.

**Batched inference.** `brains/providers.DiffusionGemmaProvider` — lazy,
GPU-only (fails clearly without CUDA), `dtype/device_map="auto"`, **thinking ON**
(model card: prepend `<|think|>` to the system prompt; the provider then drops
the thought channel, splitting on `<channel|>`, so the JSON plan parses while
the full reasoning stays in `prompts.jsonl`; `max_new_tokens=1024`). A real
`complete_batch` (left-padded, one `generate` per batch) also returns
batch size / tokens / duration / TPS / device / peak GPU memory.
`brains/llm_planner.BatchLLMPlannerBrain` is shared by every agent, and
`engine/sim.py` plans the whole tick in one batched call when one shared
batch-capable brain is in use (retries reprompt only the failures). Per-agent
prompts.jsonl is unchanged (prompt-identical to a serial run); batch metrics go
to `model_metrics.jsonl`.

**Experiment.** `configs/experiments/diffusiongemma_2x2.json` +
`experiments/diffusiongemma_matrix.py` build exactly four cells (Note off/on ×
Book none/librarian; 7 agents, 1000 ticks, seed 42, blind, replan 15, surveys
off; A00 the sole librarian writer), one model load for all four, completed
cells skipped on rerun (`--provider mock` for a local smoke).
`experiments/analyze_diffusiongemma.py` reports survival, combine attempts,
brick discoveries, shelters built, occupancy + energy saved, fire-understanding
evidence, and the Note/Book main effects + interaction — with the one-seed
exploratory caveat. `notebooks/diffusiongemma_colab_2x2.ipynb` does the
Colab-only run (clone at ref, optional Drive cache, GPU/disk report, download +
load, batch-1-vs-7 benchmark, smoke, projections, gated full run, tables/plots,
single ZIP). The local DiffusionGemma install is never touched.

**Tests.** `tests/test_shelter.py` (recipes, one-per-builder cap + retained
bricks, shared shelter, four energy cases, exact Note in/exclusion) and
`tests/test_diffusiongemma.py` (batched plan = one call/tick with fakes, retry
reprompts only failures, lazy provider, occupancy reconstruction, exactly four
configs). `test_m3` knowledge assertion now derives the recipe set from the
techtree. Full suite: 71 passing, no weights downloaded.

## v0.39 — Playtest fixes: recipe leak, WHY NOW, earshot feedback, torus doc (2026-06-13)
Claude played 15 planner calls by hand (2 agents × ~65 ticks, central book,
seed 42) through `scratch/play.py` — a replay driver where the operator IS
the brain: it reruns the sim from t0 against recorded replies and prints the
exact prompt of the first unanswered call. Context held up; five fixes came
out of it.

**Recipe leak (measurement bug):** the format example in `_SCHEMA_DOC`
showed `combine {"wood": 2, "stone": 1}` — which IS the axe. Models copy
examples verbatim, so every prior LLM run got a free "discovery" (suspect
in gemma's axe@t81/pickaxe@t39, 06-12 run). Example is now
`{"berries": 1, "wood": 1}` and `test_schema_example_combine_is_not_a_real_recipe`
resolves the example against the techtree under all tile/tool contexts so
a future techtree edit can't silently re-introduce a leak.

**WHY NOW (prompt section):** agents couldn't tell "plan finished" from
"plan discarded mid-experiment". New line after STATUS states the trigger:
first decision / plan finished / step failed / N-tick check-in with steps
set aside / someone spoke to you / survival reflex took over. The reflex
case was found live: A01's auto-eat at t58 silently ate a 2-step experiment.
Two new trigger tokens in `planner_called` payloads: `step_failed`, `reflex`.

**Earshot feedback:** a directed say to an out-of-range target journals
"(X was too far away to hear you.)" for the speaker. Before, "You said to
A00: …" read as success and produced false-confidence coordination
(observed at t33 and t55 in the playtest). In-range behavior unchanged.

**Torus + voice documented (system prompt):** "The world is a {W}x{H} grid
whose edges wrap around…" and "your voice only carries a few tiles" join
the survival basics — embodied common sense, same rationale as berries=food
(agents at y=4 saw NEARBY wood at y=31 with no way to read it as close).
Observation gains `world_size` and `replan_reason` fields (default-empty:
old constructors unaffected).

**Cosmetics:** world-placed structures render without ", built by world";
"a axe" → "an axe" (`_an()` in events.py).

**Verified:** 61 tests pass (11 new in `tests/test_prompt_fixes.py`); M1
determinism PASS, M2 determinism PASS. M1 hash moves `f10d156c…` →
`cd920f0d…` solely from the new `step_failed`/`reflex` trigger tokens in
`planner_called` events (scripted M1 agents do hit failed steps); no rng
touched, world events unchanged. `AGENT_EXPERIENCE.md` added: section-by-
section anatomy of the prompt with the played timeline.

## v0.38 — Epistemics instruments: the Book, world-questions, survey judge, ti.py CLI (2026-06-12)
From the handwritten notes ("The epistemics of artificial society"): can a
society identify how the world works — time, regeneration, terrain coverage,
exact recipes, loss-on-failure — and *when* does it answer each question,
under five book conditions?

**World-questions (`configs/questions_v0.json` + `engine/questions.py`):**
6 static questions + one per techtree recipe (auto-expands with swapped
trees). Ground truth derived per run — coverage is *measured* on the
generated map, not nominal — and written to `runs/<id>/ground_truth.json`.
Never enters any agent prompt.

**The Book (`book_condition` knob):** `none` (control) / `central` (placed on
the nearest grass tile to map centre; read/write standing on or adjacent;
marker structure visible in NEARBY, ASCII `B`, viewer sprite) / `librarian`
(central + only `book_manager` — default first agent — may write; everyone
may read at it; system prompt tells both sides) / `personal` (private book
each, usable anywhere) / `cloud` (shared, usable anywhere). Per the notes:
note entries are append-only with author+tick (never editable); per-question
answers are editable, last write wins, author+tick recorded. New actions
`book_note(text)` / `book_answer(question, answer)`; events
`book_note_written` / `book_answer_written`; prompt gains a `== BOOK ==`
section (full contents when readable, location-only pointer when a central
book is out of reach). `none` is byte-identical to v0.37 (M1 hash
`f10d156c…` reproduced; no rng touched).

**Survey judge (`brains/survey.py`):** every N ticks (default 10) a separate
free-model chain (owl-alpha first, so judging never eats the planners' gemma
quota) reads each agent's beliefs+notes (and the shared book's answers, as
subject "book") next to the ground truth and returns BINARY verdicts per
question → `survey.jsonl`. Out-of-band: agents are never prompted, so
measurement can't contaminate behavior. Change-gated: unchanged text is
never re-judged. Analysis primitive: first-correct tick per question/subject.

**ti.py — the experiment CLI** (phone/SSH-friendly, plain text):
`run` (provider/preset/knowledge/book/survey/freeform/seed), `list`,
`status`, `summary` (discoveries + final book answers + survey
first-correct), `survey` (who answered what, when), `screenshot` (stdlib-only
PNG of any tick of any run — viewer-replay-style reconstruction from
config+events; prints the path so a remote agent can embed it in chat),
`models`. Events flush every tick → status/screenshot work on live runs.

**Freeform-thoughts knob** (notes p.2 "is this helpful?"): `run --freeform`
lets models reason in prose around the JSON object (parser was already
tolerant; ollama json_format auto-disabled). Strict remains default.

**Models (`configs/openrouter.json`):** named presets + `make_provider`
support (`--model` > `--preset` > `tier=paid` > default chain). Default stays
gemma-4-31b:free (best tps); `owl` = openrouter/owl-alpha (smart free, 1M
ctx); `nemotron-ultra` (~7 tps, overnight only); paid default now
openai/gpt-5-nano ($0.05/M in). Judge chain: owl-alpha → gemma-4-26b → 31b.
OPENROUTER_API_KEY set in the user env; key verified live (real completion,
$9.78/$15 credits remaining); `openrouterapi.txt` gitignored.

**Verified:** 50 tests pass (14 new: book conditions/access/immutability,
ground truth, judge grading + change-gating, PNG determinism, freeform);
M1 determinism PASS byte-identical, M2 PASS; mock E2E run with cloud book
through the CLI; REAL openrouter run (4 agents × 40 ticks, central book,
survey on): 0 parse failures, 12/12 judge calls ok on owl-alpha, 0 rate
limits, $0.00, all-false verdicts (correct for 40-tick-old beliefs).

## v0.37 — Housekeeping: shared defaults, tests/, encoding fix (2026-06-11)
**Encoding bugfix (behavioral):** `techtree.py` and `run_m2.py` opened text
files without `encoding="utf-8"` — on this machine (cp1253 default) the
techtree's non-ASCII characters were silently mojibaked into the feedback
strings agents see, and `run_m2.py`'s stats reader crashed outright on
UTF-8 event logs. This is the only change in this version that alters run
output: M2 event hashes shift because feedback text is now decoded
correctly (verified by isolation: HEAD + encoding fix alone reproduces the
new hash exactly; seed-42 events `816a565b…`). M1 logs are byte-identical
to pre-change runs (`f10d156c…`).

**Shared config:** new `engine/defaults.py` — WORLD params, BASE (M3),
TECHTREE_PATH and ALL_RECIPES (now derived from techtree_v0.json instead of
hardcoded). run_m1/m2/m3 and the viewer import it; `viewer/server.py` no
longer imports from the run_m3 script. Config content is unchanged
(M1 determinism hash proves it).

**Layout:** tests moved to tests/ (`py -m pytest` from the root as before);
README refreshed (viewer, run_m3, tests/, current status). server.py:
duplicated latency-percentile and spend-aggregation code extracted into
`latency_stats()` / `spend_totals()`, shared by the live stats API and
summary.json.

**Verified:** 35 tests pass from the new layout; M1 determinism PASS with
byte-identical logs; M2 determinism PASS; 60t mock M3 run through
run_m3.py; viewer imports clean.

**Docs (follow-up):** SPEC §6 layout and IMPLEMENTATION brought in line with
reality — defaults.py/tests/ added, world_default.yaml and the FUTURE.md
reference dropped, viewer marked shipped (v0.31–0.36, M4 gap = analysis/),
golden-fixture test noted as not yet implemented, pyyaml dropped.

## v0.36 — Field station v2: one tabbed console, named agents, replay (2026-06-11)
**Layout:** the two side columns are gone. One rounded console card on the
right, drag-resizable (splitter, persisted), with browser-style tabs:
**base** (mini agent cards + discoveries + key numbers + highlights),
**agents** (portrait, energy, Minecraft-style inventory slot grid, beliefs,
dropdowns for plan/knows/notes/journal/last LLM call), **events** (full feed,
per-agent colors on every row, agent-chip + category filters: social/gather/
craft/trouble/planner), **stats** (latency avg/p50/p95 from planner calls,
calls-per-tick + gathers-per-tick sparklines, gathered-by-resource, spend,
baseline table, walk heatmap), **replay**, **run** (setup moved here),
**files** (runs/ browser), **about** (purpose, prompt contract, tools, plans,
roadmap). Trails/heat are toggleable overlays on the main map.

**Named agents:** `cfg["agent_names"]` — names ARE the agent ids, so they
flow into prompts ("You are Hild, …"), say/give targets, events.jsonl and
journals natively; engine validates uniqueness. Editable in the run tab
before start (server sanitizes to `[A-Za-z0-9_-]{1,14}`). Default roster
Asha…Lumi. Engine change is 4 lines in sim.__init__; A00-format remains the
default for CLI runs.

**Replay:** a tab that never touches the live run. Server rebuilds the
initial world + spawn positions deterministically from config.json (same rng
sequence as Simulation.__init__) and ships the full event list
(/api/replay?run=…); the client reconstructs any tick — tiles from
depleted/regrown, structures from built/expired, agents from every
positioned event. Selecting the tab takes over the BIG center canvas with an
amber REPLAY frame; leaving snaps back to live. Works on pre-rename runs
(A00 ids). KNOWN LIMIT: targeted-gather travel emits no moved events
(actions.py returns False, [] while walking), so a traveling agent holds its
last evented tile until arrival — measured 5/10 agents a few tiles short at
the final tick of a 120t mock run, exact everywhere an event exists. Fixing
it would add moved events to observers' journals (= prompt change), so it
stays until a deliberate engine rev.

**Durable run record:** runs/<name>/summary.json written at run end —
outcome, ticks, duration, final agent states (energy/inventory/tools/
knowledge/beliefs/notes), discoveries, counters, latency percentiles, spend.
The files tab lists runs/ (summary.json when present, light event scan
otherwise) with per-run detail + "open in replay".

**Verified:** test_chain/run_m1 pass; 120t mock run via the new API with
named agents; all 8 tabs exercised headless (Playwright) with zero console
errors; replay reconstruction cross-checked against summary.json finals.

## v0.35 — gather target fix, validated (2026-06-11)
**Fixed:** `_validate_step` stripped x/y from gather steps — every targeted
gather reached the engine untargeted and died "no wood here", which is why
v032_smoke_local (6 × 40, gemma4:e2b) still replanned every tick (231/232
plans dead on step 1, 0.4% completion) despite the v0.32 engine fix.

**Result (v035_smoke_local, same setup):** 49 planner calls in 40 ticks
(was 232), 28.6% plan completion (was 0.4%), 150 gathers/100 ticks (no-LLM
baseline: 139), 0 deaths, 0 parse errors, and THREE blind discoveries by
three different agents: axe A04@t7, basket A05@t14, pickaxe A02@t37.
Remaining interrupts are legitimate (depleted tiles, mis-counted combine
materials) and now land in the journal. Wall time ~7.5 min vs ~37 min for
the broken run — the experiment is now latency-bound by genuine replans only.

## v0.34 — Run configurator, model mixes, baseline panel (2026-06-11)
**Models:** gemma-4 is now the main OpenRouter model
(google/gemma-4-31b-it:free, 26b-a4b:free as rate-limit fallback);
qwen/qwen3.6-35b-a3b ($0.15/M in, $1.00/M out) is the paid tier
(`configs/openrouter.json` "paid_models", `make_provider(tier="paid")`).
Viewer run configurator is defaults-first: a "models" preset select —
**local-e2b** (ollama gemma4:e2b, the current acceptance setup),
**openrouter-90-10** (floor(10%) of agents on paid qwen3.6 — A00.. first —
rest on free gemma-4, both billing into one costs.jsonl), mock, or custom
provider/model. Ticks, agents, seed, condition and ALL map parameters
(grid size, terrain fractions, berry/wood/stone knobs) are editable in the
form, prefilled from engine defaults via /api/defaults; only changed values
are sent. Missing OPENROUTER_API_KEY fails at /api/start, not at tick 1.

**Measurement:** new "vs no-LLM runs" panel — live run's deaths/1k ticks,
plan completion %, gathers/100 ticks and recipes found next to the same
numbers computed from all scripted/mock runs in runs/ (classified by
prompts.jsonl provider; tick-weighted). Discovery chips now name the
discoverer: "axe A06 t63" (✎ = taught), tooltip has the source. Agent
inspector shows which model produced the last exchange (matters in mixes).

## v0.32 — Engine fixes from the first LLM run (2026-06-11, uncommitted until now)
The m3_accept_ui_135429 run (gemma4:e2b, died at t78, 2 deaths): 607 of 631
plans died on step 1, 590 of them "no wood/berries here" — models assume
gather(x, y) walks to the tile. Fixes: `gather` takes an optional target and
walks there first (one multi-tick primitive); `plan_interrupted` IS journaled
(deviating from spec §3b — without the failure reason in context, models
repeat the identical broken step forever); survival basics (berries are food,
+25; wood/stone are not) moved into the system prompt — withholding them
measured confusion, not discovery; FORMAT_DOC tells the model planning ahead
is cheaper than being re-consulted. Planning is fanned out to threads when
every brain declares parallel_safe (network providers do, mock doesn't — it
mutates the shared seeded rng); with 10 agents on a 10 s model that is the
difference between ~100 s and ~10 s per tick.

## v0.33 — OpenRouter backend + spend ledger (2026-06-11)
**Added:** `OpenRouterProvider` in `brains/providers.py` — ordered model
fallback chain (`configs/openrouter.json` is the configurator; CLI/UI
`--model "a,b,c"` overrides it). A model answering 429 (rate limit) or 402
(free credits gone) goes on cooldown (Retry-After / X-RateLimit-Reset
honored, defaults 65 s / 1 h) and the next model takes over; the preferred
model is retried once its cooldown expires; if the whole chain is cooling,
complete() blocks until the soonest reset. This is the one sanctioned
fallback — any other error still raises. Every reply's usage (OpenRouter
reports cost in USD with `usage.include`) is appended to `costs.jsonl` next
to `prompts.jsonl` with running totals; run_m3 prints the spend summary, the
viewer HUD shows SPEND + RATE-LIMITED chips. Key from `OPENROUTER_API_KEY`.

**Fixed:** the 2b parallel-planning path in sim.py was dead code — nothing
ever set `parallel_safe`. LLMPlannerBrain now mirrors the provider's flag
(ollama/anthropic/openrouter True, mock False: it mutates the shared seeded
rng). prompts.jsonl appends now go through a module lock (brains share one
file; planner threads would interleave lines). One provider instance is
shared by all brains (run_m3 + viewer factories) so rate-limit cooldowns and
the spend ledger are global, not per-agent.

**Tests:** `test_openrouter.py` — fallback on 429, cooldown stickiness and
expiry, 402 long cooldown, whole-chain-limited recovery, in-body provider
errors raise, 500 raises without trying the next model, config-file
defaults + `--model` override. No network: scripted fake requests module.

## v0.31 — Field Station: live viewer (2026-06-11)
**Added:** `viewer/server.py` + `viewer/static/` — a live dashboard
(`py viewer/server.py`, http://127.0.0.1:8787). Stdlib-only HTTP server runs a
Simulation in a worker thread; browser polls /api/state. Start runs from the
UI (mock preview / the gemma4:e2b acceptance run / custom params), pause /
resume / stop, tick-delay slider. Canvas world map with hand-drawn 16×16 pixel
sprites; click an agent for the inspector: energy, inventory, tools, plan
queue (current step marked), intention, beliefs, notes, journal, knowledge,
heard-of recipes, and the agent's last raw LLM prompt + response. Colony
stats, discovery tracker, live event feed.

**Engine untouched:** events are captured by teeing the sim's events.jsonl
file object (which also means live runs now flush per event); per-agent
prompts captured by subclassing `LLMPlannerBrain._log`. Beliefs/notes are read
for display only — the viewer is measurement, like the event log.

**Note:** the first CLI overnight acceptance run (m3_gemma_blind_s42) died
silently at ~t10 (process killed hard, no traceback; runs/ dir is a useless
partial). Acceptance run to be relaunched from the UI.

**UI polish:** slim header (transport controls top-right), always-visible HUD
with tick progress bar + the acceptance-relevant numbers (alive, deaths,
parse-fail % vs the 5% bar, taught, discovered), bigger map, leaner side
decks. Desktop shortcut "Terra Incognita" launches the server minimized and
opens the browser; if the server is already up it just opens the page.

**Repo:** git init + private GitHub repo (PastelDev/terra-incognita).
runs/ and scratch/ (unrelated local files) are gitignored; experiment data
stays local, results live in this changelog.

## v0.3 — M3: Minds
**Added:** `brains/llm_planner.py` (build_prompt with the spec §3 sections
STATUS / YOU KNOW / YOU BELIEVE / NOTES / NEARBY / RECENT EVENTS / ACTIONS;
strict JSON parsing — one retry quoting the error back, then plan=[wait] +
`planner_parse_error` event); `brains/providers.py` (ollama / anthropic behind
one `complete(system, user) -> str` interface, lazy imports so the repo stays
LLM-free, plus a deterministic `MockProvider` that answers by parsing the
rendered prompt — doubling as a check the prompt carries actionable info);
`prompts.jsonl` telemetry (full prompt/response pairs per call); `say` and
`give` actions; knowledge conditions (blind / oracle / fragmented / one_expert)
injected into `agent.knowledge` at init; `run_m3.py` runner; `test_m3.py`
(23 tests, no LLM calls ever).

**Brain interface formalized:** `brains/base.py` — `Brain.plan(Observation) ->
PlanResult(plan, intention, updated_beliefs, updated_notes, parse_error)`.
The engine writes beliefs/notes back verbatim and never reads them; Observation
gained standing_on, slots, nearby structures, rendered known recipes, beliefs,
notes. Scripted brains updated to the same contract.

**Replan triggers (spec §3) now real:** plan finished, plan step failed
(plan_interrupted now CLEARS the rest of the plan — previously the engine
blindly executed the remaining steps), directed speech (`said` with target=me
sets a replan flag), and a forced replan every `replan_interval` (default 15)
ticks. `planner_called` logs which trigger fired.

**Speech (decided, open question 4):** `say` is a normal primitive — it
occupies the agent's action for the tick (economy of words). Radius 3
visibility per spec §3b; engine never validates content, so lying works.

**Taught vs discovered without reading the mind:** the engine scans heard
`said` messages for recipe-id mentions (word-boundary match) into an
engine-owned `heard_recipes` set per agent; first craft of a recipe the agent
has heard named → `recipe_learned source=taught`. Pure measurement — no world
logic reads it. The mock smoke run already shows the full chain working
(say "…axe…" → neighbor crafts → taught).

**Config knobs added:** `vis_radius`, `say_radius`, `replan_interval`,
`knowledge_condition`. Providers: ollama default qwen2.5:7b (json format
forced); anthropic default claude-haiku-4-5 per spec §3 cost plan.

**Acceptance so far (mock provider — real Ollama overnight run still TODO):**
- 8 agents x 300 ticks, mock: 744 planner calls, 0 parse failures, 3 deaths,
  axe@t11, 3 recipes learned with source=taught. Determinism with mock brains:
  byte-identical logs (tested).
- test_m3.py: 23/23. test_chain.py: 9/9 (unchanged semantics).
- M2 baseline re-run under the new engine (clear-on-interrupt + forced replans
  shift the numbers slightly, same picture): deaths 5–6/12, tier 1 + accidental
  fire found, farm/cooked/storehouse never, determinism PASS.

**Note:** repo paths de-hardcoded (`/home/claude/terra` → relative to repo
root) in run_m1/run_m2/test_chain/techtree smoke tests.

**Windows fix (2026-06-11):** all text-mode `open()` calls in the M3 path
(events.jsonl, prompts.jsonl, config.json) now pass `encoding="utf-8"` —
Windows defaults to the system codepage (cp1253 here), which crashed telemetry
writes the first time gemma4:e2b emitted a non-Latin character. Found by a
30-tick real-model pretest.

**Acceptance run model (2026-06-11):** gemma4:e2b via Ollama (qwen2.5:7b not
pulled on this machine). 30-tick pretest: 247 planner calls, 2.4% parse
failure (under the 5% bar; gemma flubs JSON on ~7% of first attempts — cut-off
strings, stray tokens after the object — and the single retry recovers about
half), 0 deaths, no crash. ~25 s/tick early on ⇒ 1000 ticks ≈ overnight.

## v0.2 — M2: Technology exists
**Added:** `combine` wired into the action layer (uses the tested tech-tree
resolver); structures live on the world (Fire expires after 100 ticks, Farm
yields 3 berries/10 ticks harvestable by anyone, Storehouse = 200 shared units
+ deposit/withdraw); tool effects (axe x3 wood, pickaxe x3 stone, basket +8
slots); cooked_food (+50 energy, never spoils); `RandomScientistBrain`
(forages, then tries random small-quantity combos when energy >= 60).
Storehouse contents do NOT spoil (granary — decided).

**Journal privacy honored:** observers of a craft see "A03 made something: a
fire." — never the ingredients. Failure feedback text is actor-only.

**Emergent rule discovered by the chain test:** since 1 wood + 2 stone on grass
always resolves to FIRE (condition-specific recipes win), a pickaxe *cannot be
crafted on grass*. Crafting location matters — a workshop mechanic we never
designed. Kept deliberately: it is exactly the kind of non-obvious physics the
project studies.

**Acceptance results:**
- Full chain gather->axe->fire->cooked_food->farm->pickaxe->storehouse:
  REACHABLE (test_chain.py, 9/9).
- Random baseline (12 agents x 1000 ticks x 3 seeds): tier 1 usually found
  (axe 3/3 seeds, pickaxe 2/3, basket 3/3), fire 3/3 (always by accident on
  grass), farm 1/3 (once, at t992), cooked_food 0/3, storehouse 0/3.
  ~135-235 combine attempts per run, 4-6% success rate.
- Mortality: 5-10 of 12 random scientists die per run (vs 2 for pure foragers).
  Blind experimentation is *expensive*: materials wasted, foraging time lost.
  This is a feature — knowledge has measurable survival value in this world —
  and gives M3 two-dimensional headroom: LLM agents should discover MORE while
  dying LESS. If they do neither, that is a publishable negative result.
- Determinism: PASS (byte-identical logs, seed 42).

**M3 baseline numbers to beat:** tier-2+ discovery beyond accidental fire is
effectively out of reach for blind luck (farm 1/3 late, cooked never), and
>=40% population death. Any competent mind should clear both bars.

## v0.12 — Food pressure: berry spoilage (current)
**Change:** Each berry in an agent's inventory has a per-tick spoil probability
(`berry_spoil_prob`, default 0.02 ⇒ expected shelf life ≈ 50 ticks). Spoiled
berries vanish with a `spoiled` event (visible only to the owner).

**Why:** v0.11 showed unlimited hoarding (agents at 30–40 berries). Tuning bush
counts/refresh rates against *scripted* foragers would be miscalibrated anyway —
LLM agents in M3 will have competing demands (wood, stone, experimentation,
talking), so the same food supply will feel scarcer to them. Spoilage is robust
to that: it bounds the useful stockpile regardless of how busy agents are, and
keeps food a *recurring errand* rather than a solved problem.
It also retroactively gives the tech tree its economy: cooked_food
(`spoils: false`) is now a real invention — food security — rather than a
numeric upgrade. Storehouse may later halt spoilage (decide at M2).

**Calibration principle (instead of "optimal berries"):** target a *food-work
fraction* — the share of an agent's ticks spent acquiring/eating food — of
roughly 25–35% pre-technology. Measured from the event log (forage-intention
planner calls + gather/eat events). Calibrate roughly now with scripted agents,
**recalibrate at M3 with real LLM agents**, and report the achieved fraction in
any paper (reviewers will ask how harsh the world was).

**Observed effect (seed 42, 500 ticks, 12 agents, spoil_prob=0.02):** end
inventories dropped from 15–40 berries to 0–7. **Two agents died** — the first
deaths in any run. Food economy: 686 gathered, 330 eaten, 334 spoiled — a 49%
waste rate, because the scripted ForagerBrain still hoards and spoilage
punishes hoarding. That's the interesting part: spoilage converts hoarding from
a neutral quirk into a *failing strategy*, which a scripted brain cannot adapt
to but an LLM brain could — giving us a concrete, measurable adaptation gap to
look for at M3 (do LLM societies discover just-in-time foraging?). Gather
actions alone = 11% of agent-ticks; with travel, food work is plausibly ~30%,
inside the target band. Determinism preserved.
**Open tuning note:** 0.02 may be slightly harsh for the dumb baseline (2/12
deaths); acceptable since M3 minds should beat it. If LLM agents also die en
masse, lower to 0.01 before blaming the models.

**Eating (decided):** dual-mode. Reflex auto-eat below energy 30 stays (engine-
level survival floor — a confused LLM holding food never starves). Planned
`eat` also stays, so minds can eat strategically (top up before a long trip).
Plans do NOT need a mandatory food-gather step; spoilage + metabolism produce
that pressure organically, which is better science than hard-coding behavior.

## v0.11 — Geography is real: water + organic regions
**Change:** (1) Region-growth map generation (seed a tile, repeatedly convert a
random grass neighbor) replaced independent blob scattering → connected,
organically shaped forests, mountain ranges, lakes. Lakes generate first so
land shapes around them. (2) Water terrain: impassable, no thirst mechanic
(terrain, not biology — a second survival meter would confound every metric).
(3) Greedy movement replaced by BFS pathfinding (deterministic neighbor order)
— required, since concave shores trap greedy walkers.

**Observed effect — a lesson in coupled systems:** adding water made hoarding
*worse* (several agents hit 30–40 berries vs ~5–29 in v0.1). Mechanism: water
removed ~10% of land while berry-patch count stayed constant ⇒ higher resource
density per capita near surviving patches ⇒ foragers camp and stuff their bags.
A "cosmetic" geography change shifted behavior through resource density.
**Methodological consequence:** experiment conditions must be compared on
identical map seeds, or geography becomes a confound. Map seed is a controlled
variable from now on, reported with every result.

## v0.1 — M1 baseline
32×32 toroidal grid; berries/wood/stone; energy metabolism (-1 idle, -2 moving,
death at 0, reflex eat <30); slot inventory (8 slots × 5 units); plan-queue
agents (sense → plan → act-blindly → interrupt/finish → sense); event sourcing
to JSONL with visibility-radius journal fan-out; scripted ForagerBrain;
determinism acceptance test (byte-identical logs on same seed) — PASS.
Known quirk: unbounded berry hoarding (fixed in v0.12).
