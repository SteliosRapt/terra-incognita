# CHANGELOG / Design Notes

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
