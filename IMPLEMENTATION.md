# IMPLEMENTATION.md — Guide for coding agents
Read `SPEC.md` first. That document is the source of truth for *what* to build; this one is about *how*.

## Ground rules
1. **The engine never imports anything LLM-related.** `engine/` must run headless with scripted agents. `brains/` plugs in via a single interface: `Brain.plan(observation: Observation) -> Plan`.
2. **Mutate the World only through `actions.py`.** Every mutation emits ≥1 Event. If you find yourself editing `world` state anywhere else, stop — it will silently break the event log, and the event log is the entire dataset of the research.
3. **The agent's mind is opaque to the engine.** Engine never reads `beliefs`/`notes` for logic; brains never read the World directly — only the rendered `Observation`. (`techtree.py` already enforces this shape: it takes a `CombineContext`, not the World.)
4. **Everything tunable lives in config** (YAML/JSON), never as constants in code: map seed, grid size, energy rates, visibility radii, failure cost, feedback verbosity, replan interval K, knowledge condition.
5. **Determinism:** one `random.Random(seed)` instance owned by the sim, passed down. Two runs with the same config+seed and scripted brains must produce byte-identical event logs — this is Milestone 1's acceptance test.
6. Python 3.11+, type hints everywhere, dataclasses for state, pytest for tests, no heavy frameworks. Dependencies allowed: pyyaml, pytest, (later) requests/anthropic/ollama client, pandas+matplotlib in `analysis/` only.

## Existing reference code
- `engine/techtree.py` — DONE, tested. Resolution rules: exact-quantity match; most-conditional recipe wins ties; near_miss = right ingredients/wrong condition; failure cost configurable. Do not change semantics without updating its smoke tests.
- `configs/techtree_v0.json` — the data contract for recipes. Schema is frozen for v0.

## Milestones (each ends runnable + tested)

### M1 — Dead world walking
`world.py` (grid gen from seed, tile types, resource spawn/regrowth), `agent.py` (state dataclass incl. notes/beliefs/journal), `actions.py` (move_to, gather, eat, drop, wait — no combine yet), `sim.py` (tick loop from spec §3c, JSONL event writer), `brains/scripted.py` (`ForagerBrain`: wander, gather, eat when hungry), ASCII renderer.
**Done when:** 12 foragers survive 500 ticks; identical seeds → identical logs; pytest covers energy math, death, toroidal movement, regrowth.

### M2 — Technology exists
Wire `techtree.py` into `actions.py` (combine), implement produces/effects: tools (gather multipliers, inventory bonus), structures (Fire lifetime, Farm yields, Storehouse shared slots), cooked food. Add `RandomScientistBrain`: forages + tries random combines when safe (energy > 50). Journal fan-out per visibility rules (spec §3b).
**Done when:** random scientists eventually discover tier-1 tools in most seeds; a hand-scripted brain can run the full chain gather→axe→farm→fire→cooked_food; events validate against the taxonomy.

### M3 — Minds
`brains/llm_planner.py`: build_prompt exactly as spec §3 (sections STATUS/YOU KNOW/YOU BELIEVE/NOTES/NEARBY/RECENT EVENTS/ACTIONS), strict JSON parsing (one retry → wait + parse_error event), `providers.py` with `ollama` and `anthropic` backends behind one interface, prompts.jsonl telemetry. say/give actions. Knowledge conditions injected at init (oracle/blind/fragmented/one-expert).
**Done when:** a 10-agent blind run completes overnight on Ollama without crashing; parse failure rate < 5%; a transcript shows at least one taught recipe (recipe_learned source=taught).

### M4 — Eyes and numbers
`viewer/` HTML canvas replay (reads events.jsonl only; agent inspector panel; scrub bar). `analysis/`: discovery curves, diffusion graph, misinformation rate, specialization entropy, survival, Gini — each a function over the event log → PNG + CSV.
**Done when:** `python analysis/report.py runs/<id>` produces all plots; viewer replays the same run.

### M5 — Science
Experiment runner: takes a conditions YAML, runs N seeds per cell (sequential is fine), aggregates. Pilot: Blind vs Oracle, 3 seeds. Then RQ1 sweep.

## Testing strategy
- Unit: engine math (energy, capacity, regrowth, visibility radius sets).
- Property: random action sequences never produce negative inventories / energy > 100 / events referencing dead agents.
- Golden: one frozen seed's first 50 ticks of events committed as fixture; CI diffs against it.
- Brain tests use a `FakeBrain` returning canned plans — never call LLMs in tests.

## Open questions (decide before the relevant milestone)
1. **(M2)** Exact-quantity combine is strict; if random scientists discover too slowly even for the baseline, fallback option: allow superset matches that consume only the recipe amounts. Decide from M2 data, and note the choice changes discovery difficulty.
2. **(M2)** At verbosity=low, near_miss is *indistinguishable* from failure for the agent (deliberate). Confirm we want the v0 default this harsh, or default medium.
3. **(M3)** Notes cap 1500 chars — kept for v0; revisit once real Ollama transcripts exist (small models may thrash).
4. **(M3) DECIDED:** `say` costs a tick (it is a normal primitive — one action per tick), i.e. costed speech / economy of words.
5. **(M3) DECIDED:** identical agents in RQ1. `LLMPlannerBrain(persona=...)` exists but defaults to None; vary later.
6. **(M5)** Stats: with 10 seeds/cell, use medians + bootstrap CIs; check with the mentoring researcher whether he wants Mann-Whitney or something fancier before the writeup.
