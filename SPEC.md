# TERRA INCOGNITA — Design Specification v0.1
### A framework for studying collective discovery in LLM agent societies

**Core idea in one sentence:** A 2D grid survival world whose crafting/technology rules ("physics") are defined in a config file and *hidden from the agents* — agents must discover them through experimentation, and we study how knowledge is discovered, shared, hoarded, and distorted across a society.

**What's new vs. prior work:** Project Sid (Altera 2024) showed specialization, economy, and culture emerge in Minecraft — but Minecraft's tech rules are fixed and known to the LLM from pretraining. TerraLingua (2026) has grid survival + artifacts but no technology progression. Concordia (DeepMind) has no grounded mechanics at all. **No existing framework treats the technology tree as (a) configurable data and (b) hidden information.** That's our niche: *the epistemics of artificial societies*.

---

## 1. The World

| Property | v0 value | Notes |
|---|---|---|
| Grid | 32 × 32, toroidal (wraps) | Standard in ALife (Sugarscape, TerraLingua); no edge effects |
| Time | Discrete ticks, turn-based | All agents act once per tick, randomized order each tick |
| Run length | 500–2000 ticks | Long enough for multi-stage discovery |
| Population | 8–20 agents | Small enough for LLM budget, large enough for social dynamics |

### Terrain & resources (3 resource types)

| Resource | Spawns on | Regrowth | Gather yield (bare hands) |
|---|---|---|---|
| **Berries** (food) | Grass tiles, clustered patches | Each empty berry tile: 5%/tick | 1 unit |
| **Wood** | Forest tiles | Depleted tree regrows after 40 ticks | 1 unit |
| **Stone** | Mountain tiles | Does NOT regrow (finite) | 1 unit |

Map generation: seeded random; ~15% forest, ~8% mountain, 3–5 berry patch clusters. Same seed ⇒ identical map (reproducibility).

### Agent state

```
Agent {
  id, name, position (x, y)
  energy: 0–100        # -1/tick idle, -2/tick if moved; 0 ⇒ death (removed)
                       # exception: starting a tick on a Shelter tile costs 0
                       # (resting) or 1 (leaving it) — see the tech tree below
  inventory: dict      # capacity 8 slots (1 slot = 5 units of one item type)
  tools: list          # equipped items (axe, pickaxe...)
  knowledge: set       # recipes this agent has CONFIRMED (by crafting or being taught)
  beliefs: list        # natural-language hypotheses ("maybe wood+stone makes something")
  journal: list        # last N events this agent witnessed (engine-written episodic memory)
  notes: str           # free-form scratchpad, AGENT-written: planner may return
                       # "updated_notes" (full rewrite, cap ~1500 chars) each call.
                       # Journal = what happened TO the agent; notes = what it CHOOSES
                       # to remember (maps, theories, who owes whom, grudges).
}
```

Eating berries: +25 energy. Cooked food (if discovered): +50. Death is permanent within a run.

---

## 2. The Tech Tree (exact v0 tree)

The tree lives in `techtree.json`. The engine reads it; **the agents never do** (except in the "oracle" control condition). 9 nodes, 3 tiers, including deliberately non-obvious recipes (a contextual condition, a tool prerequisite, and a single-item self-combine).

```
TIER 0 (innate actions — agents know these from the start)
  move, gather, eat, drop, give, say, combine

TIER 1
  [Axe]      2 wood + 1 stone  → tool: wood yield ×3
  [Pickaxe]  1 wood + 2 stone  → tool: stone yield ×3
  [Basket]   3 wood            → +8 inventory slots
  [Brick]    1 stone           → item: a brick  ← non-obvious (a single item combined BY ITSELF)

TIER 2
  [Fire]     1 wood + 2 stone, AND must be standing on a grass tile   ← non-obvious (condition!)
             → creates a Fire entity on the tile, lasts 100 ticks
  [Farm]     4 wood + 2 berries, requires Axe in tools                ← non-obvious (tool-gated)
             → tile becomes Farm: yields 3 berries every 10 ticks, harvestable by anyone
  [Shelter]  4 bricks          → PERMANENT structure on the tile
             → at most one per builder (excess attempts rejected, bricks kept);
               anyone may use any shelter — standing on one costs 0 energy/tick,
               leaving it costs 1 (vs the normal 1 idle / 2 moving)

TIER 3
  [Cooked food]  1 berries, must be adjacent to a Fire                → +50 energy food, never spoils
  [Storehouse]   6 wood + 4 stone, requires Pickaxe                   → shared 40-slot container on tile
```

The **brick → shelter** branch and the `--combine-note` hint ("An item can sometimes be combined by itself.") are the subject of the DiffusionGemma 2×2 study — see `DIFFUSIONGEMMA_EXPERIMENT_PLAN.md`.

**Why these two "non-obvious" recipes matter:** Fire requires a *contextual condition* (tile type), Farm requires a *tool prerequisite*. Pure ingredient-mixing won't find them — agents need hypothesis refinement ("wood+stone failed... but it failed *differently* on grass?"). This is the "custom physics" from your original email, made concrete and measurable.

**Crucially, the tree is swappable.** Alternative trees for later experiments:
- **Deep tree** (long chains, A→B→C→D) vs. **wide tree** (many independent recipes): does depth force cooperation?
- **Counter-intuitive tree**: recipes that contradict LLM priors (e.g., stone+stone→rope), to separate *discovery* from *recall of Minecraft knowledge*. This is an important control — without it, a reviewer will say "the LLM just knows crafting games."

### The `combine` action (the experiment mechanic)

```
combine(items: list, quantities: list)
→ engine checks against techtree.json + contextual conditions
→ SUCCESS: item created, recipe added to agent.knowledge,
           event logged: "You combined 2 wood + 1 stone and created an AXE!"
→ FAILURE: ingredients destroyed (configurable: 100% / 50% / 0% loss),
           feedback message (configurable verbosity, see §4)
```

**Failure cost and feedback verbosity are experimental knobs.** Expensive failure + vague feedback should pressure agents toward knowledge-sharing (or toward stealing knowledge instead of risking resources).

---

## 3. Agent Architecture (two-tier brain)

The cost-control insight: LLM is called only at decision points, not every tick.

```
┌─────────────────────────────────────────────────┐
│ TIER A: Planner (LLM call)                      │
│ Triggered when: current plan finished/failed,    │
│ another agent spoke to me, something surprising  │
│ happened, or every K=15 ticks (forced replan)    │
│ Output (JSON): plan = ordered list of primitive  │
│ actions + a short "intention" string + updated   │
│ beliefs list                                     │
├─────────────────────────────────────────────────┤
│ TIER B: Executor (pure Python, free)             │
│ Executes plan steps: pathfinding (A*), gather    │
│ loops, eat-when-hungry reflex (energy < 30 ⇒    │
│ interrupt plan and seek food)                    │
└─────────────────────────────────────────────────┘
```

### Exact planner prompt structure (v0)

```
SYSTEM: You are {name}, a survivor in an unknown world. You do not know how
this world works. You can discover how to create things by combining items
(action: combine), but failures may destroy ingredients. You can talk to
nearby agents. Your goals: survive, and understand this world.
[persona line, optional: "You are cautious and prefer observing others." ]

USER:
== STATUS ==        energy 64/100, position (12,7), inventory: wood×3, stone×1
== YOU KNOW ==      recipes confirmed: [Axe: 2 wood + 1 stone]
== YOU BELIEVE ==   "combining berries with wood failed on dirt — maybe needs fire?"
== NEARBY ==        Eleni at (13,7); forest tiles north; a Fire at (10,5), owner: Nikos
== RECENT EVENTS == [last 10 journal entries, e.g. "Eleni said: 'I'll trade a
                     recipe for 3 wood'", "Your combine(berries,wood) FAILED, items lost"]
== ACTIONS ==       move_to(x,y), gather(resource), eat(item), combine([items]),
                    say(agent, "message"), give(agent, item, qty), wait

Respond ONLY with JSON:
{ "reasoning": "...", "updated_beliefs": [...], "plan": [ {action}, ... ] }
```

The `say` action is free-form natural language ⇒ agents can teach recipes, negotiate, **or lie** (engine never validates speech — misinformation propagation is measurable for free, since we know ground truth).

### Models & cost

- Default: small local model via Ollama (Qwen-2.5-7B / Llama-3.1-8B) ⇒ **$0**, runs on a decent laptop, slow but fine for turn-based.
- Premium runs: Claude Haiku via API. Estimate: 12 agents × 1 call/15 ticks × 1000 ticks ≈ 800 calls/run ≈ ~1.5M tokens. Cheap enough for parameter sweeps.
- Model identity itself is a variable (do Qwen societies discover faster than Llama societies?).

---

## 3b. Event Taxonomy (complete v0 list)

Everything in the simulation is an **Event**: an append-only JSONL record
`{tick, type, actor, target, payload, position, visibility}`. The journal an agent sees is just the subset of events whose `visibility` includes it. Complete v0 list:

| Category | Event types |
|---|---|
| Movement | `moved` (logged at plan granularity, not every step) |
| Resources | `gathered`, `ate`, `dropped`, `picked_up`, `resource_depleted`, `resource_regrown` |
| Crafting | `combine_attempted`, `combine_succeeded`, `combine_failed`, `recipe_learned` (with source: discovered / taught / observed) |
| Tools/structures | `tool_equipped`, `tool_broke` (if durability enabled), `structure_built` (Fire/Farm/Storehouse), `structure_expired`, `farm_yielded`, `storehouse_deposit`, `storehouse_withdraw` |
| Social | `said` (free text), `gave`, `received` |
| Life | `energy_critical` (<20), `agent_died`, `agent_spawned` (v1, reproduction) |
| System | `planner_called`, `plan_completed`, `plan_interrupted`, `planner_parse_error` (never enter agent journals — research telemetry only) |

**Visibility rule (v0):** an event is visible to its actor, its target, and any agent within radius **R=6** tiles (line-of-sight ignored in v0). `said` events are visible within radius 3 (speech is local; broadcast is a later knob). Visibility radius is itself a config parameter: it controls how much societies learn by *observation* vs. *communication*.

Note `combine_succeeded` is visible to observers but contains only the *result*, never the ingredients. **(Decided):** observation tells you something is possible; only communication (or your own experiments) can tell you how. Knowledge therefore flows exclusively through `said` events and personal discovery — which keeps the diffusion graph clean and makes teaching/lying the only social channels for recipes.

## 3c. Tick Loop, Context Processing & State Storage

**Authoritative state lives in one place:** a single in-memory `World` object (grid, entities, agents). Nothing else is ever mutated directly.

```
for tick in range(T):
  1. WORLD PHASE: regrow resources, age fires/farms, farm yields
     → emits events
  2. AGENT PHASE (randomized order): for each living agent:
     a. Reflexes (Tier B): if energy < 30, override plan → seek food
     b. If plan empty/failed/interrupted or replan timer hit:
        → build prompt CONTEXT from state (see below) → LLM call
        → parse JSON {reasoning, updated_beliefs, updated_notes, plan}
        → beliefs/notes written back to agent state; plan queued
        → on parse failure: retry once, else plan=[wait], log parse_error
     c. Pop next primitive action from plan → VALIDATE against world
        (in range? ingredients present? tile legal?) → if valid, APPLY:
        mutate World, emit event(s); if invalid, emit plan_interrupted
  3. COMMIT PHASE: append new events to runs/<id>/events.jsonl,
     fan out events into agent journals per visibility rule
```

**Context building is a pure function** `build_prompt(world, agent) → str`: STATUS and NEARBY are *rendered live* from the World object (never trusted from the LLM), YOU KNOW from `knowledge`, YOU BELIEVE from `beliefs`, NOTES verbatim, RECENT EVENTS = last 12 journal entries rendered to one-line English. The LLM's output can only ever change three things: its own `beliefs`, its own `notes`, and its action queue. **Actions are the only channel from mind to world, and every action is validated by the engine.** An agent can *believe* it has an axe; the engine knows it doesn't. This hard mind/world separation is what makes false belief and deception well-defined and measurable.

**Storage = log + snapshots:**
- `events.jsonl` — append-only ground truth; analysis and the replay viewer read only this.
- `snapshot_<tick>.json` — full World serialization every 100 ticks (resume crashed overnight runs).
- `config.yaml` + map seed copied into the run dir ⇒ every run exactly reproducible; any state reconstructable as snapshot + replay.
- LLM telemetry (`prompts.jsonl`: full prompt/response pairs, token counts) — separate from world events; invaluable for debugging *why* an agent believed something.

## 3d. Rendering (v0)

- **For the model: text only.** Agents perceive the world purely through the rendered prompt — no images. (A vision variant is a clean later ablation: does seeing the map change discovery?)
- **For us:** Week 1 ships an ASCII renderer (print the grid every N ticks — trivial, sufficient for debugging). Week 4 ships the visual companion: an HTML/canvas **replay viewer** reading `events.jsonl` — tile map, agent sprites, speech bubbles, click an agent to inspect its notes/beliefs/journal at that tick, scrub bar over time. Because it reads the log, not the live sim, any interesting run becomes a shareable replayable file (great for the researcher pitch and STEMfy content).

## 4. Experimental Design (what we actually publish)

**RQ1 (epistemics):** How do knowledge conditions affect discovery speed and survival?
**RQ2 (social):** Does hidden knowledge produce teaching, trade, secrecy, or deception?
**RQ3 (structure):** How does tech-tree topology (deep vs. wide) shape social organization?

### Conditions matrix (v0 = just RQ1)

| Condition | Tree knowledge | Prediction |
|---|---|---|
| **Oracle** (control) | Full tree in every prompt | Upper bound on performance |
| **Blind** | Nothing | Discovery curve from zero |
| **Fragmented** | Each agent starts knowing 1 random recipe | Tests whether trade/teaching emerges |
| **One-expert** | One agent knows everything | Does an "elder/teacher" role emerge? Does it hoard? |

Crossed with: communication ON/OFF, failure cost {0%, 100%}. Each cell: **≥10 seeds** (this is the rigor the field lacks — single-run papers are the norm and the weakness).

### Metrics (all computed from the event log)

- **Discovery curve:** tick of first crafting of each node, per run (the headline plot)
- **Knowledge diffusion:** for each recipe, # agents who know it vs. time; diffusion graph (who learned from whom — taught vs. independently discovered)
- **Misinformation rate:** % of recipe statements in `say` events that contradict ground truth
- **Specialization index:** entropy of each agent's action distribution (Sid's measure, reproducible here)
- **Survival** & population energy; **Gini coefficient** over inventories
- Qualitative: annotated transcripts of notable episodes (deception, trade, teaching)

---

## 5. What It Looks Like In Action (sample trace)

> **Tick 122.** Nikos (energy 41) finishes gathering. Planner fires. Beliefs: "wood+stone made my axe; bigger things may need more stone." Plan: gather 2 more stone, then combine(1 wood, 2 stone). **Tick 131:** combine succeeds while standing on grass → **FIRE discovered**. Journal: "You created FIRE!" Eleni, 3 tiles away, sees it (event enters her journal: "Nikos created a fire at (10,5)").
> **Tick 134.** Eleni's planner fires (surprise trigger). Plan: approach Nikos, say: "How did you make that fire? I'll give you 3 wood."
> **Tick 137.** Nikos's planner: reasoning "wood is useful and Eleni may share things later" → say: "1 wood and 2 stones, but it only worked on grass." → recipe enters Eleni's *beliefs* (not knowledge — she must craft it once to confirm).
> **Tick 150.** Eleni crafts fire → knowledge confirmed. Diffusion log: FIRE: Nikos (discovered, t131) → Eleni (taught, t150).
> *In the deception variant: Nikos instead says "5 berries and a stone" — and we measure how long the false recipe circulates.*

Visualizer: simple top-down web view (HTML canvas or pygame), tiles + agent sprites + speech bubbles, scrub bar over the event log. Every run is fully replayable from the log file.

---

## 6. Architecture & Repo Layout

```
terra-incognita/
├── ti.py               # experiment CLI: run/list/status/summary/survey/screenshot
├── engine/
│   ├── world.py        # grid, tiles, resources, entities (Fire, Farm, Storehouse, Book)
│   ├── agent.py        # state, journal, executor (Tier B)
│   ├── techtree.py     # loads techtree.json, resolves combine() attempts
│   ├── actions.py      # primitive action implementations + validation
│   ├── sim.py          # tick loop, event log writer (JSONL)
│   ├── questions.py    # epistemics questionnaire + ground-truth derivation
│   └── defaults.py     # canonical run config (world params, recipe list)
├── brains/
│   ├── llm_planner.py  # prompt builder, JSON parsing, retry logic
│   ├── providers.py    # mock / ollama / anthropic / openrouter backends
│   ├── survey.py       # free-LLM judge: binary world-knowledge verdicts
│   └── scripted.py     # rule-based baseline agent (important control!)
├── configs/
│   ├── techtree_v0.json
│   ├── questions_v0.json  # the world-questions the societies try to answer
│   ├── openrouter.json    # model chains + presets (judge chain included)
│   └── experiments/    # (planned, M5) one config per condition cell
├── analysis/           # (planned, M4) discovery curves, diffusion graphs, stats
├── viewer/             # live dashboard + replay of past runs
├── tests/              # pytest suite (no LLM/network calls)
└── runs/               # event logs (JSONL), one dir per run
```

Design principles: **everything is config** (tree, costs, map seed, knowledge condition); **the log is the ground truth** (every event JSONL ⇒ analysis & replay both read it); engine has zero LLM dependencies (testable headless, scripted agents for free debugging).

### Build order (each step runs end-to-end)

1. **Week 1:** world + resources + scripted agents + event log + ASCII renderer. No LLM.
2. **Week 2:** tech tree engine + combine + tools/fire/farm effects. Scripted agents that random-experiment (this is also your null-hypothesis baseline!).
3. **Week 3:** LLM planner (Ollama first), journal/beliefs, say/give.
4. **Week 4:** visualizer + analysis scripts + first Blind vs. Oracle pilot (3 seeds).
5. Then: full RQ1 sweep → writeup.

---

## 7. Risks & Honest Limitations

- **LLM prior contamination:** models "know" Minecraft-style crafting. Mitigation: counter-intuitive tree control (§2) — this is a *feature* scientifically (recall vs. reasoning).
- **JSON output failures from small models:** retry + schema validation + fallback to `wait`. Log failure rate.
- **Slow runs:** local 7B ≈ 2–5 s/call ⇒ a 1000-tick, 12-agent run ≈ 1–2 hours. Fine overnight; parallelize seeds.
- **Scope creep:** playable human mode, combat — parked for later, not v0.
- **Committed for v1: agent reproduction.** Turns knowledge transmission into cultural *inheritance*: agents die, societies either teach the next generation or suffer "dark ages" (measurable recipe loss & rediscovery). Open design question to resolve before v1: what does a newborn inherit — blank slate, parent's notes, or a "childhood" teaching period?
- **Single-author limits:** results framed as "case study + open framework," not grand claims. The open-source framework is itself a contribution (cf. Concordia, Melting Pot).
