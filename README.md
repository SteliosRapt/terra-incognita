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
- experiments/ — the DiffusionGemma 2×2 shelter study: matrix runner + analyzer
  (see DIFFUSIONGEMMA_EXPERIMENT_PLAN.md)
- notebooks/diffusiongemma_colab_2x2.ipynb — Colab GPU runner for that study
- run_m1.py / run_m2.py / run_m3.py — milestone acceptance scripts
- tests/ — pytest suite (`py -m pytest`), no LLM or network calls

Status: M2 complete; M3 (LLM planner) in progress — acceptance run pending.
Quick start: `py run_m3.py --provider mock --ticks 300` or `py viewer/server.py`
