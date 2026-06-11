# Terra Incognita
Multi-agent simulation framework for studying collective discovery in LLM agent
societies: a survival grid world whose technology rules are hidden from the
agents, who must discover them by experimentation.

- SPEC.md — full design specification (world, tech tree, agent architecture,
  experimental design, metrics)
- IMPLEMENTATION.md — ground rules, milestones M1-M5, open questions
- CHANGELOG.md — version notes incl. measured behavioral effects of each change
- engine/ — world, agents, actions, events, tick loop (zero LLM dependencies)
- brains/ — ForagerBrain, RandomScientistBrain (M3 adds the LLM planner)
- configs/techtree_v0.json — the hidden ground-truth technology tree
- run_m1.py / run_m2.py / test_chain.py — acceptance tests (all passing)

Status: M2 complete. Next: M3 (LLM planner via Ollama/Anthropic).
Quick start: python run_m2.py
