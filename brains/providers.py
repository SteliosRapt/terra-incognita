"""providers.py — LLM backends behind one tiny interface.

The engine never imports this module; only brains do. SDK/network imports are
lazy so the repo stays importable (and testable) without any LLM installed.
Per SPEC §3: default backend is a small local model via Ollama; premium runs
use Claude Haiku via the Anthropic API.
"""
from __future__ import annotations
import json
import random
import re


class Provider:
    """One blocking call: (system, user) -> raw model text."""
    name = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OllamaProvider(Provider):
    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.7, timeout: float = 180.0,
                 json_format: bool = True):
        import requests  # lazy
        self._requests = requests
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_format = json_format
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.json_format:
            body["format"] = "json"   # constrains decoding to valid JSON
        r = self._requests.post(f"{self.host}/api/chat", json=body,
                                timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class AnthropicProvider(Provider):
    # SPEC §3 names Claude Haiku for premium runs (cost estimate is sized to it).
    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 2048):
        import anthropic  # lazy
        self.client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY env
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class MockProvider(Provider):
    """Deterministic fake LLM for pipeline smoke tests and CI. It answers by
    parsing the rendered prompt — which doubles as a check that the prompt
    actually contains actionable information. Forages, stockpiles, tries the
    occasional combine, and chats; all decisions through the shared seeded rng."""
    name = "mock"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def complete(self, system: str, user: str) -> str:
        status = next((l for l in user.splitlines() if l.startswith("== STATUS ==")), "")
        energy = int(m.group(1)) if (m := re.search(r"energy (\d+)/100", status)) else 100
        pos = ((int(m.group(1)), int(m.group(2)))
               if (m := re.search(r"position \((\d+), (\d+)\)", status)) else (0, 0))
        inv = {k: int(v) for k, v in re.findall(r"(\w+) x(\d+)", status)}
        nearby = next((l for l in user.splitlines() if l.startswith("== NEARBY ==")), "")

        def spot(res):
            m = re.search(rf"{res} at \((\d+), (\d+)\)", nearby)
            return (int(m.group(1)), int(m.group(2))) if m else None

        plan, why = None, ""
        if energy < 50 and inv.get("berries", 0) > 0:
            plan, why = [{"action": "eat", "item": "berries"}], "hungry, eating"
        elif inv.get("berries", 0) < 3 and spot("berries"):
            x, y = spot("berries")
            plan = [{"action": "move_to", "x": x, "y": y},
                    {"action": "gather", "resource": "berries"}]
            why = "foraging berries"
        elif inv.get("wood", 0) >= 2 and inv.get("stone", 0) >= 1 and self.rng.random() < 0.5:
            plan = [{"action": "combine", "items": {"wood": 2, "stone": 1}}]
            why = "trying wood+stone"
        elif (other := re.search(r"(A\d\d) at \(", nearby)) and self.rng.random() < 0.2:
            plan = [{"action": "say", "target": other.group(1),
                     "message": "Try two wood and one stone, it makes an axe."}]
            why = "sharing a tip"
        else:
            for res in ("wood", "stone"):
                if inv.get(res, 0) < 3 and spot(res):
                    x, y = spot(res)
                    plan = [{"action": "move_to", "x": x, "y": y},
                            {"action": "gather", "resource": res}]
                    why = f"stockpiling {res}"
                    break
        if plan is None:
            plan = [{"action": "move_to",
                     "x": pos[0] + self.rng.randint(-4, 4),
                     "y": pos[1] + self.rng.randint(-4, 4)},
                    {"action": "wait"}]
            why = "exploring"
        return json.dumps({
            "reasoning": why,
            "updated_beliefs": ["wood and stone seem combinable"],
            "updated_notes": f"energy {energy}, carrying {inv}",
            "plan": plan,
        })


def make_provider(spec: dict, rng: random.Random | None = None) -> Provider:
    """spec: {"backend": "ollama"|"anthropic"|"mock", ...backend kwargs}."""
    spec = dict(spec)
    backend = spec.pop("backend", "ollama")
    if backend == "ollama":
        return OllamaProvider(**spec)
    if backend == "anthropic":
        return AnthropicProvider(**spec)
    if backend == "mock":
        return MockProvider(rng or random.Random(0))
    raise ValueError(f"unknown provider backend: {backend}")
