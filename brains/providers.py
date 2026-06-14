"""providers.py — LLM backends behind one tiny interface.

The engine never imports this module; only brains do. SDK/network imports are
lazy so the repo stays importable (and testable) without any LLM installed.
Per SPEC §3: default backend is a small local model via Ollama; premium runs
use Claude Haiku via the Anthropic API.
"""
from __future__ import annotations
import json
import os
import pathlib
import random
import re
import threading
import time

from .telemetry import CompletionError, CompletionMetrics, CompletionResult


class Provider:
    """One blocking call: (system, user) -> raw model text."""
    name = "base"
    parallel_safe = False   # True = instances may be called from worker threads

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_measured(self, system: str, user: str) -> CompletionResult:
        started = time.time()
        text = self.complete(system, user)
        finished = time.time()
        return CompletionResult(text, "", CompletionMetrics(
            requested_model=self.name,
            actual_model=self.name,
            started_at=started,
            first_token_at=None,
            finished_at=finished,
        ))

    def complete_batch(self, prompts: list[tuple[str, str]]
                       ) -> tuple[list[str], dict]:
        """Plan several prompts at once. Returns (texts, metrics). The default
        just loops complete() with no batch metrics; batching backends (e.g.
        DiffusionGemma) override this with a real one-call batch."""
        return [self.complete(s, u) for s, u in prompts], {}


class OllamaProvider(Provider):
    parallel_safe = True   # blocking HTTP; Ollama serializes requests server-side

    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.7, timeout: float = 180.0,
                 json_format: bool = True, thinking: bool | None = None,
                 max_tokens: int = 2048):
        import requests  # lazy
        self._requests = requests
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_format = json_format
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        return self.complete_measured(system, user).text

    def complete_measured(self, system: str, user: str) -> CompletionResult:
        started = time.time()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "options": {"temperature": self.temperature,
                        "num_predict": self.max_tokens},
        }
        if self.thinking is not None:
            body["think"] = self.thinking
        if self.json_format:
            body["format"] = "json"   # constrains decoding to valid JSON
        try:
            r = self._requests.post(f"{self.host}/api/chat", json=body,
                                    timeout=self.timeout, stream=True)
            r.raise_for_status()
            content, reasoning = [], []
            first_token_at = None
            content_deltas = reasoning_deltas = 0
            final = {}
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                message = chunk.get("message") or {}
                thought = message.get("thinking") or ""
                text = message.get("content") or ""
                if thought or text:
                    first_token_at = first_token_at or time.time()
                if thought:
                    reasoning.append(thought)
                    reasoning_deltas += 1
                if text:
                    content.append(text)
                    content_deltas += 1
                if chunk.get("done"):
                    final = chunk
            finished = time.time()
            completion_tokens = int(final.get("eval_count", 0) or 0)
            delta_count = reasoning_deltas + content_deltas
            reasoning_tokens = (round(completion_tokens * reasoning_deltas / delta_count)
                                if delta_count else 0)
            metrics = CompletionMetrics(
                requested_model=self.model,
                actual_model=final.get("model") or self.model,
                started_at=started,
                first_token_at=first_token_at,
                finished_at=finished,
                prompt_tokens=int(final.get("prompt_eval_count", 0) or 0),
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                token_split_method="stream_delta_ratio",
            )
            return CompletionResult("".join(content), "".join(reasoning), metrics)
        except CompletionError:
            raise
        except Exception as e:
            finished = time.time()
            metrics = CompletionMetrics(
                requested_model=self.model, actual_model=self.model,
                started_at=started, first_token_at=None, finished_at=finished,
                token_split_method="stream_delta_ratio")
            raise CompletionError(f"ollama [{self.model}]: {e}", metrics) from e


class AnthropicProvider(Provider):
    # SPEC §3 names Claude Haiku for premium runs (cost estimate is sized to it).
    parallel_safe = True   # anthropic SDK client is thread-safe

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


# mix runs share one costs.jsonl between two provider instances (free + paid)
_COST_FILE_LOCK = threading.Lock()


class OpenRouterProvider(Provider):
    """OpenRouter chat-completions backend with a model fallback chain.

    `models` is an ordered preference list. A model that answers 429 (rate
    limited) or 402 (out of credits — free-tier daily cap) is put on cooldown
    and the next model in the chain takes over; when the cooldown expires the
    preferred model is tried again. This is the ONE sanctioned fallback in the
    codebase (free-tier rate limits are an environment fact, not a bug):
    every other error — 4xx, 5xx, network, malformed body — raises.

    One instance is SHARED by all agents (see run_m3/viewer factories) so the
    cooldown clock and the spend ledger are global, and complete() may be
    called from several planner threads at once: all mutable state lives
    behind self._lock. Every reply's usage block (OpenRouter reports cost in
    USD when usage.include is set) is appended to costs.jsonl next to
    prompts.jsonl, with running totals.
    """
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    parallel_safe = True

    def __init__(self, models: list[str] | str, api_key: str | None = None,
                 temperature: float = 0.7, timeout: float = 180.0,
                 max_tokens: int = 2048, cooldown_s: float = 65.0,
                 credit_cooldown_s: float = 3600.0,
                 reasoning_effort: str | None = None,
                 thinking: bool | None = None,
                 cost_path: str | None = None):
        import requests  # lazy
        self._requests = requests
        if isinstance(models, str):
            models = [m.strip() for m in models.split(",") if m.strip()]
        if not models:
            raise ValueError("openrouter: empty model list")
        self.models = models
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("openrouter: no api key (set OPENROUTER_API_KEY)")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.cooldown_s = cooldown_s
        self.credit_cooldown_s = credit_cooldown_s
        self.cost_path = cost_path
        self.name = f"openrouter:{models[0]}"
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}   # model -> time.time()
        self.totals = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                       "completion_tokens": 0, "reasoning_tokens": 0,
                       "cost_usd": 0.0, "rate_limited": 0}

    # ---- fallback chain ----------------------------------------------------
    def _next_available(self) -> str | None:
        """First model in preference order whose cooldown has expired."""
        now = time.time()
        for m in self.models:
            if self._cooldown_until.get(m, 0.0) <= now:
                return m
        return None

    def _start_cooldown(self, model: str, resp, default_s: float):
        until = time.time() + default_s
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                until = time.time() + float(retry_after)
            except ValueError:
                pass
        else:
            reset = resp.headers.get("X-RateLimit-Reset")  # epoch milliseconds
            if reset:
                try:
                    until = max(until, float(reset) / 1000.0)
                except ValueError:
                    pass
        with self._lock:
            self._cooldown_until[model] = until
            self.totals["rate_limited"] += 1

    # ---- accounting ----------------------------------------------------------
    def _record_usage(self, model: str, usage: dict):
        cached = ((usage.get("prompt_tokens_details") or {})
                  .get("cached_tokens", 0) or 0)
        reasoning = ((usage.get("completion_tokens_details") or {})
                     .get("reasoning_tokens", 0) or 0)
        with self._lock:
            self.totals["calls"] += 1
            self.totals["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.totals["cached_tokens"] += cached
            self.totals["completion_tokens"] += usage.get("completion_tokens", 0) or 0
            self.totals["reasoning_tokens"] += reasoning
            self.totals["cost_usd"] += usage.get("cost", 0.0) or 0.0
            if self.cost_path:
                rec = {"time": time.time(), "model": model,
                       "prompt_tokens": usage.get("prompt_tokens", 0),
                       "cached_tokens": cached,
                       "completion_tokens": usage.get("completion_tokens", 0),
                       "reasoning_tokens": reasoning,
                       "cost_usd": usage.get("cost", 0.0),
                       "total_cost_usd": round(self.totals["cost_usd"], 6),
                       "total_calls": self.totals["calls"]}
                with _COST_FILE_LOCK:
                    with open(self.cost_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- the call ------------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        return self.complete_measured(system, user).text

    def complete_measured(self, system: str, user: str) -> CompletionResult:
        deadline = time.time() + self.timeout
        while True:
            with self._lock:
                model = self._next_available()
                wait_until = (min(self._cooldown_until.get(m, 0.0)
                                  for m in self.models)
                              if model is None else 0.0)
            if model is None:
                # whole chain is rate limited: block until the soonest reset
                if wait_until > deadline:
                    raise RuntimeError(
                        "openrouter: all models rate limited past the timeout "
                        f"({', '.join(self.models)})")
                time.sleep(max(0.5, wait_until - time.time()))
                continue
            body = {"model": model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True,
                    "usage": {"include": True}}
            if self.thinking is not None:
                body["reasoning"] = {"enabled": self.thinking}
            elif self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort}
            started = time.time()
            try:
                resp = self._requests.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "X-Title": "terra-incognita"},
                    json=body, timeout=self.timeout, stream=True)
            except Exception as e:
                finished = time.time()
                raise CompletionError(
                    f"openrouter [{model}]: {e}",
                    CompletionMetrics(model, model, started, None, finished)) from e
            if resp.status_code == 429:
                self._start_cooldown(model, resp, self.cooldown_s)
                continue
            if resp.status_code == 402:   # free credits exhausted for the day
                self._start_cooldown(model, resp, self.credit_cooldown_s)
                continue
            try:
                resp.raise_for_status()
                content, reasoning = [], []
                usage, actual_model = {}, model
                first_token_at = None
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    actual_model = data.get("model") or actual_model
                    usage = data.get("usage") or usage
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        thought = delta.get("reasoning") or ""
                        text = delta.get("content") or ""
                        if thought or text:
                            first_token_at = first_token_at or time.time()
                        if thought:
                            reasoning.append(thought)
                        if text:
                            content.append(text)
                finished = time.time()
                cached = ((usage.get("prompt_tokens_details") or {})
                          .get("cached_tokens", 0) or 0)
                reasoning_tokens = ((usage.get("completion_tokens_details") or {})
                                    .get("reasoning_tokens", 0) or 0)
                metrics = CompletionMetrics(
                    requested_model=model,
                    actual_model=actual_model,
                    started_at=started,
                    first_token_at=first_token_at,
                    finished_at=finished,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    cached_tokens=int(cached),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    reasoning_tokens=int(reasoning_tokens),
                    cost_usd=float(usage.get("cost", 0.0) or 0.0),
                )
                self._record_usage(actual_model, usage)
                return CompletionResult("".join(content), "".join(reasoning), metrics)
            except CompletionError:
                raise
            except Exception as e:
                finished = time.time()
                metrics = CompletionMetrics(model, model, started, None, finished)
                raise CompletionError(f"openrouter [{model}]: {e}", metrics) from e


class DiffusionGemmaProvider(Provider):
    """Lazy Transformers backend for google/diffusiongemma-26B-A4B-it, built for
    BATCHED inference: one model.generate() plans every agent of a tick.

    GPU-only — fails clearly when no CUDA device is present. All weights load
    inside Colab (dtype='auto', device_map='auto'); the DiffusionGemma install
    that may already exist on this machine is never touched. Tests inject
    `model` and `processor` fakes so the batching path is exercised without
    downloading any weights.
    """
    name = "diffusiongemma"
    parallel_safe = False     # one device, batched — never call from threads
    DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 max_new_tokens: int = 1024, dtype: str = "auto",
                 device_map: str = "auto", thinking: bool = True,
                 think_token: str = "<|think|>",
                 answer_delimiter: str | None = "<channel|>",
                 quantization: str | None = None,
                 chat_template_kwargs: dict | None = None,
                 gen_kwargs: dict | None = None,
                 model=None, processor=None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.device_map = device_map
        # quantization: None (full bf16, needs ~80 GB), "4bit" (~13 GB, NF4) or
        # "8bit" (~26 GB, int8) via bitsandbytes — both Ampere-compatible, so a
        # 40 GB A100 can run the 26B model. FP8/NVFP4 checkpoints need Hopper/
        # Blackwell and are NOT handled here.
        if quantization not in (None, "4bit", "8bit"):
            raise ValueError(f"quantization must be None, '4bit' or '8bit', "
                             f"got {quantization!r}")
        self.quantization = quantization
        # Thinking ON by default (model card: enable it by placing the <|think|>
        # token at the start of the system prompt). The model then emits its
        # reasoning in a thought channel followed by the final answer;
        # answer_delimiter is the channel-close tag after which the answer (our
        # JSON plan) appears, so we drop everything up to and including it.
        # max_new_tokens is sized larger than the card's 512 to leave room for
        # the reasoning before the JSON. All three are overridable if the actual
        # chat template / channel markers differ.
        self.thinking = thinking
        self.think_token = think_token
        self.answer_delimiter = answer_delimiter
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.gen_kwargs = dict(gen_kwargs or {})
        self.model = model            # injected in tests; else lazy-loaded
        self.processor = processor
        self.name = f"diffusiongemma:{model_id}"

    def _final_answer(self, text: str) -> str:
        """With thinking on, the reply is '<thought channel> ... <delimiter>
        <answer>'. Keep only what follows the last delimiter so the JSON plan
        is what gets parsed; reasoning still lands verbatim in prompts.jsonl."""
        text = str(text)
        if self.thinking and self.answer_delimiter and self.answer_delimiter in text:
            return text.rsplit(self.answer_delimiter, 1)[-1]
        return text

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return
        import torch  # lazy
        from transformers import (DiffusionGemmaForBlockDiffusion,  # lazy
                                  AutoProcessor)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DiffusionGemmaProvider requires a CUDA GPU; none is available. "
                "Run inference inside Colab on a GPU runtime.")
        load_kwargs = {"device_map": self.device_map}
        if self.quantization:
            from transformers import BitsAndBytesConfig  # lazy (needs bitsandbytes)
            if self.quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16)
            else:  # "8bit"
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True)
        else:
            load_kwargs["dtype"] = self.dtype     # full precision
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.model_id, **load_kwargs)
        self.model.eval()

    def complete(self, system: str, user: str) -> str:
        return self.complete_batch([(system, user)])[0][0]

    def complete_batch(self, prompts: list[tuple[str, str]]
                       ) -> tuple[list[str], dict]:
        self._ensure_loaded()
        import torch  # lazy
        # thinking: prepend the <|think|> token to the system prompt (model card)
        messages = [[{"role": "system",
                      "content": (self.think_token + s) if self.thinking else s},
                     {"role": "user", "content": u}] for s, u in prompts]
        # Left-pad so one prompt-length slice strips the prompt from every row.
        tok = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
        enc = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
            **self.chat_template_kwargs)
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(enc, "to"):
            enc = enc.to(device)
        in_len = enc["input_ids"].shape[1]
        prompt_tokens = int(enc["attention_mask"].sum())

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      **self.gen_kwargs)
        dur = time.monotonic() - t0

        new = out[:, in_len:]
        # keep special tokens when thinking so the channel markers survive for
        # _final_answer to split on; drop them otherwise.
        texts = self.processor.batch_decode(new, skip_special_tokens=not self.thinking)
        texts = [self._final_answer(t) for t in texts]
        pad_id = getattr(tok, "pad_token_id", None)
        completion_tokens = (int((new != pad_id).sum()) if pad_id is not None
                             else int(new.shape[0] * new.shape[1]))
        metrics = {
            "duration_s": round(dur, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tps": round(completion_tokens / dur, 2) if dur > 0 else 0.0,
            "device": str(device) if device is not None else self.device_map,
            "device_map": self.device_map,
            "peak_gpu_mem_bytes": (int(torch.cuda.max_memory_allocated())
                                   if cuda else None),
        }
        return [str(t) for t in texts], metrics


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


OPENROUTER_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "openrouter.json"


def make_provider(spec: dict, rng: random.Random | None = None) -> Provider:
    """spec: {"backend": "ollama"|"anthropic"|"openrouter"|"mock", ...kwargs}.

    openrouter: kwargs missing from the spec are filled from
    configs/openrouter.json (the model configurator) — so the chain,
    temperature etc. live in one editable file and the CLI/UI only
    overrides what the user typed. Chain selection precedence:
    --model (explicit) > preset=<name> > tier=paid > config "models".
    """
    spec = dict(spec)
    backend = spec.pop("backend", "ollama")
    if backend == "ollama":
        return OllamaProvider(**spec)
    if backend == "anthropic":
        return AnthropicProvider(**spec)
    if backend == "openrouter":
        if OPENROUTER_CONFIG.exists():
            with open(OPENROUTER_CONFIG, encoding="utf-8") as f:
                defaults = json.load(f)
            spec = {**defaults, **spec}
        for meta in ("comment", "preset_notes", "judge_models"):
            spec.pop(meta, None)      # config metadata, not provider kwargs
        presets = spec.pop("presets", {})
        paid = spec.pop("paid_models", None)
        if spec.pop("tier", "free") == "paid":
            if not paid:
                raise ValueError("openrouter: tier=paid but no paid_models in "
                                 f"{OPENROUTER_CONFIG}")
            spec["models"] = paid
        preset = spec.pop("preset", None)
        if preset is not None:
            if preset not in presets:
                raise ValueError(f"openrouter: unknown preset {preset!r} "
                                 f"(have: {', '.join(sorted(presets))})")
            spec["models"] = presets[preset]
        if "model" in spec:           # CLI/UI single --model overrides the chain
            spec["models"] = spec.pop("model")
        return OpenRouterProvider(**spec)
    if backend == "diffusiongemma":
        return DiffusionGemmaProvider(**spec)
    if backend == "mock":
        return MockProvider(rng or random.Random(0))
    raise ValueError(f"unknown provider backend: {backend}")
