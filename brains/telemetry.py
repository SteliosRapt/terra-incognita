"""Structured timing, token, and cost telemetry for provider requests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import pathlib
import threading


@dataclass
class CompletionMetrics:
    requested_model: str
    actual_model: str
    started_at: float
    first_token_at: float | None
    finished_at: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    token_split_method: str = "provider_reported"

    @property
    def ttft_ms(self) -> int | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000)

    @property
    def duration_ms(self) -> int:
        return round((self.finished_at - self.started_at) * 1000)

    @property
    def visible_response_tokens(self) -> int:
        return max(0, self.completion_tokens - self.reasoning_tokens)

    @property
    def generation_tps(self) -> float | None:
        if self.first_token_at is None or self.finished_at <= self.first_token_at:
            return None
        return self.completion_tokens / (self.finished_at - self.first_token_at)

    def as_record(self) -> dict:
        return {
            **asdict(self),
            "ttft_ms": self.ttft_ms,
            "duration_ms": self.duration_ms,
            "visible_response_tokens": self.visible_response_tokens,
            "generation_tps": self.generation_tps,
        }


@dataclass
class CompletionResult:
    text: str
    reasoning_text: str
    metrics: CompletionMetrics


class CompletionError(RuntimeError):
    def __init__(self, message: str, metrics: CompletionMetrics):
        super().__init__(message)
        self.metrics = metrics


class RequestRecorder:
    _lock = threading.Lock()

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)

    def write(self, result: CompletionResult | CompletionError, **context):
        rec = {**context, **result.metrics.as_record()}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
