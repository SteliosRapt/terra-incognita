import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brains.telemetry import CompletionMetrics, CompletionResult, RequestRecorder
from brains.providers import OllamaProvider, OpenRouterProvider


class StreamingResponse:
    def __init__(self, lines, status=200, headers=None):
        self.lines = [line.encode("utf-8") for line in lines]
        self.status_code = status
        self.headers = headers or {}

    def iter_lines(self):
        return iter(self.lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class StreamingRequests:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []
        self.stream_flags = []

    def post(self, url, headers=None, json=None, timeout=None, stream=None):
        self.bodies.append(json)
        self.stream_flags.append(stream)
        return self.responses.pop(0)


def openrouter_provider(lines, thinking):
    provider = OpenRouterProvider("requested/gemma", api_key="test-key",
                                  thinking=thinking)
    provider._requests = StreamingRequests(StreamingResponse(lines))
    return provider


def ollama_provider(chunks, thinking):
    provider = OllamaProvider("gemma4:e2b", thinking=thinking)
    lines = [json.dumps(chunk) for chunk in chunks]
    provider._requests = StreamingRequests(StreamingResponse(lines))
    return provider


def test_completion_metrics_derive_visible_tokens_and_tps():
    metrics = CompletionMetrics(
        requested_model="google/gemma-4-31b-it",
        actual_model="google/gemma-4-31b-it",
        started_at=100.0,
        first_token_at=100.25,
        finished_at=101.25,
        prompt_tokens=600,
        cached_tokens=100,
        completion_tokens=80,
        reasoning_tokens=50,
        cost_usd=0.001,
    )

    assert metrics.ttft_ms == 250
    assert metrics.duration_ms == 1250
    assert metrics.visible_response_tokens == 30
    assert metrics.generation_tps == pytest.approx(80.0)


def test_request_recorder_merges_context_and_metrics(tmp_path):
    path = tmp_path / "requests.jsonl"
    recorder = RequestRecorder(path)
    result = CompletionResult(
        text='{"plan":[{"action":"wait"}]}',
        reasoning_text="consider options",
        metrics=CompletionMetrics(
            requested_model="m/requested",
            actual_model="m/routed",
            started_at=10.0,
            first_token_at=10.1,
            finished_at=10.5,
            prompt_tokens=20,
            cached_tokens=5,
            completion_tokens=12,
            reasoning_tokens=7,
            cost_usd=0.002,
        ),
    )

    recorder.write(result, role="planner", tick=4, subject="A02",
                   attempt=1, provider_ok=True, parse_ok=False,
                   error="JSONDecodeError: bad response")

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["role"] == "planner"
    assert row["tick"] == 4
    assert row["subject"] == "A02"
    assert row["actual_model"] == "m/routed"
    assert row["visible_response_tokens"] == 5
    assert row["parse_ok"] is False


def test_openrouter_stream_records_first_token_usage_and_reasoning():
    lines = [
        'data: {"model":"routed/gemma","choices":[{"delta":{"reasoning":"think"}}]}',
        'data: {"model":"routed/gemma","choices":[{"delta":{"content":"{\\"plan\\":"}}]}',
        'data: {"model":"routed/gemma","choices":[{"delta":{"content":"[]}"}}],"usage":{"prompt_tokens":40,"completion_tokens":9,"cost":0.0003,"prompt_tokens_details":{"cached_tokens":10},"completion_tokens_details":{"reasoning_tokens":4}}}',
        "data: [DONE]",
    ]
    provider = openrouter_provider(lines, thinking=True)

    result = provider.complete_measured("system", "user")

    assert result.text == '{"plan":[]}'
    assert result.reasoning_text == "think"
    assert result.metrics.actual_model == "routed/gemma"
    assert result.metrics.cached_tokens == 10
    assert result.metrics.reasoning_tokens == 4
    assert result.metrics.ttft_ms is not None
    assert provider._requests.stream_flags == [True]
    assert provider._requests.bodies[0]["reasoning"] == {"enabled": True}


def test_openrouter_nonthinking_sends_explicit_disable():
    lines = [
        'data: {"model":"routed/gemma","choices":[{"delta":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":0}}',
        "data: [DONE]",
    ]
    provider = openrouter_provider(lines, thinking=False)

    assert provider.complete("system", "user") == "ok"
    assert provider._requests.bodies[0]["reasoning"] == {"enabled": False}


def test_ollama_stream_assembles_thinking_content_and_native_counts():
    chunks = [
        {"message": {"thinking": "consider ", "content": ""}, "done": False},
        {"message": {"thinking": "", "content": '{"plan":[]}'}, "done": False},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 30, "eval_count": 8,
         "eval_duration": 400_000_000},
    ]
    provider = ollama_provider(chunks, thinking=True)

    result = provider.complete_measured("system", "user")

    assert result.text == '{"plan":[]}'
    assert result.reasoning_text == "consider "
    assert result.metrics.prompt_tokens == 30
    assert result.metrics.completion_tokens == 8
    assert result.metrics.reasoning_tokens == 4
    assert result.metrics.token_split_method == "stream_delta_ratio"
    assert provider._requests.bodies[0]["think"] is True
    assert provider._requests.bodies[0]["stream"] is True


def test_ollama_nonthinking_sends_explicit_disable():
    chunks = [
        {"message": {"content": "ok"}, "done": False},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 2, "eval_count": 1,
         "eval_duration": 10_000_000},
    ]
    provider = ollama_provider(chunks, thinking=False)

    assert provider.complete("system", "user") == "ok"
    assert provider._requests.bodies[0]["think"] is False
