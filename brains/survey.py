"""survey.py — the out-of-band epistemics survey (notes page 1: "when did
they answer each question?").

Every N ticks a JUDGE model (cheap/free chain, separate from the planner
provider so it doesn't eat the planners' rate limits) is shown the ground
truth and one subject's writings (an agent's beliefs+notes, or the book's
answers) and returns a binary verdict per question: does this text state a
correct answer? The judge never talks to agents and never touches the World —
measurement cannot contaminate behavior. Verdicts append to survey.jsonl.

Change-gated: a subject whose text hasn't changed since its last successful
grading is skipped, so quiet stretches cost zero calls.
"""
from __future__ import annotations
import json
import threading
import time

from .providers import Provider
from .llm_planner import _extract_json
from .telemetry import (CompletionError, CompletionMetrics, CompletionResult,
                        RequestRecorder)

JUDGE_SYSTEM = (
    "You are a strict examiner in a research experiment. Subjects live in a "
    "simulated survival world and write down beliefs about how it works. You "
    "are given the GROUND TRUTH about the world and one subject's writings. "
    "For each question, answer true ONLY if the subject's text states a "
    "correct answer to that question: numbers must be within about 10 "
    "percent (or 5 percentage points for fractions of the map), ingredient "
    "lists, quantities and special conditions must all be right. Vague, "
    "partial, missing or contradicting claims are false. Judge only from the "
    "text given. Respond ONLY with a JSON object mapping every question id "
    "to true or false, nothing else."
)

_file_lock = threading.Lock()


class SurveyJudge:
    def __init__(self, provider: Provider, questions: list[dict],
                 ground_truth: dict[str, str], out_path: str,
                 max_workers: int = 4, request_path: str | None = None):
        self.provider = provider
        self.questions = [q for q in questions if q["id"] in ground_truth]
        self.truth = ground_truth
        self.out_path = out_path
        self.max_workers = max_workers
        self.request_recorder = RequestRecorder(request_path) if request_path else None
        self._last_graded: dict[str, str] = {}   # subject -> last text graded ok

    # ---- prompt ------------------------------------------------------------
    def _user_prompt(self, text: str) -> str:
        qs = "\n".join(f"[{q['id']}] {q['text']}\n  GROUND TRUTH: {self.truth[q['id']]}"
                       for q in self.questions)
        example = ", ".join(f'"{q["id"]}": false' for q in self.questions[:2])
        return (f"== QUESTIONS ==\n{qs}\n\n== SUBJECT'S WRITINGS ==\n{text}\n\n"
                f"Verdict JSON now ({{{example}, ...}} — one key per question id):")

    # ---- one subject ---------------------------------------------------------
    def _grade_one(self, tick: int, subject: str, text: str) -> dict:
        rec = {"tick": tick, "subject": subject, "model": self.provider.name,
               "ok": False, "error": None, "results": None}
        completion = None
        provider_ok = False
        started_at = time.time()
        t0 = time.monotonic()
        try:
            prompt = self._user_prompt(text)
            if hasattr(self.provider, "complete_measured"):
                completion = self.provider.complete_measured(JUDGE_SYSTEM, prompt)
            else:
                raw = self.provider.complete(JUDGE_SYSTEM, prompt)
                completion = CompletionResult(raw, "", CompletionMetrics(
                    requested_model=self.provider.name,
                    actual_model=self.provider.name,
                    started_at=started_at,
                    first_token_at=None,
                    finished_at=time.time()))
            provider_ok = True
            raw = completion.text
            data = _extract_json(raw)
            if not isinstance(data, dict):
                raise ValueError("judge reply is not a JSON object")
            rec["results"] = {q["id"]: bool(data.get(q["id"], False))
                              for q in self.questions}
            rec["ok"] = True
            self._last_graded[subject] = text
        except CompletionError as e:
            completion = e
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        except Exception as e:   # judged again next wave; failure is recorded
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            if completion is None:
                completion = CompletionError(str(e), CompletionMetrics(
                    requested_model=self.provider.name,
                    actual_model=self.provider.name,
                    started_at=started_at,
                    first_token_at=None,
                    finished_at=time.time()))
        rec["ms"] = int((time.monotonic() - t0) * 1000)
        if self.request_recorder:
            self.request_recorder.write(
                completion, role="survey", tick=tick, subject=subject,
                attempt=1, provider_ok=provider_ok, parse_ok=rec["ok"],
                error=rec["error"])
        return rec

    # ---- one wave ------------------------------------------------------------
    def grade(self, tick: int, subjects: dict[str, str]) -> int:
        """subjects: id -> current writings. Returns # of judge calls made."""
        todo = {s: t for s, t in subjects.items()
                if t.strip() and t != self._last_graded.get(s)}
        if not todo:
            return 0
        items = sorted(todo.items())
        if len(items) > 1 and getattr(self.provider, "parallel_safe", False):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                recs = list(ex.map(lambda it: self._grade_one(tick, *it), items))
        else:
            recs = [self._grade_one(tick, s, t) for s, t in items]
        with _file_lock:
            with open(self.out_path, "a", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(recs)
