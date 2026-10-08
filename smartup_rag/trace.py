"""Per-turn pipeline traces for the /dashboard page: what every step received, did and returned, and how long it took.

The API opens a trace for each chat call (`new_trace` + `active`), runs the agent inside it, and keeps the last
`TRACE_LIMIT` traces in memory. Pipeline code records steps with `span(...)`; outside a trace (CLI, eval, tests)
`span` is a no-op, so the agent never depends on tracing.

A step's details are merged into its entry only when the step ends, under the trace's lock, so the dashboard can
serialize a trace while the agent is still writing to it; a running step is visible as `running: true` meanwhile.
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

TRACE_LIMIT = 50  # traces kept in memory (a trace with its passages and prompt is ~20–40 KB)


@dataclass
class Trace:
    id: str
    thread_id: str
    question: str
    endpoint: str
    started_at: float = field(default_factory=time.time)  # epoch seconds, for display
    t0: float = field(default_factory=time.perf_counter)
    steps: list[dict[str, Any]] = field(default_factory=list)
    running: bool = True
    status: str | None = None
    language: str | None = None
    intent: str | None = None
    total_ms: int | None = None
    first_token_ms: int | None = None
    answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "thread_id": self.thread_id[:8],  # enough to group a conversation; the full id would let a reader continue it
                "question": self.question[:200],
                "endpoint": self.endpoint,
                "started_at": self.started_at,
                "running": self.running,
                "status": self.status,
                "language": self.language,
                "intent": self.intent,
                "total_ms": self.total_ms if not self.running else self.elapsed_ms(),
                "first_token_ms": self.first_token_ms,
                "n_steps": len(self.steps),
                "current_step": next((s["step"] for s in reversed(self.steps) if s.get("running")), None),
            }

    def to_dict(self) -> dict[str, Any]:
        out = self.summary()
        with self.lock:
            out.update(
                question=self.question,
                steps=[dict(s) for s in self.steps],
                answer=self.answer,
                citations=list(self.citations),
                error=self.error,
            )
        return out


class TraceStore:
    """The last `limit` traces, newest last; safe to use from the event loop and worker threads."""

    def __init__(self, limit: int = TRACE_LIMIT) -> None:
        self.limit = limit
        self._items: OrderedDict[str, Trace] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._items[trace.id] = trace
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return self._items.get(trace_id)

    def newest_first(self) -> list[Trace]:
        with self._lock:
            return list(reversed(self._items.values()))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


STORE = TraceStore()
_current: ContextVar[Trace | None] = ContextVar("smartup_trace", default=None)
_background: ContextVar[int | None] = ContextVar("smartup_trace_background", default=None)  # depth of background work


def background_context() -> contextvars.Context:
    """A copy of the current context for work handed to another thread (the fallback searches run while v2 is being
    graded): its steps land in this trace, indented under the step that started them, and — unlike a normal nested
    step — they never make the steps that run meanwhile look nested inside them."""
    ctx = contextvars.copy_context()
    trace = _current.get()
    if trace is not None:
        with trace.lock:
            depth = sum(1 for s in trace.steps if s.get("running") and not s.get("background"))
        ctx.run(_background.set, depth)
    return ctx


def new_trace(question: str, thread_id: str, endpoint: str) -> Trace:
    trace = Trace(id=uuid.uuid4().hex[:12], thread_id=thread_id, question=question, endpoint=endpoint)
    STORE.add(trace)
    return trace


@contextmanager
def active(trace: Trace | None) -> Iterator[Trace | None]:
    """Make `trace` the one `span` writes to, for this context and every thread/task started from it."""
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


def current() -> Trace | None:
    return _current.get()


@contextmanager
def span(step: str, **data: Any) -> Iterator[dict[str, Any]]:
    """Record one pipeline step. Yields a dict the step fills in; timing is added automatically.

    Untraced (no active trace) the dict is simply discarded, so callers never need to check.
    """
    trace = _current.get()
    details: dict[str, Any] = dict(data)
    if trace is None:
        yield details
        return
    entry: dict[str, Any] = {"step": step, "start_ms": trace.elapsed_ms(), "running": True}
    bg = _background.get()
    with trace.lock:
        if bg is not None:
            entry["depth"], entry["background"] = bg, True
        else:  # nested inside every still-running step, except work running in the background
            entry["depth"] = sum(1 for s in trace.steps if s.get("running") and not s.get("background"))
        trace.steps.append(entry)
    t = time.perf_counter()
    try:
        yield details
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with trace.lock:
            entry.update(details)
            entry["duration_ms"] = int((time.perf_counter() - t) * 1000)
            entry["running"] = False


def finish(trace: Trace, result: dict[str, Any] | None = None, *, error: str | None = None) -> None:
    """Close a trace with the agent's result (or the error that ended it)."""
    result = result or {}
    with trace.lock:
        trace.running = False
        trace.total_ms = trace.elapsed_ms()
        trace.status = result.get("status") or ("error" if error else trace.status)
        trace.language = result.get("language") or trace.language
        trace.intent = result.get("intent") or trace.intent
        trace.answer = result.get("answer")
        trace.citations = [
            {k: c.get(k) for k in ("citation", "course_version", "lecture_number", "lecture_part", "lecture_title")}
            for c in (result.get("citations") or [])
            if isinstance(c, dict)
        ]
        trace.error = error


def llm_timing(message: Any) -> dict[str, Any]:
    """Token counts and durations Ollama reports for one call (a LangChain message, or its `response_metadata`)."""
    meta = message if isinstance(message, dict) else (getattr(message, "response_metadata", None) or {})
    if not meta:
        return {}
    seconds = lambda k: round((meta.get(k) or 0) / 1e9, 2)  # noqa: E731
    out_tok, out_s = meta.get("eval_count") or 0, seconds("eval_duration")
    return {
        "model": meta.get("model"),
        "prompt_tokens": meta.get("prompt_eval_count"),
        "prompt_s": seconds("prompt_eval_duration"),
        "output_tokens": out_tok,
        "output_s": out_s,
        "load_s": seconds("load_duration"),
        "tokens_per_s": round(out_tok / out_s, 1) if out_s else None,
    }
