"""smartup_rag.trace — per-turn pipeline traces for the dashboard (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from smartup_rag import trace as T


@pytest.fixture(autouse=True)
def _clean_store():
    T.STORE.clear()
    yield
    T.STORE.clear()


def test_span_outside_a_trace_is_a_no_op() -> None:
    with T.span("classify", x=1) as sp:
        sp["y"] = 2          # callers never need to check whether tracing is on
    assert T.current() is None and T.STORE.newest_first() == []


def test_span_records_timing_details_and_nesting() -> None:
    tr = T.new_trace("What is JTBD?", "thread-1", "/chat/stream")
    with T.active(tr), T.span("retrieve") as outer:
        outer["scope"] = "default"
        with T.span("search v2") as inner:
            assert tr.steps[-1]["running"] is True       # visible while it runs…
            assert "passages" not in tr.steps[-1]        # …but details land only when it ends
            inner["passages"] = [{"rank": 1}]
    retrieve, search = tr.steps
    assert retrieve["step"] == "retrieve" and retrieve["depth"] == 0 and retrieve["scope"] == "default"
    assert search["step"] == "search v2" and search["depth"] == 1 and search["passages"] == [{"rank": 1}]
    assert all(s["running"] is False and s["duration_ms"] >= 0 and s["start_ms"] >= 0 for s in tr.steps)


def test_span_records_errors_and_re_raises() -> None:
    tr = T.new_trace("q", "t", "/chat")
    with T.active(tr), pytest.raises(ValueError), T.span("classify"):
        raise ValueError("model down")
    assert tr.steps[0]["error"] == "ValueError: model down" and tr.steps[0]["running"] is False


def test_finish_and_serialisation() -> None:
    tr = T.new_trace("q", "t", "/chat/stream")
    tr.first_token_ms = 1234
    T.finish(tr, {"status": "answered", "language": "en", "intent": "conceptual", "answer": "A.",
                  "citations": [{"citation": '(v2, Lecture 1 — "T")', "course_version": "v2", "lecture_number": 1}]})
    d = tr.to_dict()
    json.dumps(d)  # the dashboard receives exactly this
    assert d["running"] is False and d["status"] == "answered" and d["first_token_ms"] == 1234
    assert d["citations"][0]["citation"] == '(v2, Lecture 1 — "T")'
    assert tr.summary()["total_ms"] == tr.total_ms


def test_store_keeps_the_newest_traces_first_and_drops_the_oldest() -> None:
    store = T.TraceStore(limit=2)
    a, b, c = (T.Trace(id=i, thread_id="t", question=i, endpoint="/chat") for i in "abc")
    for tr in (a, b, c):
        store.add(tr)
    assert [t.id for t in store.newest_first()] == ["c", "b"] and store.get("a") is None


def test_llm_timing_reads_ollama_metadata() -> None:
    meta = {"model": "qwen3.5:9b", "prompt_eval_count": 8000, "prompt_eval_duration": 12_300_000_000,
            "eval_count": 300, "eval_duration": 15_000_000_000, "load_duration": 200_000_000}
    t = T.llm_timing(SimpleNamespace(response_metadata=meta))
    assert t == T.llm_timing(meta)
    assert t["prompt_tokens"] == 8000 and t["prompt_s"] == 12.3 and t["tokens_per_s"] == 20.0
    assert T.llm_timing(None) == {} and T.llm_timing({}) == {}


def test_background_steps_do_not_nest_the_steps_that_run_meanwhile():
    """The fallback searches run in other threads while v2 is being graded: they sit under "retrieve", and the grade
    that starts meanwhile stays a top-level step."""
    import threading

    tr = T.new_trace("q", "t", "test")
    done = threading.Event()
    with T.active(tr):
        with T.span("retrieve"):
            ctx = T.background_context()

            def work():
                with T.span("search tedx"):
                    done.wait(2)

            th = threading.Thread(target=ctx.run, args=(work,))
            th.start()
        with T.span("grade v2"):
            pass
        done.set()
        th.join()
    depth = {s["step"]: s["depth"] for s in tr.steps}
    assert depth == {"retrieve": 0, "search tedx": 1, "grade v2": 0}
    assert [s.get("background", False) for s in tr.steps] == [False, True, False]
