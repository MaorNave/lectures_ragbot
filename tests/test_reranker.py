"""The local cross-encoder wrapper and the rerank step — with a stand-in model (no torch, no download)."""

from __future__ import annotations

import pytest

from smartup_rag.config import get_settings
from smartup_rag.retrieval import hybrid, reranker
from smartup_rag.retrieval.embeddings import keep_alive_seconds
from smartup_rag.retrieval.hybrid import Candidate


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kw):
        self.calls.append(list(pairs))
        return [float(len(doc)) / 100 for _, doc in pairs]


def test_score_keeps_input_order_and_loads_once(monkeypatch):
    rr = reranker.CrossEncoderReranker("fake/model", device="cpu")
    model = FakeModel()
    loads: list[int] = []
    monkeypatch.setattr(rr, "_load", lambda: loads.append(1) or model)
    assert rr.score("q", []) == [] and not loads  # nothing to score: the model is not even loaded
    assert rr.score("q", ["ab", "abcd"]) == [0.02, 0.04]
    assert rr.score("q", ["a"]) == [0.01]
    assert loads == [1] and rr.loaded and model.calls[0] == [("q", "ab"), ("q", "abcd")]


def test_a_failed_load_is_not_retried_on_every_question(monkeypatch):
    rr = reranker.CrossEncoderReranker("fake/model", device="cpu")
    attempts: list[int] = []

    def boom():
        attempts.append(1)
        raise OSError("not cached and offline")

    monkeypatch.setattr(rr, "_load", boom)
    with pytest.raises(OSError):
        rr.score("q", ["x"])
    with pytest.raises(RuntimeError, match="reranker unavailable"):
        rr.score("q", ["x"])
    assert attempts == [1] and rr.status()["error"].startswith("OSError")
    monkeypatch.setattr(reranker, "RETRY_LOAD_AFTER_S", 0.0)  # once the cool-down is over, it tries again
    with pytest.raises(OSError):
        rr.score("q", ["x"])
    assert attempts == [1, 1]


def test_explicit_devices_are_taken_as_given():
    assert reranker.resolve_device("cpu") == "cpu"
    assert reranker.resolve_device("mps") == "mps"


def _cands(*texts: str) -> list[Candidate]:
    return [Candidate(text=t, course_id="smartup-foundations-v2", course_version="v2", lecture_number=i + 1,
                      lecture_title=f"L{i + 1}", lang="en", score=1.0 / (i + 1)) for i, t in enumerate(texts)]


class _Scores:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        return list(self.scores)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("MIN_RERANK_SCORE", "0.1")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_rerank_sorts_by_score_and_drops_noise(settings, monkeypatch):
    monkeypatch.setattr(hybrid, "get_reranker", lambda: _Scores([0.05, 0.9, 0.3, 0.1]))
    top = hybrid.rerank("q", _cands("a", "b", "c", "d"), top_n=2)
    assert [c.text for c in top] == ["b", "c"] and [c.rerank_score for c in top] == [0.9, 0.3]
    top = hybrid.rerank("q", _cands("a", "b", "c", "d"), top_n=8)
    assert [c.text for c in top] == ["b", "c", "d"]  # 0.05 is below MIN_RERANK_SCORE


def test_rerank_keeps_rrf_order_when_disabled_or_failing(settings, monkeypatch):
    class Broken:
        def score(self, *_):
            raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(hybrid, "get_reranker", lambda: Broken())
    assert [c.text for c in hybrid.rerank("q", _cands("a", "b", "c"), top_n=2)] == ["a", "b"]
    monkeypatch.setenv("RERANK_ENABLED", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(hybrid, "get_reranker", lambda: pytest.fail("the model must not be called"))
    top = hybrid.rerank("q", _cands("a", "b", "c"), top_n=2)
    assert [c.text for c in top] == ["a", "b"] and all(c.rerank_score is None for c in top)
    assert hybrid.rerank("q", [], top_n=2) == []


@pytest.mark.parametrize(("raw", "seconds"), [("24h", 86400), ("30m", 1800), ("90", 90), ("45s", 45), ("2d", 172800),
                                              ("-1", -1), ("1.5h", 5400), ("soon", None), ("", None)])
def test_keep_alive_durations_become_seconds_for_the_embedding_model(raw, seconds):
    assert keep_alive_seconds(raw) == seconds


def test_break_ties_prefers_the_dense_ranking():
    hits = [("a", 2.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 0.5)]
    assert hybrid.break_ties(hits, prefer=["d", "x", "b"]) == ["a", "d", "b", "c", "e"]
