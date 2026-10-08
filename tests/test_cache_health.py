"""The shared-prompt cache health signal (agent.llm): a slow classifier prompt means the cache was lost."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smartup_rag.agent import llm as L


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(L, "shares_chat_prefix", lambda: True)
    L._cold_cache.clear()
    yield
    L._cold_cache.clear()


def _reply(prompt_seconds: float) -> SimpleNamespace:
    return SimpleNamespace(response_metadata={"prompt_eval_duration": int(prompt_seconds * 1e9)})


def test_cached_classifier_prompt_leaves_the_cache_marked_healthy() -> None:
    L.note_prompt_timing(_reply(6.1))    # measured with the warm-up checkpoint in place
    assert not L.cache_needs_warm_up()


def test_slow_classifier_prompt_marks_the_cache_cold() -> None:
    L.note_prompt_timing(_reply(28.9))   # measured after a model reload: the whole prompt was re-read
    assert L.cache_needs_warm_up()


def test_missing_or_odd_metadata_is_ignored() -> None:
    L.note_prompt_timing(None)
    L.note_prompt_timing(SimpleNamespace(response_metadata={}))
    assert not L.cache_needs_warm_up()


def test_a_separate_classifier_model_has_no_shared_cache_to_lose(monkeypatch) -> None:
    monkeypatch.setattr(L, "shares_chat_prefix", lambda: False)
    L.note_prompt_timing(_reply(28.9))
    assert not L.cache_needs_warm_up()
