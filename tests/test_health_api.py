"""GET /health: Ollama (chat + embedding model), the local index and the reranker — never raises, leaks no paths."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smartup_rag.agent import llm
from smartup_rag.api.app import create_app
from smartup_rag.config import Settings
from smartup_rag.retrieval import reranker, store

OLLAMA_OK = {"ok": True, "chat_model": "qwen3.5:9b", "model_present": True, "embed_model": "bge-m3",
             "embed_model_present": True, "model_count": 7}
INDEX_OK = {"ok": True, "collection": "smartup_passages", "lectures": 34, "passages": 3627, "vectors": 3627,
            "embed_model": "bge-m3"}


@pytest.fixture
def client():
    s = Settings(_env_file=None, warm_on_startup=False, keep_warm_minutes=0)
    reranker.get_reranker.cache_clear()
    yield TestClient(create_app(s))  # not a context manager: no lifespan, so no warm-up
    reranker.get_reranker.cache_clear()


def test_everything_up_is_ok(client, monkeypatch):
    monkeypatch.setattr(llm, "ollama_health", lambda: dict(OLLAMA_OK))
    monkeypatch.setattr(store, "index_health", lambda **_: dict(INDEX_OK))
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index"]["passages"] == 3627
    assert body["embed_model"] == "bge-m3" and body["reranker"]["loaded"] is False  # it loads on first use
    assert "elasticsearch" not in body and "path" not in body["index"]


@pytest.mark.parametrize(("ollama", "index"), [
    ({**OLLAMA_OK, "embed_model_present": False}, INDEX_OK),
    ({**OLLAMA_OK, "model_present": False}, INDEX_OK),
    ({"ok": False, "error": "ConnectError"}, INDEX_OK),
    (OLLAMA_OK, {"ok": False, "error": "index not built: run `smartup ingest`"}),
])
def test_anything_missing_is_degraded(client, monkeypatch, ollama, index):
    monkeypatch.setattr(llm, "ollama_health", lambda: dict(ollama))
    monkeypatch.setattr(store, "index_health", lambda **_: dict(index))
    assert client.get("/health").json()["status"] == "degraded"


def test_a_crashing_check_is_reported_not_raised(client, monkeypatch):
    def boom(**_):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(llm, "ollama_health", lambda: dict(OLLAMA_OK))
    monkeypatch.setattr(store, "index_health", boom)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["index"]["ok"] is False


def test_reranker_errors_are_not_exposed(client, monkeypatch):
    monkeypatch.setattr(llm, "ollama_health", lambda: dict(OLLAMA_OK))
    monkeypatch.setattr(store, "index_health", lambda **_: dict(INDEX_OK))
    reranker.get_reranker().error = "OSError: /Users/someone/.cache/huggingface/... not found"
    assert client.get("/health").json()["reranker"]["error"] == "unavailable"
