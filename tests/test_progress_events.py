"""Progress messages of the fallback chain: node → LangChain custom event → SSE `status` frame. No LLM, no network."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig, RunnableLambda

from smartup_rag.agent import graph as G
from smartup_rag.agent import nodes as N
from smartup_rag.api.app import create_app
from smartup_rag.config import Settings


def test_progress_is_a_custom_event_in_the_users_language():
    def step(x, config: RunnableConfig):
        N._progress(config, "he", "tedx", first=True)
        N._progress(config, "en", "outsource", first=False)
        return x

    async def collect():
        return [ev async for ev in RunnableLambda(step).astream_events("q", version="v2") if ev["event"] == "on_custom_event"]

    events = asyncio.run(collect())
    assert [e["name"] for e in events] == ["progress", "progress"]
    assert events[0]["data"] == {"tier": "tedx", "text": "לא נמצא בגרסה v2 של הקורס — בודק בהרצאת ה-TEDx של יונתן…"}
    assert events[1]["data"]["text"] == "Not there either — checking other lectures outside the course…"


def test_progress_without_a_listener_is_silent():
    N._progress(None, "en", "v1", first=True)  # CLI / tests: no config, no error


def test_status_events_are_streamed_to_the_browser(monkeypatch):
    async def fake_stream_chat(message, thread_id="default", stop_event=None):
        yield {"type": "status", "text": "Not found in the v2 course — checking the v1 course…", "tier": "v1"}
        yield {"type": "token", "text": "Answer."}
        yield {"type": "final", "answer": "Answer.", "status": "answered", "citations": [], "fallback_tier": "v1"}

    monkeypatch.setattr(G, "stream_chat", fake_stream_chat)
    s = Settings(_env_file=None, warm_on_startup=False, keep_warm_minutes=0, debug_dashboard="off")
    body = TestClient(create_app(s)).post("/chat/stream", json={"message": "Why do startups fail?"}).text
    status_at, token_at = body.index("event: status"), body.index("event: token")
    assert status_at < token_at
    assert "checking the v1 course" in body and '"tier": "v1"' in body.replace('":"', '": "')
    assert '"fallback_tier": "v1"' in body.replace('":"', '": "')
