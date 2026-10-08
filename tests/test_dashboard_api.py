"""/dashboard and /debug/traces: who may read traces (they hold every visitor's questions). No LLM, no network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smartup_rag import trace as T
from smartup_rag.api.app import create_app
from smartup_rag.config import Settings


def _client(mode: str) -> TestClient:
    s = Settings(_env_file=None, debug_dashboard=mode, warm_on_startup=False, keep_warm_minutes=0)
    return TestClient(create_app(s))  # not used as a context manager: no lifespan, so no model warm-up


@pytest.fixture(autouse=True)
def _one_trace():
    T.STORE.clear()
    tr = T.new_trace("What does Yonatan say about pricing?", "thread-1", "/chat/stream")
    with T.active(tr), T.span("classify") as sp:
        sp["decision"] = {"in_scope": True, "intent": "conceptual"}
    T.finish(tr, {"status": "answered", "answer": "…"})
    yield tr
    T.STORE.clear()


def test_on_mode_serves_the_page_the_list_and_one_trace(_one_trace) -> None:
    c = _client("on")
    assert c.get("/dashboard").status_code == 200 and "Pipeline dashboard" in c.get("/dashboard").text
    listing = c.get("/debug/traces").json()
    assert listing["mode"] == "on" and [t["id"] for t in listing["traces"]] == [_one_trace.id]
    full = c.get(f"/debug/traces/{_one_trace.id}").json()
    assert full["steps"][0]["step"] == "classify" and full["steps"][0]["decision"]["intent"] == "conceptual"
    assert c.get("/debug/traces/nope").status_code == 404


@pytest.mark.parametrize("mode", ["local", "off"])
def test_other_modes_hide_everything_from_a_non_local_client(mode, _one_trace) -> None:
    c = _client(mode)  # the test client's address is "testclient", i.e. not this machine
    for path in ("/dashboard", "/debug/traces", f"/debug/traces/{_one_trace.id}"):
        assert c.get(path).status_code == 404, path
