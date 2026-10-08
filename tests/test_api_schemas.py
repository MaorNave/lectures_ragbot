"""The agent→API dict contract: graph._result must round-trip into ChatResponse without losing fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smartup_rag.agent import graph as G
from smartup_rag.agent import nodes as N
from smartup_rag.api.schemas import ChatRequest, ChatResponse
from tests.test_agent_routing import doc


def test_result_round_trips_into_chat_response():
    state = {"language": "he", "docs": [doc("v2", 5, "A"), doc("v1", 13, "B")],
             "answer": 'תשובה (v2, Lecture 5 — "A") ועוד (v1, Lecture 13 — "B").',
             "classification": {"intent": "conceptual", "lecture_number": None, "version_scope": "default"},
             "used_fallback": False, "grade_v2": {"verdict": "sufficient"}}
    state.update(N.verify(state))
    result = G._result(state)
    resp = ChatResponse.from_agent(result, thread_id="abcd-1234", elapsed_ms=1234)
    assert resp.status == "answered" and resp.language == "he" and resp.intent == "conceptual"
    assert [c.citation for c in resp.citations] == [d["citation"] for d in result["citations"]]
    assert resp.citations[1].lecture_part is None and resp.citations[0].display == "Lecture 5"
    assert resp.answer == result["answer"] and resp.elapsed_ms == 1234


def test_unknown_status_maps_to_error():
    resp = ChatResponse.from_agent({"answer": "x", "status": "weird"}, thread_id="abcd-1234", elapsed_ms=0)
    assert resp.status == "error"


@pytest.mark.parametrize("tid", ["a" * 4, "cli-deadbeef", "eval-he_01", "6f1e2c3a-1111-2222-3333-444455556666"])
def test_thread_id_accepts_server_and_cli_shapes(tid):
    assert ChatRequest(message="hi", thread_id=tid).thread_id == tid


@pytest.mark.parametrize("tid", ["../../etc/passwd", "<script>", "a b", "x" * 200, "abc"])
def test_thread_id_rejects_unsafe_shapes(tid):
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", thread_id=tid)


def test_blank_thread_id_means_none_and_message_is_stripped():
    req = ChatRequest(message="  hello  ", thread_id="   ")
    assert req.thread_id is None and req.message == "hello"
