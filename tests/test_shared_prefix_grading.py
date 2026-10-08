"""Shared-prefix grading, end to end through the nodes.

The answer call reuses the grader's cached reading of the passages only if the grader's prompt opens with exactly
the tokens the answer prompt opens with: system prompt, replayed history, evidence block. These tests run the real
grade → (fallback) → generate nodes with a recording fake model and compare what each call was sent.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from smartup_rag import trace
from smartup_rag.agent import nodes as N
from smartup_rag.agent import prompts as P
from smartup_rag.agent.state import Grade
from smartup_rag.config import get_settings


class _FakeModel:
    """Stands in for every ChatOllama role: returns scripted verdicts and records every prompt it is sent."""

    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)
        self.calls: list[list] = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return {"parsed": Grade(verdict=self.verdicts.pop(0)), "raw": AIMessage(content="{}"), "parsing_error": None}

    def stream(self, messages, config=None):
        self.calls.append(messages)
        yield AIMessageChunk(content="Price high [P1] (v2, Lecture 5).")


def _docs(version: str, score: float) -> list[Document]:
    return [
        Document(page_content=f"{version} passage {i} about pricing.",
                 metadata={"citation": f'({version}, Lecture 5 — "L5")', "rerank_score": score, "lang": "en",
                           "course_version": version, "lecture_number": 5})
        for i in (1, 2)
    ]


def _state(history: list) -> dict:
    docs = _docs("v2", 0.2)  # inside the grader band: the LLM grader runs
    q = "How should I price my product?"
    return {"messages": [*history, HumanMessage(content=q)], "question": q, "query": q, "language": "en",
            "classification": {"in_scope": True, "intent": "practical", "version_scope": "default"},
            "docs_v2": docs, "docs_v1": [], "docs": docs, "notes": [], "grade_v2": {}, "grade_v1": {},
            "used_fallback": False}


HISTORY = [HumanMessage(content="What is JTBD?"), AIMessage(content="Jobs to be Done is …")]


@pytest.fixture
def fake(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setattr(N, "shares_chat_prefix", lambda: True)

    def install(verdicts):
        model = _FakeModel(verdicts)
        monkeypatch.setattr(N, "get_llm", lambda role="chat": model)
        return model

    yield install
    get_settings.cache_clear()


def _assert_same_opening(grader_msgs, answer_msgs, docs):
    assert [type(m) for m in grader_msgs[:-1]] == [type(m) for m in answer_msgs[:-1]]
    assert [m.content for m in grader_msgs[:-1]] == [m.content for m in answer_msgs[:-1]], "system + history differ"
    block = P.evidence_block(docs)
    assert grader_msgs[-1].content.startswith(block), "grader does not open with the answer's evidence block"
    assert answer_msgs[-1].content.startswith(block)


@pytest.mark.parametrize("verdict", ["sufficient", "partial"])
def test_v2_grader_and_answer_open_identically(fake, verdict):
    model = fake([verdict])
    state = _state(HISTORY)
    state.update(N._grade_v2(state, None))
    assert state["grade_v2"]["method"] == "llm"
    assert state["grade_v2"]["prompt_layout"].startswith("evidence block")
    if verdict == "partial":
        assert any("partially" in n for n in state["notes"]), "the partial note goes after the shared opening"
    tr = trace.new_trace(state["question"], "t", "test")
    with trace.active(tr):
        N.generate(state, None)
    grader_msgs, answer_msgs = model.calls
    _assert_same_opening(grader_msgs, answer_msgs, state["docs"])
    assert len(answer_msgs) == 4, "system + two history messages + user turn"
    assert tr.steps[-1]["after_grader_reading"] is True


def test_v1_fallback_grader_and_answer_open_identically(fake, monkeypatch):
    model = fake(["insufficient", "sufficient"])
    v1 = _docs("v1", 0.2)
    monkeypatch.setattr(N, "_safe_retrieve", lambda *args, **kwargs: list(v1))
    state = _state(HISTORY)
    state.update(N._grade_v2(state, None))
    assert N.route_after_grade(state) == "fallback"
    state.update(N._fallback(state, None))
    assert state["docs"] == v1 and state["used_fallback"]
    tr = trace.new_trace(state["question"], "t", "test")
    with trace.active(tr):
        N.generate(state, None)
    _v2_grader, v1_grader, answer_msgs = model.calls
    _assert_same_opening(v1_grader, answer_msgs, v1)
    assert tr.steps[-1]["after_grader_reading"] is True


def test_fast_path_answer_is_not_marked_as_reusing_a_grader_reading(fake):
    model = fake([])
    state = _state([])
    for d in state["docs_v2"]:
        d.metadata["rerank_score"] = 0.9  # fast "sufficient": no grader call
    state.update(N._grade_v2(state, None))
    assert state["grade_v2"]["method"] == "fast"
    tr = trace.new_trace(state["question"], "t", "test")
    with trace.active(tr):
        N.generate(state, None)
    assert len(model.calls) == 1
    assert tr.steps[-1]["after_grader_reading"] is False
