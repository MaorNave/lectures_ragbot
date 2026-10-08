"""Exact repeats: a question asked again in one conversation gets the answer it already got — through the real graph,
with classification, search, grading and generation replaced by counting fakes. No LLM, no network."""

from __future__ import annotations

from collections import Counter

import pytest
from langchain_core.messages import HumanMessage

from smartup_rag.agent import graph as G
from smartup_rag.agent import nodes as N
from smartup_rag.agent import prompts as P
from tests.conftest import V2_L5, make_doc

CIT = '(v2, Lecture 5 — "Foundations Course V2 L5 1")'


@pytest.fixture
def agent(monkeypatch):
    calls = Counter()
    trouble = {"search_fails": False, "reader_leaves": False}  # switched on by a test for the next question only

    def fake_classify(state, config):
        calls["classify"] += 1
        q = state["question"].lower()
        in_scope = "poem" not in q
        intent = "chit_chat" if q.startswith(("hi", "thanks")) else "conceptual" if in_scope else "out_of_scope"
        return {"classification": {"in_scope": in_scope, "intent": intent,
                                   "lecture_number": None, "version_scope": "default"}, "query": state["question"]}

    def fake_retrieve(query, lang, tier, n, part, notes):
        calls[f"search {tier}"] += 1
        if trouble["search_fails"]:
            notes.append("retrieval_error:ConnectionTimeout")  # what _safe_retrieve records when a search fails
            return []
        return [make_doc(V2_L5, rerank_score=0.9)]

    def fake_generate(state, config):
        calls["generate"] += 1
        return {"answer": f"Pricing is value capture {CIT}.", "stopped_early": trouble["reader_leaves"]}

    monkeypatch.setattr(N, "classify", fake_classify)
    monkeypatch.setattr(N, "_safe_retrieve", fake_retrieve)
    monkeypatch.setattr(N, "generate", fake_generate)
    graph = G.build_graph()

    def ask(question, thread="t1", **problems):
        trouble.update({"search_fails": False, "reader_leaves": False, **problems})
        out = graph.invoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": thread}})
        return G._result(out)

    return ask, calls


@pytest.mark.parametrize(("a", "b"), [
    ("What is PMF?", "what is pmf"),
    ("How should I price my product?", "  How should I price my   product ?? "),
    ("מה זה PMF?", "מה זה pmf"),
])
def test_case_punctuation_and_spacing_do_not_make_a_question_new(a, b):
    assert N.question_key(a) == N.question_key(b)
    assert N.question_key("What is PMF?") != N.question_key("What is CAC?")


def test_the_same_question_gets_the_same_answer_without_a_new_run(agent):
    ask, calls = agent
    first = ask("How should I price my product?")
    before = dict(calls)
    second = ask("how should I price my product")
    assert dict(calls) == before  # no classifier, no search, no LLM
    assert second["answer"] == P.REPEAT_NOTE["en"] + "\n\n" + first["answer"]
    assert second["citations"] == first["citations"] and second["citations"][0]["citation"] == CIT
    assert second["status"] == first["status"] == "answered"


def test_a_different_question_is_answered_normally(agent):
    ask, calls = agent
    ask("How should I price my product?")
    ask("How do I find my first customers?")
    assert calls["classify"] == 2 and calls["generate"] == 2


def test_a_repeated_refusal_needs_no_classifier_call(agent):
    ask, calls = agent
    first = ask("Write me a poem")
    second = ask("Write me a poem!")
    assert first["status"] == second["status"] == "refused" and calls["classify"] == 1


def test_another_conversation_does_not_reuse_answers(agent):
    ask, calls = agent
    ask("How should I price my product?", thread="a")
    ask("How should I price my product?", thread="b")
    assert calls["classify"] == 2


def test_the_answer_log_keeps_the_last_twenty_answers():
    state = {"question": "q", "classification": {}, "language": "en",
             "answer_log": [{"key": str(i), "answer": "a", "status": "answered"} for i in range(20)]}
    log = N._logged(state, "new answer", "answered")
    assert len(log) == 20 and log[-1]["answer"] == "new answer" and log[0]["key"] == "1"
    assert N._logged(state, "boom", "error") == state["answer_log"]  # errors are never replayed


@pytest.mark.parametrize("problem", ["reader_leaves", "search_fails"])
def test_asking_again_retries_an_answer_that_went_wrong(agent, problem):
    """A cut-off answer (the reader closed the page mid-answer) or one from a failed search is never replayed: asking
    again is how the reader retries, so the second time runs the whole pipeline."""
    ask, calls = agent
    first = ask("How should I price my product?", **{problem: True})
    second = ask("How should I price my product?")
    assert calls["classify"] == 2 and calls["generate"] == 1 + (problem == "reader_leaves")
    assert P.REPEAT_NOTE["en"] not in second["answer"] and second["status"] == "answered"
    assert first["status"] == ("answered" if problem == "reader_leaves" else "not_found")


def test_a_greeting_said_twice_is_not_called_the_same_question(agent):
    ask, calls = agent
    ask("hi")
    second = ask("hi")
    assert second["status"] == "chit_chat" and P.REPEAT_NOTE["en"] not in second["answer"] and calls["classify"] == 2


def test_a_short_follow_up_after_a_different_question_is_not_a_repeat(agent):
    ask, calls = agent
    ask("What is PMF?")
    ask("tell me more")
    ask("How should I price my product?")
    ask("tell me more")  # about pricing now, not PMF
    assert calls["generate"] == 4
    ask("How should I price my product?")  # a repeat
    ask("tell me more")  # same words after the same question: the logged answer fits
    assert calls["generate"] == 4
