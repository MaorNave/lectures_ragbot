"""The classifier/grader prompt must adapt to whether it shares the chat model's cached prefix.

Sending the ~6 000-token system prompt is a win ONLY when the routing call runs on the chat model,
because Ollama then reuses that model's cached prompt prefix (~27 s cold → ~2.4 s warm). With a
separate CLASSIFIER_MODEL there is no shared cache, so the short task prompt must be sent instead.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartup_rag.agent import llm as L
from smartup_rag.agent import prompts as P
from smartup_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clear_caches():
    """Settings and the ChatOllama factory are lru_cached; reset around every test."""
    get_settings.cache_clear()
    L.get_llm.cache_clear()
    yield
    get_settings.cache_clear()
    L.get_llm.cache_clear()


def _docs() -> list[Document]:
    return [
        Document(
            page_content="Profitability comes before growth.",
            metadata={"citation": '(v2, Lecture 1 — "Foundations Course V2 L1")'},
        )
    ]


# --------------------------------------------------------------------------------------
# shares_chat_prefix
# --------------------------------------------------------------------------------------
def test_shares_prefix_when_no_separate_classifier(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_MODEL", raising=False)
    assert L.shares_chat_prefix() is True


def test_shares_prefix_when_classifier_equals_chat_model(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("CLASSIFIER_MODEL", "qwen3.5:9b")
    assert L.shares_chat_prefix() is True


def test_does_not_share_prefix_with_a_different_classifier_model(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("CLASSIFIER_MODEL", "qwen3.5:4b")
    assert L.shares_chat_prefix() is False


# --------------------------------------------------------------------------------------
# classifier prompt
# --------------------------------------------------------------------------------------
def test_classifier_shared_prefix_sends_the_full_system_prompt():
    msgs = P.classifier_messages("Summarize lecture 3", [], shared_prefix=True)
    assert isinstance(msgs[0], SystemMessage) and isinstance(msgs[1], HumanMessage)
    assert msgs[0].content == P.system_prompt()  # byte-identical: that is what makes the cache hit
    assert P.CLASSIFIER_SYSTEM in msgs[1].content, "routing instructions must move into the user turn"
    assert "Summarize lecture 3" in msgs[1].content


def test_classifier_without_shared_prefix_sends_only_the_short_prompt():
    msgs = P.classifier_messages("Summarize lecture 3", [], shared_prefix=False)
    assert msgs[0].content == P.CLASSIFIER_SYSTEM
    assert P.load_instructions() not in msgs[0].content
    assert len(msgs[0].content) < len(P.system_prompt()) / 4
    assert "Summarize lecture 3" in msgs[1].content


# --------------------------------------------------------------------------------------
# grader prompt
# --------------------------------------------------------------------------------------
def test_grader_shared_prefix_sends_the_full_system_prompt():
    msgs = P.grader_messages("does the course cover pricing?", _docs(), shared_prefix=True)
    assert msgs[0].content == P.system_prompt()
    assert P.GRADER_SYSTEM in msgs[1].content
    assert "[P1]" in msgs[1].content and "Foundations Course V2 L1" in msgs[1].content


def _history() -> list:
    return [HumanMessage(content="What is JTBD?"), AIMessage(content="Jobs to be Done is …"),
            HumanMessage(content="And pricing?"), AIMessage(content="Price high.")]


def test_grader_reads_exactly_the_opening_of_the_answer_prompt():
    """The answer call can resume from the grader's cached reading only if every token before the grading task —
    system prompt, replayed history, evidence block — is identical in both prompts."""
    docs, history, q = _docs() * 3, _history(), "How should I price my product?"
    grader = P.grader_messages(q, docs, history, 6, shared_prefix=True)
    answer = P.answer_messages(q, "en", docs, history, ["Say explicitly: partial."], 6)
    assert [type(m) for m in grader[:-1]] == [type(m) for m in answer[:-1]]
    assert [m.content for m in grader[:-1]] == [m.content for m in answer[:-1]]
    block = P.evidence_block(docs)
    assert answer[-1].content.startswith(block) and grader[-1].content.startswith(block)
    assert P.GRADER_SYSTEM in grader[-1].content[len(block):] and q in grader[-1].content[len(block):]
    assert grader[-1].content[len(block):].endswith("Return only the JSON object.")


def test_grader_sees_whole_passages_in_the_shared_layout():
    long_doc = [Document(page_content="x" * 1500 + " the key sentence.", metadata={"citation": "(v2, Lecture 10)"})]
    shared = P.grader_messages("q?", long_doc, shared_prefix=True)[-1].content
    compact = P.grader_messages("q?", long_doc, shared_prefix=False)[-1].content
    assert "the key sentence." in shared
    assert "the key sentence." not in compact  # the compact prompt cuts passages to 700 characters


def test_grading_task_fits_inside_the_checkpoint_window():
    """llama.cpp saves a resumable checkpoint 512 tokens before a prompt's end, so everything after the evidence
    block must stay under 512 tokens: the task text (241 tokens, 269 on a follow-up) plus the question twice — with
    the model's tokenizer a 250-character Hebrew question measured 491 tokens, a 220-character follow-up 489."""
    block = P.evidence_block(_docs())
    first = P.grader_messages("", _docs(), shared_prefix=True)[-1].content[len(block):]
    follow_up = P.grader_messages("", _docs(), _history(), 6, shared_prefix=True)[-1].content[len(block):]
    assert len(first) < 1300, "grading task grew: the answer call would stop reusing the grader's reading"
    assert len(follow_up) - len(first) == len(P.GRADER_HISTORY_RULE) + 1
    assert P.GRADER_SHARED_MAX_QUESTION_CHARS <= 250 and P.GRADER_SHARED_MAX_FOLLOW_UP_CHARS <= 220
    q = "How should I price my product?"
    assert P.grader_messages(q, _docs(), shared_prefix=True)[-1].content.count(q) == 2


def test_only_a_follow_up_is_told_that_earlier_answers_are_not_evidence():
    """A first question's grading prompt is exactly what it was before the rule existed (its verdicts were measured
    with it); only a turn with a replayed conversation carries the rule."""
    q = "What does the Dunkin' Donuts example teach about value propositions?"
    first = P.grader_messages(q, _docs(), [], 6, shared_prefix=True)[-1].content
    follow_up = P.grader_messages(q, _docs(), _history(), 6, shared_prefix=True)[-1].content
    assert P.GRADER_HISTORY_RULE not in first and P.GRADER_HISTORY_RULE in follow_up
    assert follow_up.replace(P.GRADER_HISTORY_RULE + "\n", "") == first
    assert first.endswith(f"{P.GRADER_SYSTEM}\n\nJudge the passages above against the user's question: {q}\n"
                          "Return only the JSON object.")


def test_a_follow_up_question_gets_the_shorter_cap():
    q = "How should I price my product? " * 7 + "Thanks!"  # 224 characters
    assert 220 < len(q) <= 250
    assert P.grader_reads_evidence_block(q) and not P.grader_reads_evidence_block(q, follow_up=True)
    assert P.grader_messages(q, _docs(), [], 6, shared_prefix=True)[-1].content.startswith(P.evidence_block(_docs()))
    assert not P.grader_messages(q, _docs(), _history(), 6, shared_prefix=True)[-1].content.startswith("Retrieved")


def test_a_very_long_question_gets_the_compact_grader_prompt():
    q = "How should I price? " * 30  # 600 characters
    msgs = P.grader_messages(q, _docs(), _history(), 6, shared_prefix=True)
    assert len(msgs) == 2 and msgs[0].content == P.system_prompt()
    assert not msgs[1].content.startswith("Retrieved lecture passages")
    assert not P.grader_reads_evidence_block(q)


def test_grader_without_shared_prefix_sends_only_the_short_prompt():
    msgs = P.grader_messages("does the course cover pricing?", _docs(), _history(), 6, shared_prefix=False)
    assert msgs[0].content == P.GRADER_SYSTEM
    assert len(msgs) == 2, "a separate grader model shares no cache: no history replay"
    assert "[P1]" in msgs[1].content


def test_both_prompt_shapes_carry_the_same_task_content():
    """Switching models must change only the framing, never what the model is asked to decide."""
    shared = P.classifier_messages("What about pricing?", [], shared_prefix=True)
    short = P.classifier_messages("What about pricing?", [], shared_prefix=False)
    for fragment in ("Conversation so far:", "Latest user message:", "What about pricing?", "Return only the JSON object."):
        assert fragment in shared[1].content
        assert fragment in short[1].content
