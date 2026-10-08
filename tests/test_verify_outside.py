"""nodes.verify for the outside sources (TEDx, the supplemental talks, Digital Innovation) and fallback answers."""

from __future__ import annotations

from smartup_rag.agent import prompts as P
from smartup_rag.agent.nodes import verify
from tests.conftest import DI_L3, TALK_NINE, TALK_START, TEDX, V1_L3, make_doc, make_source_doc

CIT_TEDX = '(TEDx — "Why fail if you can succeed? - TEDx")'
CIT_START = '(Talk — "How to Start a StartUp")'
CIT_NINE = '(Talk — "Nine Out of Ten Startups Fail: Why Fail When You Can Succeed?")'
CIT_DI3 = '(Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")'


def _state(answer, docs, language="en"):
    return {"language": language, "docs": docs, "answer": answer}


def test_tedx_citation_is_kept_and_reported_with_its_tier():
    out = verify(_state(f"Become profitable fast {CIT_TEDX}.", [make_source_doc(TEDX)]))
    assert out["status"] == "answered"
    assert out["answer"] == f"Become profitable fast {CIT_TEDX}."
    (c,) = out["citations"]
    assert c["tier"] == "tedx" and c["source"] == "TEDx" and c["display"] == ""
    assert c["lecture_key"] == "smartup-tedx:supplemental:L1"


def test_the_two_talks_are_told_apart_by_title():
    docs = [make_source_doc(TALK_START), make_source_doc(TALK_NINE, "Nine of ten fail.")]
    answer = f"Start with marketing {CIT_START}. Most fail {CIT_NINE}."
    out = verify(_state(answer, docs))
    assert [c["lecture_title"] for c in out["citations"]] == [TALK_START["lecture_title"], TALK_NINE["lecture_title"]]
    assert "\n- Talk — \"How to Start a StartUp\"" in out["answer"]  # two sources → Sources list with outside lines


def test_numbered_outside_course_and_normalised_title():
    out = verify(_state('Innovation is cheap to test (Digital Innovation IDC, Lecture 3 – "DI lecture three").',
                        [make_source_doc(DI_L3)]))
    assert CIT_DI3 in out["answer"] and "DI lecture three" not in out["answer"]
    assert out["citations"][0]["display"] == "Lecture 3"


def test_an_outside_citation_that_was_not_retrieved_is_removed():
    out = verify(_state(f"Become profitable fast {CIT_TEDX}. Also {CIT_START}.", [make_source_doc(TEDX)]))
    assert CIT_START not in out["answer"] and "How to Start" not in out["answer"]
    assert [c["tier"] for c in out["citations"]] == ["tedx"]


def test_ordinary_parentheses_survive_next_to_outside_citations():
    text = f"Talk to people first (Talk to customers — they know) {CIT_TEDX}."
    out = verify(_state(text, [make_source_doc(TEDX)]))
    assert "(Talk to customers — they know)" in out["answer"]


def test_a_fallback_answer_without_inline_citations_is_not_mistaken_for_a_decline():
    """The note that opens a fallback answer says "not covered directly in v2" — that is not the model declining."""
    for tier, doc in (("v1", make_doc(V1_L3)), ("tedx", make_source_doc(TEDX)), ("outsource", make_source_doc(DI_L3))):
        for lang in ("en", "he"):
            answer = P.FALLBACK_NOTES[tier][lang] + "\n\nThe answer itself, written without an inline citation."
            out = verify(_state(answer, [doc], language=lang))
            assert out["status"] == "answered", (tier, lang)
            assert out["citations"], (tier, lang)


def test_a_real_decline_is_still_not_found():
    out = verify(_state("I could not find this topic in the SmartUp lecture index.", [make_source_doc(TEDX)]))
    assert out["status"] == "not_found" and not out["citations"]
