"""smartup_rag.lang — script-based language detection, citation parsing, lecture-number extraction."""

from __future__ import annotations

import pytest

from smartup_rag.lang import CITATION_RE, detect_language, extract_citations, lecture_number_in_text


# ----------------------------------------------------------------------------------------
# detect_language
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("What does Yonatan say about pricing?", "en"),
        ("מה יונתן אומר על תמחור?", "he"),
        ("מה זה PMF?", "he"),  # Hebrew question with an English acronym
        ("איך מחשבים CAC ו-LTV בסטארטאפ?", "he"),  # several acronyms, still Hebrew-dominant
        ("What does Yonatan say about תמחור strategy in the course?", "en"),  # one Hebrew word in English
        ("hi", "en"),
        ("", "en"),  # no letters at all → default English
        ("12345 ???", "en"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# ----------------------------------------------------------------------------------------
# extract_citations
# ----------------------------------------------------------------------------------------
def test_extract_single_citation() -> None:
    text = 'Yonatan frames pricing as value capture (v2, Lecture 5 — "Foundations Course V2 L5 1").'
    assert extract_citations(text) == [("v2", 5, None, "Foundations Course V2 L5 1")]


def test_extract_citation_with_part_and_curly_quotes() -> None:
    text = "Oded describes the exit (v1, Lecture 11 Part 2 — “Foundations 11 Part 2 Oded”) in detail."
    assert extract_citations(text) == [("v1", 11, "2", "Foundations 11 Part 2 Oded")]


def test_extract_multiple_citations_and_dash_variants() -> None:
    text = (
        'First (v2, Lecture 5 — "Foundations Course V2 L5 1"), then (V1, Lecture 3 - "Foundations Lecture 3") '
        "and finally (v1, Lecture 11 Part 1 – 'Foundations 11 Part 1')."
    )
    assert extract_citations(text) == [
        ("v2", 5, None, "Foundations Course V2 L5 1"),
        ("v1", 3, None, "Foundations Lecture 3"),
        ("v1", 11, "1", "Foundations 11 Part 1"),
    ]


def test_extract_ignores_non_citations() -> None:
    assert extract_citations("Lecture 5 talks about pricing (see the course).") == []
    assert extract_citations('(v3, Lecture 5 — "Nope")') == []  # only v1/v2 exist
    assert extract_citations("") == []


def test_citation_regex_tolerates_spacing() -> None:
    assert CITATION_RE.search('( v2 ,  Lecture 5  —  "Foundations Course V2 L5 1" )') is not None


# ----------------------------------------------------------------------------------------
# lecture_number_in_text
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Summarize lecture 3 please", 3),
        ("What is the homework of Lecture 7?", 7),
        ("מה שיעורי הבית של הרצאה 3?", 3),
        ("תסכם לי את שיעור 12", 12),
        ("lesson 4 recap", 4),
        ("class #9 — what was covered?", 9),
        ("הרצאה מספר 5", 5),
        ("What does Yonatan say about pricing?", None),
        ("The 3 lectures on marketing", None),  # number before the word is not a lecture reference
        ("", None),
    ],
)
def test_lecture_number_in_text(text: str, expected: int | None) -> None:
    assert lecture_number_in_text(text) == expected


def test_a_comma_before_part_keeps_the_part():
    m = CITATION_RE.search('(v1, Lecture 11, Part 2 — "Foundations 11")')
    assert m["part"] == "2" and m["title"] == "Foundations 11"
