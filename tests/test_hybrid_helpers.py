"""smartup_rag.retrieval.hybrid — pure helpers and the Candidate → Document mapping (no network)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from smartup_rag.retrieval.hybrid import (
    Candidate,
    dedupe,
    display_lecture,
    drop_translations,
    is_translation,
    latin_tokens,
    lecture_part_from_title,
    make_lecture_key,
)


# ----------------------------------------------------------------------------------------
# lecture_part_from_title / make_lecture_key / display_lecture
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "title, expected",
    [
        ("Foundations 11 Part 1", "1"),
        ("Foundations 11 Part 2 Oded", "2"),
        ("foundations 11 PART 2", "2"),
        ("Lecture 11 part2", "2"),
        ("Foundations Course V2 L5 1", None),
        ("Foundations Lecture 3", None),
        ("Partnerships 101", None),  # 'Part' inside another word is not a part marker
        ("", None),
        (None, None),
    ],
)
def test_lecture_part_from_title(title: str | None, expected: str | None) -> None:
    assert lecture_part_from_title(title) == expected  # type: ignore[arg-type]


def test_make_lecture_key() -> None:
    assert make_lecture_key("v2", 5, None) == "v2:L5"
    assert make_lecture_key("v1", 11, "2") == "v1:L11p2"
    assert make_lecture_key("v1", 11, "") == "v1:L11"  # empty part behaves like None


def test_display_lecture() -> None:
    assert display_lecture(5, None) == "Lecture 5"
    assert display_lecture(11, "2") == "Lecture 11 Part 2"


# ----------------------------------------------------------------------------------------
# Candidate
# ----------------------------------------------------------------------------------------
def _cand(text: str = "Pricing is value capture.", **kw) -> Candidate:
    base = {
        "text": text,
        "course_id": "smartup-foundations",
        "course_version": "v2",
        "lecture_number": 5,
        "lecture_title": "Foundations Course V2 L5 1",
        "lang": "en",
        "score": 0.03,
        "lecture_subjects": ["Pricing", "Business Model"],
        "source_index": "local:smartup_passages",
    }
    base.update(kw)
    return Candidate(**base)


def test_candidate_key_and_citation() -> None:
    c = _cand()
    assert c.lecture_key == "v2:L5"
    assert c.citation == '(v2, Lecture 5 — "Foundations Course V2 L5 1")'
    p = _cand(
        course_version="v1", lecture_number=11, lecture_part="2", lecture_title="Foundations 11 Part 2 Oded"
    )
    assert p.lecture_key == "v1:L11p2"
    assert p.citation == '(v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")'


def test_candidate_to_document_metadata_keys() -> None:
    c = _cand(rerank_score=0.71, section_topic="Pricing models")
    doc = c.to_document()
    assert isinstance(doc, Document)
    assert doc.page_content == "Pricing is value capture."
    assert set(doc.metadata) == {
        "lecture_key",
        "course_id",
        "course_version",
        "tier",
        "source_label",
        "lecture_number",
        "lecture_part",
        "lecture_title",
        "lecture_subjects",
        "lang",
        "score",
        "rerank_score",
        "section_topic",
        "source_index",
        "citation",
    }
    m = doc.metadata
    assert m["lecture_key"] == "v2:L5" and m["lecture_part"] is None and m["lang"] == "en"
    assert m["rerank_score"] == 0.71 and m["score"] == 0.03 and m["section_topic"] == "Pricing models"
    assert m["citation"] == c.citation
    assert m["lecture_subjects"] == ["Pricing", "Business Model"]
    assert m["lecture_subjects"] is not c.lecture_subjects  # copied, not shared


# ----------------------------------------------------------------------------------------
# dedupe
# ----------------------------------------------------------------------------------------
def test_dedupe_drops_same_lecture_same_text_prefix() -> None:
    a = _cand("Pricing   is value capture.\nReally.")
    b = _cand("pricing is VALUE capture. really.")  # whitespace/case differences within the first 120 chars
    kept = dedupe([a, b])
    assert kept == [a]


def test_dedupe_keeps_same_text_from_different_lectures() -> None:
    a = _cand()
    b = _cand(lecture_number=6, lecture_title="Foundations Course V2 L6")
    assert dedupe([a, b]) == [a, b]


def test_dedupe_keeps_different_passages_of_one_lecture_and_preserves_order() -> None:
    a = _cand("First passage.")
    b = _cand("Second passage.")
    c = _cand("First passage.")
    assert dedupe([a, b, c]) == [a, b]


def test_dedupe_distinguishes_lecture_parts() -> None:
    p1 = _cand(
        course_version="v1", lecture_number=11, lecture_part="1", lecture_title="Foundations 11 Part 1"
    )
    p2 = _cand(
        course_version="v1", lecture_number=11, lecture_part="2", lecture_title="Foundations 11 Part 2 Oded"
    )
    assert len(dedupe([p1, p2])) == 2


def test_dedupe_empty() -> None:
    assert dedupe([]) == []


# ----------------------------------------------------------------------------------------
# cross-language duplicates (the same stretch of a lecture in English and in Hebrew)
# ----------------------------------------------------------------------------------------
EN_STAGE4 = "Stage 4 — Pricing Page. Here too you can test, and you can put tons of options: Base44, Stripe, $12 per visitor."
HE_STAGE4 = "שלב 4 — Pricing Page. גם פה אפשר לבדוק, ואפשר לשים המון אופציות: Base44, Stripe, $12 לכל מבקר."
HE_OTHER = "בהמשך ההרצאה דיברנו על גיוס הון, על משקיעים ועל הדרך לבנות צוות מייסדים."  # no Latin tokens
LENGTHS = {("v2:L7", "en"): 30_000, ("v2:L7", "he"): 20_000}


def _passage(text: str, lang: str, start: int, end: int, section: int = 0) -> Candidate:
    return _cand(text, lang=lang, lecture_number=7, lecture_title="Foundations Course V2 L7 1",
                 section_index=section, char_start=start, char_end=end)


def test_latin_tokens_are_names_numbers_and_english_terms() -> None:
    assert latin_tokens(HE_STAGE4) == {"4", "pricing", "page", "base", "44", "stripe", "12"}


def test_same_stretch_in_both_languages_is_a_translation() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)   # 50 % into the English transcript
    he = _passage(HE_STAGE4, "he", 10_050, 10_450)   # ~50 % into the Hebrew transcript
    assert is_translation(he, en, LENGTHS) and is_translation(en, he, LENGTHS)


def test_same_words_elsewhere_in_the_lecture_is_not_a_translation() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)
    he = _passage(HE_STAGE4, "he", 2_000, 2_400)     # 10 % in: a different part of the lecture
    assert not is_translation(he, en, LENGTHS)


def test_too_few_latin_tokens_never_counts_as_a_translation() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)
    he = _passage(HE_OTHER, "he", 10_050, 10_450)
    assert not is_translation(he, en, LENGTHS)


def test_summary_and_transcript_passage_are_never_the_same_passage() -> None:
    en_summary = _passage(EN_STAGE4, "en", 0, 600, section=-1)
    he = _passage(HE_STAGE4, "he", 10_050, 10_450)
    assert not is_translation(he, en_summary, LENGTHS)
    assert is_translation(_passage(HE_STAGE4, "he", 0, 400, section=-1), en_summary, LENGTHS)


def test_drop_translations_keeps_the_question_language_and_the_order() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)
    he = _passage(HE_STAGE4, "he", 10_050, 10_450)
    other = _passage(HE_OTHER, "he", 1_000, 1_400)
    assert drop_translations([he, en, other], "en", LENGTHS) == [en, other]   # English question: Hebrew twin goes
    assert drop_translations([he, en, other], "he", LENGTHS) == [he, other]   # Hebrew question: English twin goes


def test_drop_translations_leaves_other_lectures_and_unknown_lengths_alone() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)
    he_other_lecture = _cand(HE_STAGE4, lang="he", lecture_number=8, lecture_title="L8", section_index=0,
                             char_start=10_050, char_end=10_450)
    assert drop_translations([en, he_other_lecture], "en", LENGTHS) == [en, he_other_lecture]
    assert drop_translations([en, _passage(HE_STAGE4, "he", 10_050, 10_450)], "en", {}) == [
        en, _passage(HE_STAGE4, "he", 10_050, 10_450)
    ]  # no transcript lengths → no position check → keep both


def test_dropped_twin_passes_its_higher_score_to_the_kept_passage() -> None:
    en = _passage(EN_STAGE4, "en", 15_000, 15_600)
    en.rerank_score = 0.38
    he = _passage(HE_STAGE4, "he", 10_050, 10_450)
    he.rerank_score = 0.45   # the Hebrew twin matched the English question best
    other = _passage("Unrelated English passage about fundraising.", "en", 1_000, 1_400)
    other.rerank_score = 0.40
    kept = drop_translations([he, other, en], "en", LENGTHS)
    assert kept == [en, other] and en.rerank_score == 0.45   # same best score → same grading path
