"""smartup_rag.tiers — the source order v2 → TEDx → v1 → outsource, its filters and its citation strings."""

from __future__ import annotations

import pytest

from smartup_rag import tiers as T
from smartup_rag.retrieval import queries as Q
from smartup_rag.retrieval.hybrid import Candidate, lecture_key_for
from tests.conftest import DI_L3, TALK_NINE, TALK_START, TEDX


@pytest.mark.parametrize(("course_id", "version", "tier"), [
    ("smartup-foundations-v2", "v2", "v2"),
    ("smartup-foundations-v1", "v1", "v1"),
    ("smartup-tedx", "supplemental", "tedx"),
    ("smartup-how-to-start", "supplemental", "outsource"),
    ("digital-innovation-idc", "supplemental", "outsource"),
])
def test_tier_of(course_id, version, tier):
    assert T.tier_of(course_id, version) == tier


def test_search_order_is_the_one_the_instructions_define():
    assert T.TIER_ORDER == ("v2", "tedx", "v1", "outsource")
    assert T.FALLBACK_TIERS == ("tedx", "v1", "outsource")


def test_tier_conditions_select_each_source_without_a_tier_field():
    assert Q.conditions("v2") == [("course_version", "eq", "v2")]
    assert Q.conditions("tedx") == [("course_id", "eq", "smartup-tedx")]
    assert Q.conditions("outsource", 3) == [
        ("course_version", "eq", "supplemental"),
        ("course_id", "ne", "smartup-tedx"),
        ("lecture_number", "eq", 3),
    ]
    assert T.tier_conditions("v7") == [("course_version", "eq", "v7")]  # not a tier name: a plain course version


def test_citation_strings_per_source():
    assert T.citation_for("smartup-foundations-v2", "v2", 5, None, "Foundations Course V2 L5 1") == \
        '(v2, Lecture 5 — "Foundations Course V2 L5 1")'
    assert T.citation_for("smartup-foundations-v1", "v1", 11, "2", "Foundations 11 Part 2 Oded") == \
        '(v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")'
    assert T.citation_for(**_args(TEDX)) == '(TEDx — "Why fail if you can succeed? - TEDx")'
    assert T.citation_for(**_args(TALK_START)) == '(Talk — "How to Start a StartUp")'
    assert T.citation_for(**_args(DI_L3)) == '(Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")'


def _args(src):
    return {"course_id": src["course_id"], "course_version": src["course_version"], "lecture_number": src["lecture_number"],
            "part": None, "title": src["lecture_title"], "course_name": src["course_name"]}


def test_outside_lecture_keys_never_collide():
    keys = {lecture_key_for(s["course_id"], s["course_version"], s["lecture_number"], None) for s in (TEDX, TALK_START, TALK_NINE)}
    assert len(keys) == 3
    assert lecture_key_for("smartup-foundations-v2", "v2", 3, None) == "v2:L3"  # the course keys are unchanged
    c = Candidate(text="x", course_id=TEDX["course_id"], course_version="supplemental", lecture_number=1,
                  lecture_title=TEDX["lecture_title"], lang="en", score=0.1)
    assert c.lecture_key == "smartup-tedx:supplemental:L1" and c.tier == "tedx"


@pytest.mark.parametrize("text", [
    '(TEDx — "Why fail if you can succeed? - TEDx")',
    '(Talk – “How to Start a StartUp”)',
    '(Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")',
])
def test_outside_citation_pattern_matches_the_three_shapes(text):
    assert T.OUTSIDE_CITATION_RE.fullmatch(text)


@pytest.mark.parametrize("text", [
    "(Talk to customers — they know what they need)",
    "(TEDx talks are short)",
    "(for example — the first customer)",
])
def test_outside_citation_pattern_ignores_ordinary_parentheses(text):
    assert not T.OUTSIDE_CITATION_RE.search(text)
