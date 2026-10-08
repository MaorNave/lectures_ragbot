"""smartup_rag.retrieval.queries — filters, FTS5 expressions and RRF as pure functions (no index, no network)."""

from __future__ import annotations

import sqlite3

import pytest

from smartup_rag.retrieval import queries as Q

QUESTION = "What does Yonatan say about pricing?"


# ----------------------------------------------------------------------------------------
# conditions
# ----------------------------------------------------------------------------------------
def test_conditions_empty_when_nothing_given() -> None:
    assert Q.conditions() == []


def test_conditions_cover_every_filter_in_order() -> None:
    assert Q.conditions("v2", 5, "smartup-foundations-v2", "he", exclude_course_ids=["b", "a", "a"]) == [
        ("course_version", "eq", "v2"),
        ("lecture_number", "eq", 5),
        ("course_id", "eq", "smartup-foundations-v2"),
        ("lang", "eq", "he"),
        ("course_id", "nin", ["a", "b"]),
    ]


def test_lecture_zero_is_not_dropped_and_number_is_cast() -> None:
    assert Q.conditions(lecture_number=0) == [("lecture_number", "eq", 0)]
    assert Q.conditions(lecture_number="7") == [("lecture_number", "eq", 7)]  # type: ignore[arg-type]


def test_other_lang() -> None:
    assert Q.other_lang("en") == "he"
    assert Q.other_lang("he") == "en"


# ----------------------------------------------------------------------------------------
# Chroma where / SQL where
# ----------------------------------------------------------------------------------------
def test_chroma_where_shapes() -> None:
    assert Q.chroma_where([]) is None
    assert Q.chroma_where(Q.conditions("v2")) == {"course_version": {"$eq": "v2"}}
    assert Q.chroma_where(Q.conditions("outsource", 3, exclude_course_ids=["digital-innovation-idc"])) == {"$and": [
        {"course_version": {"$eq": "supplemental"}},
        {"course_id": {"$ne": "smartup-tedx"}},
        {"lecture_number": {"$eq": 3}},
        {"course_id": {"$nin": ["digital-innovation-idc"]}},
    ]}


def test_sql_where_is_parameterised() -> None:
    sql, params = Q.sql_where(Q.conditions("outsource", 3, lang="en", exclude_course_ids=["x", "y"]))
    assert sql == ("p.course_version = ? AND p.course_id != ? AND p.lecture_number = ? AND p.lang = ? "
                   "AND p.course_id NOT IN (?, ?)")
    assert params == ["supplemental", "smartup-tedx", 3, "en", "x", "y"]
    assert Q.sql_where([]) == ("", [])
    assert Q.sql_where(Q.conditions("v1"), alias="l") == ("l.course_version = ?", ["v1"])


def test_sql_where_handles_empty_lists() -> None:
    assert Q.sql_where([("course_id", "in", [])]) == ("0", [])  # x IN () → nothing matches
    assert Q.sql_where([("course_id", "nin", [])]) == ("", [])  # x NOT IN () → no restriction


@pytest.mark.parametrize("cond", [("text", "eq", "x"), ("course_id; DROP TABLE passages", "eq", "x"), ("lang", "like", "e%")])
def test_unknown_fields_and_operators_are_rejected(cond) -> None:
    with pytest.raises(ValueError):
        Q.sql_where([cond])
    with pytest.raises(ValueError):
        Q.chroma_where([cond])


# ----------------------------------------------------------------------------------------
# FTS5 terms and MATCH expressions
# ----------------------------------------------------------------------------------------
def test_query_terms_are_lowercase_words_without_repeats() -> None:
    assert Q.query_terms(QUESTION) == ["what", "does", "yonatan", "say", "about", "pricing"]
    assert Q.query_terms("Pricing, PRICING; pricing!") == ["pricing"]


def test_query_terms_keep_digits_drop_single_letters_and_split_punctuation() -> None:
    assert Q.query_terms("Yonatan's lecture 5: Product-Market Fit (PMF) & a CAC/LTV x") == [
        "yonatan", "lecture", "5", "product", "market", "fit", "pmf", "cac", "ltv"]
    assert Q.query_terms("snake_case") == ["snake", "case"]


def test_query_terms_handle_hebrew_and_cap_the_count() -> None:
    assert Q.query_terms("מה יונתן אומר על תמחור של SaaS?") == ["מה", "יונתן", "אומר", "על", "תמחור", "של", "saas"]
    assert len(Q.query_terms(" ".join(f"w{i}" for i in range(200)))) == Q.MAX_QUERY_TERMS
    assert Q.query_terms("?!  …") == []


def test_fts_match_quotes_every_term_and_restricts_columns() -> None:
    assert Q.fts_match(["pricing", "or", "near"]) == '{text} : ("pricing" OR "or" OR "near")'
    assert Q.fts_match(["a\"b"], Q.METADATA_COLUMNS) == '{section_topic subjects_text lecture_title} : ("a""b")'
    assert Q.fts_match([]) is None
    with pytest.raises(ValueError):
        Q.fts_match(["x"], ("passage_id",))


def test_fts_match_is_valid_fts5_syntax_for_hostile_questions() -> None:
    """Whatever a visitor types, the expression must parse (FTS5 raises on syntax errors at query time)."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE passages_fts USING fts5(text, section_topic, subjects_text, lecture_title)")
    db.execute("INSERT INTO passages_fts VALUES ('Pricing is NOT about costs', '', 'Pricing', 'L5')")
    for question in ['"unbalanced', "AND OR NOT", "NEAR(a b)", "col: x", "a* -b ^c", "(((", "מה זה \"PMF\"?", QUESTION]:
        match = Q.fts_match(Q.query_terms(question))
        if match:
            db.execute("SELECT rowid FROM passages_fts WHERE passages_fts MATCH ?", (match,)).fetchall()
    hits = db.execute("SELECT rowid FROM passages_fts WHERE passages_fts MATCH ?", (Q.fts_match(["not"]),)).fetchall()
    assert hits == [(1,)]  # NOT is a word to search for, not an operator


# ----------------------------------------------------------------------------------------
# RRF
# ----------------------------------------------------------------------------------------
def test_rrf_sums_reciprocal_ranks() -> None:
    fused = dict(Q.rrf_fuse([["a", "b", "c"], ["b", "d"], []], rank_constant=60, window=10))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert list(fused) == ["b", "a", "d", "c"]


def test_rrf_window_cuts_each_leg_and_the_result() -> None:
    fused = Q.rrf_fuse([["a", "b", "c", "d"], ["d", "c", "b", "a"]], rank_constant=1, window=2)
    assert [pid for pid, _ in fused] == ["a", "d"]  # b and c were outside both legs' windows


def test_rrf_ties_keep_the_first_leg_first() -> None:
    assert [pid for pid, _ in Q.rrf_fuse([["x"], ["y"]], window=5)] == ["x", "y"]
    assert Q.rrf_fuse([], window=5) == []
