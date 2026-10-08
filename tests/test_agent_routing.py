"""Routing and verification rules from the review pass, exercised without any LLM or network."""

from __future__ import annotations

from langchain_core.documents import Document

from smartup_rag.agent import nodes as N


def doc(v: str, n: int, title: str, part: str | None = None, rerank: float | None = 0.5) -> Document:
    disp = f"Lecture {n}" + (f" Part {part}" if part else "")
    return Document(
        page_content="passage text",
        metadata={
            "lecture_key": f"{v}:L{n}" + (f"p{part}" if part else ""), "course_id": "smartup-foundations",
            "course_version": v, "lecture_number": n, "lecture_part": part, "lecture_title": title,
            "lecture_subjects": ["pricing", "exits"], "lang": "en", "score": 1.0, "rerank_score": rerank,
            "section_topic": "", "source_index": "idx", "citation": f'({v}, {disp} — "{title}")',
        },
    )


# ---------------------------------------------------------------- version scope (§4.2 / §6)
def test_comparison_of_both_versions_is_not_downgraded_to_v1():
    assert N.normalise_version_scope("What changed between v1 and v2 lecture 1?", "default", "comparison") == "both"
    assert N.normalise_version_scope("Compare v1 and v2 on fundraising", "default", "conceptual") == "both"
    assert N.normalise_version_scope("מה ההבדל בין גרסה 1 לגרסה 2 של ההרצאה?", "default", "conceptual") == "both"


def test_explicit_v1_alone_becomes_v1_and_plain_questions_stay_default():
    assert N.normalise_version_scope("What does v1 lecture 5 say about pricing?", "default", "lecture_specific") == "v1"
    assert N.normalise_version_scope("What does the course say about pricing?", "default", "conceptual") == "default"
    assert N.normalise_version_scope("anything", "v2", "conceptual") == "v2"  # classifier verdicts other than default are kept


# ---------------------------------------------------------------- never generate from nothing (§4.5)
def test_explicit_scope_with_no_passages_routes_to_not_found_not_generate():
    state = {"classification": {"version_scope": "v1", "lecture_number": 12}, "docs": [], "docs_v2": [], "language": "en", "notes": []}
    g = N.grade_v2(state, None)
    assert g["grade_v2"]["verdict"] == "insufficient"
    state.update(g)
    assert N.route_after_grade(state) == "not_found"


def test_explicit_scope_with_passages_still_generates():
    state = {"classification": {"version_scope": "both"}, "docs": [doc("v2", 1, "A"), doc("v1", 1, "B")], "docs_v2": [doc("v2", 1, "A")], "language": "en", "notes": []}
    state.update(N.grade_v2(state, None))
    assert N.route_after_grade(state) == "generate"


def test_named_lecture_without_coverage_now_tries_v1_instead_of_stopping():
    state = {"classification": {"version_scope": "default", "lecture_number": 11}, "docs": [], "docs_v2": [doc("v2", 11, "X")],
             "grade_v2": {"verdict": "insufficient"}, "language": "en"}
    assert N.route_after_grade(state) == "fallback"


# ---------------------------------------------------------------- scope guard fails closed (§0.4)
def test_missing_or_false_scope_decision_refuses():
    assert N.route_after_classify({"classification": {}}) == "refuse"
    assert N.route_after_classify({"classification": {"in_scope": False, "intent": "out_of_scope"}}) == "refuse"
    assert N.route_after_classify({"classification": {"in_scope": True, "intent": "conceptual"}}) == "retrieve"
    assert N.route_after_classify({"classification": {"intent": "chit_chat"}}) == "chit_chat"


# ---------------------------------------------------------------- verify() (§9, §10, §12)
def test_decline_answer_gets_no_sources_and_is_reported_not_found():
    out = N.verify({"language": "en", "docs": [doc("v2", 5, "A"), doc("v1", 13, "B")],
                    "answer": "The lectures do not directly address this question. The closest related topic is pricing."})
    assert "Sources" not in out["answer"] and out["citations"] == [] and out["status"] == "not_found"


def test_grounded_answer_without_inline_citations_still_lists_sources():
    out = N.verify({"language": "en", "docs": [doc("v2", 5, "A"), doc("v2", 1, "B")],
                    "answer": "Yonatan says profitability comes first and growth follows from a working business model."})
    assert "## Sources" in out["answer"] and len(out["citations"]) == 2 and out["status"] == "answered"


def test_multi_space_title_survives_removal_of_another_citation():
    title = "Foundations   Lecture 13"
    ans = f'Exits matter (v1, Lecture 13 — "{title}"). Burn matters (v2, Lecture 12 — "Made Up").'
    out = N.verify({"language": "en", "docs": [doc("v1", 13, title)], "answer": ans})
    assert f'(v1, Lecture 13 — "{title}")' in out["answer"]
    assert "Made Up" not in out["answer"]
    assert out["answer"].startswith("Exits matter") and "Burn matters." in out["answer"]


def test_sources_heading_followed_by_prose_is_not_truncated():
    ans = 'First (v2, Lecture 5 — "A").\n\n## Sources\n- v2, Lecture 5 — "A"\n\nAnd here is one more important paragraph.'
    out = N.verify({"language": "he", "docs": [doc("v2", 5, "A")], "answer": ans})
    assert "one more important paragraph" in out["answer"]


def test_trailing_model_sources_block_is_replaced_not_duplicated():
    ans = 'First (v2, Lecture 5 — "A") then (v2, Lecture 1 — "B").\n\n**Sources:**\n- v2, Lecture 5 — "A"\n- v2, Lecture 1 — "B"\n'
    out = N.verify({"language": "en", "docs": [doc("v2", 5, "A"), doc("v2", 1, "B")], "answer": ans})
    assert out["answer"].count("Sources") == 1 and "**Sources:**" not in out["answer"]


def test_comma_and_unquoted_citation_shapes_are_normalised_and_verified():
    ans = 'See (v2, Lecture 5, "A") and (v2, Lecture 99: Invented) and (v2, Lecture 1).'
    out = N.verify({"language": "en", "docs": [doc("v2", 5, "A"), doc("v2", 1, "B")], "answer": ans})
    assert '(v2, Lecture 5 — "A")' in out["answer"] and '(v2, Lecture 1 — "B")' in out["answer"]
    assert "Invented" not in out["answer"] and len(out["citations"]) == 2


def test_hebrew_decline_is_recognised():
    out = N.verify({"language": "he", "docs": [doc("v2", 5, "A")], "answer": "לא מצאתי את הנושא הזה בהרצאות SmartUp."})
    assert out["status"] == "not_found" and "מקורות" not in out["answer"]
