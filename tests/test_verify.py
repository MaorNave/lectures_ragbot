"""smartup_rag.agent.nodes.verify — mechanical enforcement of the citation rules (instructions §9–10).

The node is exercised directly with fake state dicts; no LLM, no index.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from smartup_rag.agent.nodes import verify
from smartup_rag.lang import extract_citations
from tests.conftest import V1_L3, V1_L11P1, V1_L11P2, V2_L5, V2_L6, V2_L9, citation_for, make_doc

CIT_L5 = citation_for("v2", 5, None, V2_L5["lecture_title"])
CIT_L6 = citation_for("v2", 6, None, V2_L6["lecture_title"])
CIT_L9 = citation_for("v2", 9, None, V2_L9["lecture_title"])
CIT_L11P2 = citation_for("v1", 11, "2", V1_L11P2["lecture_title"])


def _state(answer: str, docs: list, language: str = "en") -> dict:
    return {"language": language, "docs": docs, "answer": answer}


def _bullets(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if line.startswith("- ")]


# ----------------------------------------------------------------------------------------
# (a) citations to lectures that were not retrieved are removed
# ----------------------------------------------------------------------------------------
def test_unretrieved_citation_is_removed_and_text_is_tidy() -> None:
    docs = [make_doc(V2_L5), make_doc(V2_L5, "A second passage from the same lecture about discounts.")]
    answer = (
        f"Yonatan explains pricing as value capture {CIT_L5}. "
        f"He also warns against discounting {CIT_L9} . Growth comes later."
    )
    out = verify(_state(answer, docs))

    assert out["status"] == "answered"
    assert "Lecture 9" not in out["answer"]
    assert V2_L9["lecture_title"] not in out["answer"]
    assert " ." not in out["answer"] and "  " not in out["answer"]
    assert "He also warns against discounting. Growth comes later." in out["answer"]
    # the valid citation survives untouched, and only that lecture is reported
    assert extract_citations(out["answer"]) == [("v2", 5, None, V2_L5["lecture_title"])]
    assert [c["lecture_key"] for c in out["citations"]] == ["v2:L5"]
    # a single cited lecture with inline citations does not get a Sources block
    assert "## Sources" not in out["answer"]


def test_citation_entries_have_the_api_shape() -> None:
    out = verify(_state(f"Pricing {CIT_L5}.", [make_doc(V2_L5)]))
    (c,) = out["citations"]
    assert set(c) == {
        "lecture_key",
        "course_id",
        "course_version",
        "tier",
        "source",
        "lecture_number",
        "lecture_part",
        "lecture_title",
        "display",
        "citation",
        "source_line",
    }
    assert c["display"] == "Lecture 5" and c["citation"] == CIT_L5 and c["lecture_part"] is None
    assert c["tier"] == "v2" and c["source"] == "v2" and c["source_line"] == CIT_L5[1:-1]
    # the verified answer is also appended to the conversation as the assistant turn
    assert isinstance(out["messages"][0], AIMessage) and out["messages"][0].content == out["answer"]


# ----------------------------------------------------------------------------------------
# (b) wrong title, right lecture → canonical title
# ----------------------------------------------------------------------------------------
def test_wrong_title_is_normalised_to_retrieved_title() -> None:
    answer = 'Pricing is value capture (v2, Lecture 5 — "Pricing Lecture").'
    out = verify(_state(answer, [make_doc(V2_L5)]))
    assert CIT_L5 in out["answer"]
    assert "Pricing Lecture" not in out["answer"]
    assert out["citations"][0]["lecture_title"] == V2_L5["lecture_title"]


def test_curly_quotes_and_dash_variants_are_normalised() -> None:
    answer = "Pricing is value capture (V2, Lecture 5 - “Foundations Course V2 L5 1”)."
    out = verify(_state(answer, [make_doc(V2_L5)]))
    assert out["answer"] == f"Pricing is value capture {CIT_L5}."


# ----------------------------------------------------------------------------------------
# (c) two distinct lectures → Sources section (EN / HE heading)
# ----------------------------------------------------------------------------------------
def test_two_lectures_get_sources_section_english() -> None:
    docs = [make_doc(V2_L5), make_doc(V2_L6, "Branding is a promise you keep.")]
    answer = f"Pricing captures value {CIT_L5}. Branding is a promise {CIT_L6}."
    out = verify(_state(answer, docs))

    body, _, sources = out["answer"].partition("\n\n## Sources\n")
    assert body == answer  # inline text unchanged
    bullets = _bullets(sources)
    assert bullets == [
        f'- v2, Lecture 5 — "{V2_L5["lecture_title"]}"',
        f'- v2, Lecture 6 — "{V2_L6["lecture_title"]}"',
    ]
    assert [c["lecture_key"] for c in out["citations"]] == ["v2:L5", "v2:L6"]


def test_two_lectures_get_hebrew_sources_heading() -> None:
    docs = [make_doc(V2_L5, lang="he"), make_doc(V2_L6, lang="he")]
    answer = f"תמחור הוא לכידת ערך {CIT_L5}. מיתוג הוא הבטחה {CIT_L6}."
    out = verify(_state(answer, docs, language="he"))
    assert "## מקורות\n" in out["answer"]
    assert "## Sources" not in out["answer"]
    assert len(_bullets(out["answer"])) == 2


# ----------------------------------------------------------------------------------------
# (d) a model-written Sources section is stripped and rebuilt from valid citations
# ----------------------------------------------------------------------------------------
def test_model_written_sources_section_is_rebuilt() -> None:
    docs = [make_doc(V2_L5), make_doc(V2_L6)]
    answer = (
        f"Pricing captures value {CIT_L5}. Branding is a promise {CIT_L6}.\n\n"
        "## Sources\n"
        f'- v2, Lecture 5 — "{V2_L5["lecture_title"]}"\n'
        '- v2, Lecture 9 — "Some Invented Title"\n'
    )
    out = verify(_state(answer, docs))
    assert out["answer"].count("## Sources") == 1
    assert "Lecture 9" not in out["answer"] and "Invented" not in out["answer"]
    assert len(_bullets(out["answer"])) == 2
    assert [c["lecture_number"] for c in out["citations"]] == [5, 6]


def test_plain_sources_label_is_also_stripped() -> None:
    docs = [make_doc(V2_L5), make_doc(V2_L6)]
    answer = (
        f"Pricing {CIT_L5}. Branding {CIT_L6}.\n\nSources:\n- v2, Lecture 5\n- v2, Lecture 6\n- v1, Lecture 3"
    )
    out = verify(_state(answer, docs))
    assert "Sources:" not in out["answer"]
    assert "v1, Lecture 3" not in out["answer"]
    assert out["answer"].count("## Sources") == 1
    assert len(_bullets(out["answer"])) == 2


# ----------------------------------------------------------------------------------------
# (e) no citations at all → Sources of the top retrieved lectures
# ----------------------------------------------------------------------------------------
def test_uncited_answer_gets_sources_of_top_docs() -> None:
    docs = [
        make_doc(V2_L5),
        make_doc(V2_L6, "Branding text."),
        make_doc(V1_L11P2, "Oded on exits.", lang="he"),
        make_doc(V1_L3, "Should not be listed: only the top three lectures are exposed."),
    ]
    out = verify(_state("Pricing is about value, branding is a promise, and exits take years.", docs))

    assert "## Sources" in out["answer"]
    bullets = _bullets(out["answer"])
    assert bullets == [
        f'- v2, Lecture 5 — "{V2_L5["lecture_title"]}"',
        f'- v2, Lecture 6 — "{V2_L6["lecture_title"]}"',
        f'- v1, Lecture 11 Part 2 — "{V1_L11P2["lecture_title"]}"',
    ]
    assert [c["lecture_key"] for c in out["citations"]] == ["v2:L5", "v2:L6", "v1:L11p2"]
    assert extract_citations(out["answer"]) == []  # nothing was injected inline


def test_uncited_answer_single_lecture_still_lists_its_source() -> None:
    out = verify(_state("Pricing is about value.", [make_doc(V2_L5), make_doc(V2_L5, "more")]))
    assert _bullets(out["answer"]) == [f'- v2, Lecture 5 — "{V2_L5["lecture_title"]}"']
    assert len(out["citations"]) == 1


def test_no_docs_and_no_citations_leaves_answer_alone() -> None:
    out = verify(_state("Plain text.", []))
    assert out["answer"] == "Plain text."
    assert out["citations"] == []


# ----------------------------------------------------------------------------------------
# (f) v1 Lecture 11 without a Part
# ----------------------------------------------------------------------------------------
def test_v1_lecture_11_without_part_resolves_to_the_single_retrieved_part() -> None:
    answer = 'Oded describes the exit process (v1, Lecture 11 — "Foundations 11 Part 2 Oded").'
    out = verify(_state(answer, [make_doc(V1_L11P2)]))
    assert out["answer"] == f"Oded describes the exit process {CIT_L11P2}."
    (c,) = out["citations"]
    assert c["lecture_part"] == "2" and c["display"] == "Lecture 11 Part 2" and c["lecture_key"] == "v1:L11p2"


def test_v1_lecture_11_without_part_is_ambiguous_when_both_parts_retrieved() -> None:
    docs = [make_doc(V1_L11P1, "Part one."), make_doc(V1_L11P2, "Part two.")]
    answer = 'The exit story (v1, Lecture 11 — "Foundations 11").'
    out = verify(_state(answer, docs))
    # the ambiguous citation is dropped inline; both retrieved parts are exposed as sources instead
    assert extract_citations(out["answer"]) == []
    assert out["answer"].startswith("The exit story.")
    assert set(re.findall(r"Lecture 11 Part (\d)", out["answer"])) == {"1", "2"}
    assert {c["lecture_part"] for c in out["citations"]} == {"1", "2"}
