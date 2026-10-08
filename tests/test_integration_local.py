"""Checks against the real index — skipped unless ``RUN_INTEGRATION=1`` (see conftest).

They need the index built by `smartup ingest`, Ollama with the embedding model, and the cross-encoder weights.
Nothing is written and no LLM is involved.
"""

from __future__ import annotations

import pytest

from smartup_rag.retrieval.store import index_health

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_index():
    h = index_health()
    if not h.get("ok"):
        pytest.skip(f"the local index is not usable: {h.get('error')}")


def test_index_holds_the_whole_corpus() -> None:
    h = index_health(detail=True)
    assert h["lectures"] == 34 and h["passages"] == h["vectors"] == 3627


def test_get_lectures_corpus_shape() -> None:
    from smartup_rag.retrieval.retriever import get_lectures

    lectures = get_lectures(include_summaries=False)
    assert len(lectures) == 34
    by_tier = {t: [lec for lec in lectures if lec["tier"] == t] for t in ("v2", "tedx", "v1", "outsource")}
    assert [len(by_tier[t]) for t in ("v2", "tedx", "v1", "outsource")] == [11, 1, 15, 7]
    parts = {lec["lecture_part"] for lec in by_tier["v1"] if lec["lecture_number"] == 11}
    assert parts == {"1", "2"}
    assert all("english_summary" not in lec for lec in lectures)
    assert by_tier["tedx"][0]["citation"] == '(TEDx — "Why fail if you can succeed? - TEDx")'
    assert len({lec["lecture_key"] for lec in lectures}) == 34  # the supplemental talks do not collide


def test_retrieve_returns_documents_with_citation_metadata() -> None:
    from smartup_rag.retrieval.hybrid import retrieve

    docs = retrieve("What does Yonatan say about pricing?", "en", "v2")
    assert docs, "expected at least one passage for a core course topic"
    for d in docs:
        m = d.metadata
        assert m["course_version"] == "v2"
        assert m["citation"].startswith("(v2, Lecture ")
        assert m["lecture_key"].startswith("v2:L")
        assert m["rerank_score"] is not None
        assert len(d.page_content) >= 40


def test_a_hebrew_question_finds_the_lecture_that_answers_it() -> None:
    from smartup_rag.retrieval.hybrid import retrieve

    docs = retrieve("מהם ארבעת הממדים לבניית מותג (Reach, Attention, Repetition, Cost) לפי הקורס?", "he", "v2")
    assert docs and docs[0].metadata["lecture_key"] == "v2:L6"


@pytest.mark.parametrize(("tier", "question", "label"), [
    ("tedx", "Why should you not be first?", "(TEDx — "),
    ("outsource", "What is the Business Model Canvas?", "(Digital Innovation IDC, Lecture "),
])
def test_each_outside_tier_is_searchable(tier, question, label) -> None:
    from smartup_rag.retrieval.hybrid import retrieve

    docs = retrieve(question, "en", tier)
    assert docs and all(d.metadata["tier"] == tier for d in docs)
    assert any(d.metadata["citation"].startswith(label) for d in docs)
