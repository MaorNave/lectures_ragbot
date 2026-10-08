"""Shared pytest configuration and helpers.

* No network and no models: the default test run needs neither Ollama nor the cross-encoder weights. Index tests
  build a real Chroma + SQLite index in a temporary directory with a deterministic fake embedder.
* Tests marked ``@pytest.mark.integration`` read the real index built by ``smartup ingest`` and call Ollama
  (bge-m3) and the reranker; they are skipped unless ``RUN_INTEGRATION=1`` is set in the environment.
* ``make_doc`` builds LangChain ``Document`` objects with the exact metadata shape produced by
  ``smartup_rag.retrieval.hybrid.Candidate.to_document`` so agent nodes can be tested in isolation.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.documents import Document

from smartup_rag.retrieval.hybrid import display_lecture, make_lecture_key

# Realistic lectures from the corpus (titles as the loader normalises them).
V2_L5 = {
    "course_version": "v2",
    "lecture_number": 5,
    "lecture_part": None,
    "lecture_title": "Foundations Course V2 L5 1",
}
V2_L6 = {
    "course_version": "v2",
    "lecture_number": 6,
    "lecture_part": None,
    "lecture_title": "Foundations Course V2 L6",
}
V2_L9 = {
    "course_version": "v2",
    "lecture_number": 9,
    "lecture_part": None,
    "lecture_title": "Foundations Course V2 L9",
}
V1_L3 = {
    "course_version": "v1",
    "lecture_number": 3,
    "lecture_part": None,
    "lecture_title": "Foundations Lecture 3",
}
V1_L11P1 = {
    "course_version": "v1",
    "lecture_number": 11,
    "lecture_part": "1",
    "lecture_title": "Foundations 11 Part 1",
}
V1_L11P2 = {
    "course_version": "v1",
    "lecture_number": 11,
    "lecture_part": "2",
    "lecture_title": "Foundations 11 Part 2 Oded",
}


def citation_for(
    course_version: str, lecture_number: int, lecture_part: str | None, lecture_title: str
) -> str:
    """Canonical citation string, e.g. ``(v2, Lecture 5 — "Foundations Course V2 L5 1")``."""
    return f'({course_version}, {display_lecture(lecture_number, lecture_part)} — "{lecture_title}")'


def make_doc(
    lecture: dict[str, Any] = V2_L5,
    text: str = "Pricing is about capturing the value you create for the customer, not about your costs.",
    *,
    lang: str = "en",
    rerank_score: float | None = 0.55,
    score: float = 0.032,
    section_topic: str = "",
    course_id: str = "smartup-foundations",
    source_index: str = "local:smartup_passages",
    subjects: list[str] | None = None,
) -> Document:
    """A retrieved passage with the same metadata keys the real retriever emits."""
    version, number, part, title = (
        lecture["course_version"],
        lecture["lecture_number"],
        lecture.get("lecture_part"),
        lecture["lecture_title"],
    )
    return Document(
        page_content=text,
        metadata={
            "lecture_key": make_lecture_key(version, number, part),
            "course_id": course_id,
            "course_version": version,
            "tier": version,
            "source_label": version,
            "lecture_number": number,
            "lecture_part": part,
            "lecture_title": title,
            "lecture_subjects": list(subjects or ["Pricing", "Business Model"]),
            "lang": lang,
            "score": score,
            "rerank_score": rerank_score,
            "section_topic": section_topic,
            "source_index": source_index,
            "citation": citation_for(version, number, part, title),
        },
    )


@pytest.fixture
def doc_factory():
    """Fixture form of :func:`make_doc` (``doc_factory(V2_L6, "text", lang="he")``)."""
    return make_doc


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test (needs the built index, Ollama and the reranker): set RUN_INTEGRATION=1")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# Outside sources (tier "tedx" / "outsource"), as the lecture catalogue stores them.
TEDX = {"course_id": "smartup-tedx", "course_name": "SmartUp TEDx Talks", "course_version": "supplemental",
        "lecture_number": 1, "lecture_title": "Why fail if you can succeed? - TEDx"}
TALK_START = {"course_id": "smartup-how-to-start", "course_name": "SmartUp Supplemental Talks",
              "course_version": "supplemental", "lecture_number": 1, "lecture_title": "How to Start a StartUp"}
TALK_NINE = {"course_id": "smartup-nine-out-of-ten", "course_name": "SmartUp Supplemental Talks",
             "course_version": "supplemental", "lecture_number": 1,
             "lecture_title": "Nine Out of Ten Startups Fail: Why Fail When You Can Succeed?"}
DI_L3 = {"course_id": "digital-innovation-idc", "course_name": "Digital Innovation IDC", "course_version": "supplemental",
         "lecture_number": 3, "lecture_title": "Digital Innovation IDC Lecture 3"}


def make_source_doc(source: dict[str, Any], text: str = "Become profitable as quickly as possible.", *,
                    lang: str = "en", rerank_score: float | None = 0.55) -> Document:
    """A retrieved passage of any source, built by the real retriever code (Candidate.to_document)."""
    from smartup_rag.retrieval.hybrid import Candidate

    return Candidate(text=text, lang=lang, score=0.03, rerank_score=rerank_score, lecture_part=source.get("lecture_part"),
                     **{k: source[k] for k in ("course_id", "course_name", "course_version", "lecture_number",
                                              "lecture_title")}).to_document()
