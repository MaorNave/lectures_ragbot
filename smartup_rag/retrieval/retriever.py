"""LangChain-facing retrieval API.

`LectureRetriever` is a standard `BaseRetriever`, so it composes with any LangChain chain
(`retriever.invoke(question)`), while `get_lectures()` serves lecture-level lookups
("summarize lecture 3", "which lectures exist?") from the lecture catalogue in SQLite.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from smartup_rag.lang import Lang
from smartup_rag.retrieval import queries as Q
from smartup_rag.retrieval.hybrid import retrieve
from smartup_rag.retrieval.store import get_index
from smartup_rag.tiers import citation_for, lecture_label, source_label, tier_of


class LectureRetriever(BaseRetriever):
    """Tier-scoped (v2 / tedx / v1 / outsource), language-routed hybrid retriever over the SmartUp lectures."""

    lang: Lang = "en"
    tier: str | None = "v2"
    lecture_number: int | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return retrieve(query, self.lang, self.tier, self.lecture_number)


def get_lectures(
    tier: str | None = None,
    lecture_number: int | None = None,
    *,
    include_summaries: bool = True,
) -> list[dict[str, Any]]:
    """Lectures (metadata + summaries) from the catalogue, sorted by version/number/part; `tier` filters."""
    where_sql, params = Q.sql_where(Q.conditions(tier, lecture_number), alias="l")
    out: list[dict[str, Any]] = []
    for row in get_index().lectures(where_sql, params):
        course_id, version = row["course_id"], row["course_version"]
        num, part, title = int(row["lecture_number"]), row["lecture_part"] or None, row["lecture_title"]
        item = {
            "lecture_key": row["lecture_key"],
            "course_id": course_id,
            "course_name": row["course_name"],
            "course_version": version,
            "tier": tier_of(course_id, version),
            "source_label": source_label(course_id, version, row["course_name"]),
            "lecture_number": num,
            "lecture_part": part,
            "lecture_title": title,
            "display": lecture_label(course_id, version, num, part),
            "citation": citation_for(course_id, version, num, part, title, row["course_name"]),
            "lecture_subjects": list(row["lecture_subjects"]),
        }
        if include_summaries:
            item["english_summary"] = row["english_summary"] or ""
            item["hebrew_summary"] = row["hebrew_summary"] or ""
        out.append(item)
    return out
