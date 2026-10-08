"""Domain models shared by ingestion, retrieval and the agent.

Field names deliberately follow the naming used in `instructions.md` (course_version, lecture_number,
lecture_title, ...) so the LLM sees the same vocabulary it was instructed with. Citation strings are built in
`smartup_rag.tiers`, the one place that knows each source's citation shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["en", "he"]

# `section_topic` value of the per-lecture summary passages written by ingest and recognised by the agent
SUMMARY_TOPIC = "lecture summary"
SUMMARY_SECTION_INDEX = -1


class Section(BaseModel):
    """A topical section of a lecture. The Drive "Naive" exports have none, so a lecture is chunked as one section
    (index 0, no topic); the model stays so a source with real sections can label its passages."""

    section_index: int
    topic: str = ""
    context: str = ""
    en_text: str = ""
    he_text: str = ""
    en_summary: str = ""
    he_summary: str = ""


class Lecture(BaseModel):
    """One lecture (or lecture part) — one Drive JSON document, one row of the lecture catalogue."""

    lecture_key: str  # our deterministic id: "v2:L3", "v1:L11p2", "smartup-tedx:supplemental:L1"
    course_id: str
    course_name: str
    course_version: str  # "v1" | "v2" | "supplemental"
    lecture_number: int
    lecture_part: str | None = None  # "1" / "2" for v1 Lecture 11, else None
    lecture_title: str
    lecture_subjects: list[str] = Field(default_factory=list)
    lecturer_name: str = "Yonatan Stern"
    english_summary: str = ""
    hebrew_summary: str = ""
    english_transcription: str = ""
    hebrew_transcription: str = ""
    sections: list[Section] = Field(default_factory=list)
    source_file: str = ""  # path relative to DATA_DIR, e.g. "course_v2/smartup-foundations-v2-lec05-….json"


class Passage(BaseModel):
    """A retrieval unit: ~1 200 characters of transcript (or a whole summary) in one language, with lecture metadata."""

    passage_id: str
    lecture_key: str
    course_id: str
    course_name: str
    course_version: str
    lecture_number: int
    lecture_part: str | None = None
    lecture_title: str
    lecture_subjects: list[str] = Field(default_factory=list)
    section_index: int
    section_topic: str = ""
    lang: Lang
    position: int
    text: str
    char_start: int = 0
    char_end: int = 0
