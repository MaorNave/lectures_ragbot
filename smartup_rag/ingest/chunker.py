"""Lecture → passages with LangChain's `RecursiveCharacterTextSplitter`.

A passage is ~`passage_chars` characters of transcript in ONE language, tagged with the lecture
metadata and the topical section it came from, so the retriever can cite
`(v2, Lecture 5 — "Title")` and the LLM sees the section topic as context.

Rules:
  - Each language is split separately. A lecture with topical sections keeps `section_index` / `topic` per
    passage; the Drive exports have none, so the full transcription of each language is section 0.
  - The lecture summary becomes one extra passage per language (`section_index=-1`,
    `section_topic="lecture summary"`), so "summarize lecture 3" hits a summary directly.
  - Passages shorter than `MIN_PASSAGE_CHARS` are dropped (headers, stray speaker labels).
  - `passage_id = "{lecture_key}:{lang}:{section_index}:{position}"` is deterministic, so a
    re-run overwrites instead of duplicating.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartup_rag.config import get_settings
from smartup_rag.schemas import SUMMARY_SECTION_INDEX, SUMMARY_TOPIC, Lang, Lecture, Passage, Section

log = logging.getLogger(__name__)

SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]
MIN_PASSAGE_CHARS = 80


def make_splitter(passage_chars: int | None = None, overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    """Character splitter (paragraph → line → sentence → word) with the configured size/overlap."""
    s = get_settings()
    size = int(passage_chars or s.passage_chars)
    ov = s.passage_overlap if overlap is None else int(overlap)
    if size <= 0:
        raise ValueError("passage_chars must be positive")
    ov = max(0, min(ov, size // 2))  # the splitter requires overlap < chunk size
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=ov,
        separators=SEPARATORS,
        keep_separator="end",  # sentences keep their full stop instead of the next chunk starting with ". "
        add_start_index=True,
        length_function=len,
    )


def _passage(
    lecture: Lecture,
    section_index: int,
    topic: str,
    lang: Lang,
    position: int,
    text: str,
    char_start: int,
    char_end: int,
) -> Passage:
    return Passage(
        passage_id=f"{lecture.lecture_key}:{lang}:{section_index}:{position}",
        lecture_key=lecture.lecture_key,
        course_id=lecture.course_id,
        course_name=lecture.course_name,
        course_version=lecture.course_version,
        lecture_number=lecture.lecture_number,
        lecture_part=lecture.lecture_part,
        lecture_title=lecture.lecture_title,
        lecture_subjects=list(lecture.lecture_subjects),
        section_index=section_index,
        section_topic=topic,
        lang=lang,
        position=position,
        text=text,
        char_start=char_start,
        char_end=char_end,
    )


def split_section(
    lecture: Lecture,
    section: Section,
    lang: Lang,
    text: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Passage]:
    """Split one language of one section; `char_start/char_end` are offsets into `text`."""
    text = text or ""
    if len(text.strip()) < MIN_PASSAGE_CHARS:
        return []
    out: list[Passage] = []
    position = 0
    for doc in splitter.create_documents([text]):
        chunk = doc.page_content
        body = chunk.strip()
        if len(body) < MIN_PASSAGE_CHARS:
            continue
        start = int(doc.metadata.get("start_index", -1))
        if start >= 0:
            start += len(chunk) - len(chunk.lstrip())
            end = start + len(body)
        else:  # splitter could not locate the chunk (should not happen with add_start_index)
            start = end = 0
        out.append(_passage(lecture, section.section_index, section.topic, lang, position, body, start, end))
        position += 1
    return out


def lecture_to_passages(
    lecture: Lecture,
    passage_chars: int | None = None,
    overlap: int | None = None,
) -> list[Passage]:
    """All passages of one lecture: per-section (or whole-transcript) chunks plus summary passages."""
    splitter = make_splitter(passage_chars, overlap)
    sections = lecture.sections or [
        Section(
            section_index=0,
            topic="",
            en_text=lecture.english_transcription,
            he_text=lecture.hebrew_transcription,
        )
    ]
    passages: list[Passage] = []
    for sec in sections:
        for lang, text in (("en", sec.en_text), ("he", sec.he_text)):
            passages.extend(split_section(lecture, sec, lang, text, splitter))
    for lang, summary in (("en", lecture.english_summary), ("he", lecture.hebrew_summary)):
        summary = (summary or "").strip()
        if len(summary) >= MIN_PASSAGE_CHARS:
            passages.append(_passage(lecture, SUMMARY_SECTION_INDEX, SUMMARY_TOPIC, lang, 0, summary, 0, len(summary)))
    return passages


def lectures_to_passages(
    lectures: Iterable[Lecture],
    passage_chars: int | None = None,
    overlap: int | None = None,
) -> list[Passage]:
    """Convenience wrapper over `lecture_to_passages` with per-lecture logging."""
    out: list[Passage] = []
    for lec in lectures:
        ps = lecture_to_passages(lec, passage_chars, overlap)
        log.info("%-28s %-45s sections=%-2d passages=%d", lec.lecture_key, lec.lecture_title[:45], len(lec.sections), len(ps))
        out.extend(ps)
    return out
