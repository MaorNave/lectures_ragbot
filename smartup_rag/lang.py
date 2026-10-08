"""Language utilities: Hebrew/English detection, citation parsing, lecture-number and part extraction.

No external dependency: Hebrew is detected by Unicode block, which is far more reliable than
statistical language ID on short chat messages that mix Hebrew with English startup jargon
(ICP, PMF, "Jobs to be Done" — exactly the terms instructions.md §13.4 tells users to keep in English).
"""

from __future__ import annotations

import re
from typing import Literal

Lang = Literal["en", "he"]

_HEBREW_RE = re.compile(r"[֐-׿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_HEBREW_WORD_RE = re.compile(r"[֐-׿]+")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*")

# Accepts every citation shape the model has been seen to produce, so none can bypass verification:
#   (v2, Lecture 6 — "Title")   (v1, Lecture 11 Part 2 – 'Title')   (v2, Lecture 5, "Title")
#   (v2, Lecture 5: **Title**)  (v2, Lecture 5)                      (v2, הרצאה 5 — "כותרת")
# The title is lazy and cannot cross ')' ; quotes / markdown emphasis on either side are optional.
CITATION_RE = re.compile(
    r"\(\s*(?P<version>v[12])\s*,\s*(?:Lecture|Lesson|הרצאה|שיעור)\s*(?P<num>\d+)"
    r"(?:\s*,?\s*(?:Part|חלק)\s*(?P<part>\d+))?"
    r"\s*(?:[—–\-:,]\s*)?"
    r"(?:[\"“”'’]|\*{1,2})?(?P<title>[^)]*?)(?:[\"“”'’]|\*{1,2})?\s*\)",
    re.IGNORECASE,
)
_TITLE_TRIM = " *\"“”'’\t"

_LECTURE_NUM_RE = re.compile(
    r"(?:lecture|lesson|class|הרצאה|שיעור)\s*(?:מס(?:פר)?\.?\s*)?#?\s*(\d{1,2})", re.IGNORECASE
)
_PART_RE = re.compile(r"(?:\bpart\s*(\d)\b|חלק\s*(\d|א|ב)\b)", re.IGNORECASE)
_HEBREW_PART = {"א": "1", "ב": "2"}


def detect_language(text: str) -> Lang:
    """Return 'he' when the message is Hebrew, else 'en'.

    Counts WORDS, not letters, and is deliberately asymmetric: a Hebrew speaker constantly types
    multi-word English terms ("מה זה Product-Market Fit?"), while an English speaker essentially never
    types Hebrew letters. Hebrew therefore wins at half the Latin word count. Mixed messages resolve to
    the dominant language (instructions §20.1).
    """
    if not _HEBREW_RE.search(text):
        return "en"
    if not _LATIN_RE.search(text):
        return "he"
    he_words = len(_HEBREW_WORD_RE.findall(text))
    la_words = len(_LATIN_WORD_RE.findall(text))
    return "he" if he_words * 2 >= la_words else "en"


def extract_citations(text: str) -> list[tuple[str, int, str | None, str]]:
    """Parse citations from an answer → [(version, lecture_number, part|None, title), ...]."""
    out = []
    for m in CITATION_RE.finditer(text):
        out.append(
            (m.group("version").lower(), int(m.group("num")), m.group("part"), m.group("title").strip(_TITLE_TRIM))
        )
    return out


def lecture_number_in_text(text: str) -> int | None:
    """Explicit lecture number in EN or HE ("lecture 3", "הרצאה 3", "שיעור 3"), else None."""
    m = _LECTURE_NUM_RE.search(text)
    return int(m.group(1)) if m else None


def lecture_part_in_text(text: str) -> str | None:
    """Explicit lecture part ("part 2", "חלק 2", "חלק ב") → "1"/"2", else None."""
    m = _PART_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return _HEBREW_PART.get(raw, raw)
