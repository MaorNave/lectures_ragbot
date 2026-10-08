"""Source tiers — where evidence may come from, in the order the assistant searches them (instructions §4, §7).

    tier        what                                          how it is recognised in the index
    ----------  --------------------------------------------  ------------------------------------------------
    v2          SmartUp Foundations course, current version   course_version = "v2"
    tedx        Yonatan Stern's TEDx talk                     course_id = "smartup-tedx"
    v1          SmartUp Foundations course, first version     course_version = "v1"
    outsource   every other outside lecture (supplemental     course_version = "supplemental", except TEDx:
                talks, Digital Innovation IDC)                "How to Start a StartUp", "Nine Out of Ten
                                                              Startups Fail", Digital Innovation IDC 1–5

The index carries no tier field: lectures and passages keep the Drive documents' own fields, and the tier is derived
from `course_id` / `course_version`, here and only here. `tier_conditions` states the rule once, as plain
(field, operator, value) triples that `retrieval.queries` turns into a Chroma `where` filter and a SQL clause.

Citation strings (what the model must copy, and what `nodes.verify` checks):
    (v2, Lecture 5 — "Foundations Course V2 L5 1")          Foundations courses: unchanged
    (v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")
    (TEDx — "Why fail if you can succeed? - TEDx")           a single talk: no lecture number
    (Talk — "How to Start a StartUp")
    (Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")   a course with several lectures
"""

from __future__ import annotations

import re
from typing import Any, Literal

Tier = Literal["v2", "tedx", "v1", "outsource"]
#: One filter condition on a lecture / passage field: (field, "eq" | "ne" | "in" | "nin", value).
Condition = tuple[str, str, Any]

#: The full search order (instructions §4.1–4.2).
TIER_ORDER: tuple[Tier, ...] = ("v2", "tedx", "v1", "outsource")
#: The tiers searched after v2 when it has no strong match.
FALLBACK_TIERS: tuple[Tier, ...] = ("tedx", "v1", "outsource")
FOUNDATIONS_TIERS: frozenset[str] = frozenset({"v1", "v2"})

TEDX_COURSE_ID = "smartup-tedx"
SUPPLEMENTAL_VERSION = "supplemental"

# Citation label per outside course. A course not listed here is cited by its course_name.
SOURCE_LABELS: dict[str, str] = {
    TEDX_COURSE_ID: "TEDx",
    "smartup-how-to-start": "Talk",
    "smartup-nine-out-of-ten": "Talk",
    "digital-innovation-idc": "Digital Innovation IDC",
}
#: Outside courses with several lectures: their citations carry "Lecture N".
NUMBERED_OUTSIDE_COURSES: frozenset[str] = frozenset({"digital-innovation-idc"})

TIER_NAMES = {
    "en": {"v2": "the v2 course", "tedx": "Yonatan's TEDx talk", "v1": "the v1 course", "outsource": "other lectures outside the course"},
    "he": {"v2": "גרסה v2 של הקורס", "tedx": "הרצאת ה-TEDx של יונתן", "v1": "גרסה v1 של הקורס", "outsource": "הרצאות נוספות מחוץ לקורס"},
}


def tier_of(course_id: str | None, course_version: str | None) -> Tier:
    """The tier of one lecture document."""
    version = (course_version or "").strip().lower()
    if version in FOUNDATIONS_TIERS:
        return version  # type: ignore[return-value]
    if (course_id or "") == TEDX_COURSE_ID:
        return "tedx"
    return "outsource"


def tier_conditions(tier: str) -> list[Condition]:
    """The conditions selecting one tier (the same on the lecture catalogue and on the passages).

    A value that is not a tier name is treated as a plain `course_version` (the historic behaviour).
    """
    if tier == "tedx":
        return [("course_id", "eq", TEDX_COURSE_ID)]
    if tier == "outsource":
        return [("course_version", "eq", SUPPLEMENTAL_VERSION), ("course_id", "ne", TEDX_COURSE_ID)]
    return [("course_version", "eq", tier)]


def source_label(course_id: str | None, course_version: str | None, course_name: str | None = None) -> str:
    """The first element of a citation: 'v2' / 'v1' for the course, else the outside source's label."""
    tier = tier_of(course_id, course_version)
    if tier in FOUNDATIONS_TIERS:
        return tier
    return SOURCE_LABELS.get(course_id or "", (course_name or "").strip() or "Lecture")


def lecture_label(course_id: str | None, course_version: str | None, lecture_number: int, part: str | None) -> str:
    """'Lecture 11 Part 2' for the course and numbered outside courses; '' for a single outside talk."""
    tier = tier_of(course_id, course_version)
    if tier not in FOUNDATIONS_TIERS and (course_id or "") not in NUMBERED_OUTSIDE_COURSES:
        return ""
    return f"Lecture {lecture_number}" + (f" Part {part}" if part else "")


def source_line(
    course_id: str | None, course_version: str | None, lecture_number: int, part: str | None, title: str,
    course_name: str | None = None,
) -> str:
    """A citation without its parentheses — also one line of the Sources list."""
    label = source_label(course_id, course_version, course_name)
    lec = lecture_label(course_id, course_version, lecture_number, part)
    return f'{label}, {lec} — "{title}"' if lec else f'{label} — "{title}"'


def citation_for(
    course_id: str | None, course_version: str | None, lecture_number: int, part: str | None, title: str,
    course_name: str | None = None,
) -> str:
    return f"({source_line(course_id, course_version, lecture_number, part, title, course_name)})"


def _outside_labels() -> list[str]:
    return sorted({*SOURCE_LABELS.values()}, key=len, reverse=True)


# Outside-source citations. Stricter than the course pattern on purpose — a known label, then a dash, then a QUOTED
# title — so an ordinary parenthesis such as "(Talk to customers — they know)" is never mistaken for a citation.
OUTSIDE_CITATION_RE = re.compile(
    r"\(\s*(?P<label>" + "|".join(re.escape(lbl) for lbl in _outside_labels()) + r")\s*"
    r"(?:,\s*(?:Lecture|Lesson|הרצאה|שיעור)\s*(?P<num>\d+))?"
    r"\s*[—–\-:]\s*[\"“”'’](?P<title>[^)\"“”]*?)[\"“”'’]\s*\)",
    re.IGNORECASE,
)


def normalise_title(title: str) -> str:
    return re.sub(r"[^\w]+", " ", (title or "").lower()).strip()
