"""Load the lectures from the Drive "Naive" exports — the one and only data source of this project.

    data/raw/drive_09_2026/            (DATA_DIR)
      course_v2/   11 documents   SmartUp Foundations v2, Lectures 1–11
      course_v1/   15 documents   SmartUp Foundations v1, Lectures 1–10, 11 Part 1, 11 Part 2, 13–15
      outsource/    8 documents   the TEDx talk, "How to Start a StartUp", "Nine Out of Ten Startups Fail",
                                  Digital Innovation IDC Lectures 1–5

Each file is one lecture: metadata (course_id, course_name, course_version, lecture_number, lecture_title,
lecture_subjects, lecturer_name, …), the full English and Hebrew transcriptions and an English and a Hebrew summary.
The files are never modified; three data-quality rules are applied while loading (verified on the 2026-09-24 export):

  - The Digital Innovation files say ``course_id: "`Digital Innovation Idc`"`` and ``course_version: "v1"``, and reuse
    the `lecture_id`s of Foundations v1 Lectures 1–5. As-is they would be searched as course v1, so they become
    `digital-innovation-idc` / "Digital Innovation IDC" / `supplemental` — the labels the other outside talks carry.
  - Foundations v1 titles carry export noise ("Foundations   Lecture 8   Nice Ai Transcript (1)", "Transcript No Tp",
    plain "Lecture"). Titles are what students see in citations, so the noise is stripped and an empty result becomes
    "Foundations Lecture N". "Part N" and speaker names ("… Part 2 Oded") are kept.
  - v1 Lecture 11 is two documents with the same number; the part comes from the title ("Part 1" / "Part 2"), and
    lecture keys ('v2:L3', 'v1:L11p2', 'smartup-tedx:supplemental:L1') are unique across all sources.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from smartup_rag.config import get_settings
from smartup_rag.retrieval.hybrid import lecture_key_for, lecture_part_from_title
from smartup_rag.schemas import Lecture

log = logging.getLogger(__name__)

DRIVE_FOLDERS = ("course_v2", "course_v1", "outsource")

DIGITAL_INNOVATION_COURSE_ID = "digital-innovation-idc"
DIGITAL_INNOVATION_COURSE_NAME = "Digital Innovation IDC"
SUPPLEMENTAL_VERSION = "supplemental"
DEFAULT_LECTURER = "Yonatan Stern"

_WS_RE = re.compile(r"\s+")
_TITLE_NOISE_RE = re.compile(r"(?:\bnice\s+)?\bai\s+transcript\b|\btranscript\s+no\s+tp\b|\(\d+\)", re.IGNORECASE)
_BARE_NUMBER_TITLE_RE = re.compile(r"^Foundations\s+(\d+)$", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "")


# the subject tagger's own chatter, left in some exports (v1 Lecture 6: "Looking at this lecture transcript", …)
_SUBJECT_CHATTER_RE = re.compile(r"^(?:looking at|i need to|let me|here (?:are|is))\b", re.IGNORECASE)


def _subjects(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    out = []
    for v in value or []:
        v = _WS_RE.sub(" ", str(v)).strip().strip('[]"').strip()  # a JSON list split into items keeps its brackets
        if v and not _SUBJECT_CHATTER_RE.match(v):
            out.append(v)
    return out


def is_digital_innovation(doc: dict[str, Any]) -> bool:
    blob = " ".join(str(doc.get(k) or "") for k in ("course_id", "course_name", "lecture_slug")).lower()
    return "digital innovation" in blob.replace("-", " ").replace("_", " ").replace("`", " ")


def clean_title(title: str, *, course_id: str, lecture_number: int) -> str:
    """Strip transcript-export noise; fall back to the course's plain lecture name when nothing is left."""
    t = _WS_RE.sub(" ", _TITLE_NOISE_RE.sub(" ", title or "")).strip(" ,;:-–—")
    if course_id == DIGITAL_INNOVATION_COURSE_ID:
        return f"{DIGITAL_INNOVATION_COURSE_NAME} Lecture {lecture_number}"
    if course_id.startswith("smartup-foundations"):
        m = _BARE_NUMBER_TITLE_RE.match(t)
        if m:
            t = f"Foundations Lecture {m.group(1)}"
        if not t or t.lower() == "lecture":
            t = f"Foundations Lecture {lecture_number}"
    return t


def normalise_document(raw: dict[str, Any]) -> dict[str, Any]:
    """One Drive JSON document → the same fields, normalised (the input dict is not modified)."""
    out = dict(raw)
    out["course_id"] = str(raw.get("course_id") or "").strip().strip("`").strip()
    out["course_name"] = _WS_RE.sub(" ", str(raw.get("course_name") or "")).strip()
    out["course_version"] = str(raw.get("course_version") or "").strip().lower()
    if is_digital_innovation(raw):
        out["course_id"] = DIGITAL_INNOVATION_COURSE_ID
        out["course_name"] = DIGITAL_INNOVATION_COURSE_NAME
        out["course_version"] = SUPPLEMENTAL_VERSION
    out["lecture_number"] = int(raw.get("lecture_number") or 0)
    out["lecture_title"] = clean_title(
        str(raw.get("lecture_title") or ""), course_id=out["course_id"], lecture_number=out["lecture_number"]
    )
    return out


def lecture_from_document(doc: dict[str, Any], *, source_file: str = "") -> Lecture:
    """A normalised document → `Lecture` (the lecture part comes from the title: "… Part 2 …")."""
    course_id, version, number = doc["course_id"], doc["course_version"], int(doc["lecture_number"])
    title = doc["lecture_title"]
    part = lecture_part_from_title(title)
    return Lecture(
        lecture_key=lecture_key_for(course_id, version, number, part),
        course_id=course_id,
        course_name=doc.get("course_name") or "",
        course_version=version,
        lecture_number=number,
        lecture_part=part,
        lecture_title=title,
        lecture_subjects=_subjects(doc.get("lecture_subjects")),
        lecturer_name=str(doc.get("lecturer_name") or "").strip() or DEFAULT_LECTURER,
        english_summary=_text(doc.get("english_summary")).strip(),
        hebrew_summary=_text(doc.get("hebrew_summary")).strip(),
        english_transcription=_text(doc.get("english_transcription")),
        hebrew_transcription=_text(doc.get("hebrew_transcription")),
        source_file=source_file,
    )


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _completeness(lec: Lecture) -> int:
    return len(lec.english_transcription) + len(lec.hebrew_transcription)


def dedupe_lectures(lectures: Iterable[Lecture]) -> list[Lecture]:
    """One `Lecture` per lecture_key: two files describing the same lecture keep the more complete transcript."""
    best: dict[str, Lecture] = {}
    for lec in lectures:
        cur = best.get(lec.lecture_key)
        if cur is None:
            best[lec.lecture_key] = lec
            continue
        keep, drop = (lec, cur) if _completeness(lec) > _completeness(cur) else (cur, lec)
        best[lec.lecture_key] = keep
        log.warning("two files for lecture %s: keeping %s (%d chars), ignoring %s (%d chars)",
                    lec.lecture_key, keep.source_file, _completeness(keep), drop.source_file, _completeness(drop))
    return list(best.values())


def sort_lectures(lectures: Iterable[Lecture]) -> list[Lecture]:
    return sorted(lectures, key=lambda lec: (lec.course_id, lec.lecture_number, lec.lecture_part or ""))


def read_documents(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """(path relative to `root`, raw JSON) for every lecture file of the three export folders, in folder order.

    A missing folder is an error (a partial export would silently drop a whole source); a JSON file that is not a
    lecture document is skipped with a warning.
    """
    docs: list[tuple[str, dict[str, Any]]] = []
    for folder in DRIVE_FOLDERS:
        path = root / folder
        if not path.is_dir():
            raise FileNotFoundError(f"missing export folder: {path}")
        for f in sorted(path.glob("*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"unreadable lecture file {f}: {exc}") from exc
            if not isinstance(raw, dict) or "lecture_number" not in raw:
                log.warning("%s/%s is not a lecture document; skipping", folder, f.name)
                continue
            docs.append((f"{folder}/{f.name}", raw))
    return docs


def load_lectures(root: Path | None = None, include_digital_innovation: bool | None = None) -> list[Lecture]:
    """Every lecture of the export as a normalised `Lecture`; Digital Innovation only when enabled."""
    s = get_settings()
    root = Path(root) if root is not None else s.data_dir
    include_di = s.include_digital_innovation if include_digital_innovation is None else include_digital_innovation
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")
    lectures = [lecture_from_document(normalise_document(raw), source_file=rel) for rel, raw in read_documents(root)]
    if not include_di:
        n_di = sum(1 for lec in lectures if lec.course_id == DIGITAL_INNOVATION_COURSE_ID)
        lectures = [lec for lec in lectures if lec.course_id != DIGITAL_INNOVATION_COURSE_ID]
        if n_di:
            log.info("skipped %d Digital Innovation lecture(s) (INCLUDE_DIGITAL_INNOVATION=false)", n_di)
    lectures = dedupe_lectures(lectures)
    log.info("loaded %d lectures from %s", len(lectures), root)
    return sort_lectures(lectures)
