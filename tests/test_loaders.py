"""smartup_rag.ingest.loaders — the Drive "Naive" exports → normalised `Lecture`s (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartup_rag.config import PROJECT_ROOT
from smartup_rag.ingest import loaders as L
from smartup_rag.tiers import tier_of

DI_RAW = {  # the Digital Innovation export as it is on Drive: backticks, labelled v1, junk title
    "course_id": "`Digital Innovation Idc`", "course_name": "Digital Innovation Idc", "course_version": "v1",
    "lecture_id": "b3349ac3-5e38-51ee-9300-bb60c4a2d02a", "lecture_slug": "lec01-digital-innovation-i",
    "lecture_number": 1, "lecture_title": ", Lecture Transcript L1",
}


def _v1(number, title, **extra):
    return {"course_id": "smartup-foundations-v1", "course_name": "SmartUp Foundations", "course_version": "v1",
            "lecture_number": number, "lecture_title": title, "lecture_slug": f"v1-lec{number}", **extra}


def test_digital_innovation_becomes_an_outside_course():
    doc = L.normalise_document(DI_RAW)
    assert (doc["course_id"], doc["course_name"], doc["course_version"]) == (
        "digital-innovation-idc", "Digital Innovation IDC", "supplemental")
    assert doc["lecture_title"] == "Digital Innovation IDC Lecture 1"
    assert doc["lecture_id"] == DI_RAW["lecture_id"]  # other fields are kept as they are
    assert DI_RAW["course_version"] == "v1"  # the input is not modified


@pytest.mark.parametrize(("number", "raw", "clean"), [
    (8, "Foundations   Lecture 8   Nice Ai Transcript (1)", "Foundations Lecture 8"),
    (2, "Foundations   Lecture 2   Ai Transcript", "Foundations Lecture 2"),
    (7, "Transcript No Tp", "Foundations Lecture 7"),
    (1, "Lecture", "Foundations Lecture 1"),
    (15, "Foundations 15   Nice Ai Transcript", "Foundations Lecture 15"),
    (11, "Foundations 11 Part 2 Oded  Nice Ai Transcript (1)", "Foundations 11 Part 2 Oded"),
])
def test_foundations_titles_lose_the_export_noise(number, raw, clean):
    assert L.normalise_document(_v1(number, raw))["lecture_title"] == clean


def test_other_titles_are_untouched():
    doc = {"course_id": "smartup-tedx", "course_name": "SmartUp TEDx Talks", "course_version": "supplemental",
           "lecture_number": 1, "lecture_title": "Why fail if you can succeed? - TEDx"}
    assert L.normalise_document(doc)["lecture_title"] == doc["lecture_title"]


def test_lecture_parts_and_keys_come_from_the_titles():
    p1 = L.lecture_from_document(L.normalise_document(_v1(11, "Foundations 11   Part 1   Yonatan   Nice Ai Transcript")))
    p2 = L.lecture_from_document(L.normalise_document(_v1(11, "Foundations 11 Part 2 Oded  Nice Ai Transcript (1)")))
    assert (p1.lecture_part, p1.lecture_key) == ("1", "v1:L11p1")
    assert (p2.lecture_part, p2.lecture_key) == ("2", "v1:L11p2")
    di = L.lecture_from_document(L.normalise_document(DI_RAW))
    assert di.lecture_key == "digital-innovation-idc:supplemental:L1" and di.lecturer_name == "Yonatan Stern"


def _export(root: Path, files: dict[str, dict]) -> Path:
    for folder in L.DRIVE_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for rel, doc in files.items():
        (root / rel).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return root


def test_load_lectures_reads_the_three_folders(tmp_path):
    root = _export(tmp_path, {
        "course_v1/a.json": _v1(11, "Foundations 11 Part 1", english_transcription="x" * 10),
        "course_v1/b.json": _v1(11, "Foundations 11 Part 2 Oded"),
        "outsource/lec01.json": DI_RAW,
        "outsource/notes.json": {"hello": "not a lecture"},
    })
    lectures = L.load_lectures(root, include_digital_innovation=True)
    assert [lec.lecture_key for lec in lectures] == ["digital-innovation-idc:supplemental:L1", "v1:L11p1", "v1:L11p2"]
    assert {lec.source_file for lec in lectures} == {"course_v1/a.json", "course_v1/b.json", "outsource/lec01.json"}
    assert [lec.lecture_key for lec in L.load_lectures(root, include_digital_innovation=False)] == ["v1:L11p1", "v1:L11p2"]


def test_two_files_for_one_lecture_keep_the_more_complete_one(tmp_path):
    root = _export(tmp_path, {
        "course_v1/short.json": _v1(3, "Foundations Lecture 3", english_transcription="short"),
        "course_v1/long.json": _v1(3, "Foundations Lecture 3", english_transcription="a much longer transcript"),
    })
    [lec] = L.load_lectures(root)
    assert lec.source_file == "course_v1/long.json"


def test_a_missing_export_folder_is_an_error(tmp_path):
    (tmp_path / "course_v2").mkdir()
    with pytest.raises(FileNotFoundError):
        L.load_lectures(tmp_path)


def test_an_unreadable_lecture_file_is_an_error(tmp_path):
    root = _export(tmp_path, {})
    (root / "course_v2" / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        L.load_lectures(root)


def test_the_shipped_export_is_the_34_lecture_corpus():
    """The data this repository ships (data/raw/drive_09_2026): four sources, unique keys, both v1 Lecture 11 parts."""
    root = PROJECT_ROOT / "data" / "raw" / "drive_09_2026"
    if not root.is_dir():
        pytest.skip("data/raw/drive_09_2026 is not present")
    lectures = L.load_lectures(root, include_digital_innovation=True)
    assert len(lectures) == 34 and len({lec.lecture_key for lec in lectures}) == 34
    by_tier = {t: [lec for lec in lectures if tier_of(lec.course_id, lec.course_version) == t] for t in ("v2", "tedx", "v1", "outsource")}
    assert [len(by_tier[t]) for t in ("v2", "tedx", "v1", "outsource")] == [11, 1, 15, 7]
    assert {lec.lecture_part for lec in by_tier["v1"] if lec.lecture_number == 11} == {"1", "2"}
    assert all(lec.english_transcription and lec.hebrew_transcription and lec.english_summary for lec in lectures)
    assert not any("transcript" in lec.lecture_title.lower() for lec in by_tier["v1"])
