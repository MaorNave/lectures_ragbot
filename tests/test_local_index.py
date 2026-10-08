"""The local index end to end: ingest → Chroma + SQLite FTS5 → hybrid search → rerank → Documents.

Real Chroma, real SQLite, the real loader and chunker. Only the two models are replaced: embeddings by a
deterministic bag-of-words hash (passages that share words are close), the cross-encoder by a word-overlap scorer.
No Ollama, no torch, no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from smartup_rag.config import get_settings
from smartup_rag.ingest import indexer
from smartup_rag.retrieval import embeddings, hybrid, reranker, store

DIM = 64
_WORD_RE = re.compile(r"[^\W_]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def fake_vector(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in _words(text):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeReranker:
    """Share of the question's words found in the passage (0–1): relevant passages score high, others ~0."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        q = set(_words(query))
        return [len(q & set(_words(t))) / (len(q) or 1) for t in texts]

    def status(self) -> dict:
        return {"model": "fake", "device": "cpu", "loaded": True, "load_s": 0.0, "error": None}


def _para(*sentences: str, n: int = 6) -> str:
    return "\n\n".join(" ".join(sentences) for _ in range(n))


LECTURES = {
    "course_v2/v2-l1.json": {
        "course_id": "smartup-foundations-v2", "course_name": "SmartUp Foundations", "course_version": "v2",
        "lecture_number": 1, "lecture_title": "Foundations Course V2 L1", "lecture_subjects": ["Profitability", "Startup failure"],
        "english_transcription": _para("Profitability changes everything for a startup.", "Nine out of ten startups fail",
                                       "because they chase growth before profit and run out of money."),
        "hebrew_transcription": _para("רווחיות משנה הכל עבור סטארטאפ.", "תשע מתוך עשר חברות נכשלות כי הן רודפות אחרי צמיחה."),
        "english_summary": "This lecture explains why profitability matters more than growth for young startups and founders.",
        "hebrew_summary": "ההרצאה מסבירה למה רווחיות חשובה יותר מצמיחה עבור סטארטאפים צעירים ומייסדים.",
    },
    "course_v2/v2-l5.json": {
        "course_id": "smartup-foundations-v2", "course_name": "SmartUp Foundations", "course_version": "v2",
        "lecture_number": 5, "lecture_title": "Foundations Course V2 L5 1", "lecture_subjects": ["Pricing", "Branding"],
        "english_transcription": _para("Pricing is about capturing value, not about your costs.",
                                       "Do not be afraid to demand a high price from your first customers."),
        "hebrew_transcription": _para("תמחור הוא לכידת ערך ולא חישוב עלויות.", "אל תפחדו לדרוש מחיר גבוה מהלקוחות הראשונים."),
        "english_summary": "This lecture covers pricing strategy and branding before the product is ready for customers.",
        "hebrew_summary": "ההרצאה עוסקת באסטרטגיית תמחור ובמיתוג עוד לפני שהמוצר מוכן עבור הלקוחות.",
    },
    "course_v1/v1-l11a.json": {
        "course_id": "smartup-foundations-v1", "course_name": "SmartUp Foundations", "course_version": "v1",
        "lecture_number": 11, "lecture_title": "Foundations 11   Part 1   Yonatan   Nice Ai Transcript",
        "lecture_subjects": ["Lean startup", "Beachhead market"],
        "english_transcription": _para("A beachhead market is the first narrow market you can dominate.",
                                       "Lean startup means testing before building anything big."),
        "hebrew_transcription": _para("שוק ראש חוף הוא השוק הצר הראשון שאפשר לשלוט בו.", "סטארטאפ רזה פירושו לבדוק לפני שבונים."),
        "english_summary": "Part one discusses lean startup methods and choosing a beachhead market carefully.",
        "hebrew_summary": "החלק הראשון עוסק בשיטות סטארטאפ רזה ובבחירה זהירה של שוק ראש חוף.",
    },
    "course_v1/v1-l11b.json": {
        "course_id": "smartup-foundations-v1", "course_name": "SmartUp Foundations", "course_version": "v1",
        "lecture_number": 11, "lecture_title": "Foundations 11 Part 2 Oded  Nice Ai Transcript (1)",
        "lecture_subjects": ["Metaverse", "Guest lecture"],
        "english_transcription": _para("Oded talked about the metaverse and virtual worlds as a new market.",
                                       "Virtual goods can be sold to millions of users."),
        "hebrew_transcription": _para("עודד דיבר על המטאברס ועל עולמות וירטואליים כשוק חדש.", "אפשר למכור מוצרים וירטואליים."),
        "english_summary": "Part two is a guest lecture by Oded about the metaverse and virtual worlds as markets.",
        "hebrew_summary": "החלק השני הוא הרצאת אורח של עודד על המטאברס ועל עולמות וירטואליים כשווקים.",
    },
    "outsource/tedx.json": {
        "course_id": "smartup-tedx", "course_name": "SmartUp TEDx Talks", "course_version": "supplemental",
        "lecture_number": 1, "lecture_title": "Why fail if you can succeed? - TEDx", "lecture_subjects": ["Do not be first"],
        "english_transcription": _para("Do not be first to market: let others educate the market for you.",
                                       "Use off the shelf technology and disruptive business models."),
        "hebrew_transcription": _para("אל תהיו ראשונים לשוק, תנו לאחרים לחנך את השוק.", "השתמשו בטכנולוגיה מהמדף."),
        "english_summary": "The TEDx talk presents the core principles: profit early, marketing first, and do not be first.",
        "hebrew_summary": "הרצאת ה-TEDx מציגה את העקרונות: רווח מוקדם, שיווק קודם, ואל תהיו ראשונים.",
    },
    "outsource/di-l3.json": {
        "course_id": "`Digital Innovation Idc`", "course_name": "Digital Innovation Idc", "course_version": "v1",
        "lecture_slug": "lec03-digital-innovation-i", "lecture_number": 3, "lecture_title": "Digital Innovation IDC Lecture Transcript L3",
        "lecture_subjects": ["Hard disks", "Innovator's dilemma"],
        "english_transcription": _para("The hard disk industry shows the innovator's dilemma.",
                                       "Smaller disks looked worse until they were better for new customers."),
        "hebrew_transcription": _para("תעשיית הדיסקים הקשיחים מדגימה את הדילמה של החדשן.", "דיסקים קטנים נראו גרועים בהתחלה."),
        "english_summary": "The lecture uses the hard disk industry to explain the innovator's dilemma and disruption.",
        "hebrew_summary": "ההרצאה משתמשת בתעשיית הדיסקים הקשיחים כדי להסביר את הדילמה של החדשן ואת השיבוש.",
    },
}


def _write_export(root: Path, lectures: dict[str, dict]) -> Path:
    for folder in ("course_v2", "course_v1", "outsource"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for rel, doc in lectures.items():
        (root / rel).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return root


def _clear_caches() -> None:
    get_settings.cache_clear()
    store.get_index.cache_clear()
    embeddings.get_embedder.cache_clear()
    embeddings._query_vector.cache_clear()
    reranker.get_reranker.cache_clear()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temporary export + index directory, fake models, settings pointing at both."""
    data = _write_export(tmp_path / "export", LECTURES)
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("COLLECTION_NAME", "test_passages")
    monkeypatch.setenv("EMBED_MODEL", "fake-embed")
    monkeypatch.setenv("INCLUDE_DIGITAL_INNOVATION", "true")
    monkeypatch.setenv("CROSS_LINGUAL", "true")
    monkeypatch.setenv("MIN_RERANK_SCORE", "0.1")
    monkeypatch.setenv("TOP_K_PASSAGES", "4")
    monkeypatch.setenv("RANK_WINDOW", "30")
    _clear_caches()
    fake = FakeReranker()
    embedded: list[int] = []

    def fake_documents(texts, batch_size=None):
        embedded.append(len(texts))
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(hybrid, "embed_query", fake_vector)
    monkeypatch.setattr(indexer, "embed_documents", fake_documents)
    monkeypatch.setattr(hybrid, "get_reranker", lambda: fake)
    yield {"data": data, "index": tmp_path / "index", "reranker": fake, "embedded": embedded}
    store.get_index().close()
    _clear_caches()


def _ingest(**kw):
    out = indexer.ingest(**kw)
    store.get_index.cache_clear()  # what restarting the API does: the reader sees the new index
    return out


# ----------------------------------------------------------------------------------------
# ingestion
# ----------------------------------------------------------------------------------------
def test_ingest_builds_both_stores_consistently(env):
    out = _ingest()
    assert out["lectures"] == 6 and out["embedded"] == 6 and out["unchanged"] == 0
    assert out["passages"] == out["stored_passages"] == out["vectors"] > 12
    health = store.index_health(detail=True)
    assert health["ok"], health
    assert health["lectures"] == 6 and health["embed_model"] == "fake-embed"
    assert health["lectures_by_course"]["supplemental/digital-innovation-idc"] == 1


def test_reingest_skips_unchanged_lectures_and_follows_the_export(env):
    _ingest()
    again = _ingest()
    assert (again["embedded"], again["unchanged"], again["removed"], again["rebuilt"]) == (0, 6, 0, False)

    changed = dict(LECTURES["course_v2/v2-l5.json"], english_summary="A new summary about pricing and value capture.")
    (env["data"] / "course_v2/v2-l5.json").write_text(json.dumps(changed), encoding="utf-8")
    (env["data"] / "outsource/tedx.json").unlink()
    third = _ingest()
    assert (third["embedded"], third["unchanged"], third["removed"]) == (1, 4, 1)
    assert third["vectors"] == third["stored_passages"]
    assert [lec["tier"] for lec in hybrid_lectures() if lec["tier"] == "tedx"] == []


def hybrid_lectures(**kw):
    from smartup_rag.retrieval.retriever import get_lectures

    return get_lectures(**kw)


def test_another_embedding_model_forces_a_full_rebuild(env, monkeypatch):
    _ingest()
    monkeypatch.setenv("EMBED_MODEL", "another-model")
    get_settings.cache_clear()
    health = store.index_health()
    assert not health["ok"] and "EMBED_MODEL=another-model" in health["error"]
    out = _ingest()
    assert out["rebuilt"] and out["embedded"] == 6
    assert store.index_health()["ok"]


def test_an_inconsistent_index_is_reported_and_rebuilt(env):
    _ingest()
    store.get_index().collection().delete(ids=["v2:L5:en:-1:0"])  # a vector lost by an interrupted run
    health = store.index_health()
    assert not health["ok"] and "vectors for" in health["error"]
    out = _ingest()
    assert out["rebuilt"] and out["vectors"] == out["stored_passages"]


def test_dry_run_writes_nothing(env):
    out = _ingest(dry_run=True)
    assert out["dry_run"] and out["lectures"] == 6 and not env["embedded"]
    assert not store.index_health()["ok"]


def test_a_missing_index_is_reported_not_crashing(env):
    health = store.index_health()
    assert not health["ok"] and "smartup ingest" in health["error"]
    with pytest.raises(store.IndexMissingError):
        hybrid.search_passages("pricing", "en", "v2")


# ----------------------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(("tier", "keys"), [
    ("v2", {"v2:L1", "v2:L5"}),
    ("v1", {"v1:L11p1", "v1:L11p2"}),
    ("tedx", {"smartup-tedx:supplemental:L1"}),
    ("outsource", {"digital-innovation-idc:supplemental:L3"}),
    (None, None),
])
def test_every_leg_respects_the_source_tier(env, tier, keys):
    _ingest()
    stats: dict = {}
    cands = hybrid.search_passages("pricing profitability metaverse first market disks", "en", tier, stats=stats)
    assert cands and stats["legs"]["dense"] > 0 and stats["legs"]["bm25"] > 0
    found = {c.lecture_key for c in cands}
    if keys is None:
        assert len(found) == 6
    else:
        assert found == keys


def test_lecture_number_filter_and_both_languages(env):
    _ingest()
    cands = hybrid.search_passages("beachhead market", "en", "v1", 11)
    assert {c.lecture_key for c in cands} == {"v1:L11p1", "v1:L11p2"}
    assert {c.lang for c in cands} == {"en", "he"}  # cross-lingual: both transcripts are searched


def test_digital_innovation_can_be_hidden(env, monkeypatch):
    _ingest()
    monkeypatch.setenv("INCLUDE_DIGITAL_INNOVATION", "false")
    get_settings.cache_clear()
    assert hybrid.search_passages("hard disk innovator dilemma", "en", "outsource") == []


def test_keyword_legs_answer_when_the_embedding_model_is_down(env, monkeypatch):
    _ingest()

    def down(_text):
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(hybrid, "embed_query", down)
    stats: dict = {}
    cands = hybrid.search_passages("metaverse virtual worlds", "en", "v1", stats=stats)
    assert "ConnectionError" in stats["dense_error"] and stats["legs"]["dense"] == 0
    assert cands and cands[0].lecture_key == "v1:L11p2"


def test_hebrew_keywords_match_hebrew_passages(env):
    _ingest()
    cands = hybrid.search_passages("מה זה תמחור?", "he", "v2")
    assert cands[0].lecture_key == "v2:L5"


# ----------------------------------------------------------------------------------------
# retrieve → Documents
# ----------------------------------------------------------------------------------------
def test_retrieve_returns_reranked_documents_with_citations(env):
    _ingest()
    docs = hybrid.retrieve("Do not be afraid to demand a high price", "en", "v2")
    assert 0 < len(docs) <= 4 and env["reranker"].calls == 1
    m = docs[0].metadata
    assert m["lecture_key"] == "v2:L5" and m["citation"] == '(v2, Lecture 5 — "Foundations Course V2 L5 1")'
    assert m["rerank_score"] >= 0.1 and m["source_index"] == "local:test_passages" and m["tier"] == "v2"
    scores = [d.metadata["rerank_score"] for d in docs]
    assert scores == sorted(scores, reverse=True) and min(scores) >= 0.1  # MIN_RERANK_SCORE drops the noise


def test_lecture_part_is_a_soft_filter(env):
    _ingest()
    part2 = hybrid.retrieve("metaverse virtual worlds market", "en", "v1", 11, "2")
    assert part2 and {d.metadata["lecture_part"] for d in part2} == {"2"}
    assert part2[0].metadata["citation"] == '(v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")'


def test_retrieve_falls_back_to_rrf_order_when_the_reranker_fails(env, monkeypatch):
    _ingest()

    class Broken:
        def score(self, *_a):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(hybrid, "get_reranker", lambda: Broken())
    docs = hybrid.retrieve("pricing value", "en", "v2")
    assert docs and all(d.metadata["rerank_score"] is None for d in docs)  # → the LLM grader decides


def test_outside_sources_cite_in_their_own_shape(env):
    _ingest()
    docs = hybrid.retrieve("Why should you not be first to market?", "en", "tedx")
    assert docs[0].metadata["citation"] == '(TEDx — "Why fail if you can succeed? - TEDx")'
    docs = hybrid.retrieve("What do hard disks teach about the innovator's dilemma?", "en", "outsource")
    assert docs[0].metadata["citation"] == '(Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")'


# ----------------------------------------------------------------------------------------
# lecture catalogue
# ----------------------------------------------------------------------------------------
def test_lecture_catalogue_comes_from_sqlite(env):
    _ingest()
    lectures = hybrid_lectures()
    assert [lec["lecture_key"] for lec in lectures] == [  # course version, then number, then part
        "smartup-tedx:supplemental:L1", "digital-innovation-idc:supplemental:L3", "v1:L11p1", "v1:L11p2", "v2:L1", "v2:L5"]
    v2 = hybrid_lectures(tier="v2", lecture_number=5)
    assert len(v2) == 1 and v2[0]["english_summary"].startswith("This lecture covers pricing")
    assert v2[0]["display"] == "Lecture 5" and v2[0]["lecture_subjects"] == ["Pricing", "Branding"]
    assert "english_summary" not in hybrid_lectures(include_summaries=False)[0]
    tedx = hybrid_lectures(tier="tedx")
    assert tedx[0]["display"] == "" and tedx[0]["citation"] == '(TEDx — "Why fail if you can succeed? - TEDx")'


def test_transcript_lengths_cover_both_languages(env):
    _ingest()
    lengths = hybrid.transcript_lengths()
    assert lengths[("v2:L5", "en")] > 300 and lengths[("v2:L5", "he")] > 100
    assert ("v2:L5", "fr") not in lengths
