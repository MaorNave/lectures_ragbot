"""Hybrid retrieval: dense (Chroma) + BM25 (SQLite FTS5) + metadata legs → RRF → local cross-encoder → Documents.

    question ─┬─ bge-m3 embedding ──► Chroma nearest passages ─────────────┐
              ├─ FTS5 BM25 over the passage text ──────────────────────────┼─► RRF (top RANK_WINDOW)
              └─ FTS5 BM25 over section topic / lecture subjects / title ──┘          │
                                                                                      ▼
            LangChain Documents ◄─ drop cross-language twins ◄─ cross-encoder rerank (top TOP_K_PASSAGES)

Every leg applies the same filters (source tier, lecture number), so the agent's source order (v2 → TEDx → v1 →
outside lectures) is enforced inside the search, not after it. The output shape (`Candidate` → `Document`) is the
one the agent has always read, with the exact citation string in the metadata.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from smartup_rag import trace
from smartup_rag.config import get_settings
from smartup_rag.lang import Lang
from smartup_rag.retrieval import queries as Q
from smartup_rag.retrieval.embeddings import embed_query
from smartup_rag.retrieval.reranker import get_reranker
from smartup_rag.retrieval.store import IndexMissingError, get_index
from smartup_rag.tiers import FOUNDATIONS_TIERS, citation_for, source_label, tier_of

log = logging.getLogger(__name__)

_PART_RE = re.compile(r"\bpart\s*(\d)\b", re.IGNORECASE)
# The metadata leg scores every passage of a lecture alike (same subjects, same title); this many hits are read so
# that ties can be broken by the dense leg before the leg is cut to RANK_WINDOW.
METADATA_POOL = 400


def lecture_part_from_title(title: str) -> str | None:
    """v1 Lecture 11 is split in two documents whose titles carry 'Part 1' / 'Part 2' (instructions §5)."""
    m = _PART_RE.search(title or "")
    return m.group(1) if m else None


def make_lecture_key(course_version: str, lecture_number: int, part: str | None) -> str:
    return f"{course_version}:L{lecture_number}" + (f"p{part}" if part else "")


def lecture_key_for(course_id: str, course_version: str, lecture_number: int, part: str | None) -> str:
    """'v2:L3' / 'v1:L11p2' for the Foundations course; any other course is prefixed with its course_id
    ('smartup-tedx:supplemental:L1'), so outside lectures of the same number never share a key."""
    key = make_lecture_key(course_version, lecture_number, part)
    return key if tier_of(course_id, course_version) in FOUNDATIONS_TIERS else f"{course_id}:{key}"


def display_lecture(lecture_number: int, part: str | None) -> str:
    return f"Lecture {lecture_number}" + (f" Part {part}" if part else "")


@dataclass
class Candidate:
    text: str
    course_id: str
    course_version: str
    lecture_number: int
    lecture_title: str
    lang: Lang
    score: float  # RRF score
    lecture_part: str | None = None
    course_name: str = ""
    lecture_subjects: list[str] = field(default_factory=list)
    rerank_score: float | None = None
    section_topic: str = ""
    source_index: str = ""
    lecture_key_override: str | None = None  # the index stores the ingest-time key (course-prefixed for non-Foundations)
    # where the passage sits in its lecture (section -1 = lecture summary)
    section_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    passage_id: str = ""

    @property
    def lecture_key(self) -> str:
        return self.lecture_key_override or lecture_key_for(self.course_id, self.course_version, self.lecture_number,
                                                            self.lecture_part)

    @property
    def tier(self) -> str:
        return tier_of(self.course_id, self.course_version)

    @property
    def citation(self) -> str:
        return citation_for(self.course_id, self.course_version, self.lecture_number, self.lecture_part, self.lecture_title,
                            self.course_name)

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "lecture_key": self.lecture_key,
                "course_id": self.course_id,
                "course_version": self.course_version,
                "tier": self.tier,
                "source_label": source_label(self.course_id, self.course_version, self.course_name),
                "lecture_number": self.lecture_number,
                "lecture_part": self.lecture_part,
                "lecture_title": self.lecture_title,
                "lecture_subjects": list(self.lecture_subjects),
                "lang": self.lang,
                "score": self.score,
                "rerank_score": self.rerank_score,
                "section_topic": self.section_topic,
                "source_index": self.source_index,
                "citation": self.citation,
            },
        )


def candidate_from_row(row: dict[str, Any], score: float, max_chars: int, source_index: str) -> Candidate:
    return Candidate(
        text=(row.get("text") or "").strip()[:max_chars],
        course_id=row.get("course_id") or "",
        course_name=row.get("course_name") or "",
        course_version=row.get("course_version") or "",
        lecture_number=int(row.get("lecture_number") or 0),
        lecture_part=row.get("lecture_part") or None,
        lecture_title=(row.get("lecture_title") or "").strip(),
        lecture_subjects=list(row.get("lecture_subjects") or []),
        lang=row.get("lang") or "en",
        score=score,
        section_topic=row.get("section_topic") or "",
        source_index=source_index,
        lecture_key_override=row.get("lecture_key") or None,
        section_index=row.get("section_index"),
        char_start=row.get("char_start"),
        char_end=row.get("char_end"),
        passage_id=row.get("passage_id") or "",
    )


# --------------------------------------------------------------------------------------
# The three legs + fusion
# --------------------------------------------------------------------------------------
def break_ties(hits: list[tuple[str, float]], prefer: list[str]) -> list[str]:
    """Order keyword hits by score; equal scores (every passage of one lecture, in the metadata leg) by their rank in
    `prefer` (the dense leg), then as they came. Returns the ids."""
    pos = {pid: i for i, pid in enumerate(prefer)}
    order = {pid: i for i, (pid, _) in enumerate(hits)}
    return [pid for pid, _ in sorted(hits, key=lambda h: (-round(h[1], 9), pos.get(h[0], len(pos)), order[h[0]]))]


def search_passages(
    query: str,
    lang: Lang,
    tier: str | None = None,
    lecture_number: int | None = None,
    *,
    stats: dict[str, Any] | None = None,
) -> list[Candidate]:
    """One hybrid search → up to RANK_WINDOW fused candidates (RRF order, not reranked yet).

    A failing leg degrades the search instead of failing it: without Ollama's embedding model the keyword legs
    still answer, and the reason is recorded in `stats` (shown on the dashboard).
    """
    s = get_settings()
    idx = get_index()
    stats = stats if stats is not None else {}
    window = s.rank_window
    conds = Q.conditions(
        tier, lecture_number, None, None if s.cross_lingual else lang,
        exclude_course_ids=[] if s.include_digital_innovation else [Q.DIGITAL_INNOVATION_COURSE_ID],
    )

    dense: list[tuple[str, float]] = []
    t = time.perf_counter()
    try:
        vector = embed_query(query)
        stats["embed_ms"] = int((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        dense = idx.dense_search(vector, Q.chroma_where(conds), window)
        stats["dense_ms"] = int((time.perf_counter() - t) * 1000)
    except IndexMissingError:
        raise
    except Exception as exc:  # Ollama down or the embedding model not pulled: keep the keyword legs
        stats["dense_error"] = f"{type(exc).__name__}: {exc}"
        log.warning("dense leg skipped (%s: %s)", type(exc).__name__, exc)
    dense_ids = [pid for pid, _ in dense]

    terms = Q.query_terms(query)
    where_sql, params = Q.sql_where(conds)
    bm25_ids: list[str] = []
    meta_ids: list[str] = []
    t = time.perf_counter()
    if terms:
        bm25 = idx.keyword_search(Q.fts_match(terms, ("text",)), Q.BM25_TEXT_WEIGHTS, where_sql, params, window)
        bm25_ids = [pid for pid, _ in bm25]
        meta = idx.keyword_search(Q.fts_match(terms, Q.METADATA_COLUMNS), Q.BM25_METADATA_WEIGHTS, where_sql, params,
                                  METADATA_POOL)
        meta_ids = break_ties(meta, dense_ids)[:window]
    stats["keyword_ms"] = int((time.perf_counter() - t) * 1000)
    stats["legs"] = {"dense": len(dense_ids), "bm25": len(bm25_ids), "metadata": len(meta_ids)}

    fused = Q.rrf_fuse([dense_ids, bm25_ids, meta_ids], s.rank_constant, window)
    rows = idx.passages(pid for pid, _ in fused)
    source = f"local:{s.collection_name}"
    return [candidate_from_row(rows[pid], score, s.max_passage_chars, source) for pid, score in fused if pid in rows]


# --------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------
def dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for c in cands:
        key = (c.lecture_key, re.sub(r"\s+", " ", c.text[:120]).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --------------------------------------------------------------------------------------
# Cross-language duplicates
# --------------------------------------------------------------------------------------
# Every lecture exists in English and Hebrew, so the top passages often hold the same stretch of a lecture twice —
# once per language — and the model then reads it twice. A passage in the other language than the question is
# dropped when its translation in the question's language is already in the list.
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]+|\d[\d,.]*\d|\d")


def latin_tokens(text: str) -> set[str]:
    """Names, numbers and English terms: the Latin-script tokens a Hebrew transcript keeps untranslated."""
    return {t.lower().strip(".,'’-") for t in _LATIN_RE.findall(text or "")}


def transcript_lengths() -> dict[tuple[str, str], int]:
    """(lecture_key, lang) → transcript length in the index (largest passage end offset); read once per process."""
    return get_index().transcript_lengths()


def is_translation(a: Candidate, b: Candidate, lengths: dict[tuple[str, str], int]) -> bool:
    """Are `a` and `b` (one English, one Hebrew) the same stretch of the same lecture?

    Two cheap signals must agree. Checked against cross-lingual embedding cosine on the eval set: none of the 23
    pairs below cosine 0.70 passed both, 22 of the 26 pairs above 0.80 did.
      * position: both cover the same part of the lecture (relative offsets overlap, allowing 2 % drift);
      * names/numbers: at least 75 % of the Hebrew passage's Latin-script tokens (at least 3) occur in the English one.
    """
    if a.lecture_key != b.lecture_key or a.lang == b.lang:
        return False
    he, en = (a, b) if a.lang == "he" else (b, a)
    if (he.section_index is None) or (en.section_index is None):
        return False
    if (he.section_index < 0) != (en.section_index < 0):  # a lecture summary is never a transcript passage
        return False
    if he.section_index >= 0:
        # some v1 Hebrew transcripts cover only the end of the lecture (a fifth of the English length, not ~¾):
        # relative positions do not line up there, so no pair is called a translation
        he_total, en_total = lengths.get((he.lecture_key, "he")) or 0, lengths.get((en.lecture_key, "en")) or 0
        if not he_total or not en_total or he_total / en_total < 0.5:
            return False
        spans = []
        for c in (he, en):
            total = lengths.get((c.lecture_key, c.lang)) or 0
            if not total or c.char_start is None or c.char_end is None:
                return False
            spans.append((c.char_start / total - 0.02, c.char_end / total + 0.02))
        if min(spans[0][1], spans[1][1]) <= max(spans[0][0], spans[1][0]):
            return False
    marks = latin_tokens(he.text)
    return len(marks) >= 3 and len(marks & latin_tokens(en.text)) / len(marks) >= 0.75


def drop_translations(cands: list[Candidate], lang: Lang, lengths: dict[tuple[str, str], int]) -> list[Candidate]:
    """Drop an other-language passage whose translation in `lang` is also in the list.

    The kept twin inherits the dropped one's rerank score when that was higher — it is the same content, and the
    best score drives the grading fast path, so removing a duplicate must never send a question to the LLM grader.
    """
    same = [c for c in cands if c.lang == lang]
    kept: list[Candidate] = []
    for c in cands:
        twin = None if c.lang == lang else next((t for t in same if is_translation(c, t, lengths)), None)
        if twin is None:
            kept.append(c)
        elif c.rerank_score is not None and (twin.rerank_score is None or c.rerank_score > twin.rerank_score):
            twin.rerank_score = c.rerank_score
    return sorted(kept, key=lambda c: -(c.rerank_score or 0.0))


# Every passage is Yonatan Stern's own words, but a transcript rarely says so. Without that context a cross-encoder
# reads "What does Yonatan say about pricing?" literally and prefers any passage that names him over the lecture that
# teaches pricing. One line naming the speaker and the source, read with the passage, fixes it. Measured on the eval
# set with bge-reranker-v2-m3 (README §3): the weakest best score of a source that answers a question rose from 0.03
# to 0.19, while sources the LLM grader had rejected stayed below 0.13.
LECTURER = "Yonatan Stern (יונתן שטרן)"


def rerank_header(c: Candidate) -> str:
    """The context line the cross-encoder reads before a passage (never shown to the answer model)."""
    tier = c.tier
    lec = f"Lecture {c.lecture_number}" + (f" Part {c.lecture_part}" if c.lecture_part else "")
    if tier in FOUNDATIONS_TIERS:
        source = f"SmartUp Foundations course ({tier}), {lec}"
    elif tier == "tedx":
        source = "TEDx talk"
    elif c.course_id == Q.DIGITAL_INNOVATION_COURSE_ID:
        source = f"Digital Innovation IDC course, {lec}"
    else:
        source = "talk"
    return f"{LECTURER}, {source}:\n"


def rerank(query: str, cands: list[Candidate], top_n: int) -> list[Candidate]:
    """Cross-encoder rerank with the local model; falls back to RRF order when it is disabled or cannot run."""
    s = get_settings()
    if not cands:
        return []
    if not s.rerank_enabled:
        return cands[:top_n]
    try:
        scores = get_reranker().score(query, [rerank_header(c) + c.text for c in cands])
    except Exception as exc:  # a missing model or an out-of-memory GPU must never break the chat
        log.warning("rerank failed (%s: %s); keeping RRF order", type(exc).__name__, exc)
        return cands[:top_n]
    for c, sc in zip(cands, scores, strict=True):
        c.rerank_score = sc
    ranked = sorted(cands, key=lambda c: -(c.rerank_score or 0.0))
    return [c for c in ranked if (c.rerank_score or 0.0) >= s.min_rerank_score][:top_n]


def _passage_view(c: Candidate, rank: int | None = None) -> dict:
    """A candidate as the dashboard shows it (full text: the passages ARE what the model reads)."""
    return {
        "rank": rank,
        "lecture_key": c.lecture_key,
        "citation": c.citation,
        "lang": c.lang,
        "section_topic": c.section_topic,
        "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
        "rrf_score": round(c.score, 5),
        "chars": len(c.text),
        "text": c.text,
    }


def retrieve(
    query: str,
    lang: Lang,
    tier: str | None = None,
    lecture_number: int | None = None,
    lecture_part: str | None = None,
) -> list[Document]:
    """One retrieval pass over one source tier (v2 / tedx / v1 / outsource; None = all) → reranked Documents.

    `lecture_part` ("1"/"2", v1 Lecture 11 only) is a *soft* filter applied before reranking: candidates of
    the requested part win when any exist, but a mis-detected part degrades to the unfiltered pool instead
    of producing a spurious not-found.
    """
    s = get_settings()
    with trace.span(f"search {tier or 'all'}") as sp:
        sp.update(
            index=f"local:{s.collection_name}",
            query=query,
            lang=lang,
            filters={"tier": tier, "lecture_number": lecture_number, "lecture_part": lecture_part},
            legs=[f"dense: {s.embed_model} → Chroma (both languages)" if s.cross_lingual
                  else f"dense: {s.embed_model} → Chroma ({lang} only)",
                  "BM25: SQLite FTS5 over the passage text", "metadata: FTS5 over section topic / lecture subjects / title"],
        )
        stats: dict[str, Any] = {}
        t = time.perf_counter()
        cands = search_passages(query, lang, tier, lecture_number, stats=stats)
        sp["search_ms"] = int((time.perf_counter() - t) * 1000)
        sp.update(stats)
        sp["candidates"] = len(cands)
        cands = dedupe(cands)
        sp["after_dedupe"] = len(cands)
        if lecture_part:
            exact = [c for c in cands if c.lecture_part == lecture_part]
            cands = exact or cands
        t = time.perf_counter()
        top = rerank(query, cands, s.top_k_passages)
        sp["rerank_ms"] = int((time.perf_counter() - t) * 1000)
        sp["reranker"] = s.rerank_model if s.rerank_enabled else None
        sp["reranked_kept"] = len(top)
        dropped: list[Candidate] = []
        if any(c.lang != lang for c in top):
            try:
                deduped = drop_translations(top, lang, transcript_lengths())
                kept_ids = {id(c) for c in deduped}
                dropped, top = [c for c in top if id(c) not in kept_ids], deduped
            except Exception as exc:  # a failed length lookup only costs the saving, never the answer
                log.warning("translation dedupe skipped (%s: %s)", type(exc).__name__, exc)
        sp["passages"] = [_passage_view(c, i) for i, c in enumerate(top, 1)]
        sp["dropped_translations"] = [_passage_view(c) for c in dropped]
        sp["thresholds"] = {"min_rerank_score": s.min_rerank_score, "top_k_passages": s.top_k_passages}
    log.info(
        "retrieve lang=%s tier=%s lecture=%s candidates=%d kept=%d dropped_translations=%d",
        lang,
        tier,
        lecture_number,
        len(cands),
        len(top),
        len(dropped),
    )
    return [c.to_document() for c in top]
