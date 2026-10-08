"""The local index: Chroma (passage vectors) + SQLite (passages, lecture catalogue, FTS5 keyword index).

    data/index/
      chroma/            Chroma persistent client — collection `COLLECTION_NAME` (cosine; bge-m3 vectors).
                         id = passage_id; metadata = the fields filters need (course_id, course_version,
                         lecture_number, lecture_key, lang, section_index). No text: SQLite holds it.
      smartup.sqlite3    lectures       one row per lecture: metadata, both summaries, source file, content hash
                         passages       one row per passage: metadata, text, character offsets
                         passages_fts   FTS5 (unicode61 tokenizer) over text / section topic / subjects / title —
                                        an external-content index kept in sync with `passages` by triggers
                         meta           how the index was built: schema version, embedding model, chunking, time

SQLite is the source of truth for passages; Chroma only answers "which vectors are nearest". `smartup ingest`
writes a lecture to Chroma first and commits it to SQLite last, with the lecture's content hash: a crash in
between leaves the old hash in SQLite, and the next run simply redoes that lecture.

Readers use one read-only SQLite connection per thread (the API answers in a thread pool, and the fallback chain
searches source tiers in parallel threads); Chroma's client is shared. Rebuilding the index while the API runs is
not picked up by it — restart the API after `smartup ingest`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from smartup_rag.config import get_settings
from smartup_rag.schemas import Lecture, Passage

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SQLITE_NAME = "smartup.sqlite3"
CHROMA_DIR = "chroma"
CHROMA_ADD_BATCH = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lectures (
    lecture_key      TEXT PRIMARY KEY,
    course_id        TEXT NOT NULL,
    course_name      TEXT NOT NULL,
    course_version   TEXT NOT NULL,
    lecture_number   INTEGER NOT NULL,
    lecture_part     TEXT,
    lecture_title    TEXT NOT NULL,
    lecture_subjects TEXT NOT NULL,   -- JSON list
    lecturer_name    TEXT NOT NULL,
    english_summary  TEXT NOT NULL,
    hebrew_summary   TEXT NOT NULL,
    source_file      TEXT NOT NULL,   -- relative to DATA_DIR
    content_hash     TEXT NOT NULL,
    passages         INTEGER NOT NULL,
    indexed_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
    rowid            INTEGER PRIMARY KEY,
    passage_id       TEXT NOT NULL UNIQUE,
    lecture_key      TEXT NOT NULL,
    course_id        TEXT NOT NULL,
    course_name      TEXT NOT NULL,
    course_version   TEXT NOT NULL,
    lecture_number   INTEGER NOT NULL,
    lecture_part     TEXT,
    lecture_title    TEXT NOT NULL,
    lecture_subjects TEXT NOT NULL,   -- JSON list
    subjects_text    TEXT NOT NULL,   -- the subjects as one string, for the FTS metadata leg
    section_index    INTEGER NOT NULL,
    section_topic    TEXT NOT NULL,
    lang             TEXT NOT NULL,
    position         INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    text             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_by_lecture ON passages(lecture_key);
CREATE INDEX IF NOT EXISTS passages_by_source ON passages(course_version, course_id, lecture_number);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text, section_topic, subjects_text, lecture_title,
    content='passages', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS passages_fts_insert AFTER INSERT ON passages BEGIN
    INSERT INTO passages_fts(rowid, text, section_topic, subjects_text, lecture_title)
    VALUES (new.rowid, new.text, new.section_topic, new.subjects_text, new.lecture_title);
END;
CREATE TRIGGER IF NOT EXISTS passages_fts_delete AFTER DELETE ON passages BEGIN
    INSERT INTO passages_fts(passages_fts, rowid, text, section_topic, subjects_text, lecture_title)
    VALUES ('delete', old.rowid, old.text, old.section_topic, old.subjects_text, old.lecture_title);
END;
"""
_DROP = """
DROP TRIGGER IF EXISTS passages_fts_insert;
DROP TRIGGER IF EXISTS passages_fts_delete;
DROP TABLE IF EXISTS passages_fts;
DROP TABLE IF EXISTS passages;
DROP TABLE IF EXISTS lectures;
DROP TABLE IF EXISTS meta;
"""

PASSAGE_COLUMNS = (
    "passage_id", "lecture_key", "course_id", "course_name", "course_version", "lecture_number", "lecture_part",
    "lecture_title", "lecture_subjects", "subjects_text", "section_index", "section_topic", "lang", "position",
    "char_start", "char_end", "text",
)
LECTURE_COLUMNS = (
    "lecture_key", "course_id", "course_name", "course_version", "lecture_number", "lecture_part", "lecture_title",
    "lecture_subjects", "lecturer_name", "english_summary", "hebrew_summary", "source_file", "content_hash",
    "passages", "indexed_at",
)


class IndexMissingError(RuntimeError):
    """The local index has not been built (or not completely): run `smartup ingest`."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def chroma_metadata(p: Passage) -> dict[str, Any]:
    """What Chroma stores next to a vector: the fields a filter can name (no None values)."""
    return {
        "lecture_key": p.lecture_key,
        "course_id": p.course_id,
        "course_version": p.course_version,
        "lecture_number": int(p.lecture_number),
        "lang": p.lang,
        "section_index": int(p.section_index),
    }


def passage_row(p: Passage) -> tuple[Any, ...]:
    subjects = list(p.lecture_subjects)
    return (
        p.passage_id, p.lecture_key, p.course_id, p.course_name, p.course_version, int(p.lecture_number),
        p.lecture_part, p.lecture_title, json.dumps(subjects, ensure_ascii=False), " ; ".join(subjects),
        int(p.section_index), p.section_topic, p.lang, int(p.position), int(p.char_start), int(p.char_end), p.text,
    )


class LocalIndex:
    """One index directory: a Chroma collection plus the SQLite database next to it."""

    def __init__(self, index_dir: Path | str, collection_name: str = "smartup_passages") -> None:
        self.index_dir = Path(index_dir)
        self.sqlite_path = self.index_dir / SQLITE_NAME
        self.chroma_path = self.index_dir / CHROMA_DIR
        self.collection_name = collection_name
        self._local = threading.local()
        self._lock = threading.Lock()
        self._client: Any = None
        self._collection: Any = None
        self._lengths: dict[tuple[str, str], int] | None = None

    # ------------------------------------------------------------------ connections
    def exists(self) -> bool:
        return self.sqlite_path.is_file() and self.chroma_path.is_dir()

    def _read(self) -> sqlite3.Connection:
        """This thread's read-only connection (opened on first use)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.sqlite_path.is_file():
                raise IndexMissingError(f"no index at {self.index_dir}: build it with `smartup ingest`")
            conn = sqlite3.connect(self.sqlite_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0,
                                   check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def connect_writer(self) -> sqlite3.Connection:
        """A read-write connection with the schema in place (ingestion only)."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def _chroma_client(self) -> Any:
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path),
                                                     settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def collection(self, *, create: bool = False) -> Any:
        """The passages collection. `create=True` (ingestion) makes it when missing; readers never create one."""
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    if not create and not self.chroma_path.is_dir():
                        raise IndexMissingError(f"no vector store at {self.chroma_path}: build it with `smartup ingest`")
                    client = self._chroma_client()
                    if create:
                        self._collection = client.get_or_create_collection(
                            self.collection_name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
                        )
                    else:
                        try:
                            self._collection = client.get_collection(self.collection_name, embedding_function=None)
                        except Exception as exc:
                            raise IndexMissingError(
                                f"collection {self.collection_name!r} not found in {self.chroma_path}: run `smartup ingest`"
                            ) from exc
        return self._collection

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ search
    def dense_search(self, vector: Sequence[float], where: dict[str, Any] | None, n: int) -> list[tuple[str, float]]:
        """Nearest passages to `vector` among those matching `where` → [(passage_id, cosine similarity)], best first."""
        res = self.collection().query(query_embeddings=[list(vector)], n_results=max(1, n), where=where,
                                      include=["distances"])
        ids, dists = (res.get("ids") or [[]])[0], (res.get("distances") or [[]])[0]
        return [(pid, 1.0 - float(d)) for pid, d in zip(ids, dists, strict=False)]

    def keyword_search(self, match: str, weights: Sequence[float], where_sql: str, params: Sequence[Any],
                       n: int) -> list[tuple[str, float]]:
        """FTS5 BM25 → [(passage_id, score)], best first; `score` is -bm25(), so higher is better."""
        w = ", ".join(f"{float(x):g}" for x in weights)
        sql = (
            f"SELECT p.passage_id, bm25(passages_fts, {w}) AS s FROM passages_fts "
            "JOIN passages p ON p.rowid = passages_fts.rowid WHERE passages_fts MATCH ?"
            + (f" AND {where_sql}" if where_sql else "")
            + " ORDER BY s, p.rowid LIMIT ?"
        )
        rows = self._read().execute(sql, (match, *params, int(n))).fetchall()
        return [(r[0], -float(r[1])) for r in rows]

    def passages(self, passage_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(passage_ids))
        if not ids:
            return {}
        marks = ", ".join("?" for _ in ids)
        rows = self._read().execute(f"SELECT * FROM passages WHERE passage_id IN ({marks})", ids).fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            d["lecture_subjects"] = json.loads(d["lecture_subjects"] or "[]")
            out[d["passage_id"]] = d
        return out

    def lectures(self, where_sql: str = "", params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Lecture catalogue rows (with both summaries), ordered by course version, number, part and title."""
        sql = ("SELECT * FROM lectures l" + (f" WHERE {where_sql}" if where_sql else "")
               + " ORDER BY l.course_version, l.lecture_number, COALESCE(l.lecture_part, ''), l.lecture_title")
        out = []
        for r in self._read().execute(sql, tuple(params)).fetchall():
            d = dict(r)
            d["lecture_subjects"] = json.loads(d["lecture_subjects"] or "[]")
            out.append(d)
        return out

    def transcript_lengths(self) -> dict[tuple[str, str], int]:
        """(lecture_key, lang) → transcript length (largest passage end offset); read once per process."""
        if self._lengths is None:
            rows = self._read().execute(
                "SELECT lecture_key, lang, MAX(char_end) FROM passages WHERE section_index >= 0 GROUP BY lecture_key, lang"
            ).fetchall()
            self._lengths = {(r[0], r[1]): int(r[2] or 0) for r in rows}
        return self._lengths

    def meta(self) -> dict[str, str]:
        try:
            return {r[0]: r[1] for r in self._read().execute("SELECT key, value FROM meta").fetchall()}
        except sqlite3.Error:
            return {}

    def stats(self) -> dict[str, Any]:
        """Counts of both stores, for health checks (raises IndexMissingError when the index is not built)."""
        conn = self._read()
        lectures = conn.execute("SELECT COUNT(*) FROM lectures").fetchone()[0]
        passages = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        by_tier_rows = conn.execute(
            "SELECT course_version, course_id, COUNT(*) FROM lectures GROUP BY course_version, course_id"
        ).fetchall()
        return {
            "lectures": int(lectures),
            "passages": int(passages),
            "vectors": int(self.collection().count()),
            "lectures_by_course": {f"{r[0]}/{r[1]}": int(r[2]) for r in by_tier_rows},
            "meta": self.meta(),
        }

    # ------------------------------------------------------------------ writing (ingestion)
    def reset(self) -> None:
        """Drop everything: the Chroma collection and every SQLite table."""
        client = self._chroma_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:  # nothing to delete
            log.debug("collection %s did not exist", self.collection_name)
        self._collection = None
        if self.sqlite_path.is_file():
            conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
            try:
                conn.executescript(_DROP)
                conn.commit()
            finally:
                conn.close()
        self._lengths = None

    @staticmethod
    def stored_hashes(conn: sqlite3.Connection) -> dict[str, str]:
        return {r[0]: r[1] for r in conn.execute("SELECT lecture_key, content_hash FROM lectures").fetchall()}

    @staticmethod
    def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
        return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta").fetchall()}

    @staticmethod
    def write_meta(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
        with conn:
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in values.items()],
            )

    def write_lecture(
        self,
        conn: sqlite3.Connection,
        lecture: Lecture,
        passages: Sequence[Passage],
        vectors: Sequence[Sequence[float]],
        *,
        content_hash: str,
        source_file: str,
    ) -> None:
        """Replace one lecture in both stores: vectors first, then the SQLite rows and the content hash in one
        transaction (the hash is what tells the next run the lecture is done)."""
        if len(passages) != len(vectors):
            raise ValueError(f"{lecture.lecture_key}: {len(passages)} passages but {len(vectors)} vectors")
        col = self.collection(create=True)
        col.delete(where={"lecture_key": lecture.lecture_key})
        for i in range(0, len(passages), CHROMA_ADD_BATCH):
            chunk = passages[i : i + CHROMA_ADD_BATCH]
            col.upsert(ids=[p.passage_id for p in chunk], embeddings=[list(v) for v in vectors[i : i + CHROMA_ADD_BATCH]],
                       metadatas=[chroma_metadata(p) for p in chunk])
        with conn:
            conn.execute("DELETE FROM passages WHERE lecture_key = ?", (lecture.lecture_key,))
            conn.execute("DELETE FROM lectures WHERE lecture_key = ?", (lecture.lecture_key,))
            conn.execute(
                f"INSERT INTO lectures({', '.join(LECTURE_COLUMNS)}) VALUES ({', '.join('?' for _ in LECTURE_COLUMNS)})",
                (
                    lecture.lecture_key, lecture.course_id, lecture.course_name, lecture.course_version,
                    int(lecture.lecture_number), lecture.lecture_part, lecture.lecture_title,
                    json.dumps(list(lecture.lecture_subjects), ensure_ascii=False), lecture.lecturer_name,
                    lecture.english_summary, lecture.hebrew_summary, source_file, content_hash, len(passages), _now(),
                ),
            )
            conn.executemany(
                f"INSERT INTO passages({', '.join(PASSAGE_COLUMNS)}) VALUES ({', '.join('?' for _ in PASSAGE_COLUMNS)})",
                [passage_row(p) for p in passages],
            )
        self._lengths = None

    def delete_lectures(self, conn: sqlite3.Connection, lecture_keys: Iterable[str]) -> None:
        keys = sorted(set(lecture_keys))
        if not keys:
            return
        col = self.collection(create=True)
        col.delete(where={"lecture_key": {"$in": keys}})
        marks = ", ".join("?" for _ in keys)
        with conn:
            conn.execute(f"DELETE FROM passages WHERE lecture_key IN ({marks})", keys)
            conn.execute(f"DELETE FROM lectures WHERE lecture_key IN ({marks})", keys)
        self._lengths = None


@lru_cache
def get_index() -> LocalIndex:
    """The process-wide index the retriever reads (paths from settings)."""
    s = get_settings()
    return LocalIndex(s.index_dir, s.collection_name)


def index_health(*, detail: bool = False) -> dict[str, Any]:
    """Is the index built, complete (one vector per passage) and built with the configured embedding model?

    `detail=True` (the CLI) adds paths and build metadata; the API's /health leaves them out.
    """
    s = get_settings()
    idx = get_index()
    out: dict[str, Any] = {"ok": False, "collection": s.collection_name}
    if detail:
        out["path"] = str(idx.index_dir)
    if not idx.exists():
        out["error"] = "index not built: run `smartup ingest`"
        return out
    try:
        st = idx.stats()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    meta = st.pop("meta")
    out.update(lectures=st["lectures"], passages=st["passages"], vectors=st["vectors"],
               embed_model=meta.get("embed_model"))
    if detail:
        out.update(lectures_by_course=st["lectures_by_course"], built_at=meta.get("built_at"),
                   schema_version=meta.get("schema_version"), passage_chars=meta.get("passage_chars"),
                   passage_overlap=meta.get("passage_overlap"))
    problems = []
    if not st["passages"]:
        problems.append("the index is empty: run `smartup ingest`")
    if st["vectors"] != st["passages"]:
        problems.append(f"{st['vectors']} vectors for {st['passages']} passages: run `smartup ingest --recreate`")
    if meta.get("embed_model") and meta.get("embed_model") != s.embed_model:
        problems.append(f"built with {meta.get('embed_model')} but EMBED_MODEL={s.embed_model}: run `smartup ingest --recreate`")
    if meta.get("schema_version") and meta.get("schema_version") != str(SCHEMA_VERSION):
        problems.append("built by another version of this code: run `smartup ingest --recreate`")
    out["ok"] = not problems
    if problems:
        out["error"] = "; ".join(problems)
    return out
