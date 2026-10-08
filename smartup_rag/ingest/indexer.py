"""Build the local index from the Drive exports: lectures → passages → embeddings → Chroma + SQLite.

    data/raw/drive_09_2026/*/*.json ─► loaders (normalise) ─► chunker (~1 200-character passages, English and
    Hebrew, plus one summary passage per language) ─► Ollama bge-m3 embeddings ─► Chroma (vectors)
                                                                                + SQLite (passages, lecture catalogue, FTS5)

Incremental: every lecture is stored with a content hash of its normalised text and metadata, the chunking settings,
the embedding model and the index schema. A re-run skips unchanged lectures, re-embeds changed ones and removes the
lectures that are no longer in the export. `--recreate` drops the index first. An index built with another
embedding model, chunking or schema — or whose two stores disagree after an interrupted run — is rebuilt from
scratch: vectors of two models must never be mixed.

CLI: `smartup ingest [--recreate] [--dry-run]` (or `python -m smartup_rag.ingest.indexer`, which adds
`--batch-size` and `--log-level`). Restart the API afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smartup_rag.config import PROJECT_ROOT, get_settings
from smartup_rag.ingest.chunker import lecture_to_passages
from smartup_rag.ingest.loaders import load_lectures
from smartup_rag.retrieval.embeddings import embed_documents
from smartup_rag.retrieval.store import SCHEMA_VERSION, LocalIndex
from smartup_rag.schemas import Lecture

log = logging.getLogger(__name__)


def build_settings() -> dict[str, str]:
    """What an index's vectors and passages depend on; stored in its `meta` table."""
    s = get_settings()
    return {
        "schema_version": str(SCHEMA_VERSION),
        "embed_model": s.embed_model,
        "passage_chars": str(s.passage_chars),
        "passage_overlap": str(s.passage_overlap),
    }


def content_hash(lecture: Lecture) -> str:
    """Changes whenever the lecture's passages or vectors would change."""
    payload = {"build": build_settings(), "lecture": lecture.model_dump(exclude={"source_file"})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ingest(
    recreate: bool = False,
    *,
    data_dir: Path | None = None,
    include_digital_innovation: bool | None = None,
    dry_run: bool = False,
    batch_size: int | None = None,
    index: LocalIndex | None = None,
) -> dict[str, Any]:
    """Load → chunk → embed → write. Returns a summary dict.

    `dry_run=True` only loads and chunks (no Ollama call, nothing written).
    """
    s = get_settings()
    t0 = time.monotonic()
    root = Path(data_dir) if data_dir is not None else s.data_dir
    lectures = load_lectures(root, include_digital_innovation)
    chunks = {lec.lecture_key: lecture_to_passages(lec) for lec in lectures}
    n_passages = sum(len(ps) for ps in chunks.values())
    log.info("%d lectures → %d passages (data dir %s)", len(lectures), n_passages, root)
    summary: dict[str, Any] = {
        "data_dir": _relative(root),
        "lectures": len(lectures),
        "passages": n_passages,
        "dry_run": dry_run,
    }
    if dry_run:
        summary["seconds"] = round(time.monotonic() - t0, 1)
        return summary

    idx = index or LocalIndex(s.index_dir, s.collection_name)  # a fresh instance: never the API's cached reader
    build = build_settings()
    rebuilt = recreate
    if recreate:
        log.warning("dropping the index at %s", idx.index_dir)
        idx.reset()
    conn = idx.connect_writer()
    try:
        stored = idx.read_meta(conn)
        changed = sorted(k for k, v in build.items() if k in stored and stored[k] != v)
        n_vectors = idx.collection(create=True).count()
        n_rows = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        if changed or n_vectors != n_rows:
            reason = f"built with different {', '.join(changed)}" if changed else f"{n_vectors} vectors for {n_rows} passages"
            log.warning("index %s (%s): rebuilding it from scratch", idx.index_dir, reason)
            conn.close()
            idx.reset()
            conn = idx.connect_writer()
            rebuilt = True

        hashes = idx.stored_hashes(conn)
        todo = [lec for lec in lectures if hashes.get(lec.lecture_key) != content_hash(lec)]
        stale = sorted(set(hashes) - set(chunks))
        if stale:
            log.info("removing %d lecture(s) no longer in the export: %s", len(stale), ", ".join(stale))
            idx.delete_lectures(conn, stale)
        if todo:
            embed_documents(["warm-up"], batch_size=1)  # fail fast (and load the model) before the long loop
        for i, lec in enumerate(todo, 1):
            t = time.monotonic()
            passages = chunks[lec.lecture_key]
            vectors = embed_documents([p.text for p in passages], batch_size=batch_size)
            idx.write_lecture(conn, lec, passages, vectors, content_hash=content_hash(lec), source_file=lec.source_file)
            log.info("[%d/%d] %-34s %4d passages embedded and stored in %.1fs", i, len(todo), lec.lecture_key,
                     len(passages), time.monotonic() - t)
        idx.write_meta(conn, {**build, "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                              "data_dir": _relative(root), "lectures": str(len(lectures)), "passages": str(n_passages)})
        vectors_now = idx.collection(create=True).count()
        rows_now = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    finally:
        conn.close()

    summary.update(
        index=_relative(idx.index_dir),
        collection=idx.collection_name,
        embed_model=s.embed_model,
        embedded=len(todo),
        unchanged=len(lectures) - len(todo),
        removed=len(stale),
        rebuilt=rebuilt,
        vectors=vectors_now,
        stored_passages=rows_now,
        seconds=round(time.monotonic() - t0, 1),
    )
    if vectors_now != rows_now:  # should never happen; health checks would report it too
        log.error("index inconsistent after ingest: %d vectors for %d passages", vectors_now, rows_now)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m smartup_rag.ingest.indexer",
        description="Build the local index (Chroma + SQLite FTS5) from the Drive exports in DATA_DIR.",
    )
    parser.add_argument("--recreate", action="store_true", help="drop the index and rebuild everything")
    parser.add_argument("--data-dir", type=Path, default=None, help="export root (default: DATA_DIR)")
    parser.add_argument("--batch-size", type=int, default=None, help="passages per embedding request (default: EMBED_BATCH_SIZE)")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk only; do not embed or write")
    parser.add_argument("--log-level", default=None, help="override LOG_LEVEL (e.g. DEBUG)")
    args = parser.parse_args(argv)

    s = get_settings()
    logging.basicConfig(level=(args.log_level or s.log_level).upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summary = ingest(recreate=args.recreate, data_dir=args.data_dir, batch_size=args.batch_size, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("dry_run") or summary.get("vectors") == summary.get("stored_passages") else 1


if __name__ == "__main__":
    raise SystemExit(main())
