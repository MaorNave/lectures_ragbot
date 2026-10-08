"""Ingestion: builds the local index the chatbot searches.

    data/raw/drive_09_2026 (JSON) → lectures → passages (LangChain splitter) → bge-m3 vectors → Chroma + SQLite FTS5

Run it with `smartup ingest` (or `python -m smartup_rag.ingest.indexer`).
"""

from typing import Any

from smartup_rag.ingest.chunker import lecture_to_passages, lectures_to_passages
from smartup_rag.ingest.loaders import load_lectures

__all__ = [
    "content_hash",
    "ingest",
    "lecture_to_passages",
    "lectures_to_passages",
    "load_lectures",
]


def __getattr__(name: str) -> Any:
    # `indexer` is imported on first use, so `python -m smartup_rag.ingest.indexer` does not find it already imported
    if name in ("ingest", "content_hash"):
        from smartup_rag.ingest import indexer

        return getattr(indexer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
