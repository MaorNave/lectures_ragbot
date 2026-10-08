"""Text embeddings from Ollama (`EMBED_MODEL`, default bge-m3).

bge-m3 is multilingual *and* cross-lingual: a Hebrew question lands next to the English passage that says the same
thing, which is what lets one search cover both transcripts of every lecture. Ollama returns L2-normalised
vectors; the Chroma collection uses cosine distance.

The model runs in the same Ollama server as the chat model (~0.7 GB next to qwen3.5:9b's ~6 GB) and is kept loaded
for `CHAT_KEEP_ALIVE`, so a question never pays a model load. Query vectors are cached per process: the fallback
chain searches up to four source tiers with the same question.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from smartup_rag.config import get_settings

log = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def keep_alive_seconds(raw: str) -> int | None:
    """"24h" → 86400, "30m" → 1800, "-1" → -1 (keep loaded). OllamaEmbeddings takes whole seconds only;
    an unparsable value → None (Ollama's default of 5 minutes)."""
    m = _DURATION_RE.match(raw or "")
    if not m:
        return None
    return int(float(m.group(1)) * _SECONDS[m.group(2).lower()])


@lru_cache
def get_embedder() -> OllamaEmbeddings:
    s = get_settings()
    return OllamaEmbeddings(
        model=s.embed_model,
        base_url=s.ollama_base_url,
        keep_alive=keep_alive_seconds(s.chat_keep_alive),
        client_kwargs={"timeout": s.embed_timeout_s},
    )


@lru_cache(maxsize=512)
def _query_vector(text: str) -> tuple[float, ...]:
    return tuple(get_embedder().embed_query(text))


def embed_query(text: str) -> list[float]:
    """One question → one vector (cached per process)."""
    return list(_query_vector(text))


def embed_documents(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Passages → vectors, in batches of `EMBED_BATCH_SIZE` (one Ollama request each)."""
    size = max(1, int(batch_size or get_settings().embed_batch_size))
    out: list[list[float]] = []
    embedder = get_embedder()
    for i in range(0, len(texts), size):
        out.extend(embedder.embed_documents(texts[i : i + size]))
    return out
