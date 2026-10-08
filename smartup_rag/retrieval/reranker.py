"""Local cross-encoder reranking (sentence-transformers `CrossEncoder`, default `BAAI/bge-reranker-v2-m3`).

A cross-encoder reads the question and one passage *together* and scores how well the passage answers it — far
more precise than comparing two embeddings, and far too slow to run over the whole index, so it only re-scores the
RRF candidates (`RANK_WINDOW`, default 30). Scores are sigmoid outputs in [0, 1]; the agent's grading fast path
compares the best one with `GRADE_STRONG_THRESHOLD` / `GRADE_WEAK_THRESHOLD`.

bge-reranker-v2-m3 is multilingual (Hebrew and English, also across the two) and Apache-2.0 licensed. On an Apple
M1 Pro it scores 30 passages in ~2 s on the GPU (MPS, float16) and ~8 s on the CPU. The weights (~2.3 GB) are
downloaded from the Hugging Face Hub on first use (`smartup models` does it ahead of time) and read from the local
cache afterwards, without network access.

torch and sentence-transformers are imported only when the model is first needed, so the CLI and the tests never
pay for them.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Any

from smartup_rag.config import get_settings

log = logging.getLogger(__name__)

RETRY_LOAD_AFTER_S = 300.0  # a failed load (offline and not cached, out of memory) is not retried on every question


def resolve_device(preference: str = "auto") -> str:
    """"auto" → "cuda" when available, else "mps" (Apple silicon), else "cpu"; anything else is taken as given."""
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """Lazily loaded, thread-safe wrapper around one `CrossEncoder`.

    The model is loaded on first use (or by `load()`, which the API calls in the background at startup). Scoring is
    serialised with a lock: the fallback chain searches several source tiers in parallel threads, and one model on
    one GPU gains nothing from concurrent calls.
    """

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.device: str | None = None
        self.load_seconds: float | None = None
        self.error: str | None = None
        self._failed_at = 0.0
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if self.error and time.monotonic() - self._failed_at < RETRY_LOAD_AFTER_S:
                    raise RuntimeError(f"reranker unavailable: {self.error}")
                try:
                    self._model = self._load()
                    self.error = None
                except Exception as exc:
                    self.error, self._failed_at = f"{type(exc).__name__}: {exc}", time.monotonic()
                    raise
        return self._model

    def _load(self) -> Any:
        import torch
        from sentence_transformers import CrossEncoder

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()  # "Loading weights: 100%|…" would land in the API log on every start
        except Exception:  # cosmetic only
            pass
        t0 = time.perf_counter()
        device = resolve_device(self.device_preference)
        kwargs: dict[str, Any] = {"device": device, "max_length": self.max_length}
        if device in ("cuda", "mps"):
            kwargs["model_kwargs"] = {"dtype": torch.float16}  # same scores as float32 (±0.001), half the memory
        try:  # the local cache first: no network round trip on every start once the weights are there
            model = CrossEncoder(self.model_name, local_files_only=True, **kwargs)
        except Exception:
            log.info("reranker %s is not cached yet; downloading it from the Hugging Face Hub (~2.3 GB once)", self.model_name)
            model = CrossEncoder(self.model_name, **kwargs)
        self.device = device
        self.load_seconds = round(time.perf_counter() - t0, 1)
        log.info("reranker %s loaded on %s in %.1fs", self.model_name, device, self.load_seconds)
        return model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in input order (sigmoid scores in [0, 1])."""
        if not texts:
            return []
        model = self.load()
        with self._lock:
            scores = model.predict(
                [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
        return [float(x) for x in scores]

    def status(self) -> dict[str, Any]:
        return {"model": self.model_name, "device": self.device or self.device_preference, "loaded": self.loaded,
                "load_s": self.load_seconds, "error": self.error}


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    s = get_settings()
    return CrossEncoderReranker(s.rerank_model, s.rerank_device, s.rerank_max_length, s.rerank_batch_size)


def is_cached(model_name: str) -> bool:
    """Are the model's weights in the local Hugging Face cache (no network access needed)?"""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        hit = try_to_load_from_cache(model_name, name)
        if isinstance(hit, str):
            return True
    return False
