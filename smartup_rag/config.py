"""Central configuration (12-factor style): every knob is an environment variable or `.env` entry.

Import `get_settings()` anywhere; it is cached so the `.env` file is parsed once.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama (local LLM) ---------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3.5:9b"
    chat_temperature: float = 0.1
    chat_num_ctx: int = 16384
    chat_think: bool = False  # disable "thinking" tokens: answers are grounded by retrieval, latency matters
    chat_timeout_s: float = 240.0
    warm_on_startup: bool = True  # prime the model + system-prompt KV cache (and load the retrieval models) at startup
    # How long Ollama keeps the model (and its cached prompt) loaded after the last call. An unload costs the next
    # visitor ~30 s (reload + cold prompt), so the default outlasts ordinary quiet periods. "-1" = until Ollama restarts.
    chat_keep_alive: str = "24h"
    # While no chat runs, the API re-primes the shared-prompt cache every N minutes and right after a call shows it
    # was lost (see agent.llm.note_prompt_timing). 0 disables both.
    keep_warm_minutes: float = 10.0
    classifier_model: str | None = None  # small model for scope/intent classification; None → chat_model

    # --- Local index: Chroma (vectors) + SQLite (passages, lecture catalogue, FTS5 BM25) ----------------
    # The only input of the index: the Drive "Naive" exports, one JSON document per lecture, in
    # course_v2/, course_v1/ and outsource/ below this folder.
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "drive_09_2026"
    # Built by `smartup ingest` (git-ignored): chroma/ holds the vectors, smartup.sqlite3 the passages, the lecture
    # catalogue and the FTS5 keyword index. Restart the API after rebuilding it.
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    collection_name: str = "smartup_passages"
    # Embeddings come from Ollama. bge-m3 is multilingual and cross-lingual (a Hebrew question finds English passages
    # and vice versa); changing the model means re-embedding everything (`smartup ingest --recreate`).
    embed_model: str = "bge-m3"
    embed_batch_size: int = 32
    embed_timeout_s: float = 120.0

    # --- Local cross-encoder reranker (sentence-transformers) -----------------------------
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # multilingual (Hebrew + English), Apache-2.0
    rerank_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"  # auto: CUDA, else Apple MPS, else CPU
    rerank_max_length: int = 512  # tokens of question + passage the cross-encoder reads
    rerank_batch_size: int = 16

    # --- Retrieval tuning ------------------------------------------------------
    top_k_passages: int = 8  # passages handed to the LLM after rerank
    rank_window: int = 30  # hits taken from each leg, and candidates fused by RRF and sent to the reranker
    rank_constant: int = 60  # RRF k
    cross_lingual: bool = True  # search passages of both languages (bge-m3 is cross-lingual); false = question's only
    # Thresholds on the cross-encoder's sigmoid score, calibrated for bge-reranker-v2-m3 (with its context line) against
    # the LLM grader's verdicts on 202 question/source pairs (README §10). A different RERANK_MODEL needs its own.
    min_rerank_score: float = 0.02  # passages below this rerank score are dropped as noise
    # Grading fast path (instructions §8 "strong match"): rerank score of the best passage. In the calibration, over the
    # decisions the agent makes, the grader accepted nothing below 0.042 and rejected nothing above 0.61.
    grade_strong_threshold: float = 0.70  # ≥ → the source is sufficient without asking the LLM
    grade_weak_threshold: float = 0.03  # < → nothing here; go to the next source without asking the LLM
    max_passage_chars: int = 2500  # truncation before rerank / prompt
    # Fallback-chain speed-ups (the v2 → TEDx → v1 → outsource order itself never changes)
    prefetch_fallback: bool = True  # search the fallback sources in the background while v2 is being graded
    # A fallback source whose best rerank score is below this is rejected without an LLM grader call, but only when a
    # later source already has a strong match (≥ grade_strong_threshold). 0 turns the rule off. Set just below the
    # lowest score the grader accepted in the calibration — v1 at 0.046, the outside lectures at 0.065, the TEDx talk
    # at 0.096 — so the rule only ever skips a source the grader would have rejected. Never applied to v2.
    skip_weak_grade_below: float = 0.04
    skip_weak_grade_below_tedx: float = 0.09

    # --- Ingestion ---------------------------------------------------------------------------
    instructions_path: Path = PROJECT_ROOT / "instructions.md"
    include_digital_innovation: bool = True  # the Digital Innovation IDC lectures are part of the "outsource" tier
    passage_chars: int = 1200
    passage_overlap: int = 200

    # --- API ------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # NoDecode: pydantic-settings would JSON-decode a list field before the validator runs and reject "*" or "a,b"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])  # "*" is a development default; set your site in prod
    max_history_turns: int = 6
    api_docs_enabled: bool = True  # expose /docs and /openapi.json (disable on a public host)
    # /dashboard: per-call pipeline traces (questions, passages, prompts). "local" = only requests from this machine,
    # "on" = anyone who can reach the API, "off" = no tracing at all. Behind a reverse proxy on the same host every
    # request looks local — use "off" there.
    debug_dashboard: Literal["local", "on", "off"] = "local"
    chat_rate_limit_per_minute: int = 12  # per client IP, across /chat and /chat/stream; 0 disables
    max_request_bytes: int = 65_536  # a 2 000-char message needs a few KB; larger bodies are rejected before parsing
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        """Accept `*`, a comma-separated list (as documented) or a JSON list."""
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                return json.loads(s)
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    @field_validator("data_dir", "index_dir", "instructions_path", mode="before")
    @classmethod
    def _absolutize(cls, v):
        p = Path(v).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def sqlite_path(self) -> Path:
        return self.index_dir / "smartup.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.index_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
