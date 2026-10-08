"""ChatOllama factory.

One process-wide client per role. `reasoning=False` sends Ollama's `think: false`, which
suppresses the hidden chain-of-thought of Qwen3/Qwen3.5/Gemma4 models: answers are grounded
by retrieved passages, so the extra latency of "thinking" buys little here.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Literal

from langchain_ollama import ChatOllama

from smartup_rag.config import get_settings

log = logging.getLogger(__name__)

Role = Literal["chat", "classifier", "grader"]

# Ollama runs Qwen3.5 through llama.cpp, which can resume a cached prompt only from a saved checkpoint. The warm-up's
# short prompt leaves one just inside the shared instructions; a model reload, or any prompt that does not start with
# them, destroys it — and from then on every call re-reads all its tokens (classifier ~29 s instead of ~6 s, answer
# prompt 35–49 s instead of 11–20 s). A classifier call that slow sets this flag; the API re-primes when idle.
COLD_CLASSIFIER_PROMPT_S = 15.0
_cold_cache = threading.Event()


def keep_alive_value(raw: str) -> int | str:
    """Ollama takes a duration ("24h") or a number of seconds; a negative NUMBER keeps the model loaded forever,
    while the string "-1" is rejected as a duration without a unit — so plain numbers are sent as numbers."""
    raw = raw.strip()
    return int(raw) if raw.lstrip("-").isdigit() else raw


@lru_cache
def get_llm(role: Role = "chat") -> ChatOllama:
    s = get_settings()
    model = s.chat_model
    temperature = s.chat_temperature
    num_predict = 1200
    if role != "chat":
        model = s.classifier_model or s.chat_model
        temperature = 0.0
        num_predict = 400
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        num_ctx=s.chat_num_ctx,
        num_predict=num_predict,
        reasoning=None if s.chat_think else False,
        keep_alive=keep_alive_value(s.chat_keep_alive),
        client_kwargs={"timeout": s.chat_timeout_s},
    )


def shares_chat_prefix() -> bool:
    """True when classifier/grader run on the chat model, so they can reuse its cached prompt prefix.

    With a separate CLASSIFIER_MODEL there is no shared cache, and sending the ~6 000-token system
    prompt to that model would cost a full prefill for nothing — the short task prompt is used instead.
    """
    s = get_settings()
    return not s.classifier_model or s.classifier_model == s.chat_model


def note_prompt_timing(message: Any) -> None:
    """Flag a lost shared-prompt cache from a classifier reply (Ollama reports prompt time in response_metadata)."""
    meta = getattr(message, "response_metadata", None) or {}
    seconds = (meta.get("prompt_eval_duration") or 0) / 1e9
    if seconds > COLD_CLASSIFIER_PROMPT_S and shares_chat_prefix():
        if not _cold_cache.is_set():
            log.warning("classifier prompt took %.1fs (cached ≈ 6 s): shared-prompt cache lost, re-priming when idle", seconds)
        _cold_cache.set()


def cache_needs_warm_up() -> bool:
    return _cold_cache.is_set()


def warm_up() -> float:
    """Load the model and prime Ollama's KV cache for the shared system-prompt prefix.

    Every LLM call in the pipeline (classifier, grader, generator) starts with the same ~6 000-token
    system prompt, and Ollama reuses the cached prefix of the previous request. Cold, that prefix costs
    ~27 s to evaluate; warm, ~2.4 s. Without this, the first visitor after a restart or an idle period
    pays the cold price even for a one-line scope refusal. Returns the elapsed seconds.
    """
    import time

    from langchain_core.messages import HumanMessage, SystemMessage

    from smartup_rag.agent.prompts import system_prompt

    t0 = time.perf_counter()
    llm = get_llm("chat").model_copy(update={"num_predict": 1})  # same num_ctx, so no model reload
    llm.invoke([SystemMessage(content=system_prompt()), HumanMessage(content="Reply with one word: ready.")])
    _cold_cache.clear()
    return time.perf_counter() - t0


def ollama_health() -> dict:
    """Is Ollama up, and are the configured chat and embedding models pulled?"""
    import httpx

    s = get_settings()
    try:
        r = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}

        def pulled(model: str) -> bool:
            return model in names or f"{model}:latest" in names

        return {"ok": True, "chat_model": s.chat_model, "model_present": pulled(s.chat_model),
                "embed_model": s.embed_model, "embed_model_present": pulled(s.embed_model), "model_count": len(names)}
    except Exception as exc:
        return {"ok": False, "chat_model": s.chat_model, "embed_model": s.embed_model,
                "error": f"{type(exc).__name__}: {exc}"}
