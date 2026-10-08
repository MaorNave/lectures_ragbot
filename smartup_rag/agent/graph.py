"""LangGraph wiring + the two entry points used by the CLI and the API.

    run_chat(message, thread_id)      → dict (blocking)
    stream_chat(message, thread_id)   → async iterator of {"type": "token"|"final", ...}

Multi-turn memory: the compiled graph carries a checkpointer keyed by `thread_id`
(MemorySaver by default — swap for langgraph-checkpoint-sqlite/postgres in production).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from smartup_rag.agent import nodes as N
from smartup_rag.agent.state import AgentState

log = logging.getLogger(__name__)


def build_graph(checkpointer: Any = None):
    g = StateGraph(AgentState)
    g.add_node("prepare", N.prepare)
    g.add_node("repeat", N.repeat)
    g.add_node("classify", N.classify)
    g.add_node("retrieve", N.retrieve_node)
    g.add_node("grade", N.grade_v2)
    g.add_node("fallback", N.fallback)
    g.add_node("generate", N.generate)
    g.add_node("verify", N.verify)
    g.add_node("refuse", N.refuse)
    g.add_node("chit_chat", N.chit_chat)
    g.add_node("not_found", N.not_found)

    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", N.route_after_prepare, {"repeat": "repeat", "classify": "classify"})
    g.add_conditional_edges("classify", N.route_after_classify, {"refuse": "refuse", "chit_chat": "chit_chat", "retrieve": "retrieve"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", N.route_after_grade, {"generate": "generate", "fallback": "fallback", "not_found": "not_found"}
    )
    g.add_conditional_edges("fallback", N.route_after_fallback, {"generate": "generate", "not_found": "not_found"})
    g.add_edge("generate", "verify")
    for terminal in ("verify", "refuse", "chit_chat", "not_found", "repeat"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _result(state: dict[str, Any]) -> dict[str, Any]:
    c = state.get("classification") or {}
    return {
        "answer": state.get("answer", ""),
        "status": state.get("status", "error"),
        "language": state.get("language"),
        "intent": c.get("intent"),
        "lecture_number": c.get("lecture_number"),
        "version_scope": c.get("version_scope"),
        "used_fallback": bool(state.get("used_fallback")),
        "fallback_tier": state.get("fallback_tier"),
        "fallback_tries": list(state.get("fallback_tries") or []),
        "grade_v2": (state.get("grade_v2") or {}).get("verdict"),
        "grade_v1": (state.get("grade_v1") or {}).get("verdict"),
        "grade_fallback": (state.get("grade_fallback") or {}).get("verdict"),
        "citations": state.get("citations", []),
        "n_passages": len(state.get("docs") or []),
    }


def _config(thread_id: str, stop_event: threading.Event | None = None) -> dict:
    cfg: dict[str, Any] = {"thread_id": thread_id}
    if stop_event is not None:
        cfg["stop_event"] = stop_event  # read by nodes.generate to stop the model when the client is gone
    return {"configurable": cfg}


def run_chat(message: str, thread_id: str = "default") -> dict[str, Any]:
    state = get_graph().invoke({"messages": [HumanMessage(content=message)]}, config=_config(thread_id))
    return _result(state)


async def stream_chat(
    message: str, thread_id: str = "default", stop_event: threading.Event | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield answer tokens as they are generated, then one 'final' event with the verified answer.

    While the fallback chain searches the TEDx talk, v1 and the outside lectures, 'status' events describe it
    ("Not found in the v2 course — checking Yonatan's TEDx talk…"), in the question's language.

    `stop_event`, when set by the caller (the API does this on client disconnect), makes the generate node
    stop pulling tokens from Ollama, which ends the model call instead of letting it run to completion.
    """
    graph = get_graph()
    cfg = _config(thread_id, stop_event)
    final: dict[str, Any] | None = None
    async for ev in graph.astream_events({"messages": [HumanMessage(content=message)]}, config=cfg, version="v2"):
        kind = ev.get("event")
        if kind == "on_chat_model_stream" and ev.get("metadata", {}).get("langgraph_node") == "generate":
            chunk = ev["data"]["chunk"]
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                yield {"type": "token", "text": text}
        elif kind == "on_custom_event" and ev.get("name") == "progress":  # dispatched by nodes._progress
            data = ev.get("data") or {}
            yield {"type": "status", "text": data.get("text", ""), "tier": data.get("tier")}
        elif kind == "on_chain_end" and ev.get("name") == "LangGraph":
            final = ev["data"].get("output")
    if not isinstance(final, dict):
        final = graph.get_state(cfg).values
    yield {"type": "final", **_result(final)}
