"""FastAPI service for the SmartUp Academy assistant.

Endpoints
    GET  /health        Ollama (chat + embedding model), local index and reranker status (never raises)
    GET  /lectures      lecture catalogue grouped by source tier (v2 → TEDx → v1 → outsource)
    POST /chat          one blocking chat turn → ChatResponse
    POST /chat/stream   same, as Server-Sent Events: `token`* → `final` (or `error`)
    GET  /widget/       static demo page + embeddable widget (web/)
    GET  /dashboard     pipeline dashboard (web/dashboard.html); GET /debug/traces[/{id}] feed it (DEBUG_DASHBOARD)

Run:  .venv/bin/python -m smartup_rag.api.app [--host H] [--port P] [--reload]

Concurrency: the local Ollama model is effectively single-tenant, so chat calls pass through an
`asyncio.Semaphore(2)`. A request that cannot take a slot immediately is answered with
HTTP 429 {"detail": "busy"} instead of queueing for minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from smartup_rag import __version__, trace
from smartup_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    HealthResponse,
    LectureGroup,
    LectureOut,
)
from smartup_rag.config import PROJECT_ROOT, Settings, get_settings
from smartup_rag.tiers import TIER_ORDER

log = logging.getLogger("smartup_rag.api")

WEB_DIR = PROJECT_ROOT / "web"
MAX_CONCURRENT_CHATS = 2
KEEP_WARM_CHECK_S = 20.0  # how often the keep-warm loop looks for a cold cache or an overdue re-prime
SSE_KEEPALIVE_S = 15.0  # idle gap after which an SSE comment is sent so proxies keep the socket open
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _configure_logging(level: str) -> None:
    """Root logger setup that does not clobber uvicorn's own handlers when they exist."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level.upper())
    logging.getLogger("smartup_rag").setLevel(level.upper())


def _new_thread_id(requested: str | None) -> str:
    return requested or str(uuid.uuid4())


def _sse(event: str, data: Any) -> str:
    """One Server-Sent Event frame. `data` must be JSON-serialisable."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


class _RateLimiter:
    """Sliding one-minute window per client IP, in memory (single process; put a real limiter at the proxy for scale)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

    def allow(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > 60.0:
            q.popleft()
        if not q:  # a client gone quiet leaves no entry behind
            del self._hits[client]
            q = self._hits[client]
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _dashboard_allowed(s: Settings, request: Request) -> bool:
    """/dashboard shows every visitor's questions and the model's inputs: this machine only, unless set to "on"."""
    return s.debug_dashboard == "on" or (s.debug_dashboard == "local" and _client_ip(request) in _LOCAL_HOSTS)


async def _try_acquire(sem: asyncio.Semaphore) -> bool:
    """Take a semaphore slot without waiting; False when every slot is busy.

    `locked()` and `acquire()` run back-to-back on the event loop: when a slot is free CPython's
    `Semaphore.acquire()` decrements the counter and returns without suspending, so the
    check-then-take pair cannot be interleaved with another coroutine.
    """
    if sem.locked():
        return False
    await sem.acquire()
    return True


async def _safe_health(fn, label: str) -> dict[str, Any]:
    try:
        out = await run_in_threadpool(fn)
        return out if isinstance(out, dict) else {"ok": True, "value": out}
    except Exception as exc:
        log.warning("health check %s failed: %s: %s", label, type(exc).__name__, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _warm_up_model(reason: str = "startup") -> None:
    """Prime the model and the shared prompt prefix so the next visitor is not charged the cold start."""
    from smartup_rag.agent.llm import warm_up

    try:
        seconds = await run_in_threadpool(warm_up)
        log.info("model warm-up (%s) finished in %.1fs", reason, seconds)
    except Exception as exc:
        log.warning("model warm-up (%s) failed (%s: %s); the next request will be slower", reason, type(exc).__name__, exc)


async def _warm_up_retrieval() -> None:
    """Load the cross-encoder and the embedding model, so the first question does not wait for them (~5–10 s)."""
    from smartup_rag.retrieval.embeddings import embed_query
    from smartup_rag.retrieval.reranker import get_reranker

    s = get_settings()
    t0 = time.perf_counter()
    try:
        await run_in_threadpool(embed_query, "warm-up")
        if s.rerank_enabled:
            await run_in_threadpool(get_reranker().load)
        log.info("retrieval models ready in %.1fs", time.perf_counter() - t0)
    except Exception as exc:
        log.warning("retrieval warm-up failed (%s: %s); the first question will load the models", type(exc).__name__, exc)


async def _keep_warm(app: FastAPI, every_s: float) -> None:
    """Keep the model loaded and its shared-prompt cache primed, but only while no chat is running.

    llama.cpp resumes a cached prompt only from a saved checkpoint, and the one every call relies on comes from the
    warm-up. A model reload, or any prompt that does not start with the shared instructions, destroys it; from then
    on each call re-reads its ~7 000–11 000 tokens (+~50 s per question) until something re-primes. So re-prime right
    after the classifier reports a cold cache, and every `every_s` seconds anyway (which also keeps the model loaded).
    """
    from smartup_rag.agent.llm import cache_needs_warm_up

    last = time.monotonic()
    while True:
        await asyncio.sleep(KEEP_WARM_CHECK_S)
        cold = cache_needs_warm_up()
        if (cold or time.monotonic() - last >= every_s) and app.state.active_chats == 0:
            await _warm_up_model("cold cache" if cold else "keep-warm")
            last = time.monotonic()
            if cache_needs_warm_up():  # the warm-up failed (Ollama down?): back off instead of retrying every check
                await asyncio.sleep(every_s)


TIER_LABELS = {
    "v2": "Foundations v2 (primary)",
    "tedx": "TEDx talk (fallback 1)",
    "v1": "Foundations v1 (fallback 2)",
    "outsource": "Other lectures outside the course (fallback 3)",
}


def _group_lectures(rows: list[dict[str, Any]]) -> list[LectureGroup]:
    groups: dict[str, list[LectureOut]] = {}
    for r in rows:
        groups.setdefault(str(r.get("tier") or r.get("course_version") or ""), []).append(
            LectureOut(
                lecture_number=int(r.get("lecture_number") or 0),
                lecture_part=r.get("lecture_part") or None,
                lecture_title=str(r.get("lecture_title") or ""),
                display=str(r.get("display") or ""),
                citation=str(r.get("citation") or ""),
                lecture_subjects=list(r.get("lecture_subjects") or []),
            )
        )
    order = {t: i for i, t in enumerate(TIER_ORDER)}
    return [LectureGroup(course_version=t, tier=t, label=TIER_LABELS.get(t, t), lectures=ls)
            for t, ls in sorted(groups.items(), key=lambda kv: order.get(kv[0], len(order)))]


# ----------------------------------------------------------------------------------------
# app factory
# ----------------------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    _configure_logging(s.log_level)

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        # Warm up in the background: the server must accept requests (and serve /health) immediately.
        tasks = []
        if s.warm_on_startup:
            tasks.append(asyncio.create_task(_warm_up_retrieval(), name="warm-up-retrieval"))
            tasks.append(asyncio.create_task(_warm_up_model("startup"), name="warm-up"))
        if s.keep_warm_minutes > 0:
            tasks.append(asyncio.create_task(_keep_warm(app_, s.keep_warm_minutes * 60), name="keep-warm"))
        try:
            yield
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    app = FastAPI(
        lifespan=lifespan,
        title="SmartUp Academy Assistant API",
        version=__version__,
        description="Chatbot that answers only from the SmartUp Academy lecture transcripts (HE / EN).",
        responses={429: {"model": ErrorDetail, "description": "The assistant is busy"}},
        docs_url="/docs" if s.api_docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if s.api_docs_enabled else None,
    )
    app.state.settings = s
    app.state.chat_slots = asyncio.Semaphore(MAX_CONCURRENT_CHATS)
    app.state.active_chats = 0  # chats holding a slot; the keep-warm loop only runs when this is 0
    app.state.rate_limiter = _RateLimiter(s.chat_rate_limit_per_minute)
    if "*" in s.cors_origins and s.api_host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "CORS_ORIGINS is '*' while binding %s: any website could drive this model. Set CORS_ORIGINS to your site.",
            s.api_host,
        )

    # -- middleware (added inner → outer; CORS last so error responses carry CORS headers too)
    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        if request.method == "POST":  # reject oversized / unbounded bodies before FastAPI buffers and parses them
            length = request.headers.get("content-length")
            if length is None and request.headers.get("transfer-encoding", "").lower() == "chunked":
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            try:
                if length is not None and int(length) > s.max_request_bytes:
                    return JSONResponse({"detail": "request body too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("%s %s → unhandled error", request.method, request.url.path)
            raise
        ms = (time.perf_counter() - t0) * 1000
        is_static_asset = request.url.path.startswith("/widget/") and request.url.path != "/widget/"
        if not is_static_asset:  # keep the log readable: skip css/js asset hits
            log.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code, ms)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(s.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=600,
    )

    # -- routes ------------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        from smartup_rag.agent.llm import ollama_health
        from smartup_rag.retrieval.reranker import get_reranker
        from smartup_rag.retrieval.store import index_health

        ollama, index = await asyncio.gather(_safe_health(ollama_health, "ollama"), _safe_health(index_health, "index"))
        reranker = {"enabled": s.rerank_enabled, **get_reranker().status()}
        for part in (ollama, index, reranker):  # internals stay in the log; the page only needs to know it is not working
            if part.get("error"):
                part["error"] = "unavailable"
        ok = (bool(ollama.get("ok")) and bool(ollama.get("model_present", True))
              and bool(ollama.get("embed_model_present", True)) and bool(index.get("ok")))
        return HealthResponse(
            status="ok" if ok else "degraded",
            ollama=ollama,
            index=index,
            reranker=reranker,
            chat_model=s.chat_model,
            embed_model=s.embed_model,
        )

    @app.get("/lectures", response_model=list[LectureGroup])
    async def lectures() -> list[LectureGroup]:
        from smartup_rag.retrieval.retriever import get_lectures

        try:
            rows = await run_in_threadpool(get_lectures, include_summaries=False)
        except Exception as exc:
            log.exception("GET /lectures failed")
            detail = f"lecture catalogue unavailable: {type(exc).__name__}"
            raise HTTPException(status_code=503, detail=detail) from exc
        return _group_lectures(rows)

    @app.post("/chat", response_model=ChatResponse, responses={429: {"model": ErrorDetail}})
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        from smartup_rag.agent.graph import run_chat

        if not app.state.rate_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limited")
        sem: asyncio.Semaphore = app.state.chat_slots
        if not await _try_acquire(sem):
            raise HTTPException(status_code=429, detail="busy")
        app.state.active_chats += 1
        thread_id = _new_thread_id(body.thread_id)
        tr = trace.new_trace(body.message, thread_id, "/chat") if s.debug_dashboard != "off" else None
        t0 = time.perf_counter()
        try:
            with trace.active(tr):  # run_in_threadpool copies the context, so the agent's steps land in `tr`
                result = await run_in_threadpool(run_chat, body.message, thread_id)
        except Exception as exc:
            log.exception("POST /chat failed (thread=%s)", thread_id)
            if tr:
                trace.finish(tr, error=f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=500, detail=f"chat failed: {type(exc).__name__}") from exc
        finally:
            app.state.active_chats -= 1
            sem.release()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if tr:
            trace.finish(tr, result)
        resp = ChatResponse.from_agent(result, thread_id=thread_id, elapsed_ms=elapsed_ms)
        if tr and _dashboard_allowed(s, request):
            resp.trace_id = tr.id
        log.info(
            "chat thread=%s status=%s lang=%s intent=%s %d ms",
            thread_id, resp.status, resp.language, resp.intent, elapsed_ms,
        )
        return resp

    @app.post(
        "/chat/stream",
        responses={200: {"content": {"text/event-stream": {}}}, 429: {"model": ErrorDetail}},
    )
    async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        from smartup_rag.agent.graph import stream_chat

        if not app.state.rate_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limited")
        sem: asyncio.Semaphore = app.state.chat_slots
        if not await _try_acquire(sem):  # decide *before* the 200 + headers go out
            raise HTTPException(status_code=429, detail="busy")
        app.state.active_chats += 1
        thread_id = _new_thread_id(body.thread_id)
        message = body.message
        stop_event = threading.Event()  # tells the generate node to stop pulling tokens if the client leaves
        tr = trace.new_trace(message, thread_id, "/chat/stream") if s.debug_dashboard != "off" else None
        show_trace = bool(tr) and _dashboard_allowed(s, request)

        async def event_source() -> AsyncIterator[str]:
            t0 = time.perf_counter()
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def pump() -> None:
                """Drain the agent stream into a queue so keep-alives can be interleaved safely.

                Owns the concurrency slot: it is released here, only once the agent run has really ended.
                A client that disconnects cannot free the slot while the model is still generating in its
                worker thread — the stop_event ends that generation early instead.
                """
                try:
                    with trace.active(tr):  # the agent's worker threads inherit this task's context
                        async for ev in stream_chat(message, thread_id, stop_event):
                            if tr and ev.get("type") == "token" and tr.first_token_ms is None:
                                tr.first_token_ms = tr.elapsed_ms()
                            elif tr and ev.get("type") == "final":
                                if tr.first_token_ms is None:  # nothing was streamed: the answer appeared at once
                                    tr.first_token_ms = tr.elapsed_ms()
                                trace.finish(tr, ev)
                            await queue.put(ev)
                except Exception as exc:
                    log.exception("POST /chat/stream failed (thread=%s)", thread_id)
                    if tr:
                        trace.finish(tr, error=f"{type(exc).__name__}: {exc}")
                    await queue.put({"type": "error", "message": f"chat failed: {type(exc).__name__}"})
                finally:
                    app.state.active_chats -= 1
                    sem.release()
                    await queue.put(None)

            task = asyncio.create_task(pump(), name=f"chat-stream-{thread_id}")
            n_tokens = 0
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if ev is None:
                        break
                    kind = ev.get("type")
                    if kind == "token":
                        n_tokens += 1
                        yield _sse("token", {"text": ev.get("text", "")})
                    elif kind == "final":
                        elapsed_ms = int((time.perf_counter() - t0) * 1000)
                        resp = ChatResponse.from_agent(ev, thread_id=thread_id, elapsed_ms=elapsed_ms)
                        if show_trace:
                            resp.trace_id = tr.id
                        log.info(
                            "chat/stream thread=%s status=%s lang=%s intent=%s tokens=%d %d ms",
                            thread_id, resp.status, resp.language, resp.intent, n_tokens, elapsed_ms,
                        )
                        yield _sse("final", resp.model_dump())
                    elif kind == "status":  # progress of the fallback chain, shown in the typing bubble
                        yield _sse("status", {"text": ev.get("text", ""), "tier": ev.get("tier")})
                    elif kind == "error":
                        yield _sse("error", {"message": ev.get("message", "chat failed")})
            finally:
                if not task.done():  # client gone mid-answer: stop the model; pump() releases the slot when it ends
                    stop_event.set()
                    log.info("chat/stream client disconnected (thread=%s); stopping generation", thread_id)

        return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)

    # -- static widget + demo page --------------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/widget", StaticFiles(directory=str(WEB_DIR), html=True), name="widget")
    else:  # pragma: no cover
        log.warning("web/ directory not found at %s — /widget/ is disabled", WEB_DIR)

    api_index = {
        "service": app.title,
        "docs": "/docs",
        "health": "/health",
        "lectures": "/lectures",
        "chat": "POST /chat",
        "chat_stream": "POST /chat/stream (text/event-stream)",
        "chat_ui": "/",
        "embed_demo": "/widget/embed-demo.html",
        "dashboard": "/dashboard (pipeline traces; see DEBUG_DASHBOARD)",
    }

    @app.get("/", include_in_schema=False)
    async def root():
        """The full-page chat UI (web/index.html); falls back to the JSON index if the page is missing."""
        page = WEB_DIR / "index.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-cache"})
        return api_index

    @app.get("/api", include_in_schema=False)
    async def api_root() -> dict[str, Any]:
        return api_index

    # -- pipeline dashboard (web/dashboard.html + the traces it reads) -------------------------
    def _require_dashboard(request: Request) -> None:
        if not _dashboard_allowed(s, request):
            raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard(request: Request):
        _require_dashboard(request)
        return FileResponse(WEB_DIR / "dashboard.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/debug/traces", include_in_schema=False)
    async def list_traces(request: Request) -> dict[str, Any]:
        _require_dashboard(request)
        return {"mode": s.debug_dashboard, "limit": trace.STORE.limit, "traces": [t.summary() for t in trace.STORE.newest_first()]}

    @app.get("/debug/traces/{trace_id}", include_in_schema=False)
    async def get_trace(trace_id: str, request: Request) -> dict[str, Any]:
        _require_dashboard(request)
        tr = trace.STORE.get(trace_id)
        if tr is None:
            raise HTTPException(status_code=404, detail="unknown trace")
        return tr.to_dict()

    log.info(
        "SmartUp API ready: model=%s embeddings=%s reranker=%s index=%s cors=%s web=%s dashboard=%s",
        s.chat_model, s.embed_model, s.rerank_model if s.rerank_enabled else "off", s.index_dir,
        ",".join(s.cors_origins), WEB_DIR if WEB_DIR.is_dir() else "-", s.debug_dashboard,
    )
    return app


app = create_app()


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    import uvicorn

    s = get_settings()
    parser = argparse.ArgumentParser(prog="smartup-api", description="Serve the SmartUp assistant API")
    parser.add_argument("--host", default=s.api_host, help=f"bind address (default {s.api_host})")
    parser.add_argument("--port", type=int, default=s.api_port, help=f"port (default {s.api_port})")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = parser.parse_args(argv)
    uvicorn.run(
        "smartup_rag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=s.log_level.lower(),
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
