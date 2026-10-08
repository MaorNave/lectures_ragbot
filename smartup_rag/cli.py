"""Command-line interface: ``smartup <command>`` (console script) or ``python -m smartup_rag.cli``.

Commands
    check      health of Ollama, the local index and the reranker, plus a settings summary (``--explain`` prints
               the filters and FTS5 queries of every retrieval leg)
    models     pull the Ollama models (``--pull``) and download the cross-encoder ahead of the first question
    ingest     build or update the local index (Chroma + SQLite FTS5) from data/raw/drive_09_2026
    ask        one-shot question through the LangGraph agent, streamed token by token
    chat       interactive REPL with streamed answers and a persistent thread (follow-ups work)
    retrieve   show the passages the hybrid retriever returns — no LLM involved
    lectures   table of the lecture corpus
    eval       run the evaluation set (delegates to smartup_rag.eval.run_eval)
    serve      start the FastAPI server with uvicorn

Heavy modules (LangGraph, the Ollama client, torch, Chroma, the API app) are imported inside the command handlers so
that fast commands such as ``lectures`` stay fast and ``serve`` never imports the API module at CLI load time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from smartup_rag import __version__

log = logging.getLogger("smartup_rag.cli")

SAMPLE_QUESTION = "What does Yonatan say about pricing?"
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio", "chromadb", "sentence_transformers", "transformers",
                 "huggingface_hub")
_WS_RE = re.compile(r"\s+")


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def setup_logging(verbosity: int) -> None:
    """``-v`` → INFO with a compact format, ``-vv`` → DEBUG; otherwise WARNING. Chatty library loggers are quieted."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _out(text: str = "") -> None:
    print(text, flush=True)


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _snippet(text: str, n: int = 120) -> str:
    text = _WS_RE.sub(" ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _fmt_score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _yes_no(flag: Any) -> str:
    return "yes" if flag else "no"


def _print_result_footer(result: dict[str, Any]) -> None:
    """Status line + citation list shared by ``ask`` and ``chat``."""
    grade = result.get("grade_v2") or "-"
    for t in result.get("fallback_tries") or []:  # v2 → TEDx → v1 → outsource, as far as the chain went
        grade += f" → {t['tier']} {t['verdict']}"
    _out(
        f"── status: {result.get('status')} · lang: {result.get('language')} · intent: {result.get('intent')}"
        f" · lecture: {result.get('lecture_number') if result.get('lecture_number') is not None else '-'}"
        f" · scope: {result.get('version_scope') or '-'} · grade: {grade}"
        f" · fallback: {_yes_no(result.get('used_fallback'))} · passages: {result.get('n_passages', 0)}"
    )
    citations = result.get("citations") or []
    if citations:
        _out("── sources:")
        for c in citations:
            _out(f"   • {c.get('citation') or c}")


def _lazy_import(module: str, attr: str, purpose: str) -> Callable[..., Any] | None:
    """Import ``module.attr`` at call time; explain clearly when the module is not there."""
    try:
        mod = __import__(module, fromlist=[attr])
    except ImportError as exc:
        _err(f"{purpose} is not available: cannot import {module} ({exc}).")
        _err("Is the package installed? (pip install -e .)")
        return None
    fn = getattr(mod, attr, None)
    if fn is None:
        _err(f"{purpose} is not available: {module} has no `{attr}` function.")
        return None
    return fn


# ----------------------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------------------
def _explain(question: str) -> None:
    """The filters and keyword queries every leg of one v2 retrieval pass uses (nothing is executed)."""
    from smartup_rag.config import get_settings
    from smartup_rag.retrieval import queries as Q

    s = get_settings()
    conds = Q.conditions("v2", None, None, None if s.cross_lingual else "en",
                         exclude_course_ids=[] if s.include_digital_innovation else [Q.DIGITAL_INNOVATION_COURSE_ID])
    terms = Q.query_terms(question)
    where_sql, params = Q.sql_where(conds)
    _out(f"\nRetrieval legs for: {question!r} (lang=en, tier=v2)")
    _out(f"\n# dense — {s.embed_model} vector → Chroma collection {s.collection_name!r}, top {s.rank_window}")
    _out("where = " + _dump(Q.chroma_where(conds)))
    _out(f"\n# bm25 — SQLite FTS5, weights {Q.BM25_TEXT_WEIGHTS} over {Q.FTS_COLUMNS}, top {s.rank_window}")
    _out(f"MATCH {Q.fts_match(terms, ('text',))!r}")
    _out(f"  AND {where_sql}  params={params}")
    _out(f"\n# metadata — SQLite FTS5, weights {Q.BM25_METADATA_WEIGHTS}, ties broken by the dense leg, top {s.rank_window}")
    _out(f"MATCH {Q.fts_match(terms, Q.METADATA_COLUMNS)!r}")
    _out(f"\n# fusion — RRF k={s.rank_constant}, window={s.rank_window} → {s.rerank_model if s.rerank_enabled else 'no rerank'}"
         f" → top {s.top_k_passages} (scores below {s.min_rerank_score} dropped)")


def cmd_check(args: argparse.Namespace) -> int:
    """Print Ollama / index / reranker health and a settings summary. Exit 1 when something blocks answering."""
    from smartup_rag.agent.llm import ollama_health
    from smartup_rag.config import PROJECT_ROOT, get_settings
    from smartup_rag.retrieval.reranker import is_cached, resolve_device
    from smartup_rag.retrieval.store import index_health

    s = get_settings()
    ok = True
    _out(f"SmartUp RAG {__version__} — environment check\n")

    oh = ollama_health()
    if oh.get("ok"):
        for label, model, present in (("chat model", s.chat_model, oh.get("model_present")),
                                      ("embeddings", s.embed_model, oh.get("embed_model_present"))):
            state = "present" if present else "NOT PULLED — run: ollama pull " + model
            _out(f"Ollama         {'OK  ' if present else 'MISS'}  {s.ollama_base_url} · {label} {model}: {state}")
            ok = ok and bool(present)
    else:
        ok = False
        _out(f"Ollama         DOWN  {s.ollama_base_url} · {oh.get('error')}")

    ih = index_health(detail=True)
    if ih.get("ok"):
        _out(f"Local index    OK    {ih.get('path')} · {ih.get('lectures')} lectures · {ih.get('passages')} passages"
             f" · {ih.get('vectors')} vectors · built {ih.get('built_at')} with {ih.get('embed_model')}")
        for course, n in sorted((ih.get("lectures_by_course") or {}).items()):
            _out(f"               {course}: {n}")
    else:
        ok = False
        _out(f"Local index    FAIL  {ih.get('path')} · {ih.get('error')}")

    if s.rerank_enabled:
        cached = is_cached(s.rerank_model)
        try:
            device = resolve_device(s.rerank_device)
        except Exception as exc:  # torch missing or broken
            device = f"unavailable ({type(exc).__name__})"
        state = "cached locally" if cached else "NOT DOWNLOADED — run: smartup models (or it downloads on first use)"
        _out(f"Reranker       {'OK  ' if cached else 'WARN'}  {s.rerank_model} on {device} · {state}")
    else:
        _out("Reranker       OFF   RERANK_ENABLED=false: RRF order, and the LLM grader decides every source")

    web_dir = PROJECT_ROOT / "web"
    questions = PROJECT_ROOT / "smartup_rag" / "eval" / "questions.jsonl"
    files_ok = s.instructions_path.is_file()
    n_json = len(list(s.data_dir.glob("*/*.json"))) if s.data_dir.is_dir() else 0
    _out("\nFiles")
    _out(f"  instructions.md    {'OK  ' if files_ok else 'MISSING'}  {s.instructions_path}"
         + (f" ({s.instructions_path.stat().st_size:,} bytes)" if files_ok else ""))
    _out(f"  data dir           {'OK  ' if n_json else 'MISSING'}  {s.data_dir} ({n_json} lecture files)")
    _out(f"  web/ (chat UI)     {'OK  ' if (web_dir / 'index.html').is_file() else 'MISSING'}  {web_dir}")
    _out(f"  eval questions     {'OK  ' if questions.is_file() else 'MISSING'}  {questions}")
    if not files_ok:
        ok = False

    _out("\nSettings")
    rows = [
        ("chat_model", s.chat_model),
        ("classifier_model", s.classifier_model or f"(same as chat: {s.chat_model})"),
        ("ollama_base_url", s.ollama_base_url),
        ("embed_model", s.embed_model),
        ("rerank", f"{s.rerank_model} ({'enabled' if s.rerank_enabled else 'disabled'}, device {s.rerank_device})"),
        ("index_dir", str(s.index_dir)),
        ("collection", s.collection_name),
        ("cross_lingual", _yes_no(s.cross_lingual)),
        ("top_k", f"passages={s.top_k_passages} window={s.rank_window} rrf_k={s.rank_constant}"),
        ("grading", f"strong ≥ {s.grade_strong_threshold} · weak < {s.grade_weak_threshold} · noise < {s.min_rerank_score}"),
        ("data_dir", str(s.data_dir)),
        ("api", f"{s.api_host}:{s.api_port} · cors={','.join(s.cors_origins)}"),
    ]
    for k, v in rows:
        _out(f"  {k:<18} {v}")

    if args.explain:
        _explain(SAMPLE_QUESTION)

    _out("\nResult: " + ("all good" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------
def _ollama_pull(base_url: str, model: str) -> bool:
    """Pull one model through Ollama's HTTP API (works for a remote Ollama too), printing its progress."""
    import httpx

    last = ""
    try:
        with httpx.stream("POST", f"{base_url}/api/pull", json={"model": model, "stream": True}, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("error"):
                    _err(f"  {model}: {ev['error']}")
                    return False
                status = ev.get("status", "")
                if ev.get("total"):
                    status += f" {int(100 * (ev.get('completed') or 0) / ev['total'])}%"
                if status != last:
                    _out(f"  {model}: {status}")
                    last = status
    except Exception as exc:
        _err(f"  {model}: pull failed ({type(exc).__name__}: {exc})")
        return False
    return True


def cmd_models(args: argparse.Namespace) -> int:
    """Make sure every model is available locally: the two Ollama models and the cross-encoder."""
    from smartup_rag.agent.llm import ollama_health
    from smartup_rag.config import get_settings

    s = get_settings()
    ok = True
    oh = ollama_health()
    if not oh.get("ok"):
        _err(f"Ollama is not reachable at {s.ollama_base_url}: {oh.get('error')}")
        ok = False
    else:
        for model, present in ((s.chat_model, oh.get("model_present")), (s.embed_model, oh.get("embed_model_present"))):
            if present:
                _out(f"Ollama   {model}: present")
            elif args.pull:
                _out(f"Ollama   {model}: pulling…")
                ok = _ollama_pull(s.ollama_base_url, model) and ok
            else:
                _out(f"Ollama   {model}: MISSING — run `ollama pull {model}` (or `smartup models --pull`)")
                ok = False
    if s.rerank_enabled:
        from smartup_rag.retrieval.reranker import get_reranker

        _out(f"Reranker {s.rerank_model}: loading (downloads ~2.3 GB from the Hugging Face Hub the first time)…")
        try:
            rr = get_reranker()
            rr.load()
            score = rr.score("What is pricing?", ["Pricing is how a company captures the value it creates."])[0]
            _out(f"Reranker {s.rerank_model}: ready on {rr.device} in {rr.load_seconds}s (test score {score:.3f})")
        except Exception as exc:
            _err(f"Reranker {s.rerank_model}: FAILED ({type(exc).__name__}: {exc})")
            ok = False
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------
# ask / chat
# ----------------------------------------------------------------------------------------
async def _stream_turn(message: str, thread_id: str, prefix: str = "assistant> ") -> tuple[dict[str, Any], bool]:
    """Stream one turn to stdout; return (final result, whether any token was printed)."""
    from smartup_rag.agent.graph import stream_chat

    streamed = False
    final: dict[str, Any] = {}
    async for ev in stream_chat(message, thread_id=thread_id):
        if ev.get("type") == "token":
            if not streamed and prefix:
                sys.stdout.write(prefix)
            sys.stdout.write(ev.get("text", ""))
            sys.stdout.flush()
            streamed = True
        elif ev.get("type") == "status" and not streamed:  # fallback-chain progress, before the answer starts
            _out(f"  … {ev.get('text', '')}")
        elif ev.get("type") == "final":
            final = {k: v for k, v in ev.items() if k != "type"}
    return final, streamed


def cmd_ask(args: argparse.Namespace) -> int:
    """One question → the answer streamed token by token (or the verified answer / full result at once)."""
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    if args.json or args.no_stream:
        from smartup_rag.agent.graph import run_chat

        try:
            result = run_chat(args.question, thread_id=thread_id)
        except Exception as exc:
            _err(f"ask failed: {type(exc).__name__}: {exc}")
            return 1
        if args.json:
            _out(_dump({"thread_id": thread_id, **result}))
            return 0
        _out(result.get("answer", ""))
    else:
        try:
            result, streamed = asyncio.run(_stream_turn(args.question, thread_id, prefix=""))
        except Exception as exc:
            _err(f"\nask failed: {type(exc).__name__}: {exc}")
            return 1
        if not streamed:  # refusal / not found / greeting / a repeated question: nothing was generated
            _out(result.get("answer", ""))
        else:
            _out()
    _out()
    _print_result_footer(result)
    return 0 if result.get("status") != "error" else 1


CHAT_HELP = (
    "Commands: /quit (or /exit, Ctrl-D) · /new — start a fresh thread · /json — last result as JSON · /help"
)


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL: answers are streamed token by token; the thread_id persists across turns."""
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    _out(f"SmartUp Academy chat · thread {thread_id} · {CHAT_HELP}")
    last: dict[str, Any] = {}
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            _out("\nbye")
            return 0
        if not message:
            continue
        if message.lower() in {"/quit", "/exit", "/q"}:
            _out("bye")
            return 0
        if message.lower() == "/help":
            _out(CHAT_HELP)
            continue
        if message.lower() == "/new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            _out(f"new thread {thread_id}")
            continue
        if message.lower() == "/json":
            _out(_dump(last) if last else "(no result yet)")
            continue
        try:
            last, streamed = asyncio.run(_stream_turn(message, thread_id))
        except KeyboardInterrupt:
            _out("\n(interrupted)")
            continue
        except Exception as exc:
            _err(f"\nturn failed: {type(exc).__name__}: {exc}")
            continue
        if streamed:
            _out()  # end the streamed line
        else:  # refusal / not found / chit-chat: nothing was streamed, print the final text
            _out(f"assistant> {last.get('answer', '')}")
        _out()
        _print_result_footer(last)


# ----------------------------------------------------------------------------------------
# retrieve / lectures
# ----------------------------------------------------------------------------------------
def cmd_retrieve(args: argparse.Namespace) -> int:
    """Run only the retrieval stage (three legs → RRF → dedupe → rerank) and print what the LLM would see."""
    from smartup_rag.lang import detect_language
    from smartup_rag.retrieval import hybrid as H

    lang = args.lang or detect_language(args.question)
    version = None if args.version == "all" else args.version
    stats: dict[str, Any] = {}
    try:
        cands = H.search_passages(args.question, lang, version, args.lecture, stats=stats)
    except Exception as exc:
        _err(f"retrieval failed: {type(exc).__name__}: {exc}")
        return 1
    n_raw = len(cands)
    cands = H.dedupe(cands)
    top = H.rerank(args.question, cands, args.k)
    legs = stats.get("legs") or {}
    _out(
        f"lang={lang} version={args.version} lecture={args.lecture or '-'}"
        f" · legs dense={legs.get('dense', 0)} bm25={legs.get('bm25', 0)} metadata={legs.get('metadata', 0)}"
        f" · fused={n_raw} unique={len(cands)} shown={len(top)}"
    )
    if stats.get("dense_error"):
        _out(f"(dense leg skipped: {stats['dense_error']})")
    if not top:
        _out("(no passages)")
        return 0
    for i, c in enumerate(top, 1):
        topic = f" · {c.section_topic}" if c.section_topic else ""
        _out(
            f"\n{i:>2}. {c.citation}  [{c.lang}]  rerank={_fmt_score(c.rerank_score)}  rrf={_fmt_score(c.score)}{topic}"
        )
        _out(f"    {_snippet(c.text, args.chars)}")
    return 0


def cmd_lectures(args: argparse.Namespace) -> int:
    """Table of every lecture in the corpus (version, number/part, title, number of subjects)."""
    from smartup_rag.retrieval.retriever import get_lectures

    try:
        lectures = get_lectures(include_summaries=False)
    except Exception as exc:
        _err(f"could not list lectures: {type(exc).__name__}: {exc}")
        return 1
    if args.json:
        _out(_dump(lectures))
        return 0
    width = max([len(lec["lecture_title"]) for lec in lectures] + [5])
    header = f"{'tier':<10} {'lecture':<18} {'title':<{width}} {'#subj':>5}  key"
    _out(header)
    _out("-" * len(header))
    for lec in lectures:
        _out(
            f"{lec['tier']:<10} {lec['display'] or '-':<18} {lec['lecture_title']:<{width}}"
            f" {len(lec['lecture_subjects']):>5}  {lec['lecture_key']}"
        )
    per_tier: dict[str, int] = {}
    for lec in lectures:
        per_tier[lec["tier"]] = per_tier.get(lec["tier"], 0) + 1
    _out("-" * len(header))
    _out(f"{len(lectures)} lectures · " + " · ".join(f"{t}: {n}" for t, n in sorted(per_tier.items())))
    return 0


# ----------------------------------------------------------------------------------------
# ingest / eval / serve
# ----------------------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    ingest = _lazy_import("smartup_rag.ingest.indexer", "ingest", "Ingestion")
    if ingest is None:
        return 1
    if not args.verbose_sub and not logging.getLogger().isEnabledFor(logging.INFO):
        logging.getLogger("smartup_rag.ingest").setLevel(logging.INFO)  # per-lecture progress, even without -v
        logging.getLogger().setLevel(logging.INFO)
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    try:
        result = ingest(recreate=args.recreate, data_dir=args.data_dir, dry_run=args.dry_run)
    except Exception as exc:
        _err(f"ingest failed: {type(exc).__name__}: {exc}")
        log.debug("ingest traceback", exc_info=True)
        return 1
    _out(_dump(result))
    if result.get("dry_run"):
        return 0
    ok = result.get("vectors") == result.get("stored_passages")
    if not ok:
        _out("ingest finished with an INCONSISTENT index — run `smartup ingest --recreate`")
    elif result.get("embedded") or result.get("removed") or result.get("rebuilt"):
        _out("ingest finished — restart the API to use the new index")
    else:
        _out("the index is up to date")
    return 0 if ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    run = _lazy_import("smartup_rag.eval.run_eval", "run", "Evaluation")
    if run is None:
        return 1
    try:
        result = run(full=args.full, limit=args.limit, filter_lecture=not args.no_lecture_filter)
    except Exception as exc:
        _err(f"eval failed: {type(exc).__name__}: {exc}")
        log.debug("eval traceback", exc_info=True)
        return 1
    if result is not None and args.json:
        _out(_dump(result))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI app with uvicorn (import string, so the API module loads inside uvicorn)."""
    import uvicorn

    from smartup_rag.config import get_settings

    s = get_settings()
    host = args.host or s.api_host
    port = args.port or s.api_port
    _out(f"serving smartup_rag.api.app:app on http://{host}:{port} (reload={_yes_no(args.reload)})")
    try:
        uvicorn.run(
            "smartup_rag.api.app:app",
            host=host,
            port=port,
            reload=args.reload,
            log_level=(s.log_level or "info").lower(),
        )
    except Exception as exc:
        _err(f"server failed to start: {type(exc).__name__}: {exc}")
        return 1
    return 0


# ----------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartup",
        description="SmartUp Academy RAG chatbot — local Ollama LLM over a local hybrid index (Chroma + SQLite FTS5).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="INFO logging (-vv for DEBUG)")
    parser.add_argument("--version", action="version", version=f"smartup-rag {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # `--verbose` is accepted after the sub-command too; a different dest avoids argparse's
    # sub-parser-default-overrides-parent behaviour.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", dest="verbose_sub", action="count", default=0, help=argparse.SUPPRESS
    )

    p = sub.add_parser("check", parents=[common], help="health of Ollama, the local index and the reranker")
    p.add_argument("--explain", action="store_true", help="also print the filters and FTS5 queries of every retrieval leg")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("models", parents=[common], help="make sure the Ollama models and the cross-encoder are available")
    p.add_argument("--pull", action="store_true", help="pull missing Ollama models through the Ollama API")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("ingest", parents=[common], help="build or update the local index from the Drive exports")
    p.add_argument("--recreate", action="store_true", help="drop the index and rebuild everything")
    p.add_argument("--data-dir", type=Path, default=None, help="export root (default: DATA_DIR = data/raw/drive_09_2026)")
    p.add_argument("--dry-run", action="store_true", help="load and chunk only; do not embed or write")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", parents=[common], help="ask one question; the answer is streamed")
    p.add_argument("question")
    p.add_argument("--thread", help="conversation thread id (defaults to a fresh one)")
    p.add_argument("--json", action="store_true", help="print the full verified result as JSON (no streaming)")
    p.add_argument("--no-stream", action="store_true", help="print the verified answer once it is complete")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", parents=[common], help="interactive chat (streams the answer; /quit to exit)")
    p.add_argument("--thread", help="conversation thread id to continue")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("retrieve", parents=[common], help="show retrieved passages for a question (no LLM)")
    p.add_argument("question")
    p.add_argument(
        "--lang", choices=["en", "he"], help="query language (default: detected from the question)"
    )
    p.add_argument(
        "--version", dest="version", choices=["v2", "tedx", "v1", "outsource", "all"], default="v2",
        help="source tier filter (v2 / tedx / v1 / outsource / all)"
    )
    p.add_argument("--lecture", type=int, help="restrict to one lecture number")
    p.add_argument("--k", type=int, default=8, help="passages to show after rerank (default 8)")
    p.add_argument("--chars", type=int, default=120, help="characters of each passage to print (default 120)")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("lectures", parents=[common], help="table of all lectures in the corpus")
    p.add_argument("--json", action="store_true", help="print as JSON")
    p.set_defaults(func=cmd_lectures)

    p = sub.add_parser("eval", parents=[common], help="run the evaluation set")
    p.add_argument(
        "--full", action="store_true", help="include LLM answers (slow); default is retrieval-only"
    )
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument(
        "--no-lecture-filter",
        action="store_true",
        help="retrieval mode: do NOT filter by a lecture number named in the question "
        "(harder, and unlike what the agent does)",
    )
    p.add_argument("--json", action="store_true", help="also print the whole result as JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", parents=[common], help="start the FastAPI server")
    p.add_argument("--host", help="bind host (default: API_HOST setting)")
    p.add_argument("--port", type=int, help="bind port (default: API_PORT setting)")
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(max(args.verbose, getattr(args, "verbose_sub", 0)))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _err("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
