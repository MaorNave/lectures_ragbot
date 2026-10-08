"""LangGraph nodes. Each node is a plain function: state in → partial state out.

Flow (see graph.py):
    prepare → classify ─┬─ refuse / chit_chat
                        └─ retrieve → grade ─┬─ generate → verify
                                             ├─ fallback ─┬─ generate → verify
                                             │            └─ not_found
                                             └─ not_found   (explicit v1/both scope that found nothing)

Hard rules from instructions.md are enforced here in code, not left to the model:
§0 scope guard fails closed · §4 v2 first, v1 only when justified · §4.5 never answer from nothing ·
§9–10 every citation must point to a retrieved lecture · §12 a decline carries no Sources.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from smartup_rag import trace
from smartup_rag.agent import prompts as P
from smartup_rag.agent.llm import get_llm, note_prompt_timing, shares_chat_prefix
from smartup_rag.agent.state import AgentState, Classification, FollowUpClassification, Grade
from smartup_rag.config import get_settings
from smartup_rag.lang import CITATION_RE, detect_language, lecture_number_in_text, lecture_part_in_text
from smartup_rag.retrieval.hybrid import display_lecture, retrieve
from smartup_rag.retrieval.retriever import get_lectures
from smartup_rag.schemas import SUMMARY_TOPIC
from smartup_rag.tiers import (
    FALLBACK_TIERS,
    FOUNDATIONS_TIERS,
    OUTSIDE_CITATION_RE,
    TIER_NAMES,
    normalise_title,
    source_label,
    source_line,
    tier_of,
)

log = logging.getLogger(__name__)

HOMEWORK_TERMS = {"en": " homework assignment exercise task", "he": " שיעורי בית משימה תרגיל להגיש"}

# version-scope normalisation (instructions §4.2, §6 "Version Comparison Questions")
# the Hebrew words take no leading \b: a prefix letter (בגרסה, הגרסאות) is a word character
_V1_RE = re.compile(r"\b(?:v1|version\s*1|old version|previous version)\b|גרסה\s*1\b|גרסה\s+הישנה", re.IGNORECASE)
_V2_RE = re.compile(r"\b(?:v2|version\s*2|new version|current version)\b|גרסה\s*2\b|גרסה\s+החדשה", re.IGNORECASE)
_VERSION_WORD_RE = re.compile(r"\bversions?\b|גרס(?:ה|אות)\b", re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"(\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bdiffer\w*\b|\bdifference\b|what changed|changed between"
    r"|השווה|השוו|להשוות|ההבדל|הבדלים|מה השתנה)",
    re.IGNORECASE,
)
# a question that names one outside source is answered from that source (like an explicit "v1")
_TEDX_RE = re.compile(r"\bted\s?x\b|\bted talk\b|טדקס|טד[\s-]?אקס|הרצאת\s+ה?[-–]?ted", re.IGNORECASE)
_DIGITAL_INNOVATION_RE = re.compile(r"digital\s+innovation|דיגיטל\s+אינוביישן|חדשנות\s+דיגיטלית", re.IGNORECASE)
#: scopes that name the source to use, so the v2 → TEDx → v1 → outsource chain does not run
EXPLICIT_SCOPES = ("v1", "both", "tedx", "outsource")


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _history(state: AgentState) -> list[BaseMessage]:
    """Previous turns, excluding the current human message."""
    msgs = list(state.get("messages", []))
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]
    return msgs


def _last_human(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _key(meta: dict[str, Any]) -> tuple:
    """Citation identity: (version, number, part) for the Foundations course, ("outside", lecture_key) otherwise
    — the supplemental talks are all "lecture 1", so only their lecture key tells them apart."""
    if tier_of(meta.get("course_id"), meta.get("course_version")) in FOUNDATIONS_TIERS:
        return (str(meta.get("course_version")), int(meta.get("lecture_number", 0)), meta.get("lecture_part") or None)
    return ("outside", str(meta.get("lecture_key")))


def _search_query(state: AgentState) -> str:
    """Query sent to the retriever; homework questions get the §17 keyword hints appended."""
    q = state["query"]
    if (state.get("classification") or {}).get("intent") == "homework":
        q += HOMEWORK_TERMS[state["language"]]
    return q


def normalise_version_scope(question: str, scope: str | None, intent: str | None) -> str:
    """Deterministic version-scope rule applied on top of the classifier (instructions §4.2, §6).

    Only a "default" verdict is touched. Both versions named, or comparison language plus any version
    signal → "both"; v1 named alone → "v1". A v1-vs-v2 comparison must never collapse to v1-only.
    Naming the TEDx talk → "tedx"; naming the Digital Innovation course → "outsource".
    """
    if scope not in (None, "default"):
        return scope
    v1, v2 = bool(_V1_RE.search(question)), bool(_V2_RE.search(question))
    version_mentioned = v1 or v2 or bool(_VERSION_WORD_RE.search(question))
    comparing = bool(_COMPARE_RE.search(question)) or intent == "comparison"
    if (v1 and v2) or (comparing and version_mentioned):
        return "both"
    if v1:
        return "v1"
    if _TEDX_RE.search(question):
        return "tedx"
    if _DIGITAL_INNOVATION_RE.search(question):
        return "outsource"
    return "default"


def _safe_retrieve(
    query: str, lang: str, version: str, lecture_number: int | None, lecture_part: str | None, notes: list[str]
) -> list[Document]:
    """Retrieval must never kill the run: a search error (index missing, model unavailable) degrades to 'nothing found'."""
    try:
        return retrieve(query, lang, version, lecture_number, lecture_part)
    except Exception as exc:
        log.exception("retrieval failed for %s (lecture=%s)", version, lecture_number)
        notes.append(f"retrieval_error:{type(exc).__name__}")
        return []


# --- background searches of the fallback sources --------------------------------------------------
# The fallback chain grades one source after another, but its searches do not depend on each other. When v2 looks weak
# (best score below the fast-accept bar), retrieve_node starts the TEDx / v1 / outsource searches in these threads, so
# they run while the v2 grader is thinking; the fallback node then finds them done. Results are keyed by the turn and
# kept in this process (futures do not belong in the checkpointed graph state).
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fallback-search")
_PREFETCHED: dict[str, dict[str, Future]] = {}
_PREFETCH_LOCK = threading.Lock()
_PREFETCH_KEEP = 32  # abandoned entries (v2 answered after all) are dropped oldest-first


def _search_one(query: str, lang: str, tier: str, n: int | None, part: str | None) -> tuple[list[Document], list[str]]:
    notes: list[str] = []
    return _safe_retrieve(query, lang, tier, n, part, notes), notes


def _submit_search(query: str, lang: str, tier: str, n: int | None, part: str | None) -> Future:
    # each task gets its own copy of the context, so its "search <tier>" step still lands in this turn's trace
    return _PREFETCH_POOL.submit(trace.background_context().run, _search_one, query, lang, tier, n, part)


def _start_prefetch(turn_id: str, query: str, lang: str, tiers: tuple[str, ...], n: int | None, part: str | None) -> None:
    futures = {t: _submit_search(query, lang, t, n, part) for t in tiers}
    with _PREFETCH_LOCK:
        _PREFETCHED[turn_id] = futures
        while len(_PREFETCHED) > _PREFETCH_KEEP:
            _PREFETCHED.pop(next(iter(_PREFETCHED)))


def _take_prefetch(turn_id: str | None) -> dict[str, Future]:
    if not turn_id:
        return {}
    with _PREFETCH_LOCK:
        return _PREFETCHED.pop(turn_id, None) or {}


def _progress(config: RunnableConfig | None, lang: str, tier: str, first: bool) -> None:
    """Tell the chat what the slow part is doing: "Not found in the v2 course — checking the TEDx talk…"."""
    if not config:
        return
    template = P.PROGRESS["first" if first else "next"][lang]
    try:
        dispatch_custom_event("progress", {"tier": tier, "text": template.format(source=TIER_NAMES[lang][tier])}, config=config)
    except Exception:  # no streaming listener (CLI, tests): progress is decoration, never an error
        log.debug("progress event not dispatched", exc_info=True)


def _summary_docs(
    tier: str, lecture_number: int | None, language: str, lecture_part: str | None = None, limit: int = 3
) -> list[Document]:
    """Lecture-level summaries of one tier as Documents (for 'summarize lecture N' style questions)."""
    s = get_settings()
    out: list[Document] = []
    try:
        lectures = get_lectures(tier, lecture_number)
    except Exception:
        log.exception("lecture lookup failed for %s L%s", tier, lecture_number)
        return []
    if lecture_part:
        exact = [lec for lec in lectures if lec["lecture_part"] == lecture_part]
        lectures = exact or lectures
    for lec in lectures[:limit]:
        text = lec["hebrew_summary"] if language == "he" else lec["english_summary"]
        text = text or lec["english_summary"] or lec["hebrew_summary"]
        if not text:
            continue
        out.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "lecture_key": lec["lecture_key"],
                    "course_id": lec["course_id"],
                    "course_version": lec["course_version"],
                    "tier": lec["tier"],
                    "source_label": lec["source_label"],
                    "lecture_number": lec["lecture_number"],
                    "lecture_part": lec["lecture_part"],
                    "lecture_title": lec["lecture_title"],
                    "lecture_subjects": lec["lecture_subjects"],
                    "lang": language if (language == "he" and lec["hebrew_summary"]) else "en",
                    "score": 1.0,
                    "rerank_score": None,
                    "section_topic": SUMMARY_TOPIC,
                    "source_index": f"local:{s.collection_name}:lectures",
                    "citation": lec["citation"],
                },
            )
        )
    return out


def _fast_grade(docs: list[Document], *, accept: bool = True) -> str | None:
    """Instructions §8 fast path using reranker scores; None → ask the LLM grader.

    `accept=False` keeps the fast "insufficient" but leaves a strong score to the LLM grader (see _fallback).
    """
    s = get_settings()
    if not docs:
        return "insufficient"
    scores = [d.metadata.get("rerank_score") for d in docs if d.metadata.get("rerank_score") is not None]
    if not scores:
        return None
    top = max(scores)
    if top >= s.grade_strong_threshold:
        return "sufficient" if accept else None
    if top < s.grade_weak_threshold:
        return "insufficient"
    return None


def _grade(
    question: str, docs: list[Document], history: list[BaseMessage], config: RunnableConfig | None,
    *, fast_accept: bool = True,
) -> dict[str, Any]:
    """`history` must be what generate() will replay: the grader then reads the answer prompt's exact opening."""
    fast = _fast_grade(docs, accept=fast_accept)
    if fast:
        return {"verdict": fast, "reason": "rerank-score fast path", "method": "fast"}
    try:
        s = get_settings()
        shared = shares_chat_prefix()
        grader = get_llm("grader").with_structured_output(Grade, method="json_schema", include_raw=True)
        msgs = P.grader_messages(question, docs, history, s.max_history_turns, shared_prefix=shared)
        out = grader.invoke(msgs, config=config)
        if out.get("parsed") is None:
            raise out.get("parsing_error") or ValueError("grader returned no parsable JSON")
        follow_up = bool(P.replayed_history(history, s.max_history_turns))
        layout = "evidence block shared with the answer" \
            if P.grader_reads_evidence_block(question, shared_prefix=shared, follow_up=follow_up) \
            else "compact (passages cut to 700 characters)"
        return {**out["parsed"].model_dump(), "method": "llm", "llm": trace.llm_timing(out.get("raw")), "prompt_layout": layout}
    except Exception as exc:
        # Fall back to the deterministic retrieval signal instead of throwing good passages away:
        # anything above the noise floor is treated as a partial match and answered from v2.
        s = get_settings()
        scores = [d.metadata.get("rerank_score") or 0.0 for d in docs]
        verdict = "partial" if scores and max(scores) >= s.grade_weak_threshold else "insufficient"
        log.warning("grader failed (%s); using retrieval signal → %s", type(exc).__name__, verdict)
        return {"verdict": verdict, "reason": f"grader error: {type(exc).__name__}", "method": "error"}


# --- exact repeats ------------------------------------------------------------------------------------
# A question asked again in the same conversation gets the answer it already got — the verified text with its original
# citations — instead of a new run. A new run is not only slower: the earlier answer sits in the replayed history, and
# once it made the grader accept passages it rejects in a fresh conversation (README §11).
_ANSWER_LOG_KEEP = 20
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def question_key(question: str) -> str:
    """Case, punctuation and spacing do not make a question new: "What is PMF?" == "what is pmf"."""
    return _NON_WORD_RE.sub(" ", question.casefold()).strip()


_FOLLOW_UP_MAX_WORDS = 3


def _context_key(state: AgentState) -> str:
    """The question asked before this one — what a short follow-up like "tell me more" or "why?" refers to."""
    prev = next((m for m in reversed(_history(state)) if isinstance(m, HumanMessage)), None)
    return question_key(str(prev.content)) if prev is not None else ""


def _previous_answer(state: AgentState) -> dict[str, Any] | None:
    key = question_key(state.get("question") or "")
    if not key:
        return None
    # a short question is only the same question when it follows the same one ("tell me more" after PMF ≠ after pricing)
    context = _context_key(state) if len(key.split()) <= _FOLLOW_UP_MAX_WORDS else None
    return next((e for e in reversed(state.get("answer_log") or [])
                 if e.get("key") == key and (context is None or e.get("context") == context)), None)


def _replayable(state: AgentState, status: str) -> bool:
    """Only a finished answer to a real question is worth giving again. Not an error or a greeting, not an answer cut
    off when the reader left, and not one from a turn whose search failed — asking again is how a reader retries."""
    if status in ("error", "chit_chat") or state.get("stopped_early"):
        return False
    if (state.get("classification") or {}).get("error"):  # refused only because the classifier was down
        return False
    return not any(str(n).startswith("retrieval_error:") for n in state.get("notes") or [])


def _logged(state: AgentState, answer: str, status: str, citations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The conversation's answer log with this turn's answer appended, when it may be replayed (_replayable)."""
    log_ = list(state.get("answer_log") or [])
    if _replayable(state, status):
        c = state.get("classification") or {}
        log_.append({
            "key": question_key(state.get("question") or ""),
            "context": _context_key(state),
            "question": state.get("question"),
            "answer": answer,
            "status": status,
            "citations": list(citations or []),
            "language": state.get("language"),
            "classification": {k: c.get(k) for k in ("in_scope", "intent", "lecture_number", "lecture_part", "version_scope")},
            "used_fallback": bool(state.get("used_fallback")),
            "fallback_tier": state.get("fallback_tier"),
        })
    return log_[-_ANSWER_LOG_KEEP:]


# ----------------------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------------------
def prepare(state: AgentState) -> dict:
    q = _last_human(state).strip()
    lang = detect_language(q)
    with trace.span("prepare") as sp:
        sp.update(question=q, language=lang, history_messages=len(_history(state)))
    return {
        "turn_id": uuid.uuid4().hex,  # keys this turn's background searches (see _start_prefetch)
        "question": q,
        "language": lang,
        "classification": {},
        "query": q,
        "docs_v2": [],
        "docs_v1": [],
        "docs": [],
        "grade_v2": {},
        "grade_v1": {},
        "docs_fallback": [],
        "grade_fallback": {},
        "fallback_tier": None,
        "fallback_tries": [],
        "used_fallback": False,
        "notes": [],
        "answer": "",
        "stopped_early": False,
        "citations": [],
        "status": "error",  # overwritten by every terminal node; a crash mid-run must not look like success
    }


def route_after_prepare(state: AgentState) -> str:
    return "repeat" if _previous_answer(state) else "classify"


def repeat(state: AgentState) -> dict:
    """The same question was already answered in this conversation: that answer again, with its sources — no search,
    no LLM. The note says so, so a user who repeated the question because the answer did not help can ask for more."""
    prev = _previous_answer(state) or {}
    lang = state["language"]
    text = P.REPEAT_NOTE[lang] + "\n\n" + str(prev.get("answer") or "")
    with trace.span("repeat") as sp:
        sp.update(reason="the same question was already answered in this conversation", previous_question=prev.get("question"),
                  previous_status=prev.get("status"), citations=[c.get("citation") for c in prev.get("citations") or []])
    return {
        "answer": text,
        "status": prev.get("status", "answered"),
        "citations": list(prev.get("citations") or []),
        "classification": dict(prev.get("classification") or {}),
        "used_fallback": bool(prev.get("used_fallback")),
        "fallback_tier": prev.get("fallback_tier"),
        "messages": [AIMessage(content=text)],
    }


def classify(state: AgentState, config: RunnableConfig) -> dict:
    q = state["question"]
    history = _history(state)
    with trace.span("classify") as sp:
        data: dict[str, Any] | None = None
        last_exc: Exception | None = None
        schema = FollowUpClassification if history else Classification  # the rewritten query is only used for follow-ups
        # include_raw: the reply's timing tells whether the shared-prompt cache survived (see llm.note_prompt_timing)
        clf = get_llm("classifier").with_structured_output(schema, method="json_schema", include_raw=True)
        messages = P.classifier_messages(q, history, shared_prefix=shares_chat_prefix())
        sp.update(schema=schema.__name__, shared_prefix=shares_chat_prefix(), history_messages=len(history))
        for attempt in (1, 2):  # transient Ollama / JSON-parse failures dominate; one retry is cheap
            try:
                out = clf.invoke(messages, config=config)
                note_prompt_timing(out.get("raw"))
                sp.update(llm=trace.llm_timing(out.get("raw")), attempts=attempt)
                if out.get("parsed") is None:
                    raise out.get("parsing_error") or ValueError("classifier returned no parsable JSON")
                data = out["parsed"].model_dump()
                break
            except Exception as exc:
                last_exc = exc
                log.warning("classifier attempt %d failed: %s: %s", attempt, type(exc).__name__, exc)
        if data is None:
            # §0.4: when in doubt, treat as out of scope. Failing open would disable the scope guard
            # exactly when the local model is struggling.
            log.error("classifier unavailable after retry (%s); refusing per §0.4", type(last_exc).__name__)
            data = Classification(in_scope=False, intent="out_of_scope").model_dump()
            data["error"] = f"classifier unavailable (§0.4): {type(last_exc).__name__}: {last_exc}"
        model_said = dict(data)

        # deterministic overrides the small model may miss
        n = lecture_number_in_text(q)
        if n is not None and data.get("lecture_number") is None:
            data["lecture_number"] = n
        part = lecture_part_in_text(q)
        if part and not data.get("lecture_part"):
            data["lecture_part"] = part
        if data.get("intent") == "out_of_scope":
            data["in_scope"] = False
        if data.get("intent") == "chit_chat":
            data["in_scope"] = True
        data["version_scope"] = normalise_version_scope(q, data.get("version_scope"), data.get("intent"))

        # the rewritten standalone query is only trusted for follow-ups (it resolves pronouns);
        # for a first message the user's own wording is the best search query
        query = (data.get("search_query") or "").strip()
        if not history or len(query) < 3:
            query = q
        keys = ("in_scope", "intent", "lecture_number", "lecture_part", "version_scope")
        sp.update(
            decision={k: data.get(k) for k in keys},
            model_said={k: model_said.get(k) for k in (*keys, "search_query") if k in model_said},
            code_overrides={k: [model_said.get(k), data.get(k)] for k in keys if model_said.get(k) != data.get(k)},
            search_query=query,
            error=data.get("error"),
        )
    log.info(
        "classify intent=%s in_scope=%s lecture=%s part=%s scope=%s",
        data.get("intent"), data.get("in_scope"), data.get("lecture_number"), data.get("lecture_part"),
        data.get("version_scope"),
    )
    return {"classification": data, "query": query}


def route_after_classify(state: AgentState) -> str:
    c = state["classification"]
    if c.get("intent") == "chit_chat":
        return "chit_chat"
    if not c.get("in_scope", False):  # a missing decision refuses, never retrieves
        return "refuse"
    return "retrieve"


def retrieve_node(state: AgentState) -> dict:
    c = state["classification"]
    q, lang = _search_query(state), state["language"]
    n, part = c.get("lecture_number"), c.get("lecture_part")
    scope = c.get("version_scope", "default")
    want_summary = c.get("intent") == "lecture_summary"
    notes = list(state.get("notes", []))

    summaries_added: dict[str, list[str]] = {}

    def pass_for(version: str) -> list[Document]:
        docs = _safe_retrieve(q, lang, version, n, part, notes)
        if want_summary:
            if n is not None:
                summaries = _summary_docs(version, n, lang, part)
            else:  # "summarize the course": summaries of the top retrieved lectures only
                keys = {d.metadata["lecture_key"] for d in docs[:3]}
                summaries = [d for d in _summary_docs(version, None, lang, limit=100) if d.metadata["lecture_key"] in keys]
            summaries_added[version] = [d.metadata["citation"] for d in summaries]
            docs = summaries + docs
        return docs

    with trace.span("retrieve") as sp:
        sp.update(version_scope=scope, search_query=q, lecture_filter=n, part_filter=part,
                  homework_hints=q != state["query"], lecture_summary=want_summary)
        if scope == "v1":
            d1 = pass_for("v1")
            out = {"docs_v1": d1, "docs": d1, "used_fallback": False, "notes": notes}
        elif scope in ("tedx", "outsource"):  # the user named that source: search it alone, no fallback chain
            d = pass_for(scope)
            out = {"docs_fallback": d, "docs": d, "used_fallback": False, "notes": notes}
        elif scope == "both":
            d2, d1 = pass_for("v2"), pass_for("v1")
            notes.append("Comparison requested: clearly separate v2 content, v1 content, what changed and what stayed the same.")
            out = {"docs_v2": d2, "docs_v1": d1, "docs": d2 + d1, "notes": notes}
        else:
            d2 = pass_for("v2")
            out = {"docs_v2": d2, "docs": d2, "notes": notes}
            # v2 not clearly strong → the fallback chain may run: search its sources now, while v2 is being graded
            has_summary = want_summary and any(d.metadata.get("section_topic") == SUMMARY_TOPIC for d in d2)
            weak = (_best_score(d2) or 0) < get_settings().grade_strong_threshold
            if get_settings().prefetch_fallback and state.get("turn_id") and not has_summary and weak:
                _start_prefetch(state["turn_id"], q, lang, fallback_tiers(n), n, part)
                sp["prefetched_fallback"] = list(fallback_tiers(n))
        sp.update(passages_total=len(out["docs"]), summaries_added=summaries_added)
    return out


def _best_score(docs: list[Document]) -> float | None:
    scores = [d.metadata.get("rerank_score") for d in docs if d.metadata.get("rerank_score") is not None]
    return round(max(scores), 4) if scores else None


def _grade_view(g: dict[str, Any], docs: list[Document]) -> dict[str, Any]:
    s = get_settings()
    return {"verdict": g.get("verdict"), "method": g.get("method"), "reason": g.get("reason"), "llm": g.get("llm"),
            "prompt_layout": g.get("prompt_layout"), "best_score": _best_score(docs), "passages": len(docs),
            "thresholds": {"fast_sufficient_at": s.grade_strong_threshold, "fast_insufficient_below": s.grade_weak_threshold}}


def grade_v2(state: AgentState, config: RunnableConfig) -> dict:
    with trace.span("grade v2") as sp:
        out = _grade_v2(state, config)
        sp.update(_grade_view(out["grade_v2"], state.get("docs_v2") or state.get("docs") or []))
    return out


def _grade_v2(state: AgentState, config: RunnableConfig) -> dict:
    c = state["classification"]
    lang = state["language"]
    notes = list(state.get("notes", []))
    if c.get("version_scope") in EXPLICIT_SCOPES:
        if not state["docs"]:  # §4.5: the requested version has no such material — never generate from nothing
            return {"grade_v2": {"verdict": "insufficient", "reason": "no passages in the requested version", "method": "skip"}}
        return {"grade_v2": {"verdict": "sufficient", "reason": "explicit version scope", "method": "skip"}}
    docs = state["docs_v2"]
    if c.get("intent") == "lecture_summary" and any(d.metadata.get("section_topic") == SUMMARY_TOPIC for d in docs):
        return {"grade_v2": {"verdict": "sufficient", "reason": "lecture summary available in v2", "method": "skip"}}
    g = _grade(state["query"], docs, _history(state), config)
    if g["verdict"] in ("sufficient", "partial"):
        _take_prefetch(state.get("turn_id"))  # v2 answers: the background searches are not needed
    if g["verdict"] == "partial":
        notes.append(f'Say explicitly: "{P.PARTIAL_NOTE[lang]}" and answer from the v2 passages only.')
    log.info("grade v2: %s (%s)", g["verdict"], g.get("method"))
    return {"grade_v2": g, "notes": notes}


def route_after_grade(state: AgentState) -> str:
    if state["grade_v2"].get("verdict") in ("sufficient", "partial") and state["docs"]:
        return "generate"
    if state["classification"].get("version_scope") in EXPLICIT_SCOPES:
        return "not_found"  # the requested source was already searched; the fallback chain would override the user
    return "fallback"


def fallback_tiers(lecture_number: int | None) -> tuple[str, ...]:
    """The sources searched after v2 (instructions §7): TEDx → v1 → other outside lectures.

    A question that names a lecture number is about the course's numbering, which the outside sources do not
    share — "lecture 3" of the TEDx talk does not exist — so only v1 is tried then.
    """
    return ("v1",) if lecture_number is not None else FALLBACK_TIERS


def fallback(state: AgentState, config: RunnableConfig) -> dict:
    out = _fallback(state, config)
    log.info("fallback: %s → %s", " → ".join(f"{t['tier']}={t['verdict']}" for t in out["fallback_tries"]),
             out["fallback_tier"] or "not_found")
    return out


def _skip_cutoff(tier: str) -> float:
    """Below this best score a fallback source is not worth an LLM grader call when a later source is strong."""
    s = get_settings()
    return s.skip_weak_grade_below_tedx if tier == "tedx" else s.skip_weak_grade_below


def _fallback(state: AgentState, config: RunnableConfig) -> dict:
    """Pass 2 (instructions §7): the fallback sources in order, stopping at the first one that answers.

    Each source gets its own search (with the same homework hints) and its own grade — the rerank fast path or the
    LLM grader. The searches are independent, so they run in parallel: usually retrieve_node already started them
    while v2 was being graded, and whatever is missing starts here. Knowing every source's best score up front
    allows one saving: a source below SKIP_WEAK_GRADE_BELOW (…_TEDX for the talk) is rejected without an LLM call when a *later* source
    already has a strong match — that later source answers anyway unless the weak one is accepted, and in the
    validation (README §7) the grader never accepted one.

    When the user named a lecture that exists in v2 but does not cover the topic, v1 Lecture N is a *different*
    lecture, so the bar is higher there: only a clearly sufficient v1 match may answer (§4.2 condition 2);
    otherwise the lecture-aware not-found message is returned.
    """
    s = get_settings()
    c = state["classification"]
    q, lang = _search_query(state), state["language"]
    n, part = c.get("lecture_number"), c.get("lecture_part")
    notes = list(state.get("notes", []))
    lecture_named_and_in_v2 = n is not None and bool(state.get("docs_v2"))
    tiers = fallback_tiers(n)
    futures = _take_prefetch(state.get("turn_id"))
    prefetched = set(futures)
    for tier in tiers:
        if tier not in futures:
            futures[tier] = _submit_search(q, lang, tier, n, part)
    found: dict[str, list[Document]] = {}
    for tier in tiers:
        d, search_notes = futures[tier].result()
        notes += search_notes
        if c.get("intent") == "lecture_summary" and n is not None:
            d = _summary_docs(tier, n, lang, part) + d
        found[tier] = d

    tried: list[dict[str, Any]] = []
    seen: list[Document] = []
    g: dict[str, Any] = {}
    out_v1: dict[str, Any] = {}
    for i, tier in enumerate(tiers):
        d = found[tier]
        _progress(config, lang, tier, first=i == 0)
        with trace.span(f"fallback {tier}") as sp:
            best = _best_score(d) or 0.0
            later_strong = next((t for t in tiers[i + 1:] if (_best_score(found[t]) or 0.0) >= s.grade_strong_threshold), None)
            if c.get("intent") == "lecture_summary" and not (lecture_named_and_in_v2 and tier == "v1") \
                    and any(x.metadata.get("section_topic") == SUMMARY_TOPIC for x in d):
                g = {"verdict": "sufficient", "reason": f"{tier} lecture found", "method": "skip"}
            elif later_strong and s.grade_weak_threshold <= best < _skip_cutoff(tier):
                g = {"verdict": "insufficient", "method": "skip-weak",
                     "reason": f"best score {best:.2f} < {_skip_cutoff(tier):.2f} and {later_strong} has a strong match"}
            else:
                # v1 Lecture N is not v2 Lecture N, but it is often about the same topic ("the homework of lecture 2"),
                # and the reranker scores that high whether or not it answers the question: in the calibration two
                # such v1 passes scored 0.75 and 0.87 and the grader rejected both (README §10). Only the grader decides.
                strict = lecture_named_and_in_v2 and tier == "v1"
                g = _grade(q, d, _history(state), config, fast_accept=not strict)
            ok = g["verdict"] == "sufficient" if (lecture_named_and_in_v2 and tier == "v1") \
                else g["verdict"] in ("sufficient", "partial")
            sp.update(_grade_view(g, d))
            sp["searched_while_v2_was_graded"] = tier in prefetched
            sp["decision"] = f"answer from {tier}" if ok else ("not found" if tier == tiers[-1] else "try the next source")
        tried.append({"tier": tier, "verdict": g["verdict"], "method": g.get("method"), "best_score": _best_score(d),
                      "passages": len(d)})
        seen += d
        if tier == "v1":
            out_v1 = {"docs_v1": d, "grade_v1": g}
        if ok:
            notes.append(
                f'Start your answer with this note (adapted to the specific source you use): "{P.FALLBACK_NOTES[tier][lang]}"'
            )
            return {**out_v1, "docs_fallback": seen, "docs": d, "grade_fallback": g, "fallback_tier": tier,
                    "fallback_tries": tried, "used_fallback": True, "notes": notes}
    return {**out_v1, "docs_fallback": seen, "docs": [], "grade_fallback": g, "fallback_tier": None,
            "fallback_tries": tried, "used_fallback": True, "notes": notes}


def route_after_fallback(state: AgentState) -> str:
    return "generate" if state["docs"] else "not_found"


def generate(state: AgentState, config: RunnableConfig) -> dict:
    s = get_settings()
    guidance = [n for n in state.get("notes", []) if not str(n).startswith("retrieval_error:")]
    msgs = P.answer_messages(state["question"], state["language"], state["docs"], _history(state), guidance, s.max_history_turns)
    stop = ((config or {}).get("configurable") or {}).get("stop_event")  # set by the API when the client leaves
    parts: list[str] = []
    meta: dict[str, Any] = {}
    # the grade that picked these passages; an LLM grade read this prompt's whole opening just before (see _grade)
    g = state.get("grade_fallback") if state.get("used_fallback") else state.get("grade_v2")
    after_grader_reading = bool(g) and g.get("method") == "llm" and str(g.get("prompt_layout", "")).startswith("evidence")
    with trace.span("generate") as sp:
        sp.update(
            model=s.chat_model,
            after_grader_reading=after_grader_reading,
            passages=[d.metadata.get("citation") for d in state["docs"]],
            history_messages=len(msgs) - 2,
            guidance_notes=list(state.get("notes", [])),
            system_prompt_chars=len(msgs[0].content),
            prompt=msgs[-1].content,  # the user turn: passages + allowed citations + guidance + question
        )
        stopped_early = False
        for chunk in get_llm("chat").stream(msgs, config=config):  # streaming so the API can relay tokens
            if stop is not None and stop.is_set():
                log.info("generate: client gone, stopping the model early")
                stopped_early = True
                break
            if isinstance(chunk.content, str):
                parts.append(chunk.content)
            if chunk.response_metadata:
                meta = chunk.response_metadata
        sp.update(llm=trace.llm_timing(meta), stopped_early=stopped_early, draft_chars=sum(len(p) for p in parts))
    if meta:
        log.info(
            "generate: prompt_tokens=%s output_tokens=%s prefill=%.1fs decode=%.1fs",
            meta.get("prompt_eval_count"), meta.get("eval_count"),
            (meta.get("prompt_eval_duration") or 0) / 1e9, (meta.get("eval_duration") or 0) / 1e9,
        )
    return {"answer": "".join(parts).strip(), "stopped_early": stopped_early}


# --- verification helpers -------------------------------------------------------------------
_SOURCES_HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,4}[ \t]*)?(?:\*\*)?[ \t]*(?:Sources|מקורות)[ \t]*:?[ \t]*(?:\*\*)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOURCE_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])?[ \t]*\(?\s*v[12]\b", re.IGNORECASE)
# instructions §12: an answer that declines is grounded in nothing, so it gets no Sources list
_DECLINE_RE = re.compile(
    r"(could not find|couldn't find|cannot find|do(?:es)? not (?:directly )?(?:address|cover|discuss|mention)"
    r"|not covered|is not discussed|no (?:relevant )?information"
    r"|לא מצאתי|לא מכוס|אינ[םן]? (?:מתייחס|עוסק|מכס)|לא (?:מתייחס|עוסק|נמצא|מזכיר))",
    re.IGNORECASE,
)
# the opening sentence of a fallback answer (P.FALLBACK_NOTES in either language, as the model adapts it)
_FALLBACK_NOTE_RE = re.compile(
    r"^\s*[^.\n]*(?:not covered directly|לא מכוס\S*\s+ישירות)[^.\n]*(?:drawn from|מבוסס\S*\s+על)[^.\n]*[.\n]",
    re.IGNORECASE,
)
_GAP = "\x00"  # marks where a removed citation stood; never appears in model output


def _outside_display(meta: dict[str, Any]) -> str:
    """The Sources-list label of an outside lecture: its lecture number only where the course has several."""
    line = source_line(meta.get("course_id"), meta.get("course_version"), int(meta.get("lecture_number", 0)),
                       meta.get("lecture_part"), str(meta.get("lecture_title") or ""))
    head = line.split(" — ", 1)[0]
    return head.split(", ", 1)[1] if ", " in head else ""


def _strip_model_sources(answer: str) -> str:
    """Drop a model-written Sources list — verify() rebuilds it — without swallowing real prose.

    Cuts at the first 'Sources' / 'מקורות' heading whose whole tail is blank lines, citation-shaped
    bullets or citation strings; a heading followed by anything else is left in place.
    """
    for m in _SOURCES_HEADING_RE.finditer(answer):
        tail = answer[m.end():]
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if all(_SOURCE_ITEM_RE.match(ln) or CITATION_RE.search(ln) or OUTSIDE_CITATION_RE.search(ln) for ln in lines):
            return answer[: m.start()].rstrip()
    return answer.rstrip()


def verify(state: AgentState) -> dict:
    """Instructions §9–10 and §12 enforced mechanically.

    * every citation must point to a lecture retrieved in this turn — others are removed;
    * surviving citations are normalised to the exact retrieved title;
    * the Sources list is rebuilt from the citations that survived;
    * an answer that declines (§12) carries no Sources and is reported as not_found.
    """
    lang = state["language"]
    docs = state["docs"]
    answer = state["answer"].replace(_GAP, "")

    valid: dict[tuple, dict[str, Any]] = {}
    for d in docs:
        m = d.metadata
        num, part = int(m.get("lecture_number", 0)), m.get("lecture_part")
        valid.setdefault(
            _key(m),
            {
                "lecture_key": m.get("lecture_key"),
                "course_id": m.get("course_id"),
                "course_version": m.get("course_version"),
                "tier": m.get("tier") or tier_of(m.get("course_id"), m.get("course_version")),
                "source": m.get("source_label") or source_label(m.get("course_id"), m.get("course_version")),
                "lecture_number": num,
                "lecture_part": part,
                "lecture_title": m.get("lecture_title"),
                "display": display_lecture(num, part) if _key(m)[0] != "outside" else _outside_display(m),
                "citation": m.get("citation"),
                "source_line": source_line(m.get("course_id"), m.get("course_version"), num, part,
                                           str(m.get("lecture_title") or "")),
            },
        )

    answer = _strip_model_sources(answer)
    used: dict[tuple, dict[str, Any]] = {}
    dropped: list[str] = []

    def _repl(m: re.Match) -> str:
        version, num, part = m.group("version").lower(), int(m.group("num")), m.group("part")
        key = (version, num, part)
        if key not in valid:
            same = [k for k in valid if k[0] == version and k[1] == num]
            if len(same) == 1:
                key = same[0]
            else:
                dropped.append(m.group(0))
                return _GAP
        used[key] = valid[key]
        return valid[key]["citation"]

    outside = {k: v for k, v in valid.items() if k[0] == "outside"}

    def _repl_outside(m: re.Match) -> str:
        """(TEDx — "…") / (Talk — "…") / (Digital Innovation IDC, Lecture 3 — "…"): the label and number narrow it
        down, the title decides between the supplemental talks (both are "lecture 1")."""
        label, num, title = m.group("label").lower(), m.group("num"), normalise_title(m.group("title"))
        keys = [k for k, v in outside.items() if v["source"].lower() == label and (num is None or v["lecture_number"] == int(num))]
        if len(keys) > 1:
            keys = [k for k in keys if normalise_title(outside[k]["lecture_title"]) == title] or [
                k for k in keys if title and (title in normalise_title(outside[k]["lecture_title"])
                                             or normalise_title(outside[k]["lecture_title"]) in title)]
        if len(keys) != 1:
            dropped.append(m.group(0))
            return _GAP
        used[keys[0]] = outside[keys[0]]
        return outside[keys[0]]["citation"]

    answer = CITATION_RE.sub(_repl, answer)
    answer = OUTSIDE_CITATION_RE.sub(_repl_outside, answer)
    if dropped:
        # tidy ONLY around the holes a removed citation left; titles just written by _repl and any
        # intentional spacing elsewhere (nested bullets, hard breaks) are never touched
        answer = re.sub(r"\(\s*" + _GAP + r"\s*\)", _GAP, answer)
        answer = re.sub(r"[ \t]*" + _GAP + r"[ \t]*(?=[.,;:!?])", "", answer)
        answer = re.sub(r"(?<=\S)[ \t]+" + _GAP + r"[ \t]+(?=\S)", " ", answer)
        answer = answer.replace(_GAP, "")
        log.warning("verify: removed %d citation(s) to non-retrieved lectures: %s", len(dropped), dropped)
    answer = answer.strip()

    status = "answered"
    if not used:
        # the fallback note that opens an answer drawn from TEDx / v1 / an outside lecture says "not covered
        # directly in v2" — that is not the model declining, so it is skipped before looking for a decline
        if _DECLINE_RE.search(_FALLBACK_NOTE_RE.sub("", answer, count=1)[:240]):
            status = "not_found"  # §12: the model declined; nothing grounds this answer
        elif docs:  # grounded prose without inline citations → still expose what it was based on (§22)
            for d in docs[:3]:
                used.setdefault(_key(d.metadata), valid[_key(d.metadata)])

    has_inline = bool(CITATION_RE.search(answer) or OUTSIDE_CITATION_RE.search(answer))
    appended = len(used) >= 2 or bool(used and not has_inline)
    if appended:
        lines = [P.SOURCES_HEADING[lang]] + [f"- {u['source_line']}" for u in used.values()]
        answer = answer + "\n\n" + "\n".join(lines)

    with trace.span("verify") as sp:
        sp.update(
            citations_kept=[u["citation"] for u in used.values()],
            citations_removed=dropped,
            model_declined=status == "not_found",
            sources_list_added=appended,
            status=status,
        )
    return {"answer": answer, "citations": list(used.values()), "status": status, "messages": [AIMessage(content=answer)],
            "answer_log": _logged(state, answer, status, list(used.values()))}


def refuse(state: AgentState) -> dict:
    text = P.REFUSAL[state["language"]]
    with trace.span("refuse") as sp:
        c = state.get("classification") or {}
        sp.update(reason=c.get("error") or "classifier: not about SmartUp or entrepreneurship (§0.1)", canned_text=True)
    return {"answer": text, "citations": [], "status": "refused", "messages": [AIMessage(content=text)],
            "answer_log": _logged(state, text, "refused")}


def chit_chat(state: AgentState) -> dict:
    text = P.GREETING[state["language"]]
    with trace.span("greeting") as sp:
        sp.update(reason="classifier: greeting / thanks / question about the bot", canned_text=True)
    return {"answer": text, "citations": [], "status": "chit_chat", "messages": [AIMessage(content=text)],
            "answer_log": _logged(state, text, "chit_chat")}


def not_found(state: AgentState) -> dict:
    lecture_aware = bool(
        (state.get("classification") or {}).get("lecture_number") is not None and state.get("docs_v2")
        and (state.get("classification") or {}).get("version_scope") not in ("v1", "both")
    )
    tried = ["v2", *[t["tier"] for t in state.get("fallback_tries") or []]]
    with trace.span("not found") as sp:
        sp.update(
            reason=("the named lecture exists in v2 but does not cover this" if lecture_aware
                    else f"no passages were good enough in {', '.join(TIER_NAMES['en'][t] for t in tried)} (§12)"),
            sources_tried=tried,
            canned_text=True,
        )
    return _not_found(state)


def _not_found(state: AgentState) -> dict:
    lang = state["language"]
    c = state.get("classification") or {}
    if c.get("lecture_number") is not None and state.get("docs_v2") and c.get("version_scope") not in ("v1", "both"):
        m = state["docs_v2"][0].metadata
        text = P.NOT_FOUND_IN_LECTURE[lang].format(
            lecture=P.LECTURE_WORD[lang].format(n=m.get("lecture_number")), citation=m.get("citation", "")
        )
        subjects = list(m.get("lecture_subjects") or [])
        if subjects:
            text += "\n\n" + P.LECTURE_COVERS[lang] + "\n" + "\n".join(f"- {s_}" for s_ in subjects[:8])
        return {"answer": text, "citations": [], "status": "not_found", "messages": [AIMessage(content=text)],
                "answer_log": _logged(state, text, "not_found")}
    subjects: list[str] = []
    for d in (state.get("docs_v2") or []) + (state.get("docs_fallback") or state.get("docs_v1") or []):
        for sub in d.metadata.get("lecture_subjects") or []:
            if sub not in subjects:
                subjects.append(sub)
    text = P.NOT_FOUND[lang]
    if subjects:
        text += "\n\n" + P.CLOSEST[lang] + "\n" + "\n".join(f"- {s_}" for s_ in subjects[:6])
    return {"answer": text, "citations": [], "status": "not_found", "messages": [AIMessage(content=text)],
            "answer_log": _logged(state, text, "not_found")}
