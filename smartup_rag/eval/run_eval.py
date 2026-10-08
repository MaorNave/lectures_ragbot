"""Evaluation runner for the SmartUp RAG chatbot.

Two modes, one question set (``questions.jsonl`` next to this file):

* **retrieval** (default, no LLM, ~1 s per item): for every in-scope item call
  :func:`smartup_rag.retrieval.hybrid.retrieve` with the expected course version and check whether an
  expected lecture (number, optionally part) appears among the returned passages — *hit@k* with
  ``k = settings.top_k_passages``. Items whose ``expect.version`` is ``null`` (v1-vs-v2 comparisons) run one
  pass per version and count as a hit only when the lecture is found in **both**.
* **full** (``--full``, ~60–100 s per item on this machine): additionally run the LangGraph agent
  (:func:`smartup_rag.agent.graph.run_chat`) and score the status, the citations and the v1-fallback
  behaviour against the expectation.

Results are printed as a table plus aggregates by kind and language, and written as JSON to
``data/eval_results/<timestamp>.json`` (git-ignored).

Usage::

    .venv/bin/python -m smartup_rag.eval.run_eval                 # retrieval mode, whole set
    .venv/bin/python -m smartup_rag.eval.run_eval --full --limit 2 # agent end-to-end for the first 2 items
    .venv/bin/python -m smartup_rag.eval.run_eval --ids en_fud_ibm,he_four_dimensions --full
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from smartup_rag.config import PROJECT_ROOT, get_settings
from smartup_rag.lang import Lang, lecture_number_in_text
from smartup_rag.tiers import tier_of

log = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "eval_results"

Kind = Literal[
    "conceptual", "lecture_specific", "homework", "summary", "comparison", "cross_lecture", "out_of_scope", "v1_only"
]
Status = Literal["answered", "refused", "not_found"]

# "11p2" → (11, "2"); 5 → (5, None)
_PART_SPEC_RE = re.compile(r"^(\d+)p(\d)$")


# ----------------------------------------------------------------------------------------
# Question set
# ----------------------------------------------------------------------------------------
@dataclass
class Expect:
    """What a correct system should do for one question."""

    status: Status = "answered"
    version: str | None = "v2"  # source tier the answer must come from (v2 / tedx / v1 / outsource); None → v1 + v2
    lectures: list[int | str] = field(default_factory=list)  # 3, 11, "11p1", "11p2"
    min_hits: int = 1  # distinct expected lectures that must be retrieved / cited


@dataclass
class Item:
    id: str
    question: str
    language: Lang
    kind: str
    expect: Expect
    notes: str = ""

    @property
    def in_scope(self) -> bool:
        """Retrieval mode evaluates only items that must be answered from lectures."""
        return self.expect.status == "answered" and bool(self.expect.lectures)

    @property
    def versions(self) -> list[str]:
        return [self.expect.version] if self.expect.version else ["v2", "v1"]


def parse_lecture_spec(spec: int | str) -> tuple[int, str | None]:
    """``5`` → ``(5, None)``; ``"11p2"`` → ``(11, "2")``; ``"7"`` → ``(7, None)``."""
    if isinstance(spec, int):
        return spec, None
    m = _PART_SPEC_RE.match(str(spec).strip())
    if m:
        return int(m.group(1)), m.group(2)
    return int(str(spec).strip()), None


def lecture_matches(spec: tuple[int, str | None], number: int | None, part: str | None) -> bool:
    """A spec without a part matches any part of that lecture; ``"11p2"`` matches only Part 2."""
    if number is None or int(number) != spec[0]:
        return False
    return spec[1] is None or (part or None) == spec[1]


def load_questions(path: Path = QUESTIONS_PATH) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid JSON ({exc})") from exc
        exp = obj.get("expect") or {}
        item = Item(
            id=str(obj["id"]),
            question=str(obj["question"]).strip(),
            language=obj.get("language", "en"),
            kind=obj.get("kind", "conceptual"),
            expect=Expect(
                status=exp.get("status", "answered"),
                version=exp.get("version", "v2"),
                lectures=list(exp.get("lectures") or []),
                min_hits=int(exp.get("min_hits", 1)),
            ),
            notes=obj.get("notes", ""),
        )
        if item.id in seen:
            raise ValueError(f"{path.name}:{line_no}: duplicate id {item.id!r}")
        seen.add(item.id)
        for spec in item.expect.lectures:
            parse_lecture_spec(spec)  # fail fast on typos
        items.append(item)
    return items


# ----------------------------------------------------------------------------------------
# Retrieval mode
# ----------------------------------------------------------------------------------------
def _doc_key(meta: dict[str, Any]) -> str:
    """The lecture key (course-prefixed for outside sources, so the supplemental talks — all "lecture 1" — differ)."""
    if meta.get("lecture_key"):
        return str(meta["lecture_key"])
    part = meta.get("lecture_part")
    return f"{meta.get('course_version')}:L{meta.get('lecture_number')}" + (f"p{part}" if part else "")


def _tier(c: dict[str, Any]) -> str:
    return str(c.get("tier") or tier_of(c.get("course_id"), c.get("course_version")))


def _retrieval_pass(item: Item, version: str, lecture_number: int | None) -> dict[str, Any]:
    """One `retrieve()` call for one version → hit / first_rank / top_score / returned lecture keys."""
    from smartup_rag.retrieval.hybrid import retrieve

    specs = [parse_lecture_spec(s) for s in item.expect.lectures]
    t0 = time.perf_counter()
    try:
        docs = retrieve(item.question, item.language, version, lecture_number)
        error = None
    except Exception as exc:  # network / index problems: record, keep going
        docs, error = [], f"{type(exc).__name__}: {exc}"
        log.warning("retrieve failed for %s (%s): %s", item.id, version, error)
    elapsed = time.perf_counter() - t0

    found: set[tuple[int, str | None]] = set()
    first_rank: int | None = None
    returned: list[str] = []
    for rank, d in enumerate(docs, 1):
        m = d.metadata
        key = _doc_key(m)
        if key not in returned:
            returned.append(key)
        for spec in specs:
            if lecture_matches(spec, m.get("lecture_number"), m.get("lecture_part")):
                found.add(spec)
                first_rank = first_rank or rank
    scores = [d.metadata.get("rerank_score") for d in docs if d.metadata.get("rerank_score") is not None]
    return {
        "version": version,
        "lecture_filter": lecture_number,
        "hit": len(found) >= max(1, item.expect.min_hits),
        "distinct_found": len(found),
        "first_rank": first_rank,
        "top_score": max(scores) if scores else None,
        "n_docs": len(docs),
        "returned": returned,
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }


def eval_retrieval(item: Item, *, filter_lecture: bool = True) -> dict[str, Any]:
    """hit@k for one item (one pass per expected version; comparison items must hit in every version).

    ``filter_lecture`` defaults to True so the measurement mirrors the chatbot: ``agent.nodes.classify``
    applies the same ``lecture_number_in_text`` override, so a question naming a lecture is always
    retrieved with that filter. Measuring without it understates the system — three of four apparent
    misses in the first run were this artifact.
    """
    lecture_number = lecture_number_in_text(item.question) if filter_lecture else None
    passes = [_retrieval_pass(item, v, lecture_number) for v in item.versions]
    ranks = [p["first_rank"] for p in passes if p["first_rank"] is not None]
    scores = [p["top_score"] for p in passes if p["top_score"] is not None]
    return {
        "hit": all(p["hit"] for p in passes),
        "first_rank": min(ranks) if ranks and all(p["hit"] for p in passes) else (min(ranks) if ranks else None),
        "top_score": max(scores) if scores else None,
        "elapsed_s": round(sum(p["elapsed_s"] for p in passes), 2),
        "passes": passes,
    }


# ----------------------------------------------------------------------------------------
# Full mode (agent end-to-end)
# ----------------------------------------------------------------------------------------
def _citation_ok(item: Item, c: dict[str, Any]) -> bool:
    if item.expect.version and _tier(c) != item.expect.version:
        return False
    return any(
        lecture_matches(parse_lecture_spec(s), c.get("lecture_number"), c.get("lecture_part"))
        for s in item.expect.lectures
    )


def eval_full(item: Item, *, strict_citations: bool = False) -> dict[str, Any]:
    """Run the agent once and score status / citations / fallback against the expectation.

    * ``status_ok``   – ``run_chat().status`` equals ``expect.status``.
    * ``cite_any``    – at least ``min_hits`` distinct expected lectures are cited (default pass criterion).
    * ``cite_all``    – every citation points to an expected lecture (pass criterion with ``strict_citations``).
      For refused / not_found expectations both require that nothing is cited.
    * ``fallback_ok`` – expected v1 ⇒ ``used_fallback`` or ``version_scope == "v1"``; expected v2 ⇒ no fallback;
      comparison (``version`` null) ⇒ both versions cited.
    """
    from smartup_rag.agent.graph import run_chat

    t0 = time.perf_counter()
    error: str | None = None
    try:
        res = run_chat(item.question, thread_id=f"eval-{item.id}")
    except Exception as exc:  # the agent must never abort the whole run
        error = f"{type(exc).__name__}: {exc}"
        log.warning("run_chat failed for %s: %s", item.id, error)
        res = {"answer": "", "status": "error", "citations": [], "used_fallback": False, "version_scope": None}
    elapsed = time.perf_counter() - t0

    cited: list[dict[str, Any]] = list(res.get("citations") or [])
    cited_keys = [_doc_key(c) for c in cited]
    status_ok = res.get("status") == item.expect.status

    if item.expect.status == "answered":
        matching = {
            parse_lecture_spec(s)
            for s in item.expect.lectures
            for c in cited
            if lecture_matches(parse_lecture_spec(s), c.get("lecture_number"), c.get("lecture_part"))
            and (not item.expect.version or _tier(c) == item.expect.version)
        }
        cite_any = len(matching) >= max(1, item.expect.min_hits)
        cite_all = bool(cited) and all(_citation_ok(item, c) for c in cited)
        if item.expect.version in ("tedx", "v1", "outsource"):  # the fallback chain (or an explicit scope) reached it
            fallback_ok = res.get("fallback_tier") == item.expect.version or res.get("version_scope") == item.expect.version
        elif item.expect.version == "v2":
            fallback_ok = not res.get("used_fallback")
        else:  # comparison: both versions must be represented in the citations
            fallback_ok = {_tier(c) for c in cited} >= {"v1", "v2"}
    else:  # refused / not_found: nothing may be cited; fallback irrelevant
        cite_any = cite_all = not cited
        fallback_ok = True

    citations_ok = cite_all if strict_citations else cite_any
    answer = res.get("answer") or ""
    return {
        "status": res.get("status"),
        "status_ok": status_ok,
        "cite_any": cite_any,
        "cite_all": cite_all,
        "citations_ok": citations_ok,
        "fallback_ok": fallback_ok,
        "passed": status_ok and citations_ok and fallback_ok,
        "used_fallback": bool(res.get("used_fallback")),
        "intent": res.get("intent"),
        "version_scope": res.get("version_scope"),
        "lecture_number": res.get("lecture_number"),
        "grade_v2": res.get("grade_v2"),
        "grade_v1": res.get("grade_v1"),
        "fallback_tier": res.get("fallback_tier"),
        "fallback_tries": res.get("fallback_tries"),
        "n_passages": res.get("n_passages"),
        "cited": cited_keys,
        "language": res.get("language"),
        "answer_preview": re.sub(r"\s+", " ", answer)[:300],
        "answer": answer,
        "elapsed_s": round(elapsed, 1),
        "error": error,
    }


# ----------------------------------------------------------------------------------------
# Aggregation + reporting
# ----------------------------------------------------------------------------------------
def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("retrieval")]
    hits = [r for r in scored if r["retrieval"]["hit"]]
    ranks = [r["retrieval"]["first_rank"] for r in hits if r["retrieval"]["first_rank"]]
    return {
        "n": len(scored),
        "hits": len(hits),
        "hit_rate": _rate(len(hits), len(scored)),
        # one term per scored item: 1/rank for genuine hits, 0 otherwise (a ranked-but-not-hit item is not a hit)
        "mrr": _mean(
            [
                (1.0 / r["retrieval"]["first_rank"]) if (r["retrieval"]["hit"] and r["retrieval"]["first_rank"]) else 0.0
                for r in scored
            ]
        )
        if scored
        else None,
        "mean_first_rank": _mean([float(x) for x in ranks]),
        "mean_top_score": _mean([r["retrieval"]["top_score"] for r in scored if r["retrieval"]["top_score"] is not None]),
        "misses": [r["id"] for r in scored if not r["retrieval"]["hit"]],
    }


def aggregate_full(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("full")]
    f = [r["full"] for r in scored]
    return {
        "n": len(scored),
        "passed": sum(1 for x in f if x["passed"]),
        "pass_rate": _rate(sum(1 for x in f if x["passed"]), len(f)),
        "status_ok": _rate(sum(1 for x in f if x["status_ok"]), len(f)),
        "citations_ok": _rate(sum(1 for x in f if x["citations_ok"]), len(f)),
        "cite_all": _rate(sum(1 for x in f if x["cite_all"]), len(f)),
        "fallback_ok": _rate(sum(1 for x in f if x["fallback_ok"]), len(f)),
        "errors": sum(1 for x in f if x["error"]),
        "mean_elapsed_s": _mean([x["elapsed_s"] for x in f]),
        "failed": [r["id"] for r in scored if not r["full"]["passed"]],
    }


def _group(rows: list[dict[str, Any]], key: str, fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {k: fn(v) for k, v in sorted(groups.items())}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=False)) for r in cells]
    return "\n".join([line, sep, *body])


def print_report(result: dict[str, Any], *, out=sys.stdout) -> None:
    rows = result["items"]
    full = result["mode"] == "full"
    headers = ["id", "kind", "lang", "hit", "first_rank", "top_score"]
    if full:
        headers += ["status(exp)", "status", "cite", "fb", "pass", "t(s)"]
    table_rows = []
    for r in rows:
        ret = r.get("retrieval") or {}
        row: list[Any] = [
            r["id"],
            r["kind"],
            r["language"],
            ret.get("hit") if ret else "skip",
            ret.get("first_rank") if ret else "-",
            ret.get("top_score") if ret else "-",
        ]
        if full:
            f = r.get("full") or {}
            row += [
                r["expect"]["status"],
                f.get("status", "-"),
                f.get("citations_ok") if f else "-",
                f.get("fallback_ok") if f else "-",
                f.get("passed") if f else "-",
                f.get("elapsed_s", "-"),
            ]
        table_rows.append(row)
    print(_table(headers, table_rows), file=out)

    agg = result["aggregates"]["retrieval"]
    print("\nRetrieval hit@k (k = top_k_passages) — overall:", file=out)
    o = agg["overall"]
    print(f"  n={o['n']}  hits={o['hits']}  hit_rate={_fmt(o['hit_rate'])}  mrr={_fmt(o['mrr'])}  "
          f"mean_top_score={_fmt(o['mean_top_score'])}  misses={o['misses']}", file=out)
    for label in ("by_language", "by_kind"):
        print(f"  {label}:", file=out)
        for k, v in agg[label].items():
            print(f"    {k:<18} n={v['n']:<3} hits={v['hits']:<3} hit_rate={_fmt(v['hit_rate']):<6} "
                  f"mrr={_fmt(v['mrr']):<6} misses={v['misses']}", file=out)

    if full:
        agg = result["aggregates"]["full"]
        o = agg["overall"]
        print("\nAgent end-to-end — overall:", file=out)
        print(f"  n={o['n']}  passed={o['passed']}  pass_rate={_fmt(o['pass_rate'])}  status_ok={_fmt(o['status_ok'])}  "
              f"citations_ok={_fmt(o['citations_ok'])}  cite_all={_fmt(o['cite_all'])}  fallback_ok={_fmt(o['fallback_ok'])}  "
              f"errors={o['errors']}  mean_t={_fmt(o['mean_elapsed_s'])}s  failed={o['failed']}", file=out)
        for label in ("by_language", "by_kind"):
            print(f"  {label}:", file=out)
            for k, v in agg[label].items():
                print(f"    {k:<18} n={v['n']:<3} passed={v['passed']:<3} pass_rate={_fmt(v['pass_rate']):<6} "
                      f"status_ok={_fmt(v['status_ok']):<6} citations_ok={_fmt(v['citations_ok']):<6} "
                      f"fallback_ok={_fmt(v['fallback_ok']):<6} failed={v['failed']}", file=out)
        for r in rows:
            f = r.get("full")
            if f:
                print(f"\n[{r['id']}] status={f['status']} intent={f['intent']} fallback={f['used_fallback']} "
                      f"cited={f['cited']} t={f['elapsed_s']}s", file=out)
                print(f"  {f['answer_preview']}", file=out)
    print(f"\nJSON written to {result['output_path']}", file=out)


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------
def _safe_settings() -> dict[str, Any]:
    """Settings worth recording with a result — never credentials."""
    s = get_settings()
    return {
        "index": f"local:{s.collection_name}",
        "embed_model": s.embed_model,
        "rerank_enabled": s.rerank_enabled,
        "rerank_model": s.rerank_model,
        "top_k_passages": s.top_k_passages,
        "rank_window": s.rank_window,
        "cross_lingual": s.cross_lingual,
        "min_rerank_score": s.min_rerank_score,
        "grade_strong_threshold": s.grade_strong_threshold,
        "grade_weak_threshold": s.grade_weak_threshold,
        "chat_model": s.chat_model,
        "classifier_model": s.classifier_model,
    }


def run(
    full: bool = False,
    limit: int | None = None,
    out_dir: Path | None = None,
    *,
    ids: list[str] | None = None,
    filter_lecture: bool = True,
    strict_citations: bool = False,
    questions_path: Path = QUESTIONS_PATH,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the evaluation and return the result dict (also written to ``out_dir``).

    Args:
        full: also run the agent (`run_chat`) for every selected item — slow (≈60–100 s per item).
        limit: evaluate only the first N selected items (file order).
        out_dir: where to write ``<timestamp>.json`` (default ``data/eval_results``).
        ids: restrict to these question ids (applied before ``limit``).
        filter_lecture: retrieval mode only — pass the lecture number named in the question as a filter,
            mirroring the agent's deterministic override (default on). Set False to measure pure topical
            retrieval instead, which is a harder and less representative task.
        strict_citations: full mode — require *every* citation to point to an expected lecture
            (default: at least ``min_hits`` expected lectures cited).
        questions_path: alternative question file.
        quiet: do not print the report.
    """
    items = load_questions(questions_path)
    if ids:
        wanted = [i.strip() for i in ids if i.strip()]
        unknown = sorted(set(wanted) - {it.id for it in items})
        if unknown:
            raise ValueError(f"unknown question id(s): {unknown}")
        items = [it for it in items if it.id in wanted]
    if limit is not None:
        items = items[: max(0, limit)]
    if not items:
        raise ValueError("no questions selected")

    if full:
        from smartup_rag.agent.llm import ollama_health

        health = ollama_health()
        if not health.get("ok") or not health.get("model_present"):
            raise RuntimeError(f"Ollama not ready for full mode: {health}")
        est = len(items) * 80
        print(
            f"WARNING: full mode runs the LLM for {len(items)} question(s); expect ≈60–100 s each "
            f"(≈{est // 60} min total). Model: {health['chat_model']}",
            file=sys.stderr,
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for n, item in enumerate(items, 1):
        row: dict[str, Any] = {
            "id": item.id,
            "kind": item.kind,
            "language": item.language,
            "question": item.question,
            "expect": asdict(item.expect),
            "notes": item.notes,
            "retrieval": None,
            "full": None,
        }
        if item.in_scope:
            row["retrieval"] = eval_retrieval(item, filter_lecture=filter_lecture)
        else:
            row["skipped_retrieval"] = "not answerable from lectures (expected status != answered)"
        if full:
            print(f"[{n}/{len(items)}] {item.id} …", file=sys.stderr, flush=True)
            row["full"] = eval_full(item, strict_citations=strict_citations)
            print(f"    → {row['full']['status']} in {row['full']['elapsed_s']}s", file=sys.stderr, flush=True)
        rows.append(row)

    aggregates: dict[str, Any] = {
        "retrieval": {
            "overall": aggregate_retrieval(rows),
            "by_kind": _group([r for r in rows if r["retrieval"]], "kind", aggregate_retrieval),
            "by_language": _group([r for r in rows if r["retrieval"]], "language", aggregate_retrieval),
        }
    }
    if full:
        aggregates["full"] = {
            "overall": aggregate_full(rows),
            "by_kind": _group(rows, "kind", aggregate_full),
            "by_language": _group(rows, "language", aggregate_full),
        }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}{'-full' if full else ''}.json"
    result = {
        "timestamp": stamp,
        "mode": "full" if full else "retrieval",
        "questions_file": str(questions_path),
        "options": {"limit": limit, "ids": ids, "filter_lecture": filter_lecture, "strict_citations": strict_citations},
        "settings": _safe_settings(),
        "n_items": len(rows),
        "aggregates": aggregates,
        "items": rows,
        "output_path": str(out_path),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print_report(result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SmartUp RAG retrieval (default) or the full agent (--full).")
    p.add_argument("--full", action="store_true", help="also run the agent for each item (≈60–100 s each)")
    p.add_argument("--limit", type=int, default=None, help="only the first N selected questions")
    p.add_argument("--ids", type=str, default=None, help="comma-separated question ids to run")
    p.add_argument("--out", type=Path, default=None, help=f"output directory (default {DEFAULT_OUT_DIR})")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="alternative questions.jsonl")
    p.add_argument(
        "--no-lecture-filter",
        action="store_true",
        help="retrieval mode: do not filter by the lecture number named in the question (the agent does)",
    )
    p.add_argument("--strict-citations", action="store_true", help="full mode: every citation must be expected")
    p.add_argument("-v", "--verbose", action="store_true", help="show retrieval/agent INFO logs")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run(
        full=a.full,
        limit=a.limit,
        out_dir=a.out,
        ids=a.ids.split(",") if a.ids else None,
        filter_lecture=not a.no_lecture_filter,
        strict_citations=a.strict_citations,
        questions_path=a.questions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
