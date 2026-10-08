"""Query builders for the local hybrid retriever.

Pure functions — nothing here touches the index — so they are trivial to unit-test and to print
(`smartup check --explain`).

Retrieval is *hybrid*: three ranked "legs" are fused with Reciprocal Rank Fusion (`rrf_fuse`):
  - dense:    the question's bge-m3 embedding against every passage vector in Chroma (cosine)
  - bm25:     SQLite FTS5 BM25 over the passage text
  - metadata: SQLite FTS5 BM25 over the section topic, lecture subjects and lecture title
Filters implement the instructions' hard rules — the source tier (v2 / tedx / v1 / outsource, see
`smartup_rag.tiers`) and the lecture number — and are applied inside every leg: as a Chroma `where`
filter (`chroma_where`) and as a SQL clause on the passages table (`sql_where`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from smartup_rag.lang import Lang
from smartup_rag.tiers import Condition, tier_conditions

DIGITAL_INNOVATION_COURSE_ID = "digital-innovation-idc"  # an outside course (tier "outsource"); INCLUDE_DIGITAL_INNOVATION=false hides it

#: Fields a condition may name: they exist on the passages table and in every Chroma metadata record.
FILTER_FIELDS = frozenset({"course_id", "course_version", "lecture_number", "lecture_key", "lang"})

#: FTS5 columns of `passages_fts`, in table order, and the BM25 weight of each column per leg.
FTS_COLUMNS = ("text", "section_topic", "subjects_text", "lecture_title")
BM25_TEXT_WEIGHTS = (1.0, 0.0, 0.0, 0.0)
# section_topic^2, lecture_subjects^2, lecture_title — the metadata leg of the previous Elasticsearch version
BM25_METADATA_WEIGHTS = (0.0, 2.0, 2.0, 1.0)
METADATA_COLUMNS = ("section_topic", "subjects_text", "lecture_title")

MAX_QUERY_TERMS = 48
_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)  # letters and digits of any script; "_" and punctuation split terms


def other_lang(lang: Lang) -> Lang:
    return "he" if lang == "en" else "en"


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def conditions(
    tier: str | None = None,
    lecture_number: int | None = None,
    course_id: str | None = None,
    lang: str | None = None,
    exclude_course_ids: Iterable[str] | None = None,
) -> list[Condition]:
    """`tier` is a source tier name; the course versions "v1" / "v2" are tiers too."""
    out: list[Condition] = []
    if tier:
        out.extend(tier_conditions(tier))
    if lecture_number is not None:
        out.append(("lecture_number", "eq", int(lecture_number)))
    if course_id:
        out.append(("course_id", "eq", course_id))
    if lang:
        out.append(("lang", "eq", lang))
    excluded = sorted(set(exclude_course_ids or []))
    if excluded:
        out.append(("course_id", "nin", excluded))
    return out


def _check(field: str, op: str) -> None:
    if field not in FILTER_FIELDS:
        raise ValueError(f"cannot filter on {field!r}; expected one of {sorted(FILTER_FIELDS)}")
    if op not in ("eq", "ne", "in", "nin"):
        raise ValueError(f"unknown filter operator {op!r}")


def chroma_where(conds: Sequence[Condition]) -> dict[str, Any] | None:
    """Chroma `where` filter: one condition as is, several under `$and`, none → None (no filter)."""
    clauses = []
    for field, op, value in conds:
        _check(field, op)
        clauses.append({field: {f"${op}": list(value) if op in ("in", "nin") else value}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def sql_where(conds: Sequence[Condition], alias: str = "p") -> tuple[str, list[Any]]:
    """`AND`-joined SQL predicates on the passages (or lectures) table plus their parameters; ("", []) for none."""
    parts: list[str] = []
    params: list[Any] = []
    for field, op, value in conds:
        _check(field, op)
        col = f"{alias}.{field}" if alias else field
        if op in ("in", "nin"):
            values = list(value)
            if not values:  # x IN () is always false, x NOT IN () always true
                if op == "in":
                    parts.append("0")
                continue
            marks = ", ".join("?" for _ in values)
            parts.append(f"{col} {'IN' if op == 'in' else 'NOT IN'} ({marks})")
            params.extend(values)
        else:
            parts.append(f"{col} {'=' if op == 'eq' else '!='} ?")
            params.append(value)
    return " AND ".join(parts), params


# --------------------------------------------------------------------------------------
# FTS5 keyword legs
# --------------------------------------------------------------------------------------
def query_terms(query: str, max_terms: int = MAX_QUERY_TERMS) -> list[str]:
    """The question as FTS terms: lower-cased words of any script, in order, without repeats.

    Single letters are dropped (the "s" of "Yonatan's"), single digits kept ("lecture 5"). The FTS index uses the
    `unicode61` tokenizer, which splits text on the same boundaries, so every term can match an indexed token.
    """
    seen: list[str] = []
    for t in _TERM_RE.findall((query or "").lower()):
        if (len(t) > 1 or t.isdigit()) and t not in seen:
            seen.append(t)
        if len(seen) >= max_terms:
            break
    return seen


def fts_match(terms: Sequence[str], columns: Sequence[str] = ("text",)) -> str | None:
    """FTS5 MATCH expression: any of the terms (OR, like Elasticsearch's `match` with operator "or"), restricted to
    `columns`. Every term is a quoted string, so FTS5 operators and punctuation in a question are never syntax."""
    if not terms:
        return None
    for col in columns:
        if col not in FTS_COLUMNS:
            raise ValueError(f"unknown FTS column {col!r}")
    ors = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
    return "{" + " ".join(columns) + "} : (" + ors + ")"


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------
def rrf_fuse(rankings: Sequence[Sequence[str]], rank_constant: int = 60, window: int = 30) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (rank_constant + rank of d in each leg), ranks from 1.

    Each leg contributes its first `window` ids; the fused list is cut to `window` as well (what Elasticsearch's
    `rrf` retriever did with `rank_window_size` = `size`). Ties keep the order in which ids were first seen —
    legs in the order given, best rank first — so the result is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking[:window], 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (rank_constant + rank)
    order = {pid: i for i, pid in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))[:window]
