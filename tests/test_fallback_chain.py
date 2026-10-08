"""The fallback chain v2 → TEDx → v1 → outsource (instructions §7), with search and grading mocked per tier."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from smartup_rag.agent import nodes as N
from smartup_rag.agent import prompts as P
from smartup_rag.config import get_settings
from tests.conftest import DI_L3, TEDX, V1_L3, V2_L5, make_doc, make_source_doc

DOCS = {"tedx": [make_source_doc(TEDX)], "v1": [make_doc(V1_L3)], "outsource": [make_source_doc(DI_L3)]}


def _state(question="Why do startups fail?", lecture=None, scope="default", intent="conceptual"):
    return {"messages": [HumanMessage(content=question)], "question": question, "query": question, "language": "en",
            "classification": {"in_scope": True, "intent": intent, "lecture_number": lecture, "version_scope": scope},
            "docs_v2": [make_doc(V2_L5)] if lecture else [], "docs": [], "notes": [], "grade_v2": {"verdict": "insufficient"}}


class Recorder:
    def __init__(self):
        self.searched: list[tuple[str, int | None]] = []  # every search (they run in parallel: order is not fixed)
        self.graded: list[str] = []  # the grades, in chain order
        self.progress: list[str] = []


@pytest.fixture
def chain(monkeypatch):
    """verdicts: {tier: verdict}; docs: optional {tier: [Document]} to control the rerank scores."""
    rec = Recorder()

    def install(verdicts, docs=None):
        pool = {**DOCS, **(docs or {})}

        def fake_retrieve(query, lang, tier, n, part, notes):
            rec.searched.append((tier, n))
            return list(pool[tier])

        def fake_grade(question, docs, history, config, **kw):
            tier = docs[0].metadata["tier"]
            rec.graded.append(tier)
            return {"verdict": verdicts[tier], "method": "llm"}

        monkeypatch.setattr(N, "_safe_retrieve", fake_retrieve)
        monkeypatch.setattr(N, "_grade", fake_grade)
        monkeypatch.setattr(N, "_progress", lambda config, lang, tier, first: rec.progress.append(tier))
        return rec

    return install


def test_tedx_is_the_first_fallback(chain):
    rec = chain({"tedx": "sufficient", "v1": "sufficient", "outsource": "sufficient"})
    out = N._fallback(_state(), None)
    assert {t for t, _ in rec.searched} == {"tedx", "v1", "outsource"}  # searched together …
    assert rec.graded == ["tedx"] and rec.progress == ["tedx"]  # … but graded in order, stopping at the first answer
    assert out["fallback_tier"] == "tedx" and out["docs"] == DOCS["tedx"]
    assert P.FALLBACK_NOTES["tedx"]["en"] in out["notes"][-1]
    assert N.route_after_fallback(out) == "generate"


def test_the_chain_moves_on_until_a_source_answers(chain):
    rec = chain({"tedx": "insufficient", "v1": "insufficient", "outsource": "partial"})
    out = N._fallback(_state(), None)
    assert rec.graded == ["tedx", "v1", "outsource"] and rec.progress == ["tedx", "v1", "outsource"]
    assert out["fallback_tier"] == "outsource"
    assert [t["tier"] for t in out["fallback_tries"]] == ["tedx", "v1", "outsource"]
    assert out["docs_fallback"] == DOCS["tedx"] + DOCS["v1"] + DOCS["outsource"]
    assert out["docs_v1"] == DOCS["v1"]  # the v1 step is still visible on its own


def test_nothing_anywhere_ends_in_not_found_with_topics_from_every_source(chain):
    chain({"tedx": "insufficient", "v1": "insufficient", "outsource": "insufficient"})
    state = _state()
    state.update(N._fallback(state, None))
    assert state["docs"] == [] and state["fallback_tier"] is None
    assert N.route_after_fallback(state) == "not_found"
    out = N._not_found(state)
    assert out["status"] == "not_found" and P.CLOSEST["en"] in out["answer"]


def test_a_named_lecture_only_falls_back_to_v1(chain):
    """'Lecture 3' is course numbering: the TEDx talk and the outside lectures have no such lecture."""
    rec = chain({"tedx": "sufficient", "v1": "partial", "outsource": "sufficient"})
    out = N._fallback(_state("What does lecture 3 say about pricing?", lecture=3), None)
    assert rec.searched == [("v1", 3)]
    assert out["fallback_tier"] is None  # v2 has Lecture 3, so v1 must be *sufficient* (§4.2), partial is not enough


@pytest.mark.parametrize(("question", "scope"), [
    ("What did Yonatan say in his TEDx talk about being first?", "tedx"),
    ("מה יונתן אמר בהרצאת ה-TEDx שלו?", "tedx"),
    ("What does the Digital Innovation course say about product innovation?", "outsource"),
    ("What changed between v1 and v2 lecture 1?", "both"),
    ("Why do nine out of ten startups fail?", "default"),  # the phrase alone is course content, not the talk
])
def test_explicit_sources_are_detected(question, scope):
    assert N.normalise_version_scope(question, "default", "conceptual") == scope


def test_an_explicit_outside_scope_searches_that_source_alone(monkeypatch):
    searched = []
    monkeypatch.setattr(N, "_safe_retrieve", lambda q, lang, tier, n, part, notes: searched.append(tier) or list(DOCS["tedx"]))
    state = _state("What did Yonatan say in his TEDx talk?", scope="tedx")
    state.update(N.retrieve_node(state))
    assert searched == ["tedx"] and state["docs"] == DOCS["tedx"]
    grade = N._grade_v2(state, None)["grade_v2"]
    assert grade["method"] == "skip"  # explicit source: no grading, no chain


# --------------------------------------------------------------------------------------------------------------
# speed-ups: the skip rule and the background searches
# --------------------------------------------------------------------------------------------------------------
def _weak(tier, score):
    src = {"tedx": TEDX, "outsource": DI_L3}.get(tier)
    return [make_source_doc(src, rerank_score=score)] if src else [make_doc(V1_L3, rerank_score=score)]


def _cutoffs():
    s = get_settings()
    return s.skip_weak_grade_below_tedx, s.skip_weak_grade_below, s.grade_strong_threshold + 0.1


def _in_skip_band(cutoff):
    """A best score the skip rule applies to: not low enough for the fast path to reject it, but below the cutoff."""
    return (get_settings().grade_weak_threshold + cutoff) / 2


def test_a_weak_source_is_skipped_when_a_later_one_is_strong(chain):
    # TEDx and v1 below their cutoffs, outsource strong → neither weak source costs an LLM call
    tedx_cut, v1_cut, strong = _cutoffs()
    rec = chain({"tedx": "sufficient", "v1": "sufficient", "outsource": "sufficient"},
                {"tedx": _weak("tedx", _in_skip_band(tedx_cut)), "v1": _weak("v1", _in_skip_band(v1_cut)),
                 "outsource": _weak("outsource", strong)})
    out = N._fallback(_state(), None)
    assert rec.graded == ["outsource"] and out["fallback_tier"] == "outsource"
    assert [t["method"] for t in out["fallback_tries"]] == ["skip-weak", "skip-weak", "llm"]


def test_the_v1_cutoff_is_lower_than_the_tedx_one(chain):
    # the same score is skipped for the TEDx talk but graded for v1, whose cutoff is lower
    tedx_cut, v1_cut, strong = _cutoffs()
    assert v1_cut < tedx_cut
    score = (v1_cut + tedx_cut) / 2
    rec = chain({"tedx": "insufficient", "v1": "sufficient", "outsource": "sufficient"},
                {"tedx": _weak("tedx", score), "v1": _weak("v1", score), "outsource": _weak("outsource", strong)})
    out = N._fallback(_state(), None)
    assert rec.graded == ["v1"] and out["fallback_tier"] == "v1"


def test_nothing_is_skipped_without_a_strong_later_source(chain):
    rec = chain({"tedx": "insufficient", "v1": "insufficient", "outsource": "insufficient"},
                {"tedx": _weak("tedx", 0.10), "v1": _weak("v1", 0.12), "outsource": _weak("outsource", 0.30)})
    N._fallback(_state(), None)
    assert rec.graded == ["tedx", "v1", "outsource"]


def test_weak_v2_starts_the_fallback_searches_in_the_background(chain):
    rec = chain({"tedx": "sufficient", "v1": "sufficient", "outsource": "sufficient"}, {"v2": []})
    state = _state()
    state["turn_id"] = "turn-weak"
    state.update(N.retrieve_node(state))  # the fake v2 search returns nothing: weak
    assert set(N._PREFETCHED["turn-weak"]) == {"tedx", "v1", "outsource"}
    for f in N._PREFETCHED["turn-weak"].values():
        f.result()  # let the background searches finish before counting
    rec.searched.clear()
    N._fallback(state, None)
    assert rec.searched == []  # the chain used the background results, it did not search again
    assert "turn-weak" not in N._PREFETCHED


def test_strong_v2_starts_no_background_search(chain, monkeypatch):
    chain({"tedx": "sufficient", "v1": "sufficient", "outsource": "sufficient"})
    monkeypatch.setattr(N, "_safe_retrieve", lambda *a, **k: [make_doc(V2_L5, rerank_score=0.9)])
    state = _state()
    state["turn_id"] = "turn-strong"
    N.retrieve_node(state)
    assert "turn-strong" not in N._PREFETCHED


def test_the_background_searches_are_dropped_when_v2_answers(monkeypatch):
    N._PREFETCHED["turn-v2"] = {}
    monkeypatch.setattr(N, "_grade", lambda *a, **k: {"verdict": "partial", "method": "llm"})
    state = _state()
    state.update(turn_id="turn-v2", docs_v2=[make_doc(V2_L5, rerank_score=0.2)])
    N._grade_v2(state, None)
    assert "turn-v2" not in N._PREFETCHED


def test_the_real_graph_reuses_the_background_searches(monkeypatch):
    """Through LangGraph itself: state keys the schema does not declare are dropped between nodes, so this is where
    a lost turn id would show up as every fallback source being searched twice."""
    from collections import Counter

    from smartup_rag.agent import graph as G

    searches = Counter()
    pool = {"v2": [make_doc(V2_L5, rerank_score=0.13)], **DOCS}

    def fake_retrieve(query, lang, tier, n, part, notes):
        searches[tier] += 1
        return list(pool[tier])

    monkeypatch.setattr(N, "_safe_retrieve", fake_retrieve)
    monkeypatch.setattr(N, "classify", lambda state, config: {"classification": {
        "in_scope": True, "intent": "conceptual", "lecture_number": None, "version_scope": "default"}, "query": state["question"]})
    during_v2_grade: list[set[str]] = []

    def fake_grade(q, docs, history, config, **kw):
        tier = docs[0].metadata["tier"]
        if tier == "v2":  # the background searches must already be under way while v2 is being graded
            during_v2_grade.append({t for entry in N._PREFETCHED.values() for t in entry})
        return {"verdict": "sufficient" if tier == "v1" else "insufficient", "method": "llm"}

    monkeypatch.setattr(N, "_grade", fake_grade)
    monkeypatch.setattr(N, "generate", lambda state, config: {"answer": "From v1 (v1, Lecture 3 — \"Foundations Lecture 3\")."})
    out = G.build_graph().invoke({"messages": [HumanMessage(content="Why do startups fail?")]},
                                 config={"configurable": {"thread_id": "t-prefetch"}})
    assert out["fallback_tier"] == "v1"
    assert during_v2_grade == [{"tedx", "v1", "outsource"}]  # started in the background, before the chain ran …
    assert searches == Counter({"v2": 1, "tedx": 1, "v1": 1, "outsource": 1})  # … and reused, not searched again


def test_fast_grade_can_leave_a_strong_score_to_the_grader():
    strong = get_settings().grade_strong_threshold + 0.1
    assert N._fast_grade([make_doc(V1_L3, rerank_score=strong)]) == "sufficient"
    assert N._fast_grade([make_doc(V1_L3, rerank_score=strong)], accept=False) is None
    assert N._fast_grade([make_doc(V1_L3, rerank_score=0.0)], accept=False) == "insufficient"


@pytest.mark.parametrize(("lecture", "fast_accept"), [(3, False), (None, True)])
def test_v1_lecture_n_is_judged_by_the_grader_however_high_it_scores(monkeypatch, lecture, fast_accept):
    """v1 Lecture N after v2 Lecture N did not cover the question: same number, different lecture — the rerank score
    cannot tell them apart, so the LLM grader decides. Without a named lecture the fast path stays on."""
    calls = []
    monkeypatch.setattr(N, "_safe_retrieve", lambda q, lang, tier, n, part, notes: list(DOCS[tier]) if tier != "v1"
                        else [make_doc(V1_L3, rerank_score=0.95)])

    def fake_grade(question, docs, history, config, **kw):
        calls.append((docs[0].metadata["tier"], kw.get("fast_accept", True)))
        return {"verdict": "insufficient", "method": "llm"}

    monkeypatch.setattr(N, "_grade", fake_grade)
    N._fallback(_state("What does lecture 3 say about pricing?" if lecture else "Why do startups fail?", lecture=lecture), None)
    assert ("v1", fast_accept) in calls
