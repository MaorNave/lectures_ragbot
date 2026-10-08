"""LangGraph state and the structured outputs produced by the classifier and grader."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Intent = Literal[
    "conceptual",  # "What does Yonatan say about pricing?"
    "lecture_summary",  # "Summarize lecture 3"
    "lecture_specific",  # "What does lecture 7 say about business plans?"
    "homework",  # "What was the homework in lecture 5?"
    "comparison",  # "What changed between v1 and v2 lecture 1?"
    "cross_lecture",  # "Which lectures discuss marketing?"
    "practical",  # "How should I price my product?"
    "general_advice",  # explicit request for general (non-course) advice
    "chit_chat",  # greetings / thanks / meta questions about the bot
    "out_of_scope",
]


# The structured outputs carry decisions only. The local model writes ~20 tokens/s, so free-text fields that
# nothing reads cost real time: the grader's "reason" alone averaged ~200 tokens (~10 s) per call.
class Classification(BaseModel):
    """Scope + intent decision for a first message (instructions §0, §6)."""

    in_scope: bool = Field(description="True only if the message is about SmartUp Academy, its lectures, or entrepreneurship/startup topics.")
    intent: Intent
    lecture_number: int | None = Field(default=None, description="Lecture number if the user explicitly names one, else null.")
    lecture_part: Literal["1", "2"] | None = Field(default=None, description="'1' or '2' if the user says 'part 1/2', else null.")
    version_scope: Literal["default", "v1", "v2", "both"] = Field(
        default="default",
        description="'v1' if the user explicitly asks about the old/v1 course, 'both' for v1-vs-v2 comparisons, else 'default'.",
    )


class FollowUpClassification(Classification):
    """The same decision for a follow-up, plus the follow-up rewritten as a standalone search query.

    Only follow-ups need the rewrite (it resolves "that" / "it" from the conversation); for a first message the
    user's own wording is the better query, so the field is not generated at all.
    """

    search_query: str = Field(description="The user's question rewritten as one standalone search query in the SAME language, resolving pronouns from the conversation. No answer, no commentary.")


class Grade(BaseModel):
    """Does the retrieved evidence answer the question? (instructions §8)"""

    verdict: Literal["sufficient", "partial", "insufficient"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_id: str  # keys this turn's background searches (nodes._start_prefetch); undeclared keys are dropped by LangGraph
    answer_log: list[dict[str, Any]]  # this conversation's answers, kept across turns, for exact repeats (nodes.repeat)
    question: str
    language: Literal["en", "he"]
    classification: dict[str, Any]
    query: str
    docs_v2: list[Document]
    docs_v1: list[Document]  # v1 passages (explicit v1 / both scope, or the v1 step of the fallback chain)
    docs_fallback: list[Document]  # every passage the fallback chain retrieved (TEDx → v1 → outsource), in order
    docs: list[Document]  # the passages the answer is written from
    grade_v2: dict[str, Any]
    grade_v1: dict[str, Any]
    grade_fallback: dict[str, Any]  # the grade that decided the fallback chain (the answering source's, else the last)
    fallback_tier: str | None  # the fallback source that answered, None when none did
    fallback_tries: list[dict[str, Any]]  # one {tier, verdict, method, best_score, passages} per source tried
    used_fallback: bool
    notes: list[str]
    answer: str
    stopped_early: bool  # the reader left mid-answer: the answer is cut off, so it is never replayed
    citations: list[dict[str, Any]]
    status: Literal["answered", "refused", "not_found", "chit_chat", "error"]
