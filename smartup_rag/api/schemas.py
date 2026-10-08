"""Request / response models of the HTTP API.

These are the wire types the website widget depends on. They intentionally expose only a
curated subset of what `smartup_rag.agent.graph.run_chat` returns (no grader verdicts, no
passage counts) — the widget needs the answer, its status and the verified citations.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ChatStatus = Literal["answered", "refused", "not_found", "chit_chat", "error"]

MAX_MESSAGE_CHARS = 2000
MAX_THREAD_ID_CHARS = 128


class ChatRequest(BaseModel):
    """Body of POST /chat and POST /chat/stream."""

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_CHARS,
        description="The user's message (Hebrew or English). Leading/trailing whitespace is stripped.",
    )
    thread_id: str | None = Field(
        default=None,
        min_length=4,
        max_length=MAX_THREAD_ID_CHARS,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description=(
            "Conversation id for multi-turn memory. Omitted or blank → the server creates a uuid4 and returns it. "
            "Treat it as a secret: whoever knows it can continue that conversation."
        ),
    )

    @field_validator("message", mode="before")
    @classmethod
    def _strip_message(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("thread_id", mode="before")
    @classmethod
    def _blank_thread_is_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        v = v.strip()
        return v or None


class CitationOut(BaseModel):
    """One lecture the answer was verified against (instructions §9–10)."""

    course_version: str  # "v2" / "v1" / "supplemental" (outside sources)
    tier: str = ""  # "v2" / "tedx" / "v1" / "outsource" — see smartup_rag.tiers
    source: str = ""  # the citation label: "v2", "v1", "TEDx", "Talk", "Digital Innovation IDC"
    lecture_number: int
    lecture_part: str | None = None
    lecture_title: str
    display: str  # "Lecture 11 Part 2"; "" for a single outside talk
    citation: str  # '(v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")' / '(TEDx — "Why fail if you can succeed? - TEDx")'

    @classmethod
    def from_agent(cls, c: dict[str, Any]) -> CitationOut:
        return cls(
            course_version=str(c.get("course_version") or ""),
            tier=str(c.get("tier") or ""),
            source=str(c.get("source") or c.get("course_version") or ""),
            lecture_number=int(c.get("lecture_number") or 0),
            lecture_part=c.get("lecture_part") or None,
            lecture_title=str(c.get("lecture_title") or ""),
            display=str(c.get("display") or ""),
            citation=str(c.get("citation") or ""),
        )


class ChatResponse(BaseModel):
    """Result of one chat turn (also the payload of the SSE `final` event)."""

    thread_id: str
    answer: str
    status: ChatStatus
    language: Literal["en", "he"] | None = None
    intent: str | None = None
    used_fallback: bool = False
    fallback_tier: str | None = Field(default=None, description="The fallback source that answered: tedx / v1 / outsource.")
    citations: list[CitationOut] = Field(default_factory=list)
    elapsed_ms: int = Field(ge=0, description="Server-side wall time of the turn.")
    trace_id: str | None = Field(default=None, description="Pipeline trace of this turn on /dashboard (only for callers allowed to open it).")

    @classmethod
    def from_agent(cls, result: dict[str, Any], *, thread_id: str, elapsed_ms: int) -> ChatResponse:
        status = result.get("status") or "error"
        if status not in ("answered", "refused", "not_found", "chit_chat", "error"):
            status = "error"
        return cls(
            thread_id=thread_id,
            answer=str(result.get("answer") or ""),
            status=status,
            language=result.get("language"),
            intent=result.get("intent"),
            used_fallback=bool(result.get("used_fallback")),
            fallback_tier=result.get("fallback_tier") or None,
            citations=[CitationOut.from_agent(c) for c in (result.get("citations") or [])],
            elapsed_ms=max(0, int(elapsed_ms)),
        )


class LectureOut(BaseModel):
    lecture_number: int
    lecture_part: str | None = None
    lecture_title: str
    display: str
    citation: str
    lecture_subjects: list[str] = Field(default_factory=list)


class LectureGroup(BaseModel):
    """All lectures of one source tier (v2 → TEDx → v1 → outsource: the search order), sorted by number / part."""

    course_version: str  # the tier name — "v2" / "v1" as before, "tedx" / "outsource" for the outside sources
    tier: str = ""
    label: str = ""  # "Foundations v2 (primary)", "TEDx talk (fallback 1)", …
    lectures: list[LectureOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: dict[str, Any]  # up? chat and embedding models pulled?
    index: dict[str, Any]  # the local index: built, one vector per passage, built with the configured embedding model?
    reranker: dict[str, Any]  # the local cross-encoder: enabled, loaded (it loads on first use), device
    chat_model: str
    embed_model: str


class ErrorDetail(BaseModel):
    detail: str
