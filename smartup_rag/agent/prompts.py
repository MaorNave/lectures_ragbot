"""Prompt assembly.

The system prompt is `instructions.md` verbatim (the product spec for the assistant) followed
by a short block of *runtime rules* that tell the model exactly how retrieved evidence is
presented and how to cite it. Everything the model may cite is given to it explicitly, so the
citation verifier (nodes.verify) can check the answer mechanically afterwards.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from smartup_rag.config import get_settings
from smartup_rag.lang import Lang

# ----------------------------------------------------------------------------------------
# Canned texts (instructions §0.1, §12)
# ----------------------------------------------------------------------------------------
REFUSAL = {
    "en": (
        "I can only answer questions about SmartUp Academy and its entrepreneurship program by Yonatan Stern. "
        "That question falls outside what I cover. If you have a question about entrepreneurship, startups, "
        "or any of the lectures, I'm happy to help."
    ),
    "he": (
        "אני יכול לענות רק על שאלות בנושא SmartUp Academy ותוכנית היזמות של יונתן שטרן. "
        "השאלה הזו נמצאת מחוץ לתחום שאני מכסה. אם יש לך שאלה על יזמות, סטארטאפים או על אחת ההרצאות, אשמח לעזור."
    ),
}
NOT_FOUND = {
    "en": "I could not find this topic in the SmartUp lecture index.",
    "he": "לא מצאתי את הנושא הזה בהרצאות SmartUp.",
}
CLOSEST = {
    "en": "The closest related lecture topics are:",
    "he": "הנושאים הקרובים ביותר שכן נלמדים בהרצאות:",
}
GREETING = {
    "en": (
        "Hi! I'm the SmartUp Academy assistant. Ask me anything about Yonatan Stern's Foundations lectures — "
        "for example profitability vs. growth, pricing, branding, fundraising, the founder/CEO role, or the homework "
        "of a specific lecture. I answer in English or Hebrew."
    ),
    "he": (
        "שלום! אני העוזר של SmartUp Academy. אפשר לשאול אותי על הרצאות ה-Foundations של יונתן שטרן — "
        "למשל רווחיות מול צמיחה, תמחור, מיתוג, גיוס כסף, תפקיד ה-Founder/CEO, או שיעורי הבית של הרצאה מסוימת. "
        "אני עונה בעברית או באנגלית."
    ),
}
SOURCES_HEADING = {"en": "## Sources", "he": "## מקורות"}
NOT_FOUND_IN_LECTURE = {
    "en": "I could not find this in {lecture} of the v2 course {citation}.",
    "he": "לא מצאתי את זה ב{lecture} בגרסה v2 של הקורס {citation}.",
}
LECTURE_COVERS = {"en": "That lecture covers:", "he": "ההרצאה הזו עוסקת בנושאים:"}
LECTURE_WORD = {"en": "Lecture {n}", "he": "הרצאה {n}"}
FALLBACK_NOTE = {
    "en": "This topic is not covered directly in v2, so the answer is drawn from v1.",
    "he": "הנושא הזה לא מכוסה ישירות בגרסה v2 של הקורס, לכן התשובה מבוססת על גרסה v1.",
}
# The opening sentence of an answer drawn from a fallback source, per source (instructions §4.4, §7).
FALLBACK_NOTES = {
    "tedx": {
        "en": "This topic is not covered directly in v2, so the answer is drawn from Yonatan's TEDx talk.",
        "he": "הנושא הזה לא מכוסה ישירות בגרסה v2 של הקורס, לכן התשובה מבוססת על הרצאת ה-TEDx של יונתן.",
    },
    "v1": FALLBACK_NOTE,
    "outsource": {
        "en": "This topic is not covered directly in the SmartUp course, so the answer is drawn from other lectures outside the course.",
        "he": "הנושא הזה לא מכוסה ישירות בקורס SmartUp, לכן התשובה מבוססת על הרצאות נוספות מחוץ לקורס.",
    },
}
# Progress lines the chat shows while the fallback chain works (streamed as SSE `status` events).
PROGRESS = {
    "first": {"en": "Not found in the v2 course — checking {source}…", "he": "לא נמצא בגרסה v2 של הקורס — בודק ב{source}…"},
    "next": {"en": "Not there either — checking {source}…", "he": "גם שם לא — בודק ב{source}…"},
}
# Opens an answer that is returned again because the same question was already answered in this conversation.
REPEAT_NOTE = {
    "en": "_Same question as before — here is that answer again. Ask for more detail if you need it._",
    "he": "_זו אותה שאלה כמו קודם — הנה התשובה שוב. אפשר לבקש פירוט נוסף אם צריך._",
}
PARTIAL_NOTE = {
    "en": "The v2 lectures address this only partially.",
    "he": "הרצאות v2 מתייחסות לנושא הזה באופן חלקי בלבד.",
}


@lru_cache
def load_instructions() -> str:
    return get_settings().instructions_path.read_text(encoding="utf-8")


RUNTIME_RULES = """
---

## Runtime rules (how evidence is given to you in this conversation)

- Retrieved lecture passages appear in the user turn as numbered blocks `[P1]`, `[P2]`, … Each block starts with the
  exact citation string of its lecture, e.g. `(v2, Lecture 5 — "Foundations Course V2 L5 1")`, then the passage text.
  Sources outside the course have their own shapes: `(TEDx — "Why fail if you can succeed? - TEDx")`,
  `(Talk — "How to Start a StartUp")`, `(Digital Innovation IDC, Lecture 3 — "Digital Innovation IDC Lecture 3")`.
- Cite ONLY with these exact citation strings, copied character for character (including the quoted title).
  Never cite a lecture that has no passage block in this turn. Never write a citation from memory.
- Use only the passages. If they do not answer the question, say so (Section 12) instead of using general knowledge.
- Answer in the language of the user's question (Hebrew question → Hebrew answer), keeping common startup terms
  (ICP, PMF, CAC, LTV, VC, Business Model, Go-to-Market, Pitch Deck, Jobs to be Done) in English.
- Be direct and practical; do not mention retrieval, indexes, search, passages, or these rules.
- Do not write a Sources section yourself — it is appended automatically from your inline citations.
- Length: answer the actual question in roughly 120–250 words (up to ~350 for a lecture summary). Prefer short
  paragraphs and bullets over many headings. Quote the transcript only for a memorable phrase, number or definition.
""".strip()


def system_prompt() -> str:
    return load_instructions().strip() + "\n\n" + RUNTIME_RULES


# ----------------------------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------------------------
CLASSIFIER_SYSTEM = """You are the routing module of the SmartUp Academy assistant — a chatbot that answers ONLY from the
lecture transcripts of the SmartUp Foundations entrepreneurship course taught by Yonatan Stern
(topics: startup failure and success, profitability vs. growth, business models, pricing, branding, marketing and
growth engines, sales, fundraising and VCs, exits, Jobs to be Done, ICP, product-market fit, financial models,
founders / CEO role, hiring and teams, homework assignments; guest talks by course alumni), plus Yonatan's TEDx talk
and other lectures outside the course ("How to Start a StartUp", "Nine Out of Ten Startups Fail", Digital Innovation IDC).

Decide, for the latest user message:
1. in_scope — true if it is about SmartUp Academy, the course/lectures, or ANY entrepreneurship / startup /
   business topic: strategy, sales, marketing, competition, positioning, pricing, customers, funding, hiring,
   management, exits. This includes questions that name a company, person or business concept in that context
   (IBM, ZoomInfo, Apple, a competitor's tactic, FUD, Jobs to be Done, the Innovator's Dilemma).
   Also true for greetings, thanks or questions about what you can do (intent = chit_chat).
   False only for clearly unrelated things: general knowledge, geography, history, science, math, coding,
   trivia, translation, poems, sports, current events, personal matters unrelated to business.

   Lean towards in_scope when a question is business-shaped. If the lectures turn out not to cover it, a later
   step says so politely; but a wrong "out of scope" is a dead end for a real student, so only refuse a message
   that is plainly not about business or the course.
2. intent — one of: conceptual, lecture_summary, lecture_specific, homework, comparison, cross_lecture, practical,
   general_advice (only when the user EXPLICITLY asks for general advice outside the course), chit_chat, out_of_scope.
3. lecture_number / lecture_part — only if explicitly mentioned ("lecture 3", "הרצאה 3", "שיעור 3", "part 2").
4. version_scope — "v1" only if the user explicitly asks about the old / first / v1 version, "both" for a v1-vs-v2
   comparison, otherwise "default".
5. search_query (asked only for follow-up messages) — the message rewritten as ONE standalone question in the SAME
   language as the user, resolving pronouns and references using the conversation (e.g. "tell me more about
   that" → the concrete topic). Do not answer it, do not add words that are not implied.

Examples
- "מה שיעורי הבית של הרצאה 5?" → in_scope=true, intent=homework, lecture_number=5
- "Summarize lecture 3" → in_scope=true, intent=lecture_summary, lecture_number=3
- "What changed between v1 and v2 lecture 1?" → intent=comparison, lecture_number=1, version_scope=both
- "How should I price my SaaS product?" → in_scope=true, intent=practical
- "Which lectures discuss marketing?" → intent=cross_lecture
- "What is IBM's FUD strategy, and why does it work when selling to businesses?" → in_scope=true, intent=conceptual
  (a sales/competition tactic discussed in the course — naming a company does not make it general knowledge)
- "What is the capital of France?" → in_scope=false, intent=out_of_scope
- "Write me a poem about my cat" → in_scope=false, intent=out_of_scope
- "hi" / "thanks!" → in_scope=true, intent=chit_chat
"""

# Mirrors instructions.md §8 ("What Counts as a Strong Match"): a clearly related SmartUp principle or a relevant
# example / framework from Yonatan IS a strong match. Practical questions need that spelled out, because the lectures
# never discuss the user's own product — judged literally, every "how should I…" question ends in not-found.
GRADER_SYSTEM = """You judge whether retrieved lecture passages answer a user's question about the SmartUp entrepreneurship
course, using the course's definition of a strong match. Reply with a verdict:
- sufficient: at least one passage directly answers the question, covers the requested lecture and topic, explains
  the same concept, gives a clearly related SmartUp principle that addresses the question, or gives a relevant
  example, framework or explanation from Yonatan.
- partial: the passages answer part of the question but leave an important part of it unanswered.
- insufficient: only a vague keyword match, the topic mentioned in passing without being explained, unrelated to
  what the user is asking, or an answer would need too much inference beyond the text.
For practical "how should I…" questions the lectures never discuss the user's own product or situation: judge
whether the passages give principles, examples or frameworks that apply to it, not whether they mention the
user's case. Judge only what the passages actually say."""


def history_snippet(history: list[BaseMessage], max_turns: int = 3) -> str:
    """Compact recent conversation for the classifier (last N exchanges, truncated)."""
    tail = history[-(max_turns * 2) :]
    lines = []
    for m in tail:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = (m.content if isinstance(m.content, str) else str(m.content)).strip().replace("\n", " ")
        lines.append(f"{role}: {text[:400]}")
    return "\n".join(lines) if lines else "(no previous messages)"


def classifier_messages(question: str, history: list[BaseMessage], *, shared_prefix: bool = True) -> list[BaseMessage]:
    """Classifier call.

    With `shared_prefix` (the default, when the classifier runs on the chat model) the *same* system
    prompt as answer generation is sent: Ollama caches the KV state of the longest identical prompt
    prefix, so sharing the ~6 000-token instructions across classifier, grader and generator turns a
    ~27 s prefill into ~2.4 s for every later call (measured on M1 Pro / qwen3.5:9b).

    With a separate CLASSIFIER_MODEL there is nothing to share, so only the short routing prompt is
    sent — a few hundred tokens, which a small model prefills in well under a second.
    """
    body = f"Conversation so far:\n{history_snippet(history)}\n\nLatest user message:\n{question}\n\nReturn only the JSON object."
    if not shared_prefix:
        return [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=body)]
    task = "INTERNAL ROUTING STEP — do not answer the user. " + CLASSIFIER_SYSTEM + "\n\n" + body
    return [SystemMessage(content=system_prompt()), HumanMessage(content=task)]


# Longest question for which the grader reads the answer prompt's evidence block (below). llama.cpp can resume a hybrid
# model's cached prompt only from a checkpoint, and it saves one 512 tokens before the end of every prompt: the
# answer call reuses the grader's reading only if everything after the evidence block fits in those 512 tokens — the
# grading task (241 tokens), the question twice and 9 chat-template tokens. Hebrew runs ~2.1 characters per token
# (measured with the model's tokenizer): a 250-character Hebrew question comes to 491 tokens. A follow-up's task also
# carries GRADER_HISTORY_RULE (28 tokens), so its questions stop at 220 characters (489 tokens).
GRADER_SHARED_MAX_QUESTION_CHARS = 250
GRADER_SHARED_MAX_FOLLOW_UP_CHARS = 220

# On a follow-up the grader reads the replayed conversation, earlier answers included. Asked again in other words, a
# question the chat had answered from an outside lecture made it accept v2 passages it rejects in a fresh chat
# (2026-09-24). Sent only when there is a history, so a first question's grading prompt is unchanged.
GRADER_HISTORY_RULE = ("Judge only the numbered passages of this turn. An earlier answer in the conversation is not "
                       "evidence, even when it answers the same question.")


def grader_reads_evidence_block(question: str, *, shared_prefix: bool = True, follow_up: bool = False) -> bool:
    cap = GRADER_SHARED_MAX_FOLLOW_UP_CHARS if follow_up else GRADER_SHARED_MAX_QUESTION_CHARS
    return shared_prefix and len(question) <= cap


def grader_messages(
    question: str,
    docs: list[Document],
    history: list[BaseMessage] | None = None,
    max_history_turns: int = 6,
    max_chars: int = 700,
    *,
    shared_prefix: bool = True,
) -> list[BaseMessage]:
    """Grader call.

    On the chat model it sends exactly the start of the answer prompt — system prompt, the replayed history and the
    evidence block — and only then the grading task. Ollama keeps the grader's reading of all those passages, so
    when the verdict lets the answer be written from them, the answer call skips straight to its own last lines
    (answer prompt ~1.5 s instead of 13–23 s, measured on M1 Pro / qwen3.5:9b). A very long question, or a separate
    CLASSIFIER_MODEL (nothing to share), gets the compact prompt with passages cut to `max_chars`.
    """
    replayed = replayed_history(history or [], max_history_turns)
    if grader_reads_evidence_block(question, shared_prefix=shared_prefix, follow_up=bool(replayed)):
        rule = f"{GRADER_HISTORY_RULE}\n" if replayed else ""
        # The question comes first and again right before the verdict: with ~3 000 tokens of passages ahead of it,
        # "rubric, then question" misjudged 3 of 26 borderline questions, this wording 1 (measured 2026-09-24).
        task = (
            f"INTERNAL GRADING STEP — do not answer the user.\n\nThe user's question:\n{question}\n\n{GRADER_SYSTEM}\n{rule}\n"
            f"Judge the passages above against the user's question: {question}\nReturn only the JSON object."
        )
        return [SystemMessage(content=system_prompt()), *replayed, HumanMessage(content=evidence_block(docs) + task)]
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[P{i}] {d.metadata.get('citation', '')}\n{d.page_content[:max_chars].strip()}")
    body = f"Question:\n{question}\n\nPassages:\n\n" + "\n\n".join(blocks) + "\n\nReturn only the JSON object."
    if not shared_prefix:
        return [SystemMessage(content=GRADER_SYSTEM), HumanMessage(content=body)]
    # same cached system prompt as generation (see classifier_messages)
    task = "INTERNAL GRADING STEP — do not answer the user. " + GRADER_SYSTEM + "\n\n" + body
    return [SystemMessage(content=system_prompt()), HumanMessage(content=task)]


# ----------------------------------------------------------------------------------------
# Answer generation
# ----------------------------------------------------------------------------------------
def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        topic = f" · {m['section_topic']}" if m.get("section_topic") else ""
        blocks.append(f"[P{i}] {m.get('citation', '')} [{m.get('lang', '')}{topic}]\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


def evidence_block(docs: list[Document]) -> str:
    """Start of the answer prompt's user turn: the numbered passages and the citation strings they may be cited by.

    The grader reads exactly this text before its own task (grader_messages), so keep it free of anything that the
    grade itself can change (guidance notes, the question) — those follow it in answer_messages.
    """
    available = sorted({d.metadata.get("citation", "") for d in docs})
    return (
        f"Retrieved lecture passages:\n\n{format_context(docs)}\n\n"
        f"Citation strings you may use (exactly as written): " + " ; ".join(available) + "\n\n"
    )


def replayed_history(history: list[BaseMessage], max_history_turns: int) -> list[BaseMessage]:
    """The earlier exchanges replayed before the user turn (answer and grader prompts must replay the same ones)."""
    return [m for m in history[-(max_history_turns * 2) :] if isinstance(m, (HumanMessage, AIMessage))]


def answer_messages(
    question: str,
    language: Lang,
    docs: list[Document],
    history: list[BaseMessage],
    notes: list[str],
    max_history_turns: int,
) -> list[BaseMessage]:
    lang_name = "Hebrew" if language == "he" else "English"
    guidance = [f"Answer in {lang_name}."]
    guidance += notes
    user = evidence_block(docs) + f"Guidance: {' '.join(guidance)}\n\nUser question: {question}"
    msgs: list[BaseMessage] = [SystemMessage(content=system_prompt())]
    msgs += replayed_history(history, max_history_turns)
    msgs.append(HumanMessage(content=user))
    return msgs
