# SmartUp RAG — the SmartUp Academy lecture assistant

A chatbot that answers questions about the SmartUp Academy Foundations course (taught by Yonatan Stern) **only** from
the lecture transcripts and summaries, in Hebrew or English. It knows four sources, searched in this order: the
**v2** course, Yonatan's **TEDx talk**, the **v1** course, and **other lectures outside the course** (two
supplemental talks and the Digital Innovation IDC course).

Everything runs on one machine, with no cloud service and no API key:

- **Answers** stream token by token from a local LLM, `qwen3.5:9b` on [Ollama](https://ollama.com).
- **Retrieval** is hybrid: `bge-m3` embeddings in a **Chroma** vector store, plus **SQLite FTS5** BM25 over the
  passage text and over the lecture metadata, fused with Reciprocal Rank Fusion.
- **Reranking** uses a local **cross-encoder**, `BAAI/bge-reranker-v2-m3` via sentence-transformers, on the Apple
  GPU when there is one.
- **LangChain + LangGraph** turn the behavioural spec in [`instructions.md`](instructions.md) into a pipeline that
  enforces the hard rules in code rather than only in the prompt: scope, the v2 → TEDx → v1 → outside source
  order, and "never cite what was not retrieved".

It is operated from a **chat box in the browser**: a full-page chat UI and the same widget as a floating bubble for
the SmartUp website. The CLI is for operations.

The index is built from the lecture documents in [`data/raw/drive_09_2026/`](data/raw/drive_09_2026) — the only data
source of the project — with one command (`smartup ingest`, ~4 minutes).

## How it works

```
  BROWSER                                  SERVER  (FastAPI + uvicorn)
  +-------------------------------+        +-------------------------------------------+
  | web/index.html full chat box  |  POST  | POST /chat/stream  SSE: token* -> final    |
  | web/widget.js  bubble on the  | -----> | POST /chat         blocking JSON           |
  |                SmartUp site   | <----- | GET /health /lectures /widget/* static     |
  +-------------------------------+   SSE  +--------------------+----------------------+
                                                                | Semaphore(2), else HTTP 429
                                                                v
  LANGGRAPH AGENT (smartup_rag/agent/graph.py)                     LOCAL HYBRID RETRIEVAL (smartup_rag/retrieval/)

  prepare       detect Hebrew / English                            question
     |                                                               |-- bge-m3 (Ollama) --> Chroma: 30 nearest passages
     v                                                               |-- SQLite FTS5 BM25 over the passage text: 30
  classify --+--> refuse      out-of-scope question (spec 0.1)       |-- FTS5 over topic / subjects / title: 30
  LLM, JSON  +--> chit_chat   greeting                               |   (every leg filtered by source tier + lecture)
     |                                                               v
     v                                                             RRF (k = 60) -> 30 candidates
  retrieve v2  ------------------------------------------------>     |
     |         filters: source tier, lecture_number                  v
     v                                                             cross-encoder bge-reranker-v2-m3 (local, MPS/CUDA/CPU)
  grade   rerank >= 0.70 -> sufficient                               |  -> top 8, scores < 0.02 dropped
     |      rerank <  0.03 -> insufficient                           v
     |      in between     -> LLM grader                           drop cross-language twins -> LangChain Documents
     +-- insufficient --> fallback: TEDx -> v1 -> outside lectures
     |                      (search + grade each, first that answers)     data/index/  (built by `smartup ingest` from
     |                                  +--> nothing: not_found             data/raw/drive_09_2026, ~4 minutes)
     +-- sufficient / partial ----------+                                   chroma/          3 627 passage vectors
                                        |                                   smartup.sqlite3  passages, FTS5, lectures
                                        v
                                     generate  ----------------->  OLLAMA  qwen3.5:9b
                                        |                            tokens streamed back through SSE
                                        v
                                     verify citations --> answer + rebuilt "Sources"
```

---

## 1. Prerequisites

| | |
|---|---|
| Machine | Developed on macOS / Apple silicon (M1 Pro, 16 GB RAM). The chat model, the embedding model and the reranker together need ~9 GB of free RAM. On Linux the reranker runs on CUDA or the CPU. |
| Python | 3.11 or newer (developed on 3.13). |
| [Ollama](https://ollama.com) | Running locally on `http://localhost:11434`; it serves the chat model and the embedding model. |
| Disk | ~10 GB for the models (6.6 GB `qwen3.5:9b`, 1.2 GB `bge-m3`, 2.3 GB cross-encoder), ~35 MB for the index. |
| Network | Only to download the models once. No Docker, no cloud service, no API key. |

---

## 2. Quick start

```bash
git clone https://github.com/OWNER/lectures_ragbot.git && cd lectures_ragbot
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install --no-deps -e .   # the pinned, tested versions
# or: .venv/bin/pip install -e ".[dev]"      # the newest versions within pyproject.toml's ranges (make install)
cp .env.example .env                          # optional: every value in it is already the default

.venv/bin/smartup models --pull               # qwen3.5:9b + bge-m3 into Ollama, the cross-encoder from Hugging Face
.venv/bin/smartup ingest                      # build data/index/ from data/raw/drive_09_2026 (~4 minutes)
.venv/bin/smartup check                       # Ollama, index and reranker health (make check)
.venv/bin/smartup serve                       # FastAPI on API_PORT, default 8000 (make serve)
```

On Linux without an NVIDIA GPU, install the CPU build of PyTorch first (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`); the default PyPI wheel bundles several GB of CUDA libraries.

Open **<http://localhost:8000/>**. An empty conversation shows a welcome screen with suggested questions; type a
question in Hebrew or English and watch the answer **stream in token by token** into an answer card — inline citations
coloured by source, the **Sources** under the answer, which source answered when the fallback chain was used, two
stopwatches, and a copy button. The sidebar is the lecture library: the four sources in search order, and every lecture
grouped by source with its main topic — filter it by topic or title, click a lecture for its summary. The page follows
the system's light or dark mode, Hebrew conversations switch to right-to-left, and on a phone the library becomes a
drawer. The floating-bubble version of the same widget is demonstrated at
<http://localhost:8000/widget/embed-demo.html>.

### The pipeline dashboard

**<http://localhost:8000/dashboard>** shows every chat call step by step, live while it runs: the language detected,
the classifier's decision (and any code corrections to it), the query and filters of each search, how many
candidates each retrieval leg returned and how long embedding, Chroma, FTS5 and the reranker took, every passage
that came back with its rerank score against the grading thresholds (and the duplicate translations that were
dropped), how the evidence was graded, the prompt the model received with its token counts and timings, which
citations survived verification, and the final answer — plus a timeline of where the time went. Each answer in the
chat carries a small **trace ↗** link to its own trace. Traces live in memory (the last 50 calls) and are shown only
to requests from this machine by default (`DEBUG_DASHBOARD`, §8).


### The CLI is for operations

```bash
smartup check [--explain]         # health + settings; --explain prints every retrieval leg's filter and FTS5 query
smartup models [--pull]           # are the Ollama models pulled and the cross-encoder downloaded? --pull fetches them
smartup ingest [--recreate] [--dry-run]           # build or update the local index from data/raw/drive_09_2026
smartup lectures [--json]         # the 34-lecture catalogue
smartup retrieve "question" [--lang en|he] [--version v2|tedx|v1|outsource|all] [--lecture N] [--k 8] [--chars 120]
smartup ask "question" [--thread ID] [--no-stream | --json]   # one turn through the agent, streamed token by token
smartup chat [--thread ID]        # streaming REPL; /new, /json, /quit
smartup eval [--full] [--limit N] # evaluation set
smartup serve [--host H] [--port P] [--reload]    # the API + chat UI
```

`make install | models | check | ingest [RECREATE=1] | ask Q="…" | chat | serve | eval [FULL=1] | test | lint` wrap
the same commands.

---

## 3. The models, and why

```bash
smartup models --pull     # = ollama pull qwen3.5:9b && ollama pull bge-m3 + the cross-encoder download
```

### Chat model (Ollama)

| Model | Size | Verdict |
|---|---|---|
| **`qwen3.5:9b`** (default) | 6.6 GB | **Recommended.** 201 languages including Hebrew, reliable structured output (the classifier and grader need strict JSON), "thinking" can be switched off, and it fits in 16 GB next to the retrieval models, the API and a browser. |
| `gemma4:12b` | 7.6 GB | Stronger reasoning, ~1 GB more RAM, noticeably slower per token on an M1 Pro. |
| `qwen3:8b` | 5.2 GB | Produced mixed-script Hebrew (Latin letters inside Hebrew words) in our runs. A zero-download fallback only. |
| `qwen3.5:4b` | 3.4 GB | Too weak for grounded answers; a possible `CLASSIFIER_MODEL` on a machine with more RAM (§9). |

Switching is configuration only (`CHAT_MODEL=…`, optionally `CLASSIFIER_MODEL=…`). `CHAT_THINK=false` (the default)
sends Ollama's `think: false`: answers are grounded by retrieved passages, so hidden chain-of-thought buys little
and costs a lot of latency.

### Embeddings: `bge-m3` (Ollama)

`bge-m3` (568 M parameters, 1024 dimensions, MIT licence) is multilingual **and cross-lingual**: a Hebrew question lands
next to the English passage that says the same thing, so one search covers both transcripts of every lecture
(`CROSS_LINGUAL=true`). It runs in the same Ollama server as the chat model — 0.7 GB next to qwen3.5's 6 GB, both
resident at once on a 16 GB Mac — and gets the same keep-alive, so no question waits for a model load. Query vectors
are cached per process, because the fallback chain searches up to four source tiers with the same question.
Changing `EMBED_MODEL` means re-embedding everything; the index records the model it was built with, and
`smartup check` / `smartup ingest` notice a mismatch.

### Reranker: `BAAI/bge-reranker-v2-m3` (sentence-transformers)

A cross-encoder reads the question and one passage together and scores how well the passage answers it — much
more precise than comparing embeddings, and too slow for the whole index, so it re-scores only the 30 fused
candidates. `bge-reranker-v2-m3` is multilingual (Hebrew and English, also across the two), Apache-2.0, and needs no
remote code. On an M1 Pro it scores 30 passages in ~2 s on the GPU (MPS, float16 — the same scores as float32 to
±0.001) and ~8 s on the CPU. The weights (2.3 GB) are downloaded from the Hugging Face Hub once and then loaded from
the local cache without network access; torch is imported only when the model is first needed.

**One context line per passage.** Every passage is Yonatan's own words, but a transcript rarely says so, and a
cross-encoder takes a question literally: asked "What does Yonatan say about pricing?", it preferred any passage that
*names* Yonatan over the lecture that teaches pricing ("Pricing Psychology" scored 0.001). The reranker therefore reads
each passage after one line naming the speaker and the source — `Yonatan Stern (יונתן שטרן), SmartUp Foundations course
(v2), Lecture 8:` (`hybrid.rerank_header`; the answer model never sees it). On the evaluation set this lifted the
weakest best score of a source that answers the question from 0.03 to 0.19 (tenth percentile 0.20 → 0.88), while the
25 weak sources an earlier LLM-grader validation had rejected stayed below 0.13 (AUC 0.984 → 0.990).

Alternatives measured on the same candidates: **`Qwen/Qwen3-Reranker-0.6B`** (Apache-2.0, an instruction-following LLM
reranker) showed the same name bias and is ~4× slower on MPS; **`jinaai/jina-reranker-v2-base-multilingual`** — the
model the previous, Elasticsearch-hosted version of this project used — is licensed CC-BY-NC-4.0, which rules it out
as a default for a public, commercially used deployment. `RERANK_MODEL` accepts any sentence-transformers
cross-encoder; a different model needs its own threshold calibration (§10).

---


### The index

`smartup ingest` reads the 34 lecture documents, normalises them (§5), cuts every transcript into ~1 200-character
passages per language with LangChain's `RecursiveCharacterTextSplitter` (plus one summary passage per language),
embeds them with `bge-m3` and writes two stores under `data/index/`:

- **`chroma/`** — the passage vectors, each with only the fields filters need (`course_id`, `course_version`,
  `lecture_number`, `lecture_key`, `lang`, `section_index`). No text.
- **`smartup.sqlite3`** — `passages` (text and metadata, the source of truth), `passages_fts` (an FTS5
  external-content index over text, section topic, subjects and title, kept in sync by triggers), `lectures` (the
  catalogue: metadata, both summaries, the source file and a content hash) and `meta` (how the index was built).

Ingestion is **incremental**: each lecture is stored with a hash of its normalised content, the chunking settings, the
embedding model and the schema version. A re-run skips unchanged lectures, re-embeds changed ones and removes lectures
that are no longer in the export. A lecture is written to Chroma first and committed to SQLite (with its hash) last,
so an interrupted run is simply redone next time; an index whose two stores disagree, or that was built with another
embedding model or chunking, is rebuilt from scratch. `--recreate` always starts over.

### A search

`retrieval/hybrid.py` runs one pass over one source tier:

1. **Three legs**, each returning its 30 best passages *within the filters* (source tier, lecture number, and the
   Digital Innovation switch): the question's `bge-m3` vector against Chroma; FTS5 BM25 over the passage text, with
   the question turned into quoted OR-terms (`queries.fts_match`, so FTS5 operators or punctuation in a question are
   never syntax); FTS5 BM25 over section topic / lecture subjects / title.
2. **Reciprocal Rank Fusion** → 30 candidates, then exact duplicates are removed.
3. A requested **lecture part** (v1 Lecture 11 Part 1/2) is applied as a soft filter.
4. The **cross-encoder** re-scores the candidates; the best `TOP_K_PASSAGES` (8) with a score of at least
   `MIN_RERANK_SCORE` (0.02) are kept.
5. **Cross-language twins** are dropped: an other-language passage whose translation in the question's language is
   already in the list (same stretch of the same lecture, ≥ 75 % of the Hebrew passage's names, numbers and English
   terms shared) — the model would otherwise read it twice.
6. The survivors become LangChain `Document`s whose metadata carries the exact citation string.

Failures degrade instead of breaking a chat: without the embedding model the keyword legs still answer (the dashboard
shows why the dense leg was skipped); without the reranker the RRF order is kept and the LLM grader decides; a missing
index is "nothing found" plus a clear message in `smartup check` and `/health`.

### LangChain and LangGraph

- **LangChain**: `ChatOllama` as the model client, `OllamaEmbeddings` for `bge-m3`,
  `with_structured_output(..., method="json_schema")` for the classifier and the grader, `Document` as the passage type
  carried through the whole pipeline, `BaseRetriever` (`LectureRetriever`) so the retriever composes with any chain, and
  `RecursiveCharacterTextSplitter` for ingestion.
- **LangGraph**: a `StateGraph` whose nodes *are* the workflow of `instructions.md` — scope guard, classify, v2
  retrieval, grade, the fallback chain, generate, verify — with conditional edges for the routing rules, a
  `MemorySaver` checkpointer for multi-turn threads, and `astream_events(version="v2")` to **stream generation tokens**
  out through the API (SSE) and the CLI while the graph keeps running.

Nothing else is added: FastAPI + uvicorn serve the API, `pydantic-settings` reads the configuration, `httpx` performs
the Ollama health probe, and the web UI is dependency-free vanilla JavaScript with no build step.

---

## 5. Data

### Four sources, one order

| Tier | Source | Documents | Passages | Recognised by | Cited as |
|---|---|---|---|---|---|
| **v2** (primary) | SmartUp Foundations v2, Lectures 1–11 | 11 | 873 | `course_version = v2` | `(v2, Lecture 5 — "Foundations Course V2 L5 1")` |
| **tedx** (fallback 1) | Yonatan's TEDx talk "Why fail if you can succeed?" | 1 | 17 | `course_id = smartup-tedx` | `(TEDx — "Why fail if you can succeed? - TEDx")` |
| **v1** (fallback 2) | SmartUp Foundations v1: Lectures 1–10, 11 Part 1, 11 Part 2, 13–15 (no Lecture 12) | 15 | 2 042 | `course_version = v1` | `(v1, Lecture 11 Part 2 — "Foundations 11 Part 2 Oded")` |
| **outsource** (fallback 3) | "How to Start a StartUp", "Nine Out of Ten Startups Fail", Digital Innovation IDC Lectures 1–5 | 7 | 695 | `course_version = supplemental`, not TEDx | `(Talk — "How to Start a StartUp")`, `(Digital Innovation IDC, Lecture 3 — "…")` |

The index carries no tier field — `smartup_rag/tiers.py` derives the tier from `course_id` / `course_version` and
states each tier's filter once (`tier_conditions`), which `retrieval/queries.py` turns into a Chroma `where` and a SQL
clause. The agent searches v2 first and walks down the list only while a source has no strong match (§7).

### Where the data lives

| Path | What it holds | In git |
|---|---|---|
| `data/raw/drive_09_2026/{course_v2,course_v1,outsource}/*.json` | **The only data source**: 34 lecture documents (the "Naive" exports of the SmartUp shared drive, downloaded 2026-09-24, kept untouched). Each holds the lecture metadata (`course_id`, `course_version`, `lecture_number`, `lecture_title`, `lecture_subjects`, `lecturer_name`, …), the full English and Hebrew transcriptions, and an English and a Hebrew summary. | yes (4 MB) |
| `data/index/` | The index built from it by `smartup ingest`: `chroma/` (3 627 vectors) and `smartup.sqlite3`. | no — rebuild it |
| `data/eval_results/`, `data/logs/` | Evaluation results and run logs. | no |

Three fixes are applied while loading (`ingest/loaders.py`; the files stay as they are):

- The five **Digital Innovation** files say `course_version: "v1"` and ``course_id: "`Digital Innovation Idc`"`` and
  reuse the `lecture_id`s of v1 Lectures 1–5 — as-is they would be searched as course v1. They become
  `digital-innovation-idc` / "Digital Innovation IDC" / `supplemental`, like the other outside talks.
- The **v1 titles** carry export noise ("Foundations   Lecture 8   Nice Ai Transcript (1)", "Transcript No Tp", plain
  "Lecture"). Titles are what students see in citations, so the noise is stripped: "Foundations Lecture 8". "Part N" and
  speaker names ("… Part 2 Oded") are kept.
- v1 **Lecture 11** is two documents with the same number; the part comes from the title, and lecture keys (`v2:L3`,
  `v1:L11p2`, `smartup-tedx:supplemental:L1`) are unique across all sources.

### Updating the data

Replace or add files in the three folders (same JSON shape), then run `smartup ingest` — only the lectures that
changed are re-embedded — and restart the API. A missing folder is an error rather than a silently smaller index.

---

## 6. Project structure — what each module does

### `smartup_rag/` — the package

| File | What it does |
|---|---|
| `__init__.py` | Package docstring and `__version__`. Nothing else is imported at package level, so the CLI stays fast. |
| `config.py` | The single source of configuration: a `pydantic-settings` `Settings` model whose every field maps to an environment variable or a `.env` entry, plus `PROJECT_ROOT` and the cached `get_settings()`. Validators split `CORS_ORIGINS` on commas and make relative paths absolute. |
| `schemas.py` | The domain models shared by ingestion, retrieval and the agent: `Lecture`, `Passage` and `Section`. Field names deliberately match `instructions.md`. |
| `lang.py` | Language utilities with no external dependency: `detect_language` decides Hebrew vs. English by Unicode block (a mixed message resolves to the dominant script, so "מה זה PMF?" is Hebrew), `CITATION_RE` / `extract_citations` parse citation strings tolerantly, and `lecture_number_in_text` finds "lecture 3" / "הרצאה 3" / "שיעור 3". |
| `tiers.py` | The four source tiers in search order and everything derived from them: `tier_of(course_id, course_version)`, `tier_conditions()` (each tier's filter as plain (field, operator, value) triples), the citation label and shape per source (`citation_for`, `source_line`), and `OUTSIDE_CITATION_RE`, the strict pattern the verifier uses for outside citations. |
| `trace.py` | Per-turn pipeline traces for the dashboard: `span(...)` records a step (a no-op outside a trace, so the CLI, eval and tests are unaffected); the last 50 traces are kept in memory. `background_context()` hands the fallback searches to other threads with the trace attached. |
| `cli.py` | The `smartup` console script: `check`, `models`, `ingest`, `ask`, `chat`, `retrieve`, `lectures`, `eval`, `serve`. Heavy modules (LangGraph, torch, Chroma) are imported inside each command handler. |

#### `smartup_rag/retrieval/`

| File | What it does |
|---|---|
| `store.py` | `LocalIndex`: the Chroma collection plus the SQLite database. Dense search, FTS5 keyword search, passage and lecture lookups, transcript lengths, and — for ingestion — `reset`, `write_lecture` (vectors first, SQLite + content hash last) and `delete_lectures`. One read-only SQLite connection per thread. `index_health()` checks that the index exists, holds one vector per passage and was built with the configured embedding model. |
| `embeddings.py` | `bge-m3` through LangChain's `OllamaEmbeddings`, kept loaded for `CHAT_KEEP_ALIVE`; `embed_query` is cached per process. |
| `reranker.py` | `CrossEncoderReranker`: loads the sentence-transformers cross-encoder lazily (local cache first, download only when missing), picks CUDA / MPS / CPU, uses float16 on a GPU, serialises scoring with a lock, and does not retry a failed load on every question. |
| `queries.py` | Pure functions, no I/O: filter `conditions()` → `chroma_where()` / `sql_where()`, `query_terms()` + `fts_match()` (a safe FTS5 expression), `rrf_fuse()`. |
| `hybrid.py` | The retrieval engine: `search_passages` (three legs → RRF), `dedupe`, `rerank` (with `rerank_header`), `drop_translations`, and `retrieve()`, which ties them together and returns LangChain `Document`s carrying the exact citation string. Also `lecture_key_for`, `lecture_part_from_title`, `display_lecture`. |
| `retriever.py` | The LangChain-facing surface: `LectureRetriever(BaseRetriever)` and `get_lectures()` (the catalogue with summaries, used by `/lectures`, the CLI and the lecture-summary shortcut). |

#### `smartup_rag/agent/`

| File | What it does |
|---|---|
| `llm.py` | The `ChatOllama` factory, cached per role (`chat` / `classifier` / `grader`); `shares_chat_prefix()`, `warm_up()` (loads the model and primes the shared prompt cache), `note_prompt_timing()` (detects a lost cache), and `ollama_health()` (Ollama up, chat and embedding models pulled). |
| `state.py` | The typed contract of the graph: `AgentState` plus the structured outputs `Classification`, `FollowUpClassification` and `Grade`. |
| `prompts.py` | Prompt assembly and every canned text. `system_prompt()` is `instructions.md` verbatim plus a short **runtime rules** block; the classifier and grader reuse that system prompt (shared KV-cache prefix); `grader_messages` opens with exactly the answer prompt's evidence block so the answer continues from the grader's reading (§9). |
| `nodes.py` | Every graph node as a plain function, the routing predicates, the grading fast path, the fallback chain (with background searches and the skip rule), and `verify()`, the citation checker. |
| `graph.py` | Wires the nodes into a `StateGraph`, compiles it with a `MemorySaver`, and exposes `run_chat()` (blocking) and `stream_chat()` (async iterator of `token`, `status` and `final` events). |

#### `smartup_rag/ingest/` — data/raw/drive_09_2026 → the local index

| File | What it does |
|---|---|
| `loaders.py` | Reads the three export folders and applies the data fixes of §5 (`normalise_document`, `clean_title`, `lecture_from_document`); a missing folder or an unreadable file is an error. |
| `chunker.py` | Splits each lecture into passages (`PASSAGE_CHARS` / `PASSAGE_OVERLAP`), one language at a time, dropping fragments under 80 characters, plus one `lecture summary` passage per language. `passage_id` is deterministic. |
| `indexer.py` | `ingest()`: load → chunk → embed → write, incremental by content hash, with the rebuild rules of §4. Also runnable as `python -m smartup_rag.ingest.indexer`. |

#### `smartup_rag/api/`

| File | What it does |
|---|---|
| `app.py` | The FastAPI application: `GET /health` (Ollama, index and reranker; never raises, no paths or error internals), `GET /lectures`, `POST /chat`, `POST /chat/stream` (SSE with keep-alive comments), the static `/widget` mount, `/` (the chat page) and the dashboard routes. Request logging, CORS, a per-IP rate limit, an `asyncio.Semaphore(2)` that answers 429 instead of queueing, and startup tasks that load the retrieval models and warm the chat model. |
| `schemas.py` | The wire types the widget depends on: `ChatRequest`, `ChatResponse`, `CitationOut`, `LectureOut` / `LectureGroup`, `HealthResponse`, `ErrorDetail`. |

#### `smartup_rag/eval/`

| File | What it does |
|---|---|
| `questions.jsonl` | The evaluation set: 42 questions (28 English, 14 Hebrew) with the expected status, source tier and lecture(s). |
| `run_eval.py` | Retrieval mode (default, no LLM) measures hit@k against the expected lectures; `--full` runs the whole agent and scores status, citations and fallback behaviour. Writes `data/eval_results/<timestamp>.json`. |

### Everything outside the package

| Path | What it is |
|---|---|
| `web/index.html` | The full-page chat UI served at `/`: the lecture library in a sidebar (the source order, every lecture grouped by source with its main topic, a filter; click for a summary), a status pill and a "New chat" button, and the widget rendered inline. Light and dark mode; a drawer on narrow screens. |
| `web/widget.js`, `web/widget.css` | The embeddable chat widget — vanilla JS, no dependencies, no build step. Streams `/chat/stream` (falling back to `/chat`), renders a safe subset of Markdown into answer cards, colours citations by source, shows the Sources and the source that answered, runs two stopwatches per answer, offers a copy button, shows a welcome screen with suggested questions, keeps the conversation in `sessionStorage`, switches to RTL for Hebrew, and handles 429 politely. Themed through CSS variables, with `light`, `dark` and `auto` themes. |
| `web/fonts/` | Heebo and Frank Ruhl Libre (Hebrew + Latin variable fonts, SIL OFL 1.1), self-hosted so the UI makes no third-party requests. |
| `web/embed-demo.html` | A stand-in website page showing the floating bubble. |
| `web/dashboard.html` | The pipeline dashboard served at `/dashboard`. All trace text is inserted as text, never as HTML. |
| `tests/` | The test suite (§10). |
| `docs/ARCHITECTURE.md` | Design notes: requirements, decisions and their reasons, the move from Elasticsearch to the local index, and how the reranker thresholds were calibrated. |
| `instructions.md` | The behavioural specification for the assistant, **loaded verbatim as the system prompt** — editing it changes the chatbot's behaviour without touching Python. |
| `.env.example` | The documented configuration template (every value is also the default). |
| `Makefile`, `pyproject.toml` | Thin wrappers over the CLI; package metadata, dependency ranges, the `smartup` console script, Ruff and pytest configuration. |
| `requirements.txt`, `requirements-dev.txt` | The exact versions the tests, the evaluation and the reranker calibration ran with (Python 3.13, macOS arm64); the dev file adds pytest and Ruff. |
| `LICENSE` | The MIT License. |

---

## 7. The agent flow in detail

### How `instructions.md` becomes the prompt

`prompts.system_prompt()` returns `instructions.md` **verbatim**, followed by a short *runtime rules* block that
describes the one thing the spec cannot know: how evidence is presented in this conversation — numbered
`[P1] (v2, Lecture 5 — "…")` blocks, citations copied character for character, no self-written Sources section, and
roughly how long an answer should be.

When the classifier and the grader run on the chat model (the default), they are given the *same* system prompt and put
their task in the user turn (`INTERNAL ROUTING STEP — do not answer the user. …`). Ollama caches the KV state of the
longest identical prompt prefix, so sharing the ~6 000-token instructions turns a ~27 s prefill into a few seconds for
every call after the first (§9). With a separate `CLASSIFIER_MODEL` there is nothing to share, and those two calls send
only their own short prompt.

### The nodes

| Node | What it does |
|---|---|
| `prepare` | Takes the latest human message, detects its language by script, and resets the per-turn fields of the state. |
| `repeat` | The same question asked again in one conversation (case, punctuation and spacing ignored) gets the answer it already got, with its original citations and a one-line note — no search, no LLM. Errors, greetings, answers cut off because the reader left, and turns whose search failed are never replayed. |
| `classify` | One structured-output LLM call: in scope or not, intent, an explicitly named lecture number/part, the version scope, and — for follow-ups only — the question rewritten as a standalone search query. Deterministic overrides follow (a regex lecture number fills a missing one, out-of-scope forces `in_scope=false`, "v1 / old version / גרסה 1" sets `version_scope=v1`, …). |
| `retrieve` | Runs the retrieval pass(es): v2 by default; v1 only, both versions, the TEDx talk or the outside lectures when the question names them. Homework questions get keyword hints; lecture summaries are prepended for summary questions. When v2 looks weak, the fallback searches start in the background. |
| `grade` | Decides whether v2 is enough. Skipped when the user named a source or a lecture summary was found. Otherwise the fast path: best rerank score ≥ `GRADE_STRONG_THRESHOLD` (0.70) → *sufficient*; < `GRADE_WEAK_THRESHOLD` (0.03) or no passages → *insufficient*; in between → one structured LLM call (sufficient / partial / insufficient) judged by instructions §8, reading the passages exactly as the answer prompt presents them. A *partial* verdict makes the answer open with "The v2 lectures address this only partially." |
| `fallback` | The fallback sources in order — the TEDx talk, v1, the outside lectures — each graded on its own; the first *sufficient* or *partial* one answers, opening with that source's fallback sentence. A question naming a lecture number tries v1 only, and that v1 lecture is always judged by the LLM grader, however high it scores — v1 Lecture N is a different lecture from v2 Lecture N, often on the same topic, which a rerank score cannot tell apart. A source below `SKIP_WEAK_GRADE_BELOW` (0.04; 0.09 for the TEDx talk) is rejected without an LLM call when a later source already has a strong match. The chat shows what the chain is doing ("Not found in the v2 course — checking Yonatan's TEDx talk…"). |
| `generate` | System prompt, up to `MAX_HISTORY_TURNS` previous exchanges, then the passages, the citation strings that may be used and the guidance notes; the answer is **streamed** from Ollama. |
| `verify` | The citation checker (below). |
| `refuse` / `chit_chat` | Canned bilingual texts, no LLM call. |
| `not_found` | "I could not find this topic in the SmartUp lecture index", plus the closest lecture subjects among what was retrieved; for a named lecture that does not cover the topic, the subjects it does cover. |

### Routing rules

- After `prepare`: a question this conversation already answered → `repeat`; otherwise → classify.
- After `classify`: `chit_chat` → greeting; not in scope → refusal; otherwise → retrieve.
- After `grade`: *sufficient* or *partial* → generate. *Insufficient* with an explicit source → `not_found` (the user's
  choice is not overridden). Otherwise → the fallback chain.
- In the chain, a named lecture number that v2 has → only v1 Lecture N is tried, and the LLM grader must find it
  *sufficient* (a high rerank score alone is not enough).
- After `fallback`: passages → generate; nothing → `not_found`.

### What the citation verifier guarantees

1. A trailing Sources block the model wrote itself is stripped — only when everything after the heading is
   source-shaped, so real prose is never truncated.
2. Every citation is looked up against the lectures retrieved **this turn**, accepting every shape a small model
   produces (dash, comma, colon, quotes or none, markdown emphasis, the Hebrew `הרצאה`/`חלק`); one to a lecture that was
   not retrieved is deleted and only the hole it leaves is tidied.
3. A right version and number with a wrong or missing part is repaired when exactly one retrieved lecture matches,
   and the title is always rewritten to the retrieved `lecture_title`.
4. The **Sources** block is rebuilt from the surviving citations, in the answer's language. Grounded prose without
   inline citations exposes the top three retrieved lectures; an answer that *declines* gets no Sources and is reported
   as `not_found`. Outside sources are checked through their own citation shapes.
5. The surviving citations are returned as structured data — the UI's Sources list cannot contain anything the answer
   was not grounded in.

### How failures are handled

- **Classifier unavailable** → one retry, then the turn is *refused* (§0.4: an uncertain scope is out of scope).
- **Grader unavailable** → the rerank score decides: above the noise floor counts as partial, below goes to the chain.
- **Search error** (index missing, a model unavailable) → treated as "nothing found" and logged; never a 500. A failing
  embedding model or reranker only degrades the search (§4).
- **Explicit source that finds nothing** → `not_found` immediately; the model never gets an empty context.
- **Version and source words** are normalised deterministically: "v1 and v2", "compare … versions", "מה ההבדל בין גרסה 1
  לגרסה 2" → both versions; "v1" alone → v1; "TEDx" / "הרצאת ה-TEDx" → the TEDx talk; "Digital Innovation" → the
  outside lectures.
- **Lecture parts** ("Part 2", "חלק ב") are a soft filter before reranking.
- **Language** is decided by word counts, asymmetrically: `מה זה Product-Market Fit?` is Hebrew and answered in Hebrew.

### Conversation memory

The compiled graph carries a `MemorySaver` checkpointer keyed by `thread_id`. The API creates a `uuid4` thread when the
client does not send one; the widget keeps it in the tab's `sessionStorage`. Only the last `MAX_HISTORY_TURNS`
exchanges are replayed into the prompt; the thread's last 20 answers are kept for the `repeat` node. The id is the only
key to that memory, so it is validated (`^[A-Za-z0-9_.:-]{4,128}$`) and should be treated as a secret.

---

## 8. Configuration reference

Every setting is an environment variable or a line in `.env` (same name, uppercase). Defaults below are the ones in
`smartup_rag/config.py`; `.env.example` lists the commonly changed ones.

### Ollama

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server listens (chat and embeddings). |
| `CHAT_MODEL` | `qwen3.5:9b` | Model used for answers (and for routing when `CLASSIFIER_MODEL` is unset). |
| `CHAT_TEMPERATURE` | `0.1` | Sampling temperature for answers. The classifier and grader always run at 0.0. |
| `CHAT_NUM_CTX` | `16384` | Context window. Must fit the ~6 000-token system prompt plus the passages. |
| `CHAT_THINK` | `false` | `true` re-enables the model's hidden "thinking" tokens (slower; rarely worth it here). |
| `CHAT_TIMEOUT_S` | `240.0` | HTTP timeout for a single Ollama call. |
| `CHAT_KEEP_ALIVE` | `24h` | How long Ollama keeps the chat and the embedding model loaded after the last call; `-1` = until Ollama restarts. |
| `WARM_ON_STARTUP` | `true` | At API startup, load the embedding model and the reranker and prime the chat model's prompt cache, in the background. |
| `KEEP_WARM_MINUTES` | `10` | While no chat runs, re-prime the shared-prompt cache every N minutes and right after a call shows it was lost (§9). `0` disables both. |
| `CLASSIFIER_MODEL` | *(unset → `CHAT_MODEL`)* | Optional separate model for the classifier and the grader. |

### Local index

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data/raw/drive_09_2026` | The lecture documents the index is built from (relative paths resolve from the project root). |
| `INDEX_DIR` | `data/index` | Where `smartup ingest` writes `chroma/` and `smartup.sqlite3`. |
| `COLLECTION_NAME` | `smartup_passages` | The Chroma collection. |
| `EMBED_MODEL` | `bge-m3` | Ollama embedding model. Changing it requires `smartup ingest --recreate` (`check` reports the mismatch). |
| `EMBED_BATCH_SIZE` | `32` | Passages per embedding request during ingestion. |
| `EMBED_TIMEOUT_S` | `120.0` | HTTP timeout for one embedding request. |
| `INCLUDE_DIGITAL_INNOVATION` | `true` | The 5 Digital Innovation IDC lectures belong to the outside-lectures tier; `false` hides them from search. |
| `PASSAGE_CHARS` / `PASSAGE_OVERLAP` | `1200` / `200` | Passage size and overlap in characters (a change re-embeds everything). |
| `INSTRUCTIONS_PATH` | `instructions.md` | The file loaded verbatim as the system prompt. |

### Reranker and retrieval tuning

| Variable | Default | Meaning |
|---|---|---|
| `RERANK_ENABLED` | `true` | `false` keeps the RRF order and disables the grading fast path (the LLM grader then runs every turn). |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Any sentence-transformers cross-encoder; recalibrate the thresholds below for another model. |
| `RERANK_DEVICE` | `auto` | `auto` = CUDA, else Apple MPS, else CPU; or `cuda` / `mps` / `cpu`. |
| `RERANK_MAX_LENGTH` / `RERANK_BATCH_SIZE` | `512` / `16` | Tokens of question + passage read by the cross-encoder; pairs per forward pass. |
| `TOP_K_PASSAGES` | `8` | Passages handed to the LLM after reranking (the *k* of the evaluation's hit@k). |
| `RANK_WINDOW` | `30` | Hits per leg, and candidates fused by RRF and sent to the reranker. |
| `RANK_CONSTANT` | `60` | RRF rank constant. |
| `CROSS_LINGUAL` | `true` | Search the passages of both languages; `false` = the question's language only. |
| `MIN_RERANK_SCORE` | `0.02` | Passages scoring below this after reranking are dropped as noise. |
| `GRADE_STRONG_THRESHOLD` | `0.70` | Best rerank score at or above which a source is accepted without the LLM grader. |
| `GRADE_WEAK_THRESHOLD` | `0.03` | Best rerank score below which a source is rejected without the LLM grader. |
| `PREFETCH_FALLBACK` | `true` | Search the fallback sources in the background while v2 is being graded (only when v2's best score is below the strong threshold). |
| `SKIP_WEAK_GRADE_BELOW` | `0.04` | A fallback source below this best score is rejected without an LLM call when a later source already has a strong match. `0` turns the rule off. Never applied to v2. |
| `SKIP_WEAK_GRADE_BELOW_TEDX` | `0.09` | The same cutoff for the TEDx talk. Both sit just below the lowest score the grader accepted for that source in the calibration (§10). |
| `MAX_PASSAGE_CHARS` | `2500` | Passage text is truncated to this before reranking and prompting. |

### API

| Variable | Default | Meaning |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` if the server should not be reachable from the LAN. |
| `API_PORT` | `8000` | Bind port. |
| `CORS_ORIGINS` | `*` | Comma-separated list of browser origins allowed to call the API. **Set this to your website in production.** |
| `MAX_HISTORY_TURNS` | `6` | How many previous exchanges are replayed into the answer prompt. |
| `API_DOCS_ENABLED` | `true` | Serve `/docs` and `/openapi.json`. Turn off on a public host. |
| `DEBUG_DASHBOARD` | `local` | Pipeline traces for `/dashboard`: `local` = this machine only, `on` = anyone who can reach the API, `off` = no tracing. |
| `CHAT_RATE_LIMIT_PER_MINUTE` | `12` | Per-client-IP limit across `/chat` and `/chat/stream`; `0` disables. |
| `MAX_REQUEST_BYTES` | `65536` | Largest accepted request body; bigger bodies get HTTP 413 before parsing. |
| `LOG_LEVEL` | `INFO` | Log level for the API and the package loggers. |

---

## 9. Operations

### Measured latency (Apple M1 Pro, 16 GB, `qwen3.5:9b`)

Measured 2026-10-08 through `POST /chat/stream` on a freshly started API (warm-up done), five questions:

| Step | Time |
|---|---|
| Retrieval pass, one source tier (embed the question, Chroma, FTS5, cross-encoder on 30 candidates) | **4.6–6.5 s** right after startup: embedding 0.5–1.1 s, Chroma 0.01–0.8 s, FTS5 under 0.15 s, rerank 3.2–5.7 s. A warm benchmark of the reranker alone: ~2 s per 30 passages. |
| Classifier call | **5–11 s** warm |
| Grade | **0 s** on the rerank fast path (every graded source in this run); an LLM grader call takes ~15–25 s |
| Answer generation, streamed | **26–53 s** (English 26–35 s; a long Hebrew answer 53 s) |
| First streamed token | **28–37 s** after the question was sent |
| Full answer, end to end | **45–67 s**; an out-of-scope refusal 9 s |
| A question only an outside lecture answers (v2 → TEDx → v1 → Digital Innovation IDC) | **45 s** — every source decided by the fast path, the fallback searches run in the background |
| API startup | retrieval models loaded in ~12 s and the chat model primed in ~15 s, in the background |
| `smartup ingest` | ~4 minutes for the 3 627 passages; a re-run with no changes 0.6 s |

Almost all of the wall time is the local LLM; a retrieval pass is a few seconds at most, most of it the cross-encoder.

### Memory budget

Resident while the API runs: `qwen3.5:9b` ~5.9 GB (16 384-token context) and `bge-m3` ~0.7 GB in Ollama, the
cross-encoder ~1.1 GB (float16 on MPS) plus the PyTorch runtime in the API process. That fits a 16 GB Mac next to a
browser. Do not add a second chat model on 16 GB: a separate `CLASSIFIER_MODEL` (e.g. `qwen3.5:4b`) does not fit next
to the 9b, and Ollama then evicts one model for the other on every question.

### Keeping it warm

Every LLM call starts with the same ~6 000-token instructions, and Ollama can reuse its work on them — but Qwen3.5 runs
through llama.cpp's `llama-server`, which resumes a cached prompt only from a saved *checkpoint* (512 and 4 tokens
before the end of each prompt). The warm-up's short prompt (instructions + a few words) leaves a checkpoint just inside
the instructions; every later call shares those tokens and resumes from there. Lose it — after the keep-alive expires,
when another model or another prompt evicts it, or when Ollama restarts — and every call re-reads everything (measured:
classifier prompt ~29 s instead of ~6 s, answer prompt 35–49 s instead of 11–20 s).

- `CHAT_KEEP_ALIVE=24h` keeps the models loaded for a day after the last call.
- `WARM_ON_STARTUP=true` primes the checkpoint (and loads the retrieval models) when the API starts.
- `KEEP_WARM_MINUTES=10` re-primes it every 10 idle minutes and right after a classifier call shows it was lost (a
  classifier prompt above 15 s is logged as `shared-prompt cache lost, re-priming when idle`).

**The grader reads for the answer.** The grader's prompt is the answer prompt's opening — system prompt, replayed
history, then the evidence block — followed by its short grading task, so when it lets a question through, the answer
call resumes from the grader's reading (answer prompt ~1.5 s instead of 12–23 s). A question over 250 characters (220 on
a follow-up) gets the compact grader prompt instead, because the grading task must fit in the last 512 tokens.

### The fallback chain: parallel searches, skipped grades, progress

- **Parallel searches.** When v2's best score is below the strong threshold, `retrieve` starts the TEDx, v1 and outside
  searches in background threads (`PREFETCH_FALLBACK`), so they run while the v2 grader is thinking. (The cross-encoder
  scores one batch at a time, so these searches overlap the LLM, not each other.)
- **Skipped grades.** A fallback source below `SKIP_WEAK_GRADE_BELOW` (0.04; 0.09 for the TEDx talk) is rejected
  without an LLM call when a *later* source already has a strong match. v2 is never skipped.
- **Progress messages.** Each source the chain reaches sends an SSE `status` event ("Not found in the v2 course —
  checking Yonatan's TEDx talk…"), and the answer is labelled with the source that answered.


### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `smartup check` prints `Local index FAIL … index not built` | Run `smartup ingest`. |
| `check` says `vectors for … passages` or `built with … but EMBED_MODEL=…` | The index is incomplete (an interrupted run) or from another model: `smartup ingest --recreate`. |
| `check` reports a model `NOT PULLED` | `smartup models --pull`, or `ollama pull <model>`. |
| `Reranker WARN … NOT DOWNLOADED` | `smartup models` downloads it ahead of time; otherwise the first question downloads it (~2.3 GB). |
| Every course question ends in "I could not find this topic…" | The index is missing or empty — a search error is treated as "nothing found" (§7). `smartup check`. |
| Answers miss a lecture that was just added or changed | The API reads the index it found at startup: run `smartup ingest`, then restart the API. |
| The widget says the assistant is busy (HTTP 429) | Two chats are already generating; the local LLM is single-tenant. |
| The first question takes much longer than the next ones | Cold model load plus cold prompt prefill. Keep `WARM_ON_STARTUP=true` and leave the server running. |
| Every question is refused, including obvious course questions | The classifier call is failing (Ollama down or the model evicted) and the scope guard fails closed (§0.4). `smartup check`, then look for `classifier unavailable` in the log. |
| `MPS backend out of memory` in the log | Another process holds the GPU memory; set `RERANK_DEVICE=cpu` (≈8 s per search instead of ≈2 s). |

---

## 10. Evaluation and tests

### `smartup eval`

Retrieval mode (the default, no LLM) runs every in-scope question through `retrieve()` with the expected source tier
and checks whether an expected lecture appears among the `TOP_K_PASSAGES` returned passages — hit@8 — reporting the hit
rate, MRR, mean first rank and mean best rerank score, by language and by question kind. `--full` additionally runs
the whole agent per question and scores the status, the citations and the source that answered.

Measured 2026-10-08 on the local index (37 scored questions, lecture filter on as the agent applies it):
**37/37, MRR 1.000**, in both languages and every question kind — the same as the Elasticsearch version. Without the
lecture filter (`--no-lecture-filter`, the harder measurement): **36/37**; 11 of the 12 questions that name a lecture
still find it by topic alone (the Elasticsearch version: 9 of 11).

End to end, the agent was run through the API on five questions — a v2 concept question in English, the same kind
in Hebrew, an out-of-scope question, a lecture summary, and a question only an outside lecture answers. All five ended
as expected (answered with the right citations, refused, answered from Digital Innovation IDC Lecture 5 after
v2 → TEDx → v1), with the answer streamed token by token (§9). The 42-question `smartup eval --full` run (about an hour
of LLM time) has not yet been repeated on the local index; the Elasticsearch version passed 39 of 42.

### Calibrating the grading thresholds

The fast path compares a source's best rerank score with two thresholds, and a different reranker's scores mean
something different, so the thresholds were re-derived for `bge-reranker-v2-m3` (with its context line) against the
agent's own LLM grader (`qwen3.5:9b`, the prompt the agent uses). Every in-scope evaluation question plus 20 more —
common student questions and off-topic probes such as GDPR or the best programming language for an MVP — was searched
in every source the agent could search for it (v2, TEDx, v1, outside lectures; v2 and v1 when a lecture is named):
202 question/source pairs. The grader judged the 137 pairs whose best score lay between 0.005 and 0.97; the others
were taken as clear (39 below, 26 above).

| Best rerank score | v2 | TEDx | v1 | outside lectures |
|---|---|---|---|---|
| lowest the grader accepted (sufficient or partial) | 0.042 | 0.096 | 0.046 | 0.065 |
| highest the grader rejected, in decisions the agent makes | 0.158 | 0.602 | 0.456 | 0.555 |

Hence `GRADE_WEAK_THRESHOLD=0.03`, `GRADE_STRONG_THRESHOLD=0.70`, and skip cutoffs just below the lowest accepted
score (`SKIP_WEAK_GRADE_BELOW=0.04`, `SKIP_WEAK_GRADE_BELOW_TEDX=0.09`), so the skip rule never rejects a source the
grader would accept. On the 186 pairs that are decisions the agent actually makes (a question naming a source searches
that source alone and is not graded), the fast path decides 128 (69 %) and agrees with the grader on all 128; the other
58 go to the grader. One finding changed the agent: v1 Lecture N, tried after v2 Lecture N did not cover a question,
scored 0.75 and 0.87 on two questions the grader rejected — the same lecture number and often the same topic, but a
different lecture — so that step is now always judged by the grader.

### `pytest`

```bash
.venv/bin/python -m pytest -q          # or: make test
```

**256 passed, 6 skipped** — the six skipped tests are the integration checks against the real index, Ollama and
the reranker (they pass with `RUN_INTEGRATION=1`, see below).

The default run needs no network and no model: `tests/test_local_index.py` builds a real Chroma + SQLite index in a
temporary directory with a deterministic stand-in embedder and reranker and checks ingestion (incremental updates,
removals, rebuilds after a model change or an interrupted run), every leg's filters, the degraded modes, citations and
the lecture catalogue; `tests/test_queries.py` covers the filter and FTS5 builders (including hostile questions) and
RRF; `tests/test_loaders.py` the data fixes and the shipped 34-lecture corpus; `tests/test_reranker.py` and
`tests/test_health_api.py` the cross-encoder wrapper and `/health`; the rest cover language handling, the routing
rules, the prompt-prefix logic, the source tiers and the fallback chain, repeated questions, the citation verifier, the
API schemas, configuration parsing and the dashboard. `make lint` runs Ruff.

The integration checks run against the real index, Ollama and the reranker:

```bash
RUN_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_integration_local.py
```

---

## License

The code is released under the [MIT License](LICENSE). The bundled fonts are under the SIL Open Font License 1.1
(`web/fonts/`). The lecture documents in `data/raw/drive_09_2026` are SmartUp Academy course material by Yonatan Stern,
included as this project's data; check with SmartUp Academy before reusing them elsewhere.