# SmartUp RAG — architecture and decisions

Goal: a website chatbot that answers questions about the SmartUp Academy lecture series (Yonatan Stern) strictly
from the lecture transcripts and summaries, in Hebrew or English — runnable by anyone from this repository, on one
machine, with no cloud account.

The behavioural spec is [`instructions.md`](../instructions.md). The pipeline enforces its hard rules in code, not
only in the prompt: the scope guard fails closed, sources are searched in a fixed order (v2 → TEDx → v1 → other
lectures outside the course), an explicitly requested source is never overridden, and no citation survives unless it
points to a lecture retrieved in the same turn.

---

## 1. Requirements

| Requirement | How it is met |
|---|---|
| Answers only from the lectures, with verifiable citations | Retrieval-augmented generation; `nodes.verify` rewrites or removes every citation against the retrieved lectures and rebuilds the Sources list |
| Hebrew and English, including Hebrew questions about English passages | `bge-m3` (cross-lingual embeddings) + a multilingual cross-encoder; language detection by Unicode script |
| Fixed source order with fallbacks | LangGraph: v2 pass → grade → TEDx → v1 → outside lectures, each graded on its own |
| Streams answers | LangGraph `astream_events` → Server-Sent Events (`/chat/stream`) → the widget; `smartup ask` / `chat` stream to the terminal |
| Runs locally, publishable | Ollama for both language models, Chroma + SQLite for the index, sentence-transformers for the reranker; the input data ships with the repository |
| Operable from a browser, embeddable on the website | FastAPI serves the chat page, the widget and a pipeline dashboard |

## 2. Data flow

```
data/raw/drive_09_2026/{course_v2,course_v1,outsource}/*.json      (34 lecture documents, in git)
        │  smartup ingest
        ▼
loaders: normalise (Digital Innovation relabelled, v1 title noise stripped, Lecture 11 parts)
        ▼
chunker: ~1 200-character passages per language + one summary passage per language   (3 627 passages)
        ▼
Ollama bge-m3 embeddings ──► Chroma  data/index/chroma/          vectors + filter fields
                         └─► SQLite  data/index/smartup.sqlite3   passages, FTS5 index, lecture catalogue, build metadata

question ─► classify (qwen3.5:9b) ─► retrieve per source tier:
              dense (Chroma) + BM25 (FTS5, text) + metadata (FTS5, topic/subjects/title) ─► RRF ─► cross-encoder
          ─► grade (rerank-score fast path, else LLM grader) ─► [fallback chain] ─► generate (streamed) ─► verify
```

## 3. Decisions

### 3.1 Chat model: `qwen3.5:9b` on Ollama

Newest Qwen generation available locally, strong Hebrew, reliable JSON-schema structured output for the classifier and
the grader, "thinking" can be switched off, and 6 GB resident leaves room on a 16 GB machine for the retrieval models.
All three LLM calls (classifier, grader, answer) share the same ~6 000-token system prompt, so Ollama's prompt cache
turns a ~27 s prefill into a few seconds; the API primes and re-primes that cache (README §9).

### 3.2 Vector store: Chroma, plus SQLite FTS5 for keywords

The previous version used Elastic Cloud Serverless. It was replaced because a public repository must run without a
cloud account or credentials, and because the corpus is small (3 627 passages, ~3 MB of text): an embedded store is
fast enough and has nothing to operate.

- **Chroma** (persistent client, HNSW, cosine) stores only vectors and the fields filters need. Metadata filters
  (`$and`, `$eq`, `$ne`, `$nin`) implement the source tiers inside the vector search.
- **SQLite FTS5** provides real BM25 (the `bm25()` function with per-column weights) and is part of the Python
  standard library. It holds the passages (the source of truth), an external-content FTS5 index kept in sync by
  triggers, the lecture catalogue with both summaries, and the build metadata.
- Both legs apply the same conditions, written once in `tiers.tier_conditions` and translated by
  `queries.chroma_where` / `queries.sql_where`.

A single store doing both (e.g. a vector extension inside SQLite) was not needed: Chroma answers "nearest vectors",
SQLite answers everything else, and the passage id joins them.

### 3.3 Hybrid retrieval with RRF

Three legs, as in the previous version: dense (meaning, cross-lingual), BM25 on the passage text (exact terms, names,
numbers, English jargon inside Hebrew text), and a metadata leg on section topic / lecture subjects / title. The
metadata leg scores every passage of a lecture alike, so ties are broken by the dense leg's order. Reciprocal Rank
Fusion (k = 60, 30 per leg, 30 fused) needs no score normalisation across legs. Questions are turned into quoted
OR-terms for FTS5, so a question can never be a syntax error.

### 3.4 Embeddings: `bge-m3` through Ollama

Multilingual and cross-lingual, 8k-token context, MIT licence, and already served by the same Ollama that runs the chat
model — no second runtime for embeddings, and both models stay resident together on 16 GB. Query vectors are cached
because the fallback chain searches up to four tiers with the same question.

### 3.5 Reranker: `BAAI/bge-reranker-v2-m3` with a context line

Compared on the evaluation set's candidates (the same 30 fused passages per question and source):

| Model | Hit@8 / MRR | Separation of answerable vs. unanswerable sources | Time per 30 passages (M1 Pro, MPS) | Licence |
|---|---|---|---|---|
| `bge-reranker-v2-m3`, passage only | 40/40, 1.000 | AUC 0.984; weakest correct source 0.03 | ~2 s | Apache-2.0 |
| **`bge-reranker-v2-m3`, context line + passage** | **40/40, 1.000** | **AUC 0.990; weakest correct source 0.19** | ~2 s | Apache-2.0 |
| `Qwen3-Reranker-0.6B` | same name bias on the probe questions | — (stopped: ~4× slower) | ~8 s | Apache-2.0 |
| `jina-reranker-v2-base-multilingual` | the previous (hosted) reranker | — | — | CC-BY-NC-4.0: not a default |

The context line (`Yonatan Stern (יונתן שטרן), SmartUp Foundations course (v2), Lecture 8:`) tells the cross-encoder
whose words a passage is. Without it, "What does Yonatan say about pricing?" ranked any passage *naming* Yonatan above
the lecture that teaches pricing.

### 3.6 Grading thresholds

The agent's fast path accepts a source when its best rerank score is ≥ `GRADE_STRONG_THRESHOLD`, rejects it when
< `GRADE_WEAK_THRESHOLD`, and asks the LLM grader in between. Scores of a different model are not comparable, so the
thresholds were re-derived for `bge-reranker-v2-m3` from the LLM grader's own verdicts (README §10): 202
question/source pairs, the real retrieval run and the grader asked. With weak 0.03 and strong 0.70 the fast path decides
69 % of the agent's grading decisions and agrees with the grader on all of them. The calibration also showed that a
rerank score cannot tell v1 Lecture N from v2 Lecture N (same number, often the same topic), so the v1 step for a
named lecture is always left to the grader.

### 3.7 LangChain + LangGraph

- LangChain: `ChatOllama`, `OllamaEmbeddings`, `with_structured_output` (classifier and grader), `Document`, a custom
  `BaseRetriever` (`LectureRetriever`), `RecursiveCharacterTextSplitter`.
- LangGraph: the instructions' workflow as explicit nodes and conditional edges, a `MemorySaver` checkpointer for
  multi-turn threads, `astream_events` for token streaming.
- Not used: LangChain's Chroma wrapper — the retriever needs the raw `where` filters, the SQLite join and the three-leg
  fusion, which are simpler to write directly against `chromadb` and `sqlite3`.

## 4. Guardrails in code

1. **Scope + intent classifier** (structured output; rewrites follow-ups into a standalone query); fails closed.
2. **Ordered retrieval**: v2 → grade (rerank fast path, LLM grader in the ambiguous band) → TEDx → v1 → outside
   lectures, stopping at the first source that answers; explicit source requests search that source alone.
3. **Citation verifier**: drops citations to lectures that were not retrieved, normalises titles, rebuilds Sources.
4. **Language routing** by Unicode script; Hebrew answers for Hebrew questions.

## 5. Moving from Elasticsearch — what changed

| Area | Before | Now |
|---|---|---|
| Source of the index | Elasticsearch lecture index, itself built from the Drive exports | the Drive exports in `data/raw/drive_09_2026`, directly |
| Build | `smartup build-naive` (lecture index) + `smartup ingest --source es` (passage index) | `smartup ingest` (incremental) |
| Retrieval backends | `passages` and `naive` (semantic highlighter) | one hybrid local index |
| Configuration | `ES_URL`, `ES_API_KEY`, index names, inference ids | `DATA_DIR`, `INDEX_DIR`, `EMBED_MODEL`, `RERANK_MODEL`, `RERANK_DEVICE` |
| Grading thresholds | strong 0.40 / weak 0.06, skip 0.20 / 0.25 (Jina scores) | strong 0.70 / weak 0.03, skip 0.04 / 0.09 (bge-reranker-v2-m3 with the context line) |
| `/health` | Ollama + Elasticsearch | Ollama (chat and embedding model) + index + reranker |

The agent, the prompts, the API contract, the widget and the evaluation set are unchanged; the retriever still returns
LangChain `Document`s with the same metadata.
