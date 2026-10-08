"""SmartUp RAG — local RAG chatbot over the SmartUp Academy lecture corpus.

Stack: Ollama (qwen3.5 answers, bge-m3 embeddings) · Chroma vectors + SQLite FTS5 BM25 (hybrid, RRF) ·
local cross-encoder rerank (sentence-transformers) · LangChain + LangGraph · FastAPI with streamed answers.
"""

__version__ = "0.2.0"
