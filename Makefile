# Convenience targets — everything also works through the `smartup` CLI.
PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: venv install models check ingest ask chat serve eval test lint

venv:
	python3 -m venv .venv && $(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

models:            ## pull CHAT_MODEL + EMBED_MODEL into Ollama and download the cross-encoder (once)
	$(PY) -m smartup_rag.cli models --pull

check:             ## Ollama, local index and reranker health + settings summary
	$(PY) -m smartup_rag.cli check

ingest:            ## build/update the local index from data/raw/drive_09_2026 (RECREATE=1 rebuilds it from scratch)
	$(PY) -m smartup_rag.cli ingest $(if $(RECREATE),--recreate,)

ask:               ## make ask Q="What does the course say about pricing?"
	$(PY) -m smartup_rag.cli ask "$(Q)"

chat:              ## interactive REPL with streamed answers
	$(PY) -m smartup_rag.cli chat

serve:             ## FastAPI + chat UI on API_PORT (default 8000)
	$(PY) -m smartup_rag.cli serve

eval:              ## retrieval-only evaluation (FULL=1 also runs the LLM, slow)
	$(PY) -m smartup_rag.cli eval $(if $(FULL),--full,)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check smartup_rag tests
