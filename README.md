# Agentic RAG with pgvector

Retrieval-augmented generation app built with FastAPI, LangChain, LangGraph, Claude and PostgreSQL (pgvector) on Supabase.

## Setup

```bash
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

Fill every variable in `.env` before starting. The app fails at startup if any is missing.
