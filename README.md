# Agentic RAG with pgvector

Retrieval-augmented generation app built with FastAPI, LangChain, LangGraph, Claude and PostgreSQL (pgvector).

## Setup

```bash
cp .env.example .env
docker compose up -d
uv sync
uv run uvicorn app.main:app --reload
```
