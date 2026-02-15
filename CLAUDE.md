# AquaQuery - Oceanographic Data Chatbot

## Project Overview
Interactive web chatbot for querying Argo oceanographic data through natural language.
Built as a PoC with future scalability to other spatial-temporal datasets.

## Tech Stack
- **Backend:** FastAPI + LangGraph + Claude Sonnet (Anthropic API)
- **Data:** argopy + xarray + ChromaDB (sentence-transformers for embeddings)
- **Frontend:** React 18 + TypeScript + Vite + Tailwind + shadcn/ui
- **Charts:** Plotly.js | **Maps:** Leaflet | **Streaming:** SSE

## Project Structure
```
aquaquery/
  backend/          # FastAPI app
    agents/         # LangGraph agents (supervisor, rag, query, viz)
    tools/          # LangChain tools (argo, stats, geo)
    data/           # Data loader, schema, vector indexer
    api/            # Routes, middleware
    config.py       # Pydantic Settings
    main.py         # App entrypoint
  frontend/         # React + Vite app
  tests/
    backend/        # pytest
    frontend/       # vitest
  data/
    sample/         # Cached Argo data
    embeddings/     # ChromaDB store
  docs/             # Design documentation
```

## Development Commands
```bash
# Backend
source .venv/bin/activate
cd backend && uvicorn main:app --reload --port 8000
pytest ../tests/backend -v --cov=. --cov-report=term-missing

# Frontend
cd frontend && npm run dev
npm test
```

## Conventions
- Python: type hints, async/await, Pydantic models
- TypeScript: strict mode, functional components, hooks
- Immutable data patterns (never mutate)
- TDD: write tests first, 80%+ coverage
- Conventional commits: feat:, fix:, refactor:, docs:, test:
- Small files: 200-400 lines typical, 800 max
- Abstract DataSource interface for future dataset extensibility

## Environment Variables
- `ANTHROPIC_API_KEY` - Required for Claude API access

## Key Design Decisions
- SSE over WebSocket (simpler, HTTP-based, sufficient for PoC)
- ChromaDB embedded (no server, Python-native)
- Plotly JSON from backend rendered by Plotly.js on frontend
- In-memory session storage (swap to Redis/Postgres later)
- sentence-transformers for local embeddings (free, no API key)
