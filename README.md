# AquaQuery

Natural language chatbot for querying Argo oceanographic data. Ask questions in plain English about ocean temperature, salinity, pressure, and dissolved oxygen -- get real data, statistics, and interactive charts.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and configure

```bash
git clone https://github.com/Alshehrifal/aquaquery.git
cd aquaquery
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Backend setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Frontend setup

```bash
cd frontend
npm install
cd ..
```

### 4. Run

Start the backend and frontend in separate terminals:

```bash
# Terminal 1 - Backend (port 8000)
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 - Frontend (port 5173)
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### 5. Pre-cache Argo data (recommended)

First-time queries fetch data from remote GDAC servers and can take minutes. Pre-download data to make queries fast:

```bash
source .venv/bin/activate

# Cache last 90 days for all basins (takes a while)
python -m backend.scripts.precache_argo

# Or cache specific basins only
python -m backend.scripts.precache_argo --basins north_atlantic mediterranean

# Or a shorter time window
python -m backend.scripts.precache_argo --recent-days 30
```

Cached data is stored in `data/sample/` as NetCDF files. Subsequent queries that fall within the cached region and time window will be served locally.

---

## What You Can Ask

| Query Type | Example |
|------------|---------|
| Data queries | "What's the temperature at 500m in the Atlantic?" |
| Comparisons | "Compare Pacific vs Atlantic salinity" |
| Nearby data | "Show me data near Hawaii" |
| Visualizations | "Plot oxygen levels over time at 30N, 140W" |
| Info questions | "What is the Argo program?" |
| Coverage | "What data is available?" |

---

## Architecture

```
Browser  -->  React Frontend  -->  FastAPI Backend  -->  LangGraph Agents
                                        |
                  +---------------------+---------------------+
                  |                     |                     |
            Supervisor            Tool Layer            ChromaDB
          (intent classifier)         |              (knowledge base)
                  |              argopy + xarray
         +--------+--------+         |
         |        |        |    Argo GDAC
        RAG    Query     Viz    (ocean data)
       Agent   Agent    Agent
```

### Agent Pipeline

1. **Supervisor** classifies user intent (`info`, `data`, `viz`, `clarify`)
2. Routes to the appropriate agent:
   - **RAG Agent** -- answers general questions using a ChromaDB knowledge base
   - **Query Agent** -- calls backend tools in a multi-step loop (e.g., get basin coordinates, then fetch data)
   - **Viz Agent** -- generates Plotly charts from query results
3. Response streams back to the frontend via SSE

### Query Agent Tool Loop

The Query Agent uses a ReAct-style loop (up to 5 iterations):

```
LLM -> tool call -> ToolMessage feedback -> LLM -> next tool call -> ... -> final text
```

This enables multi-step workflows like:
- `ocean_basin_bounds("atlantic")` -> get lat/lon
- `query_ocean_data(TEMP, lat/lon from above)` -> get data
- LLM summarizes results in natural language

### Available Tools

| Tool | Purpose |
|------|---------|
| `query_ocean_data` | Fetch Argo profiles by variable, region, depth, time |
| `ocean_basin_bounds` | Get lat/lon bounds for named ocean basins |
| `get_data_coverage` | Return dataset coverage information |
| `calculate_statistics` | Compute mean, median, std, percentiles |
| `detect_anomalies` | Find outlier values (z-score based) |
| `get_nearest_profiles` | Find profiles near a coordinate |

---

## Project Structure

```
aquaquery/
  backend/
    main.py                 # FastAPI app entrypoint
    config.py               # Pydantic Settings (env vars)
    agents/
      supervisor.py         # Intent classifier + LangGraph router
      query_agent.py        # Data fetching (ReAct tool loop)
      rag_agent.py          # Knowledge base Q&A
      viz_agent.py          # Plotly chart generation
      state.py              # Shared agent state schema
    tools/
      argo_tools.py         # query_ocean_data, get_data_coverage
      geo_tools.py          # ocean_basin_bounds, get_nearest_profiles
      stats_tools.py        # calculate_statistics, detect_anomalies
    data/
      loader.py             # ArgoDataLoader (argopy wrapper)
      indexer.py            # ChromaDB knowledge base indexer
      schema.py             # Pydantic models, DataSource interface
    api/
      routes.py             # REST + SSE endpoints
      middleware.py         # Rate limiting, request logging, CORS
      session.py            # In-memory session store
      sanitizer.py          # Strip tool markup from responses
  frontend/
    src/
      App.tsx               # Root component
      components/
        ChatInterface.tsx   # Main chat UI
        ChatInput.tsx       # Message input
        MessageBubble.tsx   # Message rendering (Markdown)
        DataVisualization.tsx # Plotly chart renderer
        SampleQueries.tsx   # Pre-defined query buttons
      hooks/
        useChat.ts          # Chat state management + SSE streaming
      services/
        api.ts              # API client (REST + SSE)
        sanitize.ts         # Client-side content sanitization
      types/
        index.ts            # TypeScript interfaces
  tests/
    backend/
      test_agents.py        # Agent logic, tool loop, intent classification
      test_api.py           # API endpoints, sessions
      test_data_loader.py   # Data loading, QC filtering
      test_indexer.py       # ChromaDB indexing, search
      test_sanitizer.py     # Response sanitization
      test_tools.py         # Tool execution, basin bounds, haversine
  data/
    sample/                 # Cached Argo NetCDF files (gitignored)
    embeddings/             # ChromaDB persistent store
  docs/
    architecture.md         # System diagram, tech rationale
    agents.md               # Agent specifications
    api-spec.md             # Full API reference with examples
    data-pipeline.md        # Argo data flow, variables, QC flags
```

---

## API Reference

**Base URL:** `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/message` | Send a message, get complete response |
| GET | `/chat/stream?session_id=...&message=...` | SSE streaming response |
| GET | `/chat/history/{session_id}` | Retrieve conversation history |
| GET | `/data/variables` | List available Argo variables |
| GET | `/data/metadata` | Dataset coverage bounds |
| GET | `/health` | Health check |

### Chat message example

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Temperature at 500m in the Atlantic", "session_id": "my-session"}'
```

Response:
```json
{
  "session_id": "my-session",
  "message_id": "uuid",
  "content": "Based on Argo float data, the average temperature at 500m...",
  "visualization": { "chart_type": "bar_chart", "plotly_json": {...} },
  "sources": [],
  "agent_path": ["supervisor", "query_agent"],
  "timestamp": "2026-02-16T10:30:00Z"
}
```

### SSE events

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{"agent": "query_agent", "action": "fetching data"}` | Progress update |
| `token` | `{"content": "The average"}` | Streaming text chunk |
| `visualization` | `{"chart_type": "...", "plotly_json": {...}}` | Chart data |
| `done` | `{"message_id": "uuid", "agent_path": [...]}` | Completion |
| `error` | `{"message": "Failed to fetch data"}` | Error |

See [docs/api-spec.md](docs/api-spec.md) for full specification.

---

## Argo Data

### Variables

| Variable | Name | Unit | Typical Range |
|----------|------|------|---------------|
| `TEMP` | Temperature | degC | -2 to 35 |
| `PSAL` | Salinity | PSU | 30 to 40 |
| `PRES` | Pressure | dbar | 0 to 2000 |
| `DOXY` | Dissolved Oxygen | umol/kg | 0 to 400 |

### Ocean Basins

Pre-defined bounds for: Atlantic, Pacific, Indian, Southern, Arctic, Mediterranean, North Atlantic, South Atlantic, North Pacific, South Pacific.

### Data Flow

1. User query -> Supervisor classifies intent
2. Query Agent calls tools (e.g., `ocean_basin_bounds` -> `query_ocean_data`)
3. Tool invokes `ArgoDataLoader` -> argopy fetches from Argo GDAC
4. Data cached locally as NetCDF, QC filtered (flags 1 and 2 only)
5. Statistics computed, results returned to agent
6. Agent writes natural language summary
7. Viz Agent generates Plotly JSON if visualization requested
8. Frontend renders text + interactive charts

### Knowledge Base

25+ documents indexed in ChromaDB covering:
- Argo program (overview, floats, history, data access)
- Variables (temperature, salinity, pressure, oxygen, QC flags)
- Ocean concepts (thermocline, halocline, mixed layer, water masses)
- Ocean basins (Pacific, Atlantic, Indian, Southern)
- Data concepts (profiles, climatology, ENSO)

Embeddings: sentence-transformers `all-MiniLM-L6-v2` (local, no API key).

---

## Testing

```bash
# Run all backend tests
source .venv/bin/activate
pytest tests/backend -v

# With coverage
pytest tests/backend -v --cov=backend --cov-report=term-missing
```

105 tests covering agents, API endpoints, data loading, tools, indexing, and sanitization.

---

## Configuration

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | -- | Anthropic API key for Claude |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Claude model ID |
| `ARGO_FETCH_TIMEOUT` | No | `45` | Argo data fetch timeout in seconds |

### Backend settings (backend/config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `api_port` | 8000 | Backend server port |
| `cors_origins` | localhost:5173, localhost:3000 | Allowed CORS origins |
| `rate_limit_per_minute` | 10 | Max requests per IP per minute |
| `rag_top_k` | 3 | Number of documents to retrieve |
| `rag_min_relevance` | 0.5 | Minimum similarity threshold |
| `argo_fetch_timeout` | 45 | Argo data fetch timeout in seconds |
| `chroma_collection_name` | argo_knowledge | ChromaDB collection name |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI | Async-native, automatic OpenAPI docs, SSE support |
| Agents | LangGraph | Stateful agent orchestration with conditional routing |
| LLM | Claude Sonnet 4 | Tool calling, strong reasoning |
| Data | argopy + xarray | Purpose-built Argo library with caching |
| Vector store | ChromaDB | Embedded, no server needed |
| Embeddings | sentence-transformers | Free, local, no API key |
| Streaming | SSE | Simpler than WebSocket, sufficient for chat |
| Charts | Plotly.js | Interactive, JSON-serializable |
| Frontend | React 19 + TypeScript + Vite | Fast dev, type safety |
| Styling | Tailwind CSS | Utility-first, ocean theme |

---

## Design Decisions

- **SSE over WebSocket** -- simpler, HTTP-based, sufficient for request-response streaming
- **ChromaDB embedded** -- no server needed, Python-native, good for PoC
- **Plotly JSON from backend** -- backend generates chart spec, frontend renders it
- **In-memory sessions** -- easy for PoC, swap to Redis/Postgres for production
- **sentence-transformers** -- free local embeddings, no API key required
- **DataSource interface** -- abstract base class enables future datasets (ERA5, satellite, etc.)
- **ReAct tool loop** -- allows multi-step reasoning (get coordinates, then query data)
- **Defense-in-depth sanitization** -- both backend and frontend strip tool markup

---

## Extensibility

The `DataSource` abstract interface (`backend/data/schema.py`) allows adding new datasets:

```python
class DataSource(ABC):
    def fetch_region(self, lat_bounds, lon_bounds, depth_range, time_range) -> xr.Dataset
    def get_metadata(self) -> DatasetMetadata
    def get_available_variables(self) -> list[VariableInfo]
```

Future datasets could include climate reanalysis (ERA5), satellite ocean color, seismic catalogs, or weather station data. Agents remain unchanged -- only the tool layer adapts.

---

## Further Documentation

- [Architecture](docs/architecture.md) -- system diagram, component overview, tech rationale
- [Agent Specifications](docs/agents.md) -- supervisor routing, RAG, query, viz agents
- [API Specification](docs/api-spec.md) -- full endpoint reference with examples
- [Data Pipeline](docs/data-pipeline.md) -- Argo data flow, variables, QC flags, embeddings
