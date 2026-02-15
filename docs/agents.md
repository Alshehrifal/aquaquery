# Agent Specifications

## Overview

AquaQuery uses a multi-agent architecture built on LangGraph. A supervisor agent
routes queries to specialized sub-agents based on intent classification.

## Supervisor Agent

**Purpose:** Intent classification and routing.

**Input:** User message + conversation history
**Output:** Routed to appropriate agent(s)

### Routing Table

| Intent | Route | Example Query |
|--------|-------|---------------|
| `info` | RAG Agent | "What is the Argo program?" |
| `data` | Query Agent | "Temperature at 500m in the Atlantic" |
| `viz` | Query Agent -> Viz Agent | "Plot salinity trends over time" |
| `compare` | Query Agent -> Viz Agent | "Compare Pacific vs Atlantic salinity" |
| `clarify` | Direct response | Ambiguous query, ask for details |

### Intent Classification Prompt

The supervisor uses Claude with a system prompt that classifies user intent
into one of the above categories. It considers:
- Keywords (plot, show, compare, what is, explain)
- Presence of spatial/temporal parameters
- Whether visualization is requested

### State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    data: dict          # Query results
    visualization: dict # Plotly JSON
    metadata: dict      # Session info
```

## RAG Agent

**Purpose:** Answer informational questions using retrieved context.

### Embedding Strategy
- **Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Chunk size:** ~500 tokens per document
- **Collection:** Single ChromaDB collection "argo_knowledge"

### Document Categories
1. Argo program overview (what, why, how)
2. Float technology and deployment
3. Variable descriptions (temperature, salinity, pressure, oxygen)
4. Quality control flags and levels
5. Ocean basins and circulation patterns
6. Thermocline, halocline, mixed layer concepts
7. Data access and formats (NetCDF, profiles)

### Retrieval Parameters
- Top-k: 3 documents
- Similarity metric: cosine
- Minimum relevance threshold: 0.5

### Response Format
Returns text answer with source attribution from retrieved documents.

## Query Agent

**Purpose:** Translate natural language to data operations via tool calling.

### Tool Catalog

| Tool | Purpose |
|------|---------|
| `query_ocean_data` | Fetch Argo data by region, depth, time |
| `get_data_coverage` | Return available data bounds |
| `calculate_statistics` | Mean, median, std, min, max |
| `detect_anomalies` | Find outlier values |
| `get_nearest_profiles` | Find profiles near coordinates |
| `ocean_basin_bounds` | Named basin -> lat/lon bounds |

### Spatial-Temporal Filtering
- Latitude: -90 to 90
- Longitude: -180 to 180
- Depth: 0 to 6000m (most Argo data 0-2000m)
- Time: 1999-present (Argo program start)

### Query Translation Examples

| Natural Language | Tool Call |
|-----------------|-----------|
| "Temperature in the Atlantic at 500m" | `query_ocean_data(variable="TEMP", lat_min=-60, lat_max=60, lon_min=-80, lon_max=0, depth_min=450, depth_max=550)` |
| "Average salinity near Hawaii" | `get_nearest_profiles(lat=21, lon=-157)` -> `calculate_statistics(...)` |
| "Data coverage info" | `get_data_coverage()` |

## Viz Agent

**Purpose:** Generate Plotly JSON visualizations from query results.

### Chart Type Selection

| Data Shape | Chart Type |
|-----------|------------|
| Variable vs depth (single profile) | Line chart (depth profile) |
| Variable over time (single location) | Time series line |
| Variable across lat/lon | Heatmap or scatter map |
| Comparison (2+ groups) | Grouped bar chart |
| Distribution | Histogram |
| Spatial coverage | Leaflet marker map |

### Plotly JSON Schema

The viz agent returns:
```python
{
    "chart_type": "time_series",  # For frontend routing
    "plotly_json": {
        "data": [...],            # Plotly trace objects
        "layout": {...}           # Plotly layout config
    },
    "description": "Temperature trend at 30N, 140W from 2020-2024"
}
```

### Chart Configuration
- Responsive sizing (width: 100%, height: auto)
- Ocean-themed color palette (blues, teals)
- Scientific axis labels with units
- Interactive hover tooltips
- Download button (PNG export)
