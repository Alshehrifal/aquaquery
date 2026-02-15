# API Specification

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### POST /chat/message

Send a chat message and receive a complete response.

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "What is the average temperature at 500m in the Atlantic?"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "message_id": "uuid-string",
  "content": "The average temperature at 500m depth in the Atlantic Ocean is approximately 8.2 degrees C...",
  "visualization": {
    "chart_type": "depth_profile",
    "plotly_json": {
      "data": [{"type": "scatter", "x": [...], "y": [...]}],
      "layout": {"title": "Temperature Profile", "xaxis": {"title": "Temperature (degC)"}}
    },
    "description": "Temperature depth profile for the Atlantic Ocean"
  },
  "sources": ["Argo Program Overview", "Atlantic Ocean Basin"],
  "agent_path": ["supervisor", "query_agent", "viz_agent"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid request (missing fields)
- 429: Rate limited
- 500: Internal error

---

### GET /chat/stream

SSE endpoint for streaming agent responses.

**Query Parameters:**
- `session_id` (required): Session identifier
- `message` (required): User query (URL-encoded)

**SSE Events:**
```
event: status
data: {"agent": "supervisor", "action": "classifying intent"}

event: status
data: {"agent": "query_agent", "action": "fetching data"}

event: token
data: {"content": "The average"}

event: token
data: {"content": " temperature at"}

event: visualization
data: {"chart_type": "bar", "plotly_json": {...}, "description": "..."}

event: done
data: {"message_id": "uuid", "agent_path": ["supervisor", "query_agent"]}

event: error
data: {"message": "Failed to fetch data for the specified region"}
```

---

### GET /chat/history/{session_id}

Retrieve conversation history for a session.

**Response:**
```json
{
  "session_id": "uuid-string",
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "What is Argo?",
      "timestamp": "2024-01-15T10:00:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "Argo is a global array of...",
      "visualization": null,
      "sources": ["Argo Program Overview"],
      "timestamp": "2024-01-15T10:00:02Z"
    }
  ]
}
```

---

### GET /data/variables

List available Argo variables.

**Response:**
```json
{
  "variables": [
    {
      "name": "TEMP",
      "display_name": "Temperature",
      "unit": "degC",
      "description": "In-situ sea water temperature",
      "typical_range": [-2.0, 35.0]
    },
    {
      "name": "PSAL",
      "display_name": "Salinity",
      "unit": "PSU",
      "description": "Practical salinity",
      "typical_range": [30.0, 40.0]
    },
    {
      "name": "PRES",
      "display_name": "Pressure",
      "unit": "dbar",
      "description": "Sea water pressure (approximately depth in meters)",
      "typical_range": [0.0, 2000.0]
    },
    {
      "name": "DOXY",
      "display_name": "Dissolved Oxygen",
      "unit": "umol/kg",
      "description": "Dissolved oxygen concentration",
      "typical_range": [0.0, 400.0]
    }
  ]
}
```

---

### GET /data/metadata

Return dataset coverage information.

**Response:**
```json
{
  "lat_bounds": [-90.0, 90.0],
  "lon_bounds": [-180.0, 180.0],
  "depth_range": [0.0, 2000.0],
  "time_range": ["1999-01-01", "2024-12-31"],
  "total_profiles": 2500000,
  "available_variables": ["TEMP", "PSAL", "PRES", "DOXY"],
  "data_source": "Argo GDAC",
  "last_updated": "2024-01-15"
}
```

---

## Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_REGION",
    "message": "The specified latitude range is outside valid bounds (-90 to 90)",
    "details": {
      "field": "lat_min",
      "value": -100,
      "valid_range": [-90, 90]
    }
  }
}
```

## Rate Limiting

- 10 requests per minute per IP address
- Rate limit headers included in responses:
  - `X-RateLimit-Limit: 10`
  - `X-RateLimit-Remaining: 7`
  - `X-RateLimit-Reset: 1705312800`

## CORS

Allowed origins for PoC:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (alternative dev port)

## Authentication

None for PoC. Future: API key or JWT-based auth.
