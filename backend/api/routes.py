"""FastAPI routes for AquaQuery API."""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from backend.agents.state import AgentState
from backend.api.session import SessionStore
from backend.data.schema import ChatRequest, ChatResponse, ErrorResponse, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# These are set during app startup in main.py
_graph = None
_session_store: SessionStore = SessionStore()
_data_loader = None


def set_graph(graph: Any) -> None:
    """Set the compiled LangGraph (called during app startup)."""
    global _graph
    _graph = graph


def set_session_store(store: SessionStore) -> None:
    """Set the session store (called during app startup or testing)."""
    global _session_store
    _session_store = store


def set_data_loader(loader: Any) -> None:
    """Set the data loader (called during app startup)."""
    global _data_loader
    _data_loader = loader


@router.post("/chat/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest) -> ChatResponse:
    """Send a chat message and receive a complete response."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent pipeline not initialized")

    session_id = _session_store.get_or_create_session(request.session_id)

    # Store user message
    _session_store.add_message(session_id, "user", request.message)

    # Build initial state
    state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "intent": "",
        "data": {},
        "visualization": {},
        "metadata": {},
    }

    try:
        result = await _graph.ainvoke(state)
    except Exception as e:
        logger.error("Agent pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Extract response content
    messages = result.get("messages", [])
    content = ""
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", str(last_msg))

    visualization = result.get("visualization")
    if visualization and not visualization.get("plotly_json"):
        visualization = None

    sources = result.get("data", {}).get("sources", [])
    agent_path = result.get("metadata", {}).get("agent_path", [])

    # Store assistant message
    assistant_msg = _session_store.add_message(
        session_id, "assistant", content,
        visualization=visualization,
        sources=sources,
    )

    return ChatResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        content=content,
        visualization=visualization,
        sources=sources,
        agent_path=agent_path,
        timestamp=assistant_msg.timestamp,
    )


@router.get("/chat/stream")
async def chat_stream(
    session_id: str = Query(..., description="Session ID"),
    message: str = Query(..., description="User message"),
) -> EventSourceResponse:
    """SSE endpoint for streaming agent responses."""
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent pipeline not initialized")

    sid = _session_store.get_or_create_session(session_id)
    _session_store.add_message(sid, "user", message)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "intent": "",
            "data": {},
            "visualization": {},
            "metadata": {},
        }

        try:
            yield {"event": "status", "data": json.dumps({"agent": "supervisor", "action": "classifying intent"})}

            result = await _graph.ainvoke(state)

            agent_path = result.get("metadata", {}).get("agent_path", [])
            if len(agent_path) > 1:
                yield {"event": "status", "data": json.dumps({"agent": agent_path[-1], "action": "processing"})}

            # Stream the content token by token (simulated for non-streaming LLM)
            messages = result.get("messages", [])
            content = ""
            if messages:
                content = getattr(messages[-1], "content", str(messages[-1]))

            # Send content in chunks
            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield {"event": "token", "data": json.dumps({"content": chunk})}
                await asyncio.sleep(0.02)  # Small delay for streaming effect

            # Send visualization if present
            viz = result.get("visualization")
            if viz and viz.get("plotly_json"):
                yield {"event": "visualization", "data": json.dumps(viz)}

            # Store assistant message
            sources = result.get("data", {}).get("sources", [])
            _session_store.add_message(
                sid, "assistant", content,
                visualization=viz,
                sources=sources,
            )

            yield {
                "event": "done",
                "data": json.dumps({
                    "message_id": str(uuid.uuid4()),
                    "agent_path": agent_path,
                }),
            }

        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


@router.get("/chat/history/{session_id}", response_model=list[Message])
async def chat_history(session_id: str) -> list[Message]:
    """Retrieve conversation history for a session."""
    if not _session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_store.get_history(session_id)


@router.get("/data/variables")
async def list_variables() -> dict[str, Any]:
    """List available Argo variables."""
    if _data_loader is None:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    variables = _data_loader.get_available_variables()
    return {
        "variables": [
            {
                "name": v.name,
                "display_name": v.display_name,
                "unit": v.unit,
                "description": v.description,
                "typical_range": list(v.typical_range),
            }
            for v in variables
        ]
    }


@router.get("/data/metadata")
async def dataset_metadata() -> dict[str, Any]:
    """Return dataset coverage information."""
    if _data_loader is None:
        raise HTTPException(status_code=503, detail="Data loader not initialized")

    meta = _data_loader.get_metadata()
    return {
        "lat_bounds": list(meta.lat_bounds),
        "lon_bounds": list(meta.lon_bounds),
        "depth_range": list(meta.depth_range),
        "time_range": list(meta.time_range),
        "total_profiles": meta.total_profiles,
        "available_variables": list(meta.available_variables),
        "data_source": meta.data_source,
        "last_updated": meta.last_updated,
    }
