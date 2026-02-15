"""Shared agent state definition for LangGraph."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared across all agents in the graph."""

    messages: Annotated[list, add_messages]
    intent: str
    data: dict[str, Any]
    visualization: dict[str, Any]
    metadata: dict[str, Any]
