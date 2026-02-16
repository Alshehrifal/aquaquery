"""Supervisor agent: routes queries to appropriate sub-agents via LangGraph."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from backend.agents.query_agent import QueryAgent, create_query_agent
from backend.agents.rag_agent import RagAgent, create_rag_agent
from backend.agents.state import AgentState
from backend.agents.viz_agent import VizAgent, create_viz_agent
from backend.config import Settings, get_settings
from backend.data.indexer import ArgoKnowledgeIndexer

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are an intent classifier for an oceanographic data chatbot.
Classify the user's message into exactly ONE of these intents:

- "info": General questions about Argo, oceanography, or concepts (e.g., "What is Argo?", "Explain thermocline")
- "data": Requests for specific data, measurements, or statistics (e.g., "Temperature at 500m in Atlantic", "Average salinity near Hawaii")
- "viz": Requests that explicitly ask for visualization, charts, or plots (e.g., "Plot temperature trends", "Show me a map of salinity")
- "clarify": Ambiguous queries that need more detail (e.g., "ocean data", "tell me about it")

Respond with ONLY the intent label, nothing else."""


def classify_intent(
    message: str,
    llm: ChatAnthropic,
) -> str:
    """Classify user intent using Claude."""
    response = llm.invoke([
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=message),
    ])
    intent = response.content.strip().lower().strip('"').strip("'")

    valid_intents = {"info", "data", "viz", "clarify"}
    if intent not in valid_intents:
        # Fallback heuristic -- check data keywords first (more specific)
        lower_msg = message.lower()
        words = lower_msg.split()
        if any(w in lower_msg for w in [
            "temperature", "salinity", "depth", "pressure", "oxygen",
            "average", "mean", "compare", "data",
        ]):
            return "data"
        if any(w in lower_msg for w in ["plot", "chart", "graph", "map", "visualiz"]):
            return "viz"
        if "show" in words:
            return "data"
        if any(w in lower_msg for w in ["what is", "explain", "tell me about", "describe"]):
            return "info"
        if "how" in words:
            return "info"
        return "clarify"

    return intent


def build_graph(
    settings: Settings | None = None,
    indexer: ArgoKnowledgeIndexer | None = None,
) -> StateGraph:
    """Build the LangGraph state graph for the agent pipeline.

    Returns a compiled StateGraph ready for invocation.
    """
    settings = settings or get_settings()

    rag_agent = create_rag_agent(settings=settings, indexer=indexer)
    query_agent = create_query_agent(settings=settings)
    viz_agent = create_viz_agent(settings=settings)

    classifier_llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=16,
    )

    # --- Node functions ---

    def supervisor_node(state: AgentState) -> dict[str, Any]:
        """Classify intent and set it in state."""
        messages = state["messages"]
        user_msg = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and (
                getattr(msg, "type", None) == "human"
                or getattr(msg, "role", None) == "user"
            ):
                user_msg = msg.content
                break

        intent = classify_intent(user_msg, classifier_llm)
        logger.info("Classified intent: %s for message: %s", intent, user_msg[:100])
        return {"intent": intent, "metadata": {"agent_path": ["supervisor"]}}

    async def rag_node(state: AgentState) -> dict[str, Any]:
        """Execute RAG agent."""
        result = await rag_agent.run(state)
        path = state.get("metadata", {}).get("agent_path", [])
        return {
            **result,
            "metadata": {"agent_path": [*path, "rag_agent"]},
        }

    async def query_node(state: AgentState) -> dict[str, Any]:
        """Execute Query agent."""
        result = await query_agent.run(state)
        path = state.get("metadata", {}).get("agent_path", [])
        return {
            **result,
            "metadata": {"agent_path": [*path, "query_agent"]},
        }

    async def viz_node(state: AgentState) -> dict[str, Any]:
        """Execute Viz agent."""
        result = await viz_agent.run(state)
        path = state.get("metadata", {}).get("agent_path", [])
        return {
            **result,
            "metadata": {"agent_path": [*path, "viz_agent"]},
        }

    def clarify_node(state: AgentState) -> dict[str, Any]:
        """Ask for clarification."""
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I'd like to help you explore Argo oceanographic data. "
                        "Could you be more specific? For example:\n"
                        "- Ask about ocean concepts: 'What is a thermocline?'\n"
                        "- Request data: 'What's the temperature at 500m in the Atlantic?'\n"
                        "- Ask for visualizations: 'Plot salinity trends in the Pacific'"
                    )
                )
            ],
            "metadata": {"agent_path": ["supervisor", "clarify"]},
        }

    # --- Routing function ---

    def route_by_intent(state: AgentState) -> str:
        intent = state.get("intent", "clarify")
        if intent == "info":
            return "rag"
        if intent == "data":
            return "query"
        if intent == "viz":
            return "query_for_viz"
        return "clarify"

    # --- Build graph ---

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("query", query_node)
    graph.add_node("query_for_viz", query_node)
    graph.add_node("viz", viz_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "rag": "rag",
            "query": "query",
            "query_for_viz": "query_for_viz",
            "clarify": "clarify",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("query", END)
    graph.add_edge("query_for_viz", "viz")
    graph.add_edge("viz", END)
    graph.add_edge("clarify", END)

    return graph.compile()
