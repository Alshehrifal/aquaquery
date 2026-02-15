"""Query agent for translating natural language to data operations."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from backend.agents.state import AgentState
from backend.config import Settings, get_settings
from backend.tools.argo_tools import get_data_coverage, query_ocean_data
from backend.tools.geo_tools import get_nearest_profiles, ocean_basin_bounds
from backend.tools.stats_tools import calculate_statistics, detect_anomalies

logger = logging.getLogger(__name__)

QUERY_SYSTEM_PROMPT = """You are an expert data analyst for Argo oceanographic data.
Your job is to translate natural language queries into tool calls to fetch and analyze ocean data.

Available variables: TEMP (temperature, degC), PSAL (salinity, PSU), PRES (pressure, dbar), DOXY (dissolved oxygen, umol/kg)

When a user mentions an ocean basin by name, first use the ocean_basin_bounds tool to get its coordinates,
then use query_ocean_data with those coordinates.

For queries about nearby data, use get_nearest_profiles.
For statistical analysis, use calculate_statistics or detect_anomalies on the fetched data.
For general coverage questions, use get_data_coverage.

Always provide clear, concise summaries of the results. Include units in your response.

CRITICAL FORMATTING RULES:
- Your response goes directly to the end user in a chat interface.
- NEVER include XML tags, tool calls, function calls, or any internal markup in your response.
- NEVER show <tool_call>, <function_calls>, <invoke>, <parameter>, or similar tags.
- NEVER expose internal function names, JSON parameters, or raw data structures.
- Write only clean, natural language that a non-technical user can understand.
"""

SUMMARY_SYSTEM_PROMPT = """You are summarizing ocean data query results for a chat user.
Write a clear, natural language summary of the data. Include key statistics with units.

CRITICAL: Your response goes directly to the end user. Write ONLY clean natural language.
Do NOT include any XML tags, tool calls, function invocations, code blocks with raw data,
or internal markup. Never mention tool names or function names. Just describe the findings
in plain English as a helpful oceanography expert would."""

QUERY_TOOLS = [
    query_ocean_data,
    get_data_coverage,
    calculate_statistics,
    detect_anomalies,
    get_nearest_profiles,
    ocean_basin_bounds,
]


def create_query_agent(settings: Settings | None = None) -> "QueryAgent":
    """Create a query agent instance."""
    return QueryAgent(settings=settings)


class QueryAgent:
    """Agent that translates natural language to tool-based data queries."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatAnthropic(
            model=self._settings.anthropic_model,
            api_key=self._settings.anthropic_api_key,
            max_tokens=2048,
        ).bind_tools(QUERY_TOOLS)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the query agent: invoke LLM with tools, execute tool calls, summarize."""
        messages = state["messages"]
        system_msg = SystemMessage(content=QUERY_SYSTEM_PROMPT)

        # First LLM call - may request tool calls
        response = await self._llm.ainvoke([system_msg, *messages])

        # Execute any tool calls
        tool_results = {}
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_map = {t.name: t for t in QUERY_TOOLS}
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                if tool_name in tool_map:
                    try:
                        result = await tool_map[tool_name].ainvoke(tool_args)
                        tool_results[tool_name] = result
                    except Exception as e:
                        logger.error("Tool %s failed: %s", tool_name, e)
                        tool_results[tool_name] = {"error": str(e), "success": False}

        # If we got tool results, make a follow-up call to summarize
        if tool_results:
            from langchain_core.messages import HumanMessage

            summary_prompt = (
                f"Here are the data results to summarize for the user. "
                f"Write a clear natural language summary with key statistics and units. "
                f"Do NOT include any XML, tool calls, or raw data structures.\n\n"
                f"{tool_results}"
            )
            summary_system = SystemMessage(content=SUMMARY_SYSTEM_PROMPT)
            summary_llm = ChatAnthropic(
                model=self._settings.anthropic_model,
                api_key=self._settings.anthropic_api_key,
                max_tokens=2048,
            )
            summary_response = await summary_llm.ainvoke([
                summary_system,
                *messages,
                HumanMessage(content=summary_prompt),
            ])
            return {
                "messages": [summary_response],
                "data": {"tool_results": tool_results},
            }

        return {
            "messages": [response],
            "data": {},
        }
