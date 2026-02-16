"""Query agent for translating natural language to data operations."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from backend.agents.state import AgentState
from backend.config import Settings, get_settings
from backend.tools.argo_tools import get_data_coverage, query_ocean_data
from backend.tools.geo_tools import get_nearest_profiles, ocean_basin_bounds
from backend.tools.stats_tools import calculate_statistics, detect_anomalies

logger = logging.getLogger(__name__)

QUERY_SYSTEM_PROMPT = """You are AquaQuery, an expert data analyst with DIRECT ACCESS to the Argo oceanographic database.

MANDATORY: You MUST use your tools to answer data questions. NEVER respond with generic knowledge.
- For ANY question about temperature, salinity, pressure, or oxygen: CALL query_ocean_data
- For ocean basin names (Atlantic, Pacific, etc.): FIRST call ocean_basin_bounds to get coordinates, THEN call query_ocean_data
- For comparisons: Call query_ocean_data for EACH region/variable separately
- For statistics: Use the statistics returned by query_ocean_data, or call calculate_statistics for deeper analysis

Available variables: TEMP (temperature, degC), PSAL (salinity, PSU), PRES (pressure, dbar), DOXY (dissolved oxygen, umol/kg)

You have these tools:
- query_ocean_data: Fetch Argo profiles by variable, location, depth, time
- ocean_basin_bounds: Get lat/lon bounds for named ocean basins
- get_data_coverage: Return dataset coverage information
- calculate_statistics: Compute detailed statistics on a dataset
- detect_anomalies: Find outlier values in data
- get_nearest_profiles: Find profiles near a coordinate

Example workflows:
1. "Temperature at 500m in Atlantic" -> ocean_basin_bounds("atlantic") -> query_ocean_data(variable="TEMP", depth_min=450, depth_max=550, lat_min=..., ...)
2. "Compare Pacific vs Atlantic salinity" -> ocean_basin_bounds("pacific") -> query_ocean_data for Pacific -> ocean_basin_bounds("atlantic") -> query_ocean_data for Atlantic -> summarize both
3. "Data near Hawaii" -> get_nearest_profiles(lat=21.3, lon=-157.8)

FORMATTING: Write clean natural language for the end user. Never expose XML tags, tool names, or raw data structures.
"""

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
        """Execute the query agent with a ReAct-style tool loop.

        The LLM can make multiple rounds of tool calls (up to max_iterations).
        Each tool result is fed back as a ToolMessage so the LLM can reason
        about results and make follow-up calls (e.g., ocean_basin_bounds ->
        query_ocean_data).
        """
        messages = state["messages"]
        conversation = [SystemMessage(content=QUERY_SYSTEM_PROMPT), *messages]

        all_tool_results: dict[str, Any] = {}
        max_iterations = 5

        for iteration in range(max_iterations):
            response = await self._llm.ainvoke(conversation)
            conversation.append(response)

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                return {
                    "messages": [response],
                    "data": {"tool_results": all_tool_results},
                }

            tool_map = {t.name: t for t in QUERY_TOOLS}
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                logger.info("Tool call [%d]: %s(%s)", iteration, tool_name, tool_args)

                if tool_name in tool_map:
                    try:
                        result = await tool_map[tool_name].ainvoke(tool_args)
                        all_tool_results[f"{tool_name}_{iteration}"] = result
                    except Exception as e:
                        logger.error("Tool %s failed: %s", tool_name, e)
                        result = {"error": str(e), "success": False}
                        all_tool_results[f"{tool_name}_{iteration}"] = result
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                conversation.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        # Max iterations reached -- final call without tools to force text summary
        summary_llm = ChatAnthropic(
            model=self._settings.anthropic_model,
            api_key=self._settings.anthropic_api_key,
            max_tokens=2048,
        )
        final = await summary_llm.ainvoke(conversation)
        return {
            "messages": [final],
            "data": {"tool_results": all_tool_results},
        }
