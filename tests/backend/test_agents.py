"""Tests for agents: supervisor routing, RAG retrieval, viz generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agents.query_agent import QueryAgent
from backend.agents.supervisor import classify_intent
from backend.agents.viz_agent import (
    generate_bar_chart,
    generate_depth_profile,
    generate_scatter_map,
    generate_time_series,
)


class TestGenerateDepthProfile:
    def test_returns_valid_plotly(self):
        result = generate_depth_profile(
            depths=[0, 100, 500, 1000, 2000],
            values=[25.0, 18.0, 8.0, 4.0, 2.0],
            variable="TEMP",
            unit="degC",
        )
        assert result["chart_type"] == "depth_profile"
        assert "plotly_json" in result
        assert len(result["plotly_json"]["data"]) == 1
        assert result["plotly_json"]["data"][0]["type"] == "scatter"

    def test_depth_axis_inverted(self):
        result = generate_depth_profile(
            depths=[0, 100, 500],
            values=[25.0, 18.0, 8.0],
            variable="TEMP",
            unit="degC",
        )
        layout = result["plotly_json"]["layout"]
        assert layout["yaxis"]["autorange"] == "reversed"

    def test_custom_title(self):
        result = generate_depth_profile(
            depths=[0, 100],
            values=[25.0, 18.0],
            variable="TEMP",
            unit="degC",
            title="My Custom Title",
        )
        assert result["plotly_json"]["layout"]["title"] == "My Custom Title"
        assert result["description"] == "My Custom Title"


class TestGenerateTimeSeries:
    def test_returns_valid_plotly(self):
        result = generate_time_series(
            times=["2023-01", "2023-06", "2024-01"],
            values=[20.0, 25.0, 19.0],
            variable="TEMP",
            unit="degC",
        )
        assert result["chart_type"] == "time_series"
        assert len(result["plotly_json"]["data"]) == 1
        assert result["plotly_json"]["data"][0]["x"] == ["2023-01", "2023-06", "2024-01"]

    def test_has_axis_labels(self):
        result = generate_time_series(
            times=["2023-01"],
            values=[20.0],
            variable="PSAL",
            unit="PSU",
        )
        layout = result["plotly_json"]["layout"]
        assert "Date" in layout["xaxis"]["title"]
        assert "PSU" in layout["yaxis"]["title"]


class TestGenerateBarChart:
    def test_grouped_bars(self):
        result = generate_bar_chart(
            categories=["Pacific", "Atlantic"],
            values=[[20.0, 18.0], [22.0, 19.0]],
            labels=["Surface", "500m"],
            variable="TEMP",
            unit="degC",
        )
        assert result["chart_type"] == "bar_chart"
        assert len(result["plotly_json"]["data"]) == 2
        assert result["plotly_json"]["layout"]["barmode"] == "group"

    def test_single_series(self):
        result = generate_bar_chart(
            categories=["Mean", "Median", "Max"],
            values=[[15.0, 14.5, 28.0]],
            labels=["TEMP"],
            variable="TEMP",
            unit="degC",
        )
        assert len(result["plotly_json"]["data"]) == 1


class TestGenerateScatterMap:
    def test_returns_valid_geo(self):
        result = generate_scatter_map(
            lats=[30.0, 35.0, 40.0],
            lons=[-40.0, -35.0, -30.0],
            values=[20.0, 18.0, 15.0],
            variable="TEMP",
            unit="degC",
        )
        assert result["chart_type"] == "scatter_map"
        data = result["plotly_json"]["data"][0]
        assert data["type"] == "scattergeo"
        assert data["lat"] == [30.0, 35.0, 40.0]

    def test_has_colorbar(self):
        result = generate_scatter_map(
            lats=[30.0],
            lons=[-40.0],
            values=[20.0],
            variable="TEMP",
            unit="degC",
        )
        marker = result["plotly_json"]["data"][0]["marker"]
        assert "colorbar" in marker
        assert "degC" in marker["colorbar"]["title"]

    def test_has_ocean_styling(self):
        result = generate_scatter_map(
            lats=[30.0],
            lons=[-40.0],
            values=[20.0],
            variable="TEMP",
            unit="degC",
        )
        geo = result["plotly_json"]["layout"]["geo"]
        assert geo["showocean"] is True


class TestVizAgentInferChart:
    def test_infers_scatter_map_from_locations(self):
        from backend.agents.viz_agent import VizAgent

        agent = VizAgent.__new__(VizAgent)
        data = {
            "tool_results": {
                "query_ocean_data": {
                    "success": True,
                    "variable": "TEMP",
                    "sample_locations": [
                        {"lat": 30.0, "lon": -40.0},
                        {"lat": 35.0, "lon": -35.0},
                    ],
                    "values_sample": [20.0, 18.0],
                }
            }
        }
        chart = agent._infer_chart_from_data(data)
        assert chart is not None
        assert chart["chart_type"] == "scatter_map"

    def test_infers_bar_from_statistics(self):
        from backend.agents.viz_agent import VizAgent

        agent = VizAgent.__new__(VizAgent)
        data = {
            "tool_results": {
                "query_ocean_data": {
                    "success": True,
                    "variable": "PSAL",
                    "statistics": {
                        "mean": 35.0,
                        "median": 34.8,
                        "min": 33.0,
                        "max": 37.5,
                    },
                    "sample_locations": [],
                    "values_sample": [],
                }
            }
        }
        chart = agent._infer_chart_from_data(data)
        assert chart is not None
        assert chart["chart_type"] == "bar_chart"

    def test_returns_none_for_empty_data(self):
        from backend.agents.viz_agent import VizAgent

        agent = VizAgent.__new__(VizAgent)
        chart = agent._infer_chart_from_data({})
        assert chart is None

    def test_returns_none_for_failed_result(self):
        from backend.agents.viz_agent import VizAgent

        agent = VizAgent.__new__(VizAgent)
        data = {
            "tool_results": {
                "query_ocean_data": {"success": False, "error": "Failed"}
            }
        }
        chart = agent._infer_chart_from_data(data)
        assert chart is None


# --- Helpers for QueryAgent tool-loop tests ---

def _make_ai_response(content: str = "", tool_calls: list | None = None):
    """Create a mock AIMessage with optional tool_calls."""
    msg = AIMessage(content=content)
    if tool_calls is not None:
        msg.tool_calls = tool_calls
    else:
        msg.tool_calls = []
    return msg


def _make_query_agent() -> QueryAgent:
    """Create a QueryAgent with a mocked LLM (no real API call)."""
    agent = QueryAgent.__new__(QueryAgent)
    agent._settings = MagicMock()
    agent._settings.anthropic_model = "claude-sonnet-4-20250514"
    agent._settings.anthropic_api_key = "test-key"
    agent._llm = AsyncMock()
    return agent


def _make_state(text: str = "test query") -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "intent": "data",
        "data": {},
        "visualization": {},
        "metadata": {},
    }


class TestQueryAgentToolLoop:
    """Tests for the ReAct-style tool-call loop in QueryAgent.run()."""

    @pytest.mark.asyncio
    async def test_no_tools_returns_text_immediately(self):
        """When LLM returns text without tool calls, return it directly."""
        agent = _make_query_agent()
        text_response = _make_ai_response(content="The Atlantic has varied temperatures.")
        agent._llm.ainvoke = AsyncMock(return_value=text_response)

        result = await agent.run(_make_state("What is Atlantic temperature?"))

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "The Atlantic has varied temperatures."
        assert result["data"]["tool_results"] == {}

    @pytest.mark.asyncio
    async def test_executes_single_tool_call(self):
        """LLM makes one tool call, gets result, then responds with text."""
        agent = _make_query_agent()

        tool_response = _make_ai_response(
            content="",
            tool_calls=[{
                "name": "ocean_basin_bounds",
                "args": {"basin": "atlantic"},
                "id": "call_1",
            }],
        )
        final_response = _make_ai_response(
            content="The Atlantic spans lat -60 to 70, lon -80 to 0.",
        )
        agent._llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        bounds_tool = MagicMock()
        bounds_tool.name = "ocean_basin_bounds"
        bounds_tool.ainvoke = AsyncMock(return_value={
            "success": True, "basin": "atlantic",
            "lat_min": -60, "lat_max": 70, "lon_min": -80, "lon_max": 0,
        })

        with patch("backend.agents.query_agent.QUERY_TOOLS", [bounds_tool]):
            result = await agent.run(_make_state("Atlantic bounds"))

        assert len(result["messages"]) == 1
        assert "Atlantic" in result["messages"][0].content
        assert len(result["data"]["tool_results"]) == 1

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls(self):
        """LLM calls ocean_basin_bounds, then query_ocean_data in next iteration."""
        agent = _make_query_agent()

        # Iteration 0: LLM requests ocean_basin_bounds
        step1_response = _make_ai_response(
            content="",
            tool_calls=[{
                "name": "ocean_basin_bounds",
                "args": {"basin": "atlantic"},
                "id": "call_1",
            }],
        )
        # Iteration 1: LLM requests query_ocean_data using bounds
        step2_response = _make_ai_response(
            content="",
            tool_calls=[{
                "name": "query_ocean_data",
                "args": {"variable": "TEMP", "lat_min": -60, "lat_max": 70},
                "id": "call_2",
            }],
        )
        # Iteration 2: LLM returns final text
        final_response = _make_ai_response(
            content="Average temperature in the Atlantic at 500m is 8.2 degC.",
        )

        agent._llm.ainvoke = AsyncMock(
            side_effect=[step1_response, step2_response, final_response],
        )

        bounds_tool = MagicMock()
        bounds_tool.name = "ocean_basin_bounds"
        bounds_tool.ainvoke = AsyncMock(return_value={
            "success": True, "basin": "atlantic",
            "lat_min": -60, "lat_max": 70, "lon_min": -80, "lon_max": 0,
        })

        query_tool = MagicMock()
        query_tool.name = "query_ocean_data"
        query_tool.ainvoke = AsyncMock(return_value={
            "success": True, "variable": "TEMP", "n_profiles": 42,
            "statistics": {"mean": 8.2, "std": 1.1, "min": 3.0, "max": 15.0, "median": 8.0},
            "n_measurements": 420, "region": {}, "sample_locations": [], "values_sample": [],
        })

        with patch("backend.agents.query_agent.QUERY_TOOLS", [bounds_tool, query_tool]):
            result = await agent.run(_make_state("Temperature at 500m in Atlantic"))

        assert "8.2" in result["messages"][0].content
        assert len(result["data"]["tool_results"]) == 2
        assert bounds_tool.ainvoke.await_count == 1
        assert query_tool.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        """Loop terminates after max_iterations and calls summary LLM."""
        agent = _make_query_agent()

        # LLM always returns a tool call (never finishes)
        endless_tool_call = _make_ai_response(
            content="",
            tool_calls=[{
                "name": "ocean_basin_bounds",
                "args": {"basin": "atlantic"},
                "id": "call_loop",
            }],
        )
        agent._llm.ainvoke = AsyncMock(return_value=endless_tool_call)

        bounds_tool = MagicMock()
        bounds_tool.name = "ocean_basin_bounds"
        bounds_tool.ainvoke = AsyncMock(return_value={
            "success": True, "basin": "atlantic",
            "lat_min": -60, "lat_max": 70, "lon_min": -80, "lon_max": 0,
        })

        summary_response = _make_ai_response(
            content="Here is a summary of what I found.",
        )

        with patch("backend.agents.query_agent.QUERY_TOOLS", [bounds_tool]):
            with patch("backend.agents.query_agent.ChatAnthropic") as mock_chat:
                mock_summary_llm = AsyncMock()
                mock_summary_llm.ainvoke = AsyncMock(return_value=summary_response)
                mock_chat.return_value = mock_summary_llm

                result = await agent.run(_make_state("Endless query"))

        # Should have called the tool 5 times (max_iterations)
        assert bounds_tool.ainvoke.await_count == 5
        assert result["messages"][0].content == "Here is a summary of what I found."

    @pytest.mark.asyncio
    async def test_tool_error_continues_loop(self):
        """If a tool raises an exception, error is fed back and loop continues."""
        agent = _make_query_agent()

        error_tool_call = _make_ai_response(
            content="",
            tool_calls=[{
                "name": "query_ocean_data",
                "args": {"variable": "TEMP"},
                "id": "call_err",
            }],
        )
        final_response = _make_ai_response(
            content="Sorry, the data query failed. Please try again later.",
        )
        agent._llm.ainvoke = AsyncMock(
            side_effect=[error_tool_call, final_response],
        )

        failing_tool = MagicMock()
        failing_tool.name = "query_ocean_data"
        failing_tool.ainvoke = AsyncMock(side_effect=RuntimeError("Connection timeout"))

        with patch("backend.agents.query_agent.QUERY_TOOLS", [failing_tool]):
            result = await agent.run(_make_state("Temperature query"))

        assert "failed" in result["messages"][0].content.lower()
        error_result = result["data"]["tool_results"]["query_ocean_data_0"]
        assert error_result["success"] is False
        assert "Connection timeout" in error_result["error"]


class TestClassifyIntentFallback:
    """Tests for the improved fallback heuristics in classify_intent."""

    def test_data_keywords_take_priority_over_show(self):
        """'show me temperature' should route to data, not viz."""
        mock_llm = MagicMock()
        # Return invalid intent to trigger fallback
        mock_llm.invoke.return_value = MagicMock(content="unknown")

        result = classify_intent("show me temperature data", mock_llm)
        assert result == "data"

    def test_show_alone_defaults_to_data(self):
        """'show me something' without viz keywords defaults to data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="unknown")

        result = classify_intent("show me the results", mock_llm)
        assert result == "data"

    def test_plot_keyword_routes_to_viz(self):
        """Explicit 'plot' without data keywords routes to viz."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="unknown")

        result = classify_intent("plot a nice chart", mock_llm)
        assert result == "viz"

    def test_salinity_routes_to_data(self):
        """Data-specific keywords like 'salinity' route to data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="unknown")

        result = classify_intent("What is the salinity near Hawaii?", mock_llm)
        assert result == "data"

    def test_compare_routes_to_data(self):
        """'compare' keyword routes to data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="unknown")

        result = classify_intent("compare regions", mock_llm)
        assert result == "data"

    def test_valid_intent_passes_through(self):
        """When LLM returns a valid intent, use it directly."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="data")

        result = classify_intent("temperature at 500m", mock_llm)
        assert result == "data"
