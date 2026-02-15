"""Tests for agents: supervisor routing, RAG retrieval, viz generation."""

import pytest

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
