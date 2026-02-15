"""Visualization agent for generating Plotly JSON from query results."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from backend.agents.state import AgentState
from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

VIZ_SYSTEM_PROMPT = """You are a data visualization expert for oceanographic data.
Given query results, determine the best chart type and generate a Plotly JSON specification.

Chart type selection rules:
- Variable vs depth (single profile): line chart (x=variable, y=depth inverted)
- Variable over time: time series line chart
- Variable across lat/lon: scatter map or heatmap
- Comparison of 2+ groups: grouped bar chart
- Distribution of values: histogram
- Spatial coverage: scatter geo map

Return ONLY a valid JSON object with this structure:
{
  "chart_type": "time_series|depth_profile|heatmap|scatter_map|bar_chart|histogram",
  "plotly_json": {
    "data": [...],
    "layout": {...}
  },
  "description": "Brief description of what the chart shows"
}

Use an ocean-themed color palette: blues (#0077b6, #00b4d8, #90e0ef), teals (#2a9d8f).
Always include axis labels with units. Make charts responsive.
"""


def create_viz_agent(settings: Settings | None = None) -> "VizAgent":
    """Create a visualization agent instance."""
    return VizAgent(settings=settings)


def generate_depth_profile(
    depths: list[float],
    values: list[float],
    variable: str,
    unit: str,
    title: str = "",
) -> dict[str, Any]:
    """Generate a Plotly JSON depth profile chart."""
    return {
        "chart_type": "depth_profile",
        "plotly_json": {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": values,
                    "y": depths,
                    "marker": {"color": "#0077b6", "size": 4},
                    "line": {"color": "#0077b6", "width": 2},
                    "name": variable,
                }
            ],
            "layout": {
                "title": title or f"{variable} Depth Profile",
                "xaxis": {"title": f"{variable} ({unit})"},
                "yaxis": {"title": "Depth (m)", "autorange": "reversed"},
                "template": "plotly_white",
                "height": 500,
            },
        },
        "description": title or f"Depth profile of {variable}",
    }


def generate_time_series(
    times: list[str],
    values: list[float],
    variable: str,
    unit: str,
    title: str = "",
) -> dict[str, Any]:
    """Generate a Plotly JSON time series chart."""
    return {
        "chart_type": "time_series",
        "plotly_json": {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": times,
                    "y": values,
                    "marker": {"color": "#00b4d8", "size": 4},
                    "line": {"color": "#00b4d8", "width": 2},
                    "name": variable,
                }
            ],
            "layout": {
                "title": title or f"{variable} Over Time",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": f"{variable} ({unit})"},
                "template": "plotly_white",
                "height": 400,
            },
        },
        "description": title or f"Time series of {variable}",
    }


def generate_bar_chart(
    categories: list[str],
    values: list[list[float]],
    labels: list[str],
    variable: str,
    unit: str,
    title: str = "",
) -> dict[str, Any]:
    """Generate a Plotly JSON grouped bar chart for comparisons."""
    colors = ["#0077b6", "#00b4d8", "#90e0ef", "#2a9d8f"]
    data = [
        {
            "type": "bar",
            "x": categories,
            "y": vals,
            "name": label,
            "marker": {"color": colors[i % len(colors)]},
        }
        for i, (vals, label) in enumerate(zip(values, labels))
    ]

    return {
        "chart_type": "bar_chart",
        "plotly_json": {
            "data": data,
            "layout": {
                "title": title or f"{variable} Comparison",
                "xaxis": {"title": "Category"},
                "yaxis": {"title": f"{variable} ({unit})"},
                "barmode": "group",
                "template": "plotly_white",
                "height": 400,
            },
        },
        "description": title or f"Comparison of {variable} across categories",
    }


def generate_scatter_map(
    lats: list[float],
    lons: list[float],
    values: list[float],
    variable: str,
    unit: str,
    title: str = "",
) -> dict[str, Any]:
    """Generate a Plotly JSON scatter map for geographic data."""
    return {
        "chart_type": "scatter_map",
        "plotly_json": {
            "data": [
                {
                    "type": "scattergeo",
                    "lat": lats,
                    "lon": lons,
                    "marker": {
                        "color": values,
                        "colorscale": "Viridis",
                        "colorbar": {"title": f"{variable} ({unit})"},
                        "size": 6,
                    },
                    "text": [f"{v:.2f} {unit}" for v in values],
                    "name": variable,
                }
            ],
            "layout": {
                "title": title or f"{variable} Spatial Distribution",
                "geo": {
                    "showland": True,
                    "landcolor": "#e8e8e8",
                    "showocean": True,
                    "oceancolor": "#cce5ff",
                    "projection": {"type": "natural earth"},
                },
                "template": "plotly_white",
                "height": 500,
            },
        },
        "description": title or f"Geographic distribution of {variable}",
    }


class VizAgent:
    """Agent that generates Plotly JSON visualizations from query results."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatAnthropic(
            model=self._settings.anthropic_model,
            api_key=self._settings.anthropic_api_key,
            max_tokens=2048,
        )

    def _infer_chart_from_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Try to generate a chart directly from structured tool results."""
        tool_results = data.get("tool_results", {})

        for _tool_name, result in tool_results.items():
            if not isinstance(result, dict) or not result.get("success"):
                continue

            variable = result.get("variable", "Value")
            stats = result.get("statistics", {})
            sample_locs = result.get("sample_locations", [])
            values_sample = result.get("values_sample", [])

            # If we have locations and values, make a scatter map
            if sample_locs and values_sample:
                lats = [loc["lat"] for loc in sample_locs]
                lons = [loc["lon"] for loc in sample_locs]
                n = min(len(lats), len(values_sample))
                return generate_scatter_map(
                    lats=lats[:n],
                    lons=lons[:n],
                    values=values_sample[:n],
                    variable=variable,
                    unit=self._get_unit(variable),
                )

            # If we have statistics, make a bar chart summary
            if stats and stats.get("mean") is not None:
                return generate_bar_chart(
                    categories=["Mean", "Median", "Min", "Max"],
                    values=[[
                        stats.get("mean", 0),
                        stats.get("median", 0),
                        stats.get("min", 0),
                        stats.get("max", 0),
                    ]],
                    labels=[variable],
                    variable=variable,
                    unit=self._get_unit(variable),
                    title=f"{variable} Statistics Summary",
                )

        return None

    @staticmethod
    def _get_unit(variable: str) -> str:
        units = {
            "TEMP": "degC",
            "PSAL": "PSU",
            "PRES": "dbar",
            "DOXY": "umol/kg",
        }
        return units.get(variable, "")

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Generate visualization from query results in state."""
        data = state.get("data", {})

        # Try direct chart generation from structured data
        chart = self._infer_chart_from_data(data)
        if chart:
            return {"visualization": chart}

        # Fallback: ask LLM to generate Plotly JSON
        system_msg = SystemMessage(content=VIZ_SYSTEM_PROMPT)
        from langchain_core.messages import HumanMessage

        data_prompt = f"Generate a visualization for this data:\n{data}"
        response = await self._llm.ainvoke([
            system_msg,
            HumanMessage(content=data_prompt),
        ])

        # Try to parse the response as JSON
        import json
        try:
            viz = json.loads(response.content)
            return {"visualization": viz}
        except (json.JSONDecodeError, TypeError):
            return {
                "visualization": {
                    "chart_type": "none",
                    "plotly_json": None,
                    "description": "Could not generate visualization for this data.",
                },
            }
