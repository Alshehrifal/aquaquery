"""LangChain tools for querying Argo floats by WMO ID, cycle, and trajectory."""

import logging
from typing import Any

import numpy as np
from langchain_core.tools import tool

from backend.data.argo_manager import ArgoDataManager

logger = logging.getLogger(__name__)

# Limits to prevent tool responses from exceeding LLM context window.
# Full data is still fetched and cached; only the LLM-facing response is trimmed.
_MAX_TRAJECTORY_POINTS = 50
_MAX_DEPTH_LEVELS = 200

_manager: ArgoDataManager | None = None


def _get_manager() -> ArgoDataManager:
    global _manager
    if _manager is None:
        _manager = ArgoDataManager()
    return _manager


def set_manager(manager: ArgoDataManager) -> None:
    """Set the data manager instance (for testing)."""
    global _manager
    _manager = manager


def _truncate_trajectory(
    trajectory: dict | None,
    max_points: int = _MAX_TRAJECTORY_POINTS,
) -> dict | None:
    """Downsample a trajectory to at most *max_points* evenly-spaced entries.

    Preserves first and last points so the start/end markers remain accurate.
    Returns a new dict (never mutates the input).
    """
    if trajectory is None:
        return None

    lats = trajectory.get("latitudes", [])
    lons = trajectory.get("longitudes", [])
    times = trajectory.get("timestamps", [])

    n = len(lats)
    if n <= max_points:
        return {**trajectory}

    # Evenly-spaced indices, always including first and last
    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return {
        "latitudes": [lats[i] for i in indices],
        "longitudes": [lons[i] for i in indices],
        "timestamps": [times[i] for i in indices],
        "total_points": n,
        "truncated": True,
    }


@tool
def query_by_float_id(
    wmo_id: int,
    variable: str = "TEMP",
) -> dict[str, Any]:
    """Query all profiles from an Argo float by its WMO ID.

    Returns trajectory, statistics, and profile count for the float.

    Args:
        wmo_id: World Meteorological Organization float identifier (e.g., 6902746)
        variable: Variable to compute statistics for (TEMP, PSAL, PRES, DOXY)
    """
    logger.info("query_by_float_id: wmo_id=%d, variable=%s", wmo_id, variable)

    valid_vars = {"TEMP", "PSAL", "PRES", "DOXY"}
    if variable.upper() not in valid_vars:
        return {
            "error": f"Invalid variable '{variable}'. Must be one of: {valid_vars}",
            "success": False,
        }
    variable = variable.upper()

    try:
        manager = _get_manager()
        ds = manager.get_data_by_float(wmo_id)

        if ds is None:
            return {
                "error": f"No data found for float {wmo_id}",
                "success": False,
            }

        n_profiles = ds.sizes.get("N_PROF", 0)
        trajectory = _truncate_trajectory(manager.extract_trajectory(ds))
        stats = manager.get_statistics(ds, variable)

        return {
            "success": True,
            "wmo_id": wmo_id,
            "n_profiles": n_profiles,
            "variable": variable,
            "statistics": stats,
            "trajectory": trajectory,
        }

    except TimeoutError as e:
        logger.warning("Float query timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error querying float %d: %s", wmo_id, e)
        return {"error": f"Failed to fetch float data: {e}", "success": False}


@tool
def get_float_trajectory(
    wmo_id: int,
) -> dict[str, Any]:
    """Get the trajectory (path) of an Argo float for map plotting.

    Returns ordered lat/lon/time arrays sorted chronologically.
    Use this when the user asks to plot or show a float's path on a map.

    Args:
        wmo_id: World Meteorological Organization float identifier (e.g., 6902746)
    """
    logger.info("get_float_trajectory: wmo_id=%d", wmo_id)
    try:
        manager = _get_manager()
        ds = manager.get_data_by_float(wmo_id)

        if ds is None:
            return {
                "error": f"No data found for float {wmo_id}",
                "success": False,
            }

        trajectory = _truncate_trajectory(manager.extract_trajectory(ds))
        n_profiles = ds.sizes.get("N_PROF", 0)

        return {
            "success": True,
            "wmo_id": wmo_id,
            "n_profiles": n_profiles,
            "trajectory": trajectory,
            "chart_hint": "trajectory_map",
        }

    except TimeoutError as e:
        logger.warning("Trajectory fetch timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error fetching trajectory for float %d: %s", wmo_id, e)
        return {"error": f"Failed to fetch trajectory: {e}", "success": False}


@tool
def get_floats_in_region(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """List unique Argo float WMO IDs found in a geographic region.

    Args:
        lat_min: Minimum latitude (-90 to 90)
        lat_max: Maximum latitude (-90 to 90)
        lon_min: Minimum longitude (-180 to 180)
        lon_max: Maximum longitude (-180 to 180)
        start_date: Start date as ISO string (e.g., '2023-01')
        end_date: End date as ISO string (e.g., '2024-01')
    """
    logger.info("get_floats_in_region: lat=[%s,%s], lon=[%s,%s]",
                lat_min, lat_max, lon_min, lon_max)
    try:
        manager = _get_manager()
        ds = manager.get_data(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            start_date=start_date,
            end_date=end_date,
        )

        if ds is None:
            return {
                "error": "No data found in the specified region",
                "success": False,
            }

        if "PLATFORM_NUMBER" not in ds:
            return {
                "error": "Dataset does not contain float identifiers",
                "success": False,
            }

        platform_ids = ds["PLATFORM_NUMBER"].values
        parsed_ids = set()
        for pid in platform_ids:
            try:
                pid_str = str(pid).strip()
                if pid_str:
                    parsed_ids.add(int(pid_str))
            except (ValueError, TypeError):
                continue

        unique_ids = sorted(parsed_ids)
        max_results = 100
        truncated = len(unique_ids) > max_results

        result = {
            "success": True,
            "n_floats": len(unique_ids),
            "wmo_ids": unique_ids[:max_results],
            "region": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        }

        if truncated:
            result["warning"] = (
                f"Region contains {len(unique_ids)} floats. "
                f"Showing first {max_results}."
            )

        return result

    except TimeoutError as e:
        logger.warning("Region float query timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error finding floats in region: %s", e)
        return {"error": f"Failed to find floats: {e}", "success": False}


@tool
def query_by_profile(
    wmo_id: int,
    cycle_number: int,
    variable: str = "TEMP",
) -> dict[str, Any]:
    """Query a single depth profile from an Argo float by WMO ID and cycle number.

    Returns depth and variable arrays for plotting a depth profile chart.

    Args:
        wmo_id: World Meteorological Organization float identifier (e.g., 6902746)
        cycle_number: Cycle number of the profile (e.g., 10)
        variable: Variable to extract (TEMP, PSAL, PRES, DOXY)
    """
    logger.info("query_by_profile: wmo_id=%d, cycle=%d, var=%s",
                wmo_id, cycle_number, variable)

    valid_vars = {"TEMP", "PSAL", "PRES", "DOXY"}
    variable = variable.upper()
    if variable not in valid_vars:
        return {
            "error": f"Invalid variable '{variable}'. Must be one of: {valid_vars}",
            "success": False,
        }

    try:
        manager = _get_manager()
        ds = manager.get_data_by_profile(wmo_id, cycle_number)

        if ds is None:
            return {
                "error": f"No data found for float {wmo_id} cycle {cycle_number}",
                "success": False,
            }

        depths = []
        values = []

        if "PRES" in ds and variable in ds:
            pres_vals = ds["PRES"].values.flatten()
            var_vals = ds[variable].values.flatten()
            # Filter NaN pairs
            for d, v in zip(pres_vals, var_vals):
                if not np.isnan(d) and not np.isnan(v):
                    depths.append(float(d))
                    values.append(float(v))

        total_levels = len(depths)
        truncated = total_levels > _MAX_DEPTH_LEVELS
        if truncated:
            depths = depths[:_MAX_DEPTH_LEVELS]
            values = values[:_MAX_DEPTH_LEVELS]

        result: dict[str, Any] = {
            "success": True,
            "wmo_id": wmo_id,
            "cycle_number": cycle_number,
            "variable": variable,
            "depths": depths,
            "values": values,
            "n_levels": total_levels,
            "chart_hint": "depth_profile",
        }
        if truncated:
            result["truncated"] = True
        return result

    except TimeoutError as e:
        logger.warning("Profile query timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error querying profile WMO %d cycle %d: %s",
                     wmo_id, cycle_number, e)
        return {"error": f"Failed to fetch profile: {e}", "success": False}


@tool
def compare_floats(
    wmo_ids: list[int],
    variable: str = "TEMP",
) -> dict[str, Any]:
    """Compare statistics across 2-5 Argo floats for a given variable.

    Args:
        wmo_ids: List of 2-5 WMO float identifiers to compare
        variable: Variable to compare (TEMP, PSAL, PRES, DOXY)
    """
    logger.info("compare_floats: wmo_ids=%s, variable=%s", wmo_ids, variable)

    if len(wmo_ids) < 2:
        return {
            "error": "Must provide at least 2 float IDs to compare",
            "success": False,
        }

    if len(wmo_ids) > 5:
        return {
            "error": "Cannot compare more than 5 floats at once",
            "success": False,
        }

    try:
        manager = _get_manager()
        comparisons = []

        for wmo_id in wmo_ids:
            ds = manager.get_data_by_float(wmo_id)
            if ds is None:
                comparisons.append({
                    "wmo_id": wmo_id,
                    "error": f"No data found for float {wmo_id}",
                })
                continue

            stats = manager.get_statistics(ds, variable.upper())
            n_profiles = ds.sizes.get("N_PROF", 0)
            comparisons.append({
                "wmo_id": wmo_id,
                "n_profiles": n_profiles,
                "statistics": stats,
            })

        return {
            "success": True,
            "variable": variable.upper(),
            "comparisons": comparisons,
            "chart_hint": "bar_chart",
        }

    except TimeoutError as e:
        logger.warning("Float comparison timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error comparing floats: %s", e)
        return {"error": f"Failed to compare floats: {e}", "success": False}
