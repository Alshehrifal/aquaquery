"""LangChain tools for querying Argo oceanographic data."""

import logging
from typing import Any

import numpy as np
from langchain_core.tools import tool

from backend.data.argo_manager import ArgoDataManager
from backend.data.loader import ArgoDataLoader, OCEAN_BASINS

logger = logging.getLogger(__name__)

# Module-level loader instance for get_data_coverage (static metadata only)
_loader: ArgoDataLoader | None = None

# Module-level manager instance for query_ocean_data (ERDDAP-first + caching)
_manager: ArgoDataManager | None = None


def _get_loader() -> ArgoDataLoader:
    global _loader
    if _loader is None:
        _loader = ArgoDataLoader()
    return _loader


def set_loader(loader: ArgoDataLoader) -> None:
    """Set the data loader instance (for testing)."""
    global _loader
    _loader = loader


def _get_manager() -> ArgoDataManager:
    global _manager
    if _manager is None:
        _manager = ArgoDataManager()
    return _manager


def set_manager(manager: ArgoDataManager) -> None:
    """Set the data manager instance (for testing)."""
    global _manager
    _manager = manager


@tool
def query_ocean_data(
    variable: str,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    depth_min: float = 0.0,
    depth_max: float = 2000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Query Argo oceanographic data by region, depth, and time.

    Args:
        variable: Variable name (TEMP, PSAL, PRES, DOXY)
        lat_min: Minimum latitude (-90 to 90)
        lat_max: Maximum latitude (-90 to 90)
        lon_min: Minimum longitude (-180 to 180)
        lon_max: Maximum longitude (-180 to 180)
        depth_min: Minimum depth in meters (default 0)
        depth_max: Maximum depth in meters (default 2000)
        start_date: Start date as ISO string (e.g., '2023-01')
        end_date: End date as ISO string (e.g., '2024-01')

    Returns:
        Dict with query results including profiles, statistics, and metadata.
    """
    logger.info(
        "query_ocean_data called: variable=%s, lat=[%s,%s], lon=[%s,%s], depth=[%s,%s]",
        variable, lat_min, lat_max, lon_min, lon_max, depth_min, depth_max,
    )
    valid_vars = {"TEMP", "PSAL", "PRES", "DOXY"}
    if variable.upper() not in valid_vars:
        return {
            "error": f"Invalid variable '{variable}'. Must be one of: {valid_vars}",
            "success": False,
        }

    variable = variable.upper()

    try:
        manager = _get_manager()
        ds = manager.get_data(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            depth_min=depth_min,
            depth_max=depth_max,
            start_date=start_date,
            end_date=end_date,
        )

        if ds is None:
            return {
                "error": "Failed to fetch data from all sources",
                "success": False,
            }

        if variable not in ds:
            return {
                "error": f"Variable '{variable}' not found in fetched data",
                "success": False,
            }

        values = ds[variable].values.flatten()
        valid_values = values[~np.isnan(values)]

        n_profiles = ds.sizes.get("N_PROF", 0)
        lats = ds["LATITUDE"].values.tolist() if "LATITUDE" in ds else []
        lons = ds["LONGITUDE"].values.tolist() if "LONGITUDE" in ds else []

        result = {
            "success": True,
            "variable": variable,
            "n_profiles": n_profiles,
            "n_measurements": len(valid_values),
            "statistics": {
                "mean": float(np.mean(valid_values)) if len(valid_values) > 0 else None,
                "std": float(np.std(valid_values)) if len(valid_values) > 0 else None,
                "min": float(np.min(valid_values)) if len(valid_values) > 0 else None,
                "max": float(np.max(valid_values)) if len(valid_values) > 0 else None,
                "median": float(np.median(valid_values)) if len(valid_values) > 0 else None,
            },
            "region": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "depth_min": depth_min,
                "depth_max": depth_max,
            },
            "sample_locations": [
                {"lat": lat, "lon": lon}
                for lat, lon in zip(lats[:20], lons[:20])
            ],
            "values_sample": valid_values[:100].tolist() if len(valid_values) > 0 else [],
        }

        return result

    except TimeoutError as e:
        logger.warning("Tool timed out: %s", e)
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.error("Error querying ocean data: %s", e)
        return {
            "error": f"Failed to fetch data: {str(e)}",
            "success": False,
        }


@tool
def get_data_coverage() -> dict[str, Any]:
    """Return available Argo data coverage information.

    Returns lat/lon/time bounds, available variables, and total profile count.
    """
    logger.info("get_data_coverage called")
    loader = _get_loader()
    meta = loader.get_metadata()
    variables = loader.get_available_variables()

    return {
        "success": True,
        "lat_bounds": list(meta.lat_bounds),
        "lon_bounds": list(meta.lon_bounds),
        "depth_range": list(meta.depth_range),
        "time_range": list(meta.time_range),
        "total_profiles": meta.total_profiles,
        "data_source": meta.data_source,
        "variables": [
            {
                "name": v.name,
                "display_name": v.display_name,
                "unit": v.unit,
                "description": v.description,
                "typical_range": list(v.typical_range),
            }
            for v in variables
        ],
    }
