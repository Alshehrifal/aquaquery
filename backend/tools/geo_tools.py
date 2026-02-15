"""Geographic tools for ocean data queries."""

import logging
from typing import Any

import numpy as np
from langchain_core.tools import tool

from backend.data.loader import OCEAN_BASINS, ArgoDataLoader

logger = logging.getLogger(__name__)

# Module-level loader (shares with argo_tools)
_loader: ArgoDataLoader | None = None


def _get_loader() -> ArgoDataLoader:
    global _loader
    if _loader is None:
        _loader = ArgoDataLoader()
    return _loader


def set_loader(loader: ArgoDataLoader) -> None:
    """Set the data loader instance (for testing)."""
    global _loader
    _loader = loader


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    r = 6371.0  # Earth radius in km
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return float(r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


@tool
def get_nearest_profiles(
    lat: float,
    lon: float,
    radius_km: float = 100.0,
) -> dict[str, Any]:
    """Find Argo profiles nearest to a geographic coordinate.

    Args:
        lat: Target latitude (-90 to 90)
        lon: Target longitude (-180 to 180)
        radius_km: Search radius in kilometers (default 100)

    Returns:
        Dict with nearby profiles and their distances.
    """
    # Define a bounding box from the radius (approximate)
    lat_delta = radius_km / 111.0  # ~111 km per degree latitude
    lon_delta = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.01))

    lat_min = max(-90.0, lat - lat_delta)
    lat_max = min(90.0, lat + lat_delta)
    lon_min = lon - lon_delta
    lon_max = lon + lon_delta

    try:
        loader = _get_loader()
        ds = loader.fetch_region(
            lat_bounds=(lat_min, lat_max),
            lon_bounds=(lon_min, lon_max),
        )

        if "LATITUDE" not in ds or "LONGITUDE" not in ds:
            return {"success": True, "n_profiles": 0, "profiles": []}

        lats = ds["LATITUDE"].values
        lons = ds["LONGITUDE"].values

        # Calculate distances and filter by radius
        profiles = []
        for i in range(len(lats)):
            dist = _haversine_km(lat, lon, float(lats[i]), float(lons[i]))
            if dist <= radius_km:
                profile = {
                    "index": i,
                    "latitude": float(lats[i]),
                    "longitude": float(lons[i]),
                    "distance_km": round(dist, 2),
                }
                profiles.append(profile)

        # Sort by distance
        profiles.sort(key=lambda p: p["distance_km"])

        return {
            "success": True,
            "target": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "n_profiles": len(profiles),
            "profiles": profiles[:20],  # Limit to 20 nearest
        }

    except Exception as e:
        logger.error("Error finding nearest profiles: %s", e)
        return {
            "error": f"Failed to find profiles: {str(e)}",
            "success": False,
        }


@tool
def ocean_basin_bounds(basin: str) -> dict[str, Any]:
    """Return latitude/longitude bounds for a named ocean basin.

    Args:
        basin: Ocean basin name (e.g., 'atlantic', 'pacific', 'indian',
               'southern', 'arctic', 'mediterranean', 'north_atlantic',
               'south_atlantic', 'north_pacific', 'south_pacific')

    Returns:
        Dict with lat_min, lat_max, lon_min, lon_max for the basin.
    """
    basin_key = basin.lower().strip().replace(" ", "_")

    if basin_key not in OCEAN_BASINS:
        available = ", ".join(sorted(OCEAN_BASINS.keys()))
        return {
            "error": f"Unknown basin '{basin}'. Available: {available}",
            "success": False,
        }

    bounds = OCEAN_BASINS[basin_key]
    return {
        "success": True,
        "basin": basin_key,
        "lat_min": bounds["lat_min"],
        "lat_max": bounds["lat_max"],
        "lon_min": bounds["lon_min"],
        "lon_max": bounds["lon_max"],
    }
