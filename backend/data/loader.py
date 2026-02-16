"""Argo data loader using argopy library."""

import concurrent.futures
import logging
from datetime import datetime
from typing import Any

import argopy
import numpy as np
import xarray as xr

from backend.config import Settings, get_settings
from backend.data.schema import DataSource, DatasetMetadata, OceanProfile, VariableInfo

logger = logging.getLogger(__name__)

# Variable definitions
ARGO_VARIABLES: list[VariableInfo] = [
    VariableInfo(
        name="TEMP",
        display_name="Temperature",
        unit="degC",
        description="In-situ sea water temperature",
        typical_range=(-2.0, 35.0),
    ),
    VariableInfo(
        name="PSAL",
        display_name="Salinity",
        unit="PSU",
        description="Practical salinity",
        typical_range=(30.0, 40.0),
    ),
    VariableInfo(
        name="PRES",
        display_name="Pressure",
        unit="dbar",
        description="Sea water pressure (approximately depth in meters)",
        typical_range=(0.0, 2000.0),
    ),
    VariableInfo(
        name="DOXY",
        display_name="Dissolved Oxygen",
        unit="umol/kg",
        description="Dissolved oxygen concentration",
        typical_range=(0.0, 400.0),
    ),
]

# Named ocean basin boundaries
OCEAN_BASINS: dict[str, dict[str, float]] = {
    "atlantic": {"lat_min": -60, "lat_max": 60, "lon_min": -80, "lon_max": 0},
    "pacific": {"lat_min": -60, "lat_max": 60, "lon_min": 100, "lon_max": -100},
    "indian": {"lat_min": -60, "lat_max": 30, "lon_min": 20, "lon_max": 120},
    "southern": {"lat_min": -90, "lat_max": -60, "lon_min": -180, "lon_max": 180},
    "arctic": {"lat_min": 60, "lat_max": 90, "lon_min": -180, "lon_max": 180},
    "mediterranean": {"lat_min": 30, "lat_max": 46, "lon_min": -6, "lon_max": 36},
    "north_atlantic": {"lat_min": 20, "lat_max": 60, "lon_min": -80, "lon_max": 0},
    "south_atlantic": {"lat_min": -60, "lat_max": 0, "lon_min": -70, "lon_max": 20},
    "north_pacific": {"lat_min": 20, "lat_max": 60, "lon_min": 100, "lon_max": -100},
    "south_pacific": {"lat_min": -60, "lat_max": 0, "lon_min": 150, "lon_max": -70},
}


def _apply_qc_filter(ds: xr.Dataset) -> xr.Dataset:
    """Filter dataset to keep only good quality data (QC flags 1 and 2)."""
    for var in ["TEMP", "PSAL", "PRES", "DOXY"]:
        qc_var = f"{var}_QC"
        if var in ds and qc_var in ds:
            mask = ds[qc_var].isin([1, 2])
            ds[var] = ds[var].where(mask)
    return ds


def _dataset_to_profiles(ds: xr.Dataset) -> list[OceanProfile]:
    """Convert xarray Dataset to list of OceanProfile objects."""
    profiles = []
    n_prof = ds.sizes.get("N_PROF", 0)

    for i in range(n_prof):
        try:
            lat = float(ds["LATITUDE"].values[i])
            lon = float(ds["LONGITUDE"].values[i])
            time_val = ds["JULD"].values[i] if "JULD" in ds else ds["TIME"].values[i]
            timestamp = datetime.fromisoformat(str(time_val)[:19])

            depth_levels = tuple(
                float(v) for v in ds["PRES"].values[i] if not np.isnan(v)
            )

            variables: dict[str, tuple[float, ...]] = {}
            for var in ["TEMP", "PSAL", "DOXY"]:
                if var in ds:
                    vals = ds[var].values[i]
                    variables[var] = tuple(
                        float(v) for v in vals if not np.isnan(v)
                    )

            profiles.append(
                OceanProfile(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    depth_levels=depth_levels,
                    variables=variables,
                )
            )
        except (ValueError, IndexError, KeyError) as e:
            logger.warning("Skipping profile %d: %s", i, e)
            continue

    return profiles


def _fetch_xarray_with_timeout(fetcher: Any, timeout_seconds: int) -> xr.Dataset:
    """Wrap synchronous fetcher.to_xarray() with a timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fetcher.to_xarray)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Data fetch timed out after {timeout_seconds}s. "
                "Try a smaller region or time range."
            )


class ArgoDataLoader(DataSource):
    """Argo oceanographic data loader using argopy."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._configure_argopy()

    def _configure_argopy(self) -> None:
        """Configure argopy caching and options."""
        argopy.set_options(
            cachedir=self._settings.argo_cache_dir,
            mode="standard",
        )

    def fetch_region(
        self,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        depth_range: tuple[float, float] = (0.0, 2000.0),
        time_range: tuple[str, str] | None = None,
    ) -> xr.Dataset:
        """Fetch Argo data for a geographic region.

        Args:
            lat_bounds: (lat_min, lat_max) in degrees
            lon_bounds: (lon_min, lon_max) in degrees
            depth_range: (depth_min, depth_max) in meters/dbar
            time_range: (start_date, end_date) as ISO strings, or None for all time

        Returns:
            xarray Dataset with Argo profile data, QC-filtered
        """
        region = [
            lon_bounds[0], lon_bounds[1],
            lat_bounds[0], lat_bounds[1],
            depth_range[0], depth_range[1],
        ]

        if time_range is not None:
            region.extend([time_range[0], time_range[1]])

        logger.info("Fetching Argo data for region: %s", region)

        timeout = self._settings.argo_fetch_timeout

        try:
            fetcher = argopy.DataFetcher(src="gdac").region(region)
            ds = _fetch_xarray_with_timeout(fetcher, timeout)
        except TimeoutError:
            raise
        except Exception:
            logger.warning("GDAC fetch failed, trying erddap source")
            fetcher = argopy.DataFetcher(src="erddap").region(region)
            ds = _fetch_xarray_with_timeout(fetcher, timeout)

        ds = _apply_qc_filter(ds)
        logger.info("Fetched %d profiles", ds.dims.get("N_PROF", 0))
        return ds

    def fetch_profiles_as_list(
        self,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        depth_range: tuple[float, float] = (0.0, 2000.0),
        time_range: tuple[str, str] | None = None,
    ) -> list[OceanProfile]:
        """Fetch and convert to list of OceanProfile objects."""
        ds = self.fetch_region(lat_bounds, lon_bounds, depth_range, time_range)
        return _dataset_to_profiles(ds)

    def get_metadata(self) -> DatasetMetadata:
        """Return metadata about Argo dataset coverage."""
        return DatasetMetadata(
            lat_bounds=(-90.0, 90.0),
            lon_bounds=(-180.0, 180.0),
            depth_range=(0.0, 2000.0),
            time_range=("1999-01-01", "2024-12-31"),
            available_variables=("TEMP", "PSAL", "PRES", "DOXY"),
            total_profiles=2_500_000,
            data_source="Argo GDAC",
            last_updated=datetime.now().strftime("%Y-%m-%d"),
        )

    def get_available_variables(self) -> list[VariableInfo]:
        """Return list of available Argo variables."""
        return list(ARGO_VARIABLES)
