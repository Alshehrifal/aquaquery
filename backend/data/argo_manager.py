"""Argo data manager with ERDDAP-first strategy and file-based NetCDF caching."""

import hashlib
import json
import logging
from pathlib import Path

import argopy
import numpy as np
import pandas as pd
import xarray as xr

from backend.data.loader import _apply_qc_filter, _fetch_xarray_with_timeout

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 45


class ArgoDataManager:
    """ERDDAP-first Argo data manager with file-based NetCDF caching.

    Fetches data from ERDDAP (fast, reliable) with GDAC as fallback.
    Caches results as .nc files keyed by MD5 hash of query parameters.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        if cache_dir is None:
            from backend.config import get_settings
            settings = get_settings()
            cache_dir = settings.argo_manager_cache_dir

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    def _build_cache_key(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        depth_min: float,
        depth_max: float,
        start_date: str | None,
        end_date: str | None,
    ) -> str:
        """Build a deterministic MD5 cache key from query parameters."""
        params = json.dumps(
            {
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "start_date": start_date,
                "end_date": end_date,
            },
            sort_keys=True,
        )
        return hashlib.md5(params.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.nc"

    def get_data(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        depth_min: float = 0.0,
        depth_max: float = 2000.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> xr.Dataset | None:
        """Fetch Argo data with caching. Returns xr.Dataset or None on failure.

        Strategy: cache -> ERDDAP (primary) -> GDAC (fallback).
        """
        cache_key = self._build_cache_key(
            lon_min, lon_max, lat_min, lat_max,
            depth_min, depth_max, start_date, end_date,
        )
        cached = self._cache_path(cache_key)

        # Cache hit
        if cached.exists():
            logger.info("Cache hit: %s", cached.name)
            ds = xr.open_dataset(cached)
            return ds

        # Build argopy region list
        region = [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max]
        if start_date and end_date:
            region.extend([start_date, end_date])

        logger.info("Cache miss, fetching region: %s", region)

        # Try ERDDAP first (fast, reliable)
        ds = self._try_fetch("erddap", region)

        # Fallback to GDAC
        if ds is None:
            ds = self._try_fetch("gdac", region)

        if ds is None:
            logger.error("Both ERDDAP and GDAC failed for region: %s", region)
            return None

        # Apply QC filter
        ds = _apply_qc_filter(ds)

        # Cache to disk
        try:
            ds.to_netcdf(cached)
            logger.info("Cached to: %s", cached.name)
        except Exception as e:
            logger.warning("Failed to cache dataset: %s", e)

        return ds

    def _try_fetch(self, source: str, region: list) -> xr.Dataset | None:
        """Attempt to fetch data from a single source."""
        try:
            fetcher = argopy.DataFetcher(src=source).region(region)
            ds = _fetch_xarray_with_timeout(fetcher, self._timeout)
            logger.info("Fetched %d profiles from %s", ds.sizes.get("N_PROF", 0), source)
            return ds
        except TimeoutError:
            logger.warning("%s fetch timed out after %ds", source, self._timeout)
            return None
        except Exception as e:
            logger.warning("%s fetch failed: %s", source, e)
            return None

    def get_statistics(
        self,
        ds: xr.Dataset | None,
        variable: str,
    ) -> dict:
        """Compute statistics for a variable in the dataset."""
        if ds is None:
            return {"error": "No dataset provided"}

        if variable not in ds:
            return {"error": f"Variable '{variable}' not found in dataset"}

        values = ds[variable].values.flatten()
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            return {"error": f"No valid values for '{variable}'"}

        return {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median": float(np.median(valid)),
            "count": int(len(valid)),
        }

    def to_dataframe(self, ds: xr.Dataset | None) -> pd.DataFrame:
        """Convert xarray Dataset to pandas DataFrame."""
        if ds is None:
            return pd.DataFrame()

        return ds.to_dataframe().reset_index()
