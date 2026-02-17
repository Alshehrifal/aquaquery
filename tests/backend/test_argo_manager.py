"""Tests for ArgoDataManager with ERDDAP-first strategy and file-based caching."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from backend.data.argo_manager import ArgoDataManager


def _make_sample_dataset(n_prof: int = 3, n_levels: int = 5) -> xr.Dataset:
    """Create a sample xarray Dataset mimicking Argo structure."""
    return xr.Dataset(
        {
            "TEMP": (["N_PROF", "N_LEVELS"], np.random.uniform(2, 25, (n_prof, n_levels))),
            "PSAL": (["N_PROF", "N_LEVELS"], np.random.uniform(33, 37, (n_prof, n_levels))),
            "PRES": (["N_PROF", "N_LEVELS"], np.tile(np.linspace(10, 2000, n_levels), (n_prof, 1))),
            "TEMP_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "PSAL_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "PRES_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "LATITUDE": (["N_PROF"], np.array([30.0, 35.0, 40.0][:n_prof])),
            "LONGITUDE": (["N_PROF"], np.array([-40.0, -35.0, -30.0][:n_prof])),
            "TIME": (["N_PROF"], np.array(
                ["2023-06-01", "2023-07-01", "2023-08-01"][:n_prof],
                dtype="datetime64[ns]",
            )),
        }
    )


@pytest.fixture
def manager(tmp_path):
    """Create an ArgoDataManager with a temporary cache directory."""
    return ArgoDataManager(cache_dir=str(tmp_path))


class TestCacheKey:
    def test_cache_key_deterministic(self, manager):
        """Same parameters should produce the same cache key."""
        key1 = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, None, None)
        key2 = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, None, None)
        assert key1 == key2

    def test_cache_key_different_params(self, manager):
        """Different parameters should produce different cache keys."""
        key1 = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, None, None)
        key2 = manager._build_cache_key(-40, 10, -30, 50, 0, 1000, None, None)
        assert key1 != key2

    def test_cache_key_with_dates(self, manager):
        """Date parameters should affect the cache key."""
        key1 = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, "2023-01", "2023-06")
        key2 = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, "2023-06", "2024-01")
        assert key1 != key2

    def test_cache_key_is_md5_hex(self, manager):
        """Cache key should be a valid MD5 hex string."""
        key = manager._build_cache_key(-80, 0, -60, 60, 0, 2000, None, None)
        assert len(key) == 32
        int(key, 16)  # Should not raise -- valid hex


class TestGetDataCacheHit:
    def test_loads_from_cache_when_file_exists(self, manager, tmp_path):
        """When a cached .nc file exists, load from disk instead of fetching."""
        ds = _make_sample_dataset()
        cache_key = manager._build_cache_key(-80, 0, 20, 60, 0, 2000, None, None)
        cache_path = tmp_path / f"{cache_key}.nc"
        ds.to_netcdf(cache_path)

        result = manager.get_data(-80, 0, 20, 60, 0, 2000, None, None)

        assert result is not None
        assert "TEMP" in result
        assert result.sizes["N_PROF"] == 3


class TestGetDataCacheMiss:
    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_fetches_from_erddap_and_caches(self, mock_argopy, mock_fetch, manager, tmp_path):
        """On cache miss, fetch from ERDDAP (primary) and save to disk."""
        ds = _make_sample_dataset()
        mock_fetch.return_value = ds

        result = manager.get_data(-80, 0, 20, 60, 0, 2000, None, None)

        assert result is not None
        assert "TEMP" in result
        # Verify ERDDAP was used as primary source
        mock_argopy.DataFetcher.assert_called_once_with(src="erddap")
        # Verify file was cached
        cache_key = manager._build_cache_key(-80, 0, 20, 60, 0, 2000, None, None)
        assert (tmp_path / f"{cache_key}.nc").exists()


class TestGetDataFallback:
    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_erddap_fails_gdac_fallback(self, mock_argopy, mock_fetch, manager):
        """When ERDDAP fails, fall back to GDAC."""
        ds = _make_sample_dataset()
        # First call (ERDDAP) fails, second call (GDAC) succeeds
        mock_fetch.side_effect = [Exception("ERDDAP down"), ds]

        result = manager.get_data(-80, 0, 20, 60, 0, 2000, None, None)

        assert result is not None
        assert mock_argopy.DataFetcher.call_count == 2
        calls = mock_argopy.DataFetcher.call_args_list
        assert calls[0].kwargs["src"] == "erddap"
        assert calls[1].kwargs["src"] == "gdac"

    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_both_sources_fail_returns_none(self, mock_argopy, mock_fetch, manager):
        """When both ERDDAP and GDAC fail, return None."""
        mock_fetch.side_effect = [Exception("ERDDAP down"), Exception("GDAC down")]

        result = manager.get_data(-80, 0, 20, 60, 0, 2000, None, None)

        assert result is None


class TestQCFilterApplied:
    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_qc_filter_applied_after_fetch(self, mock_argopy, mock_fetch, manager):
        """QC filtering should be applied after fetching data."""
        ds = _make_sample_dataset()
        ds["TEMP_QC"].values[0, :] = 4  # Mark first profile as bad
        mock_fetch.return_value = ds

        result = manager.get_data(-80, 0, 20, 60, 0, 2000, None, None)

        assert result is not None
        # First profile's TEMP should be NaN after QC filter
        assert np.all(np.isnan(result["TEMP"].values[0, :]))


class TestGetStatistics:
    def test_temperature_statistics(self, manager):
        """Should compute correct statistics for temperature data."""
        ds = _make_sample_dataset(n_prof=3, n_levels=5)
        stats = manager.get_statistics(ds, "TEMP")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "count" in stats
        assert stats["count"] > 0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_statistics_empty_dataset(self, manager):
        """Should return error dict for empty dataset."""
        stats = manager.get_statistics(None, "TEMP")

        assert "error" in stats

    def test_statistics_missing_variable(self, manager):
        """Should return error dict when variable is not in dataset."""
        ds = _make_sample_dataset()
        stats = manager.get_statistics(ds, "NONEXISTENT")

        assert "error" in stats


class TestToDataframe:
    def test_converts_to_dataframe(self, manager):
        """Should convert xarray Dataset to pandas DataFrame."""
        ds = _make_sample_dataset(n_prof=2, n_levels=3)
        df = manager.to_dataframe(ds)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "TEMP" in df.columns or len(df.columns) > 0

    def test_none_dataset_returns_empty_dataframe(self, manager):
        """Should return empty DataFrame for None input."""
        df = manager.to_dataframe(None)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# --- Float/Profile fetch tests ---


def _make_float_dataset(n_prof: int = 5, n_levels: int = 4) -> xr.Dataset:
    """Create a sample dataset with PLATFORM_NUMBER and CYCLE_NUMBER."""
    return xr.Dataset(
        {
            "TEMP": (["N_PROF", "N_LEVELS"], np.random.uniform(2, 25, (n_prof, n_levels))),
            "PSAL": (["N_PROF", "N_LEVELS"], np.random.uniform(33, 37, (n_prof, n_levels))),
            "PRES": (["N_PROF", "N_LEVELS"], np.tile(np.linspace(10, 2000, n_levels), (n_prof, 1))),
            "TEMP_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "PSAL_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "PRES_QC": (["N_PROF", "N_LEVELS"], np.ones((n_prof, n_levels), dtype=int)),
            "LATITUDE": (["N_PROF"], np.linspace(30.0, 40.0, n_prof)),
            "LONGITUDE": (["N_PROF"], np.linspace(-40.0, -30.0, n_prof)),
            "TIME": (["N_PROF"], np.array(
                pd.date_range("2023-01-01", periods=n_prof, freq="30D"),
                dtype="datetime64[ns]",
            )),
            "PLATFORM_NUMBER": (["N_PROF"], np.full(n_prof, "6902746")),
            "CYCLE_NUMBER": (["N_PROF"], np.arange(1, n_prof + 1)),
        }
    )


class TestGetDataByFloat:
    def test_cache_hit(self, manager, tmp_path):
        """When a cached .nc file for a float exists, load from disk."""
        ds = _make_float_dataset()
        cache_key = manager._build_float_cache_key(6902746)
        cache_path = tmp_path / f"{cache_key}.nc"
        ds.to_netcdf(cache_path)

        result = manager.get_data_by_float(6902746)

        assert result is not None
        assert "TEMP" in result
        assert result.sizes["N_PROF"] == 5

    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_erddap_success(self, mock_argopy, mock_fetch, manager, tmp_path):
        """On cache miss, fetch from ERDDAP and cache to disk."""
        ds = _make_float_dataset()
        mock_fetch.return_value = ds

        result = manager.get_data_by_float(6902746)

        assert result is not None
        assert "TEMP" in result
        mock_argopy.DataFetcher.assert_called_once_with(src="erddap")
        mock_argopy.DataFetcher.return_value.float.assert_called_once_with(6902746)
        # Verify cached
        cache_key = manager._build_float_cache_key(6902746)
        assert (tmp_path / f"{cache_key}.nc").exists()

    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_fallback_to_gdac(self, mock_argopy, mock_fetch, manager):
        """When ERDDAP fails, fall back to GDAC."""
        ds = _make_float_dataset()
        mock_fetch.side_effect = [Exception("ERDDAP down"), ds]

        result = manager.get_data_by_float(6902746)

        assert result is not None
        assert mock_argopy.DataFetcher.call_count == 2
        calls = mock_argopy.DataFetcher.call_args_list
        assert calls[0].kwargs["src"] == "erddap"
        assert calls[1].kwargs["src"] == "gdac"

    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_both_fail_returns_none(self, mock_argopy, mock_fetch, manager):
        """When both ERDDAP and GDAC fail, return None."""
        mock_fetch.side_effect = [Exception("ERDDAP down"), Exception("GDAC down")]

        result = manager.get_data_by_float(6902746)

        assert result is None


class TestGetDataByProfile:
    @patch("backend.data.argo_manager._fetch_xarray_with_timeout")
    @patch("backend.data.argo_manager.argopy")
    def test_success(self, mock_argopy, mock_fetch, manager):
        """Fetch a single profile by WMO ID and cycle number."""
        ds = _make_float_dataset(n_prof=1)
        mock_fetch.return_value = ds

        result = manager.get_data_by_profile(6902746, 10)

        assert result is not None
        mock_argopy.DataFetcher.return_value.profile.assert_called_once_with(6902746, 10)


class TestExtractTrajectory:
    def test_sorted_by_time(self, manager):
        """Trajectory should be sorted by time."""
        ds = _make_float_dataset(n_prof=5)
        # Scramble the time order
        times = ds["TIME"].values.copy()
        ds["TIME"].values[:] = times[::-1]

        traj = manager.extract_trajectory(ds)

        assert traj is not None
        assert len(traj["latitudes"]) == 5
        assert len(traj["longitudes"]) == 5
        assert len(traj["timestamps"]) == 5
        # Verify sorted ascending
        for i in range(len(traj["timestamps"]) - 1):
            assert traj["timestamps"][i] <= traj["timestamps"][i + 1]

    def test_none_dataset(self, manager):
        """Should return None for None dataset."""
        traj = manager.extract_trajectory(None)

        assert traj is None
