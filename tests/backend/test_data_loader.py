"""Tests for Argo data loader."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from backend.data.loader import (
    ARGO_VARIABLES,
    OCEAN_BASINS,
    ArgoDataLoader,
    _apply_qc_filter,
    _dataset_to_profiles,
    _fetch_xarray_with_timeout,
)
from backend.data.schema import DatasetMetadata, VariableInfo


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


class TestQCFilter:
    def test_keeps_good_data(self):
        ds = _make_sample_dataset()
        filtered = _apply_qc_filter(ds)
        assert not np.all(np.isnan(filtered["TEMP"].values))

    def test_removes_bad_data(self):
        ds = _make_sample_dataset()
        ds["TEMP_QC"].values[0, :] = 4  # Bad data
        filtered = _apply_qc_filter(ds)
        assert np.all(np.isnan(filtered["TEMP"].values[0, :]))

    def test_keeps_probably_good(self):
        ds = _make_sample_dataset()
        ds["TEMP_QC"].values[0, :] = 2  # Probably good
        filtered = _apply_qc_filter(ds)
        assert not np.all(np.isnan(filtered["TEMP"].values[0, :]))

    def test_handles_missing_qc_variable(self):
        ds = _make_sample_dataset()
        del ds["TEMP_QC"]
        filtered = _apply_qc_filter(ds)
        assert "TEMP" in filtered


class TestDatasetToProfiles:
    def test_converts_to_profiles(self):
        ds = _make_sample_dataset(n_prof=2)
        profiles = _dataset_to_profiles(ds)
        assert len(profiles) == 2

    def test_profile_has_correct_coordinates(self):
        ds = _make_sample_dataset(n_prof=1)
        profiles = _dataset_to_profiles(ds)
        assert profiles[0].latitude == 30.0
        assert profiles[0].longitude == -40.0

    def test_profile_has_variables(self):
        ds = _make_sample_dataset(n_prof=1)
        profiles = _dataset_to_profiles(ds)
        assert "TEMP" in profiles[0].variables
        assert "PSAL" in profiles[0].variables

    def test_profile_depth_levels(self):
        ds = _make_sample_dataset(n_prof=1, n_levels=5)
        profiles = _dataset_to_profiles(ds)
        assert len(profiles[0].depth_levels) == 5

    def test_handles_nan_values(self):
        ds = _make_sample_dataset(n_prof=1, n_levels=5)
        ds["TEMP"].values[0, 2] = np.nan
        profiles = _dataset_to_profiles(ds)
        assert len(profiles[0].variables["TEMP"]) == 4  # One NaN removed

    def test_empty_dataset(self):
        ds = _make_sample_dataset(n_prof=0, n_levels=0)
        profiles = _dataset_to_profiles(ds)
        assert len(profiles) == 0


class TestArgoDataLoader:
    def test_get_metadata(self):
        loader = ArgoDataLoader.__new__(ArgoDataLoader)
        loader._settings = None
        meta = ArgoDataLoader.get_metadata(loader)
        assert isinstance(meta, DatasetMetadata)
        assert meta.lat_bounds == (-90.0, 90.0)
        assert "TEMP" in meta.available_variables

    def test_get_available_variables(self):
        loader = ArgoDataLoader.__new__(ArgoDataLoader)
        loader._settings = None
        variables = ArgoDataLoader.get_available_variables(loader)
        assert len(variables) == 4
        assert all(isinstance(v, VariableInfo) for v in variables)
        names = [v.name for v in variables]
        assert "TEMP" in names
        assert "PSAL" in names


class TestFetchXarrayWithTimeout:
    def test_returns_result_within_timeout(self):
        ds = _make_sample_dataset()
        fetcher = MagicMock()
        fetcher.to_xarray.return_value = ds

        result = _fetch_xarray_with_timeout(fetcher, timeout_seconds=5)
        assert result is ds
        fetcher.to_xarray.assert_called_once()

    def test_raises_timeout_error_on_slow_fetch(self):
        def slow_fetch():
            time.sleep(5)
            return _make_sample_dataset()

        fetcher = MagicMock()
        fetcher.to_xarray.side_effect = slow_fetch

        with pytest.raises(TimeoutError, match="timed out after 1s"):
            _fetch_xarray_with_timeout(fetcher, timeout_seconds=1)

    def test_propagates_fetcher_exceptions(self):
        fetcher = MagicMock()
        fetcher.to_xarray.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            _fetch_xarray_with_timeout(fetcher, timeout_seconds=5)


class TestConstants:
    def test_argo_variables_defined(self):
        assert len(ARGO_VARIABLES) == 4
        names = {v.name for v in ARGO_VARIABLES}
        assert names == {"TEMP", "PSAL", "PRES", "DOXY"}

    def test_ocean_basins_defined(self):
        assert "atlantic" in OCEAN_BASINS
        assert "pacific" in OCEAN_BASINS
        atlantic = OCEAN_BASINS["atlantic"]
        assert atlantic["lat_min"] < atlantic["lat_max"]
