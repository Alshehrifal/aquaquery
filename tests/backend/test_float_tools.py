"""Tests for float-specific LangChain tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _make_float_dataset(n_prof: int = 5, n_levels: int = 4, wmo: str = "6902746") -> xr.Dataset:
    """Create a sample dataset with float-specific fields."""
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
            "PLATFORM_NUMBER": (["N_PROF"], np.full(n_prof, wmo)),
            "CYCLE_NUMBER": (["N_PROF"], np.arange(1, n_prof + 1)),
        }
    )


def _make_mock_manager(ds=None, trajectory=None):
    """Create a mock ArgoDataManager for float tools."""
    manager = MagicMock()
    manager.get_data_by_float.return_value = ds
    manager.get_data_by_profile.return_value = ds
    manager.extract_trajectory.return_value = trajectory
    manager.get_statistics.return_value = {
        "mean": 15.0, "std": 3.0, "min": 2.0, "max": 25.0,
        "median": 14.5, "count": 20,
    }
    return manager


class TestQueryByFloatId:
    def test_success(self):
        from backend.tools import float_tools

        ds = _make_float_dataset()
        trajectory = {
            "latitudes": [30.0, 32.5, 35.0, 37.5, 40.0],
            "longitudes": [-40.0, -37.5, -35.0, -32.5, -30.0],
            "timestamps": ["2023-01-01", "2023-01-31", "2023-03-02", "2023-04-01", "2023-05-01"],
        }
        manager = _make_mock_manager(ds=ds, trajectory=trajectory)
        float_tools.set_manager(manager)

        result = float_tools.query_by_float_id.invoke({"wmo_id": 6902746})

        assert result["success"] is True
        assert result["wmo_id"] == 6902746
        assert result["n_profiles"] == 5
        assert "trajectory" in result
        assert "statistics" in result

    def test_not_found(self):
        from backend.tools import float_tools

        manager = _make_mock_manager(ds=None)
        float_tools.set_manager(manager)

        result = float_tools.query_by_float_id.invoke({"wmo_id": 9999999})

        assert result["success"] is False
        assert "error" in result

    def test_timeout(self):
        from backend.tools import float_tools

        manager = MagicMock()
        manager.get_data_by_float.side_effect = TimeoutError("Fetch timed out")
        float_tools.set_manager(manager)

        result = float_tools.query_by_float_id.invoke({"wmo_id": 6902746})

        assert result["success"] is False
        assert "error" in result


class TestGetFloatTrajectory:
    def test_success(self):
        from backend.tools import float_tools

        ds = _make_float_dataset()
        trajectory = {
            "latitudes": [30.0, 32.5, 35.0, 37.5, 40.0],
            "longitudes": [-40.0, -37.5, -35.0, -32.5, -30.0],
            "timestamps": ["2023-01-01", "2023-01-31", "2023-03-02", "2023-04-01", "2023-05-01"],
        }
        manager = _make_mock_manager(ds=ds, trajectory=trajectory)
        float_tools.set_manager(manager)

        result = float_tools.get_float_trajectory.invoke({"wmo_id": 6902746})

        assert result["success"] is True
        assert result["chart_hint"] == "trajectory_map"
        assert "trajectory" in result

    def test_not_found(self):
        from backend.tools import float_tools

        manager = _make_mock_manager(ds=None)
        float_tools.set_manager(manager)

        result = float_tools.get_float_trajectory.invoke({"wmo_id": 9999999})

        assert result["success"] is False

    def test_timeout(self):
        from backend.tools import float_tools

        manager = MagicMock()
        manager.get_data_by_float.side_effect = TimeoutError("Timed out")
        float_tools.set_manager(manager)

        result = float_tools.get_float_trajectory.invoke({"wmo_id": 6902746})

        assert result["success"] is False


class TestGetFloatsInRegion:
    def test_success(self):
        from backend.tools import float_tools

        ds = _make_float_dataset(n_prof=3)
        # Add different platform numbers
        ds["PLATFORM_NUMBER"].values[:] = ["6902746", "6902747", "6902746"]
        manager = _make_mock_manager()
        manager.get_data.return_value = ds
        float_tools.set_manager(manager)

        result = float_tools.get_floats_in_region.invoke({
            "lat_min": 20.0, "lat_max": 50.0,
            "lon_min": -50.0, "lon_max": -20.0,
        })

        assert result["success"] is True
        assert result["n_floats"] == 2
        assert set(result["wmo_ids"]) == {6902746, 6902747}

    def test_no_data(self):
        from backend.tools import float_tools

        manager = _make_mock_manager()
        manager.get_data.return_value = None
        float_tools.set_manager(manager)

        result = float_tools.get_floats_in_region.invoke({
            "lat_min": 80.0, "lat_max": 90.0,
            "lon_min": 0.0, "lon_max": 10.0,
        })

        assert result["success"] is False

    def test_timeout(self):
        from backend.tools import float_tools

        manager = MagicMock()
        manager.get_data.side_effect = TimeoutError("Timed out")
        float_tools.set_manager(manager)

        result = float_tools.get_floats_in_region.invoke({
            "lat_min": 20.0, "lat_max": 50.0,
            "lon_min": -50.0, "lon_max": -20.0,
        })

        assert result["success"] is False


class TestQueryByProfile:
    def test_success(self):
        from backend.tools import float_tools

        ds = _make_float_dataset(n_prof=1)
        manager = _make_mock_manager(ds=ds)
        float_tools.set_manager(manager)

        result = float_tools.query_by_profile.invoke({
            "wmo_id": 6902746,
            "cycle_number": 10,
            "variable": "TEMP",
        })

        assert result["success"] is True
        assert result["chart_hint"] == "depth_profile"
        assert "depths" in result
        assert "values" in result

    def test_not_found(self):
        from backend.tools import float_tools

        manager = _make_mock_manager(ds=None)
        float_tools.set_manager(manager)

        result = float_tools.query_by_profile.invoke({
            "wmo_id": 6902746,
            "cycle_number": 999,
            "variable": "TEMP",
        })

        assert result["success"] is False

    def test_invalid_variable(self):
        from backend.tools import float_tools

        manager = _make_mock_manager()
        float_tools.set_manager(manager)

        result = float_tools.query_by_profile.invoke({
            "wmo_id": 6902746,
            "cycle_number": 10,
            "variable": "INVALID",
        })

        assert result["success"] is False

    def test_timeout(self):
        from backend.tools import float_tools

        manager = MagicMock()
        manager.get_data_by_profile.side_effect = TimeoutError("Timed out")
        float_tools.set_manager(manager)

        result = float_tools.query_by_profile.invoke({
            "wmo_id": 6902746,
            "cycle_number": 10,
            "variable": "TEMP",
        })

        assert result["success"] is False


class TestCompareFloats:
    def test_success_two_floats(self):
        from backend.tools import float_tools

        ds1 = _make_float_dataset(n_prof=3, wmo="6902746")
        ds2 = _make_float_dataset(n_prof=4, wmo="6902747")
        manager = MagicMock()
        manager.get_data_by_float.side_effect = [ds1, ds2]
        manager.get_statistics.side_effect = [
            {"mean": 15.0, "std": 3.0, "min": 2.0, "max": 25.0, "median": 14.5, "count": 12},
            {"mean": 18.0, "std": 2.0, "min": 5.0, "max": 28.0, "median": 17.5, "count": 16},
        ]
        float_tools.set_manager(manager)

        result = float_tools.compare_floats.invoke({
            "wmo_ids": [6902746, 6902747],
            "variable": "TEMP",
        })

        assert result["success"] is True
        assert result["chart_hint"] == "bar_chart"
        assert len(result["comparisons"]) == 2

    def test_too_few_floats(self):
        from backend.tools import float_tools

        manager = _make_mock_manager()
        float_tools.set_manager(manager)

        result = float_tools.compare_floats.invoke({
            "wmo_ids": [6902746],
            "variable": "TEMP",
        })

        assert result["success"] is False
        assert "2" in result["error"]

    def test_too_many_floats(self):
        from backend.tools import float_tools

        manager = _make_mock_manager()
        float_tools.set_manager(manager)

        result = float_tools.compare_floats.invoke({
            "wmo_ids": [1, 2, 3, 4, 5, 6],
            "variable": "TEMP",
        })

        assert result["success"] is False
        assert "5" in result["error"]

    def test_chart_hint_present(self):
        from backend.tools import float_tools

        ds = _make_float_dataset(n_prof=3)
        manager = MagicMock()
        manager.get_data_by_float.return_value = ds
        manager.get_statistics.return_value = {
            "mean": 15.0, "std": 3.0, "min": 2.0, "max": 25.0, "median": 14.5, "count": 12,
        }
        float_tools.set_manager(manager)

        result = float_tools.compare_floats.invoke({
            "wmo_ids": [6902746, 6902747],
            "variable": "TEMP",
        })

        assert "chart_hint" in result
        assert result["chart_hint"] == "bar_chart"

    def test_timeout(self):
        from backend.tools import float_tools

        manager = MagicMock()
        manager.get_data_by_float.side_effect = TimeoutError("Timed out")
        float_tools.set_manager(manager)

        result = float_tools.compare_floats.invoke({
            "wmo_ids": [6902746, 6902747],
            "variable": "TEMP",
        })

        assert result["success"] is False
