"""Tests for LangChain tools."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.tools.stats_tools import calculate_statistics, detect_anomalies
from backend.tools.argo_tools import query_ocean_data
from backend.tools.geo_tools import ocean_basin_bounds, get_nearest_profiles, _haversine_km


class TestCalculateStatistics:
    def test_summary_statistics(self):
        result = calculate_statistics.invoke({"data": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert result["success"] is True
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["n_values"] == 5

    def test_mean_only(self):
        result = calculate_statistics.invoke({"data": [10.0, 20.0, 30.0], "stat_type": "mean"})
        assert result["success"] is True
        assert result["mean"] == 20.0
        assert "median" not in result

    def test_median_only(self):
        result = calculate_statistics.invoke({"data": [1.0, 2.0, 3.0], "stat_type": "median"})
        assert result["success"] is True
        assert result["median"] == 2.0
        assert "mean" not in result

    def test_empty_data(self):
        result = calculate_statistics.invoke({"data": []})
        assert result["success"] is False
        assert "error" in result

    def test_single_value(self):
        result = calculate_statistics.invoke({"data": [42.0]})
        assert result["success"] is True
        assert result["mean"] == 42.0

    def test_with_percentiles(self):
        result = calculate_statistics.invoke({"data": list(range(100))})
        assert result["success"] is True
        assert "percentile_25" in result
        assert "percentile_75" in result


class TestDetectAnomalies:
    def test_no_anomalies_in_uniform_data(self):
        data = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8]
        result = detect_anomalies.invoke({"data": data})
        assert result["success"] is True
        assert result["n_anomalies"] == 0

    def test_detects_obvious_outlier(self):
        data = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 100.0]
        result = detect_anomalies.invoke({"data": data})
        assert result["success"] is True
        assert result["n_anomalies"] >= 1
        anomaly_values = [a["value"] for a in result["anomalies"]]
        assert 100.0 in anomaly_values

    def test_custom_threshold(self):
        data = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 12.0]
        # With strict threshold, 12.0 should be anomalous
        result = detect_anomalies.invoke({"data": data, "threshold": 1.5})
        assert result["success"] is True
        assert result["threshold"] == 1.5

    def test_empty_data(self):
        result = detect_anomalies.invoke({"data": []})
        assert result["success"] is False

    def test_too_few_values(self):
        result = detect_anomalies.invoke({"data": [1.0, 2.0]})
        assert result["success"] is False

    def test_returns_bounds(self):
        data = [10.0, 10.1, 9.9, 10.0, 10.2]
        result = detect_anomalies.invoke({"data": data})
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert result["lower_bound"] < result["upper_bound"]


class TestOceanBasinBounds:
    def test_atlantic(self):
        result = ocean_basin_bounds.invoke({"basin": "atlantic"})
        assert result["success"] is True
        assert result["lat_min"] < result["lat_max"]
        assert result["basin"] == "atlantic"

    def test_pacific(self):
        result = ocean_basin_bounds.invoke({"basin": "pacific"})
        assert result["success"] is True

    def test_case_insensitive(self):
        result = ocean_basin_bounds.invoke({"basin": "Atlantic"})
        assert result["success"] is True
        assert result["basin"] == "atlantic"

    def test_with_spaces(self):
        result = ocean_basin_bounds.invoke({"basin": "north atlantic"})
        assert result["success"] is True
        assert result["basin"] == "north_atlantic"

    def test_unknown_basin(self):
        result = ocean_basin_bounds.invoke({"basin": "atlantis"})
        assert result["success"] is False
        assert "error" in result

    def test_all_basins_valid(self):
        from backend.data.loader import OCEAN_BASINS
        for basin_name in OCEAN_BASINS:
            result = ocean_basin_bounds.invoke({"basin": basin_name})
            assert result["success"] is True

    def test_tropical_atlantic_exists(self):
        result = ocean_basin_bounds.invoke({"basin": "tropical_atlantic"})
        assert result["success"] is True
        assert result["basin"] == "tropical_atlantic"
        assert result["lat_min"] == -20
        assert result["lat_max"] == 20

    def test_area_deg2_in_response(self):
        result = ocean_basin_bounds.invoke({"basin": "mediterranean"})
        assert result["success"] is True
        assert "area_deg2" in result
        assert result["area_deg2"] > 0

    def test_subregions_smaller_than_full_basins(self):
        atlantic = ocean_basin_bounds.invoke({"basin": "atlantic"})
        north_atlantic = ocean_basin_bounds.invoke({"basin": "north_atlantic"})
        assert north_atlantic["area_deg2"] < atlantic["area_deg2"]


class TestQueryOceanDataSmartDefaults:
    @patch("backend.tools.argo_tools._get_manager")
    def test_applies_90_day_default_when_no_dates(self, mock_get_manager):
        """Verify manager receives non-None dates when none provided."""
        mock_manager = MagicMock()
        mock_manager.get_data.return_value = None
        mock_get_manager.return_value = mock_manager

        query_ocean_data.invoke({"variable": "TEMP"})

        call_kwargs = mock_manager.get_data.call_args[1]
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    @patch("backend.tools.argo_tools._get_manager")
    def test_preserves_explicit_dates(self, mock_get_manager):
        """Verify explicit dates pass through unchanged."""
        mock_manager = MagicMock()
        mock_manager.get_data.return_value = None
        mock_get_manager.return_value = mock_manager

        query_ocean_data.invoke({
            "variable": "TEMP",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        })

        call_kwargs = mock_manager.get_data.call_args[1]
        assert call_kwargs["start_date"] == "2020-01-01"
        assert call_kwargs["end_date"] == "2023-12-31"

    @patch("backend.tools.argo_tools._get_manager")
    def test_large_query_includes_warning(self, mock_get_manager):
        """Verify warning field present for large queries."""
        # Create a mock dataset that looks like real data
        mock_ds = MagicMock()
        mock_ds.__contains__ = lambda self, key: key in ("TEMP", "LATITUDE", "LONGITUDE")
        mock_ds.__getitem__ = lambda self, key: MagicMock(
            values=MagicMock(
                flatten=lambda: np.array([15.0, 16.0, 17.0]),
                tolist=lambda: [30.0, 31.0, 32.0],
            )
        )
        mock_ds.sizes = {"N_PROF": 3}
        mock_manager = MagicMock()
        mock_manager.get_data.return_value = mock_ds
        mock_get_manager.return_value = mock_manager

        # Full Atlantic = large query
        result = query_ocean_data.invoke({
            "variable": "TEMP",
            "lat_min": -60, "lat_max": 60,
            "lon_min": -80, "lon_max": 0,
            "start_date": "2000-01-01",
            "end_date": "2024-12-31",
        })

        assert result["success"] is True
        assert "warning" in result
        assert "query_estimate" in result


class TestQueryOceanDataTimeout:
    @patch("backend.tools.argo_tools._get_manager")
    def test_returns_error_dict_on_timeout(self, mock_get_manager):
        mock_manager = MagicMock()
        mock_manager.get_data.side_effect = TimeoutError(
            "Data fetch timed out after 45s. Try a smaller region or time range."
        )
        mock_get_manager.return_value = mock_manager

        result = query_ocean_data.invoke({"variable": "TEMP"})
        assert result["success"] is False
        assert "timed out" in result["error"]

    @patch("backend.tools.argo_tools._get_manager")
    def test_generic_exception_still_handled(self, mock_get_manager):
        mock_manager = MagicMock()
        mock_manager.get_data.side_effect = RuntimeError("connection lost")
        mock_get_manager.return_value = mock_manager

        result = query_ocean_data.invoke({"variable": "TEMP"})
        assert result["success"] is False
        assert "connection lost" in result["error"]

    @patch("backend.tools.argo_tools._get_manager")
    def test_returns_error_when_both_sources_fail(self, mock_get_manager):
        mock_manager = MagicMock()
        mock_manager.get_data.return_value = None
        mock_get_manager.return_value = mock_manager

        result = query_ocean_data.invoke({"variable": "TEMP"})
        assert result["success"] is False
        assert "Failed to fetch data" in result["error"]


class TestGetNearestProfilesTimeout:
    @patch("backend.tools.geo_tools._get_loader")
    def test_returns_error_dict_on_timeout(self, mock_get_loader):
        mock_loader = MagicMock()
        mock_loader.fetch_region.side_effect = TimeoutError(
            "Data fetch timed out after 45s. Try a smaller region or time range."
        )
        mock_get_loader.return_value = mock_loader

        result = get_nearest_profiles.invoke({"lat": 30.0, "lon": -40.0})
        assert result["success"] is False
        assert "timed out" in result["error"]


class TestHaversine:
    def test_same_point(self):
        dist = _haversine_km(0, 0, 0, 0)
        assert dist == 0.0

    def test_known_distance(self):
        # New York to London is approximately 5570 km
        dist = _haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < dist < 5700

    def test_equator_one_degree(self):
        # One degree of longitude at equator is ~111 km
        dist = _haversine_km(0, 0, 0, 1)
        assert 110 < dist < 112
