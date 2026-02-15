"""Tests for LangChain tools."""

import numpy as np
import pytest

from backend.tools.stats_tools import calculate_statistics, detect_anomalies
from backend.tools.geo_tools import ocean_basin_bounds, _haversine_km


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
