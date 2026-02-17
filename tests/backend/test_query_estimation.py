"""Tests for query estimation and smart defaults."""

from datetime import date, timedelta

import pytest

from backend.tools.query_estimation import apply_smart_date_defaults, estimate_query_size


class TestEstimateQuerySize:
    def test_small_region_not_large(self):
        """Mediterranean + 90 days = not large."""
        today = date.today()
        start = today - timedelta(days=90)
        result = estimate_query_size(
            lat_min=30, lat_max=46, lon_min=-6, lon_max=36,
            depth_min=0, depth_max=2000,
            start_date=start.isoformat(), end_date=today.isoformat(),
        )
        assert result["area_deg2"] == 16 * 42  # 672
        assert result["is_large"] is False
        assert result["is_very_large"] is False

    def test_global_no_dates_very_large(self):
        """Global region with no dates = very large."""
        result = estimate_query_size(
            lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
            depth_min=0, depth_max=2000,
            start_date=None, end_date=None,
        )
        assert result["area_deg2"] == 180 * 360  # 64800
        assert result["is_large"] is True
        assert result["is_very_large"] is True

    def test_atlantic_no_dates_very_large(self):
        """Full Atlantic with no dates is very large."""
        result = estimate_query_size(
            lat_min=-60, lat_max=60, lon_min=-80, lon_max=0,
            depth_min=0, depth_max=2000,
            start_date=None, end_date=None,
        )
        assert result["area_deg2"] == 120 * 80  # 9600
        assert result["is_very_large"] is True

    def test_narrow_depth_reduces_estimate(self):
        """Narrower depth range should reduce estimated profiles."""
        wide = estimate_query_size(
            lat_min=30, lat_max=46, lon_min=-6, lon_max=36,
            depth_min=0, depth_max=2000,
            start_date="2024-01-01", end_date="2024-04-01",
        )
        narrow = estimate_query_size(
            lat_min=30, lat_max=46, lon_min=-6, lon_max=36,
            depth_min=450, depth_max=550,
            start_date="2024-01-01", end_date="2024-04-01",
        )
        assert narrow["estimated_profiles"] < wide["estimated_profiles"]

    def test_result_contains_expected_keys(self):
        result = estimate_query_size(
            lat_min=0, lat_max=10, lon_min=0, lon_max=10,
            depth_min=0, depth_max=100,
            start_date="2024-01-01", end_date="2024-04-01",
        )
        assert "area_deg2" in result
        assert "days" in result
        assert "estimated_profiles" in result
        assert "is_large" in result
        assert "is_very_large" in result


class TestApplySmartDateDefaults:
    def test_applies_default_when_both_none(self):
        """Returns 90-day window when both dates are None."""
        start, end = apply_smart_date_defaults(None, None)
        assert start is not None
        assert end is not None
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        assert (end_date - start_date).days == 90

    def test_end_date_is_today(self):
        """Default end date should be today."""
        _, end = apply_smart_date_defaults(None, None)
        assert end == date.today().isoformat()

    def test_preserves_explicit_dates(self):
        """Doesn't override provided dates."""
        start, end = apply_smart_date_defaults("2020-01-01", "2023-12-31")
        assert start == "2020-01-01"
        assert end == "2023-12-31"

    def test_preserves_explicit_start_only(self):
        """Preserves start date when only start is provided."""
        start, end = apply_smart_date_defaults("2023-06-01", None)
        assert start == "2023-06-01"
        assert end is not None

    def test_preserves_explicit_end_only(self):
        """Preserves end date when only end is provided."""
        start, end = apply_smart_date_defaults(None, "2024-06-01")
        assert start is not None
        assert end == "2024-06-01"

    def test_custom_default_days(self):
        """Respects custom default window size."""
        start, end = apply_smart_date_defaults(None, None, default_days=30)
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        assert (end_date - start_date).days == 30
