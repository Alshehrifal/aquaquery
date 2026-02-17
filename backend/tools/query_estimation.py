"""Query size estimation and smart defaults for Argo data queries."""

from datetime import date, timedelta

# Rough Argo profile density: ~0.15 profiles per deg^2 per day (global average)
_PROFILE_DENSITY = 0.15


def estimate_query_size(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    depth_min: float,
    depth_max: float,
    start_date: str | None,
    end_date: str | None,
) -> dict:
    """Estimate the size of an Argo data query.

    Returns a dict with area_deg2, days, estimated_profiles, is_large, is_very_large.
    """
    area_deg2 = int(abs(lat_max - lat_min) * abs(lon_max - lon_min))

    if start_date and end_date:
        d0 = date.fromisoformat(start_date[:10])
        d1 = date.fromisoformat(end_date[:10])
        days = max((d1 - d0).days, 1)
    else:
        # No date filter = assume all Argo data (~25 years)
        days = 365 * 25

    # Depth fraction: narrower depth ranges mean fewer relevant measurements
    depth_fraction = min((depth_max - depth_min) / 2000.0, 1.0)

    estimated_profiles = int(area_deg2 * days * _PROFILE_DENSITY * depth_fraction)

    return {
        "area_deg2": area_deg2,
        "days": days,
        "estimated_profiles": estimated_profiles,
        "is_large": estimated_profiles > 10_000,
        "is_very_large": estimated_profiles > 50_000,
    }


def apply_smart_date_defaults(
    start_date: str | None,
    end_date: str | None,
    default_days: int = 90,
) -> tuple[str, str]:
    """Apply smart date defaults when dates are missing.

    Passes through explicit dates unchanged.
    When both are None, returns a window of default_days ending today.
    When only one is None, fills in the missing bound.
    """
    today = date.today()

    if start_date is not None and end_date is not None:
        return (start_date, end_date)

    if start_date is None and end_date is None:
        return (
            (today - timedelta(days=default_days)).isoformat(),
            today.isoformat(),
        )

    if start_date is None:
        end = date.fromisoformat(end_date[:10])
        return (
            (end - timedelta(days=default_days)).isoformat(),
            end_date,
        )

    # end_date is None
    start = date.fromisoformat(start_date[:10])
    return (
        start_date,
        min(start + timedelta(days=default_days), today).isoformat(),
    )
