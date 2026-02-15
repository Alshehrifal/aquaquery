"""Statistical analysis tools for ocean data."""

from typing import Any

import numpy as np
from langchain_core.tools import tool


@tool
def calculate_statistics(
    data: list[float],
    stat_type: str = "summary",
) -> dict[str, Any]:
    """Calculate statistical summary for a dataset.

    Args:
        data: List of numeric values to analyze
        stat_type: Type of statistics - 'summary' for all, 'mean', 'median', 'std'

    Returns:
        Dict with calculated statistics.
    """
    if not data:
        return {"error": "Empty dataset provided", "success": False}

    values = np.array([v for v in data if v is not None and not np.isnan(v)])

    if len(values) == 0:
        return {"error": "No valid values in dataset", "success": False}

    result: dict[str, Any] = {"success": True, "n_values": len(values)}

    if stat_type in ("summary", "mean"):
        result["mean"] = float(np.mean(values))
    if stat_type in ("summary", "median"):
        result["median"] = float(np.median(values))
    if stat_type in ("summary", "std"):
        result["std"] = float(np.std(values))
    if stat_type == "summary":
        result["min"] = float(np.min(values))
        result["max"] = float(np.max(values))
        result["percentile_25"] = float(np.percentile(values, 25))
        result["percentile_75"] = float(np.percentile(values, 75))

    return result


@tool
def detect_anomalies(
    data: list[float],
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Detect anomalous values beyond N standard deviations from the mean.

    Args:
        data: List of numeric values to check
        threshold: Number of standard deviations for anomaly detection (default 2.0)

    Returns:
        Dict with anomaly indices, values, and statistics.
    """
    if not data:
        return {"error": "Empty dataset provided", "success": False}

    values = np.array([v for v in data if v is not None and not np.isnan(v)])

    if len(values) < 3:
        return {"error": "Need at least 3 values for anomaly detection", "success": False}

    mean = float(np.mean(values))
    std = float(np.std(values))

    if std == 0:
        return {
            "success": True,
            "n_anomalies": 0,
            "anomalies": [],
            "mean": mean,
            "std": std,
            "threshold": threshold,
        }

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    anomaly_mask = (values < lower_bound) | (values > upper_bound)
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    anomaly_values = values[anomaly_mask].tolist()

    return {
        "success": True,
        "n_anomalies": len(anomaly_indices),
        "anomalies": [
            {"index": idx, "value": val}
            for idx, val in zip(anomaly_indices, anomaly_values)
        ],
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
