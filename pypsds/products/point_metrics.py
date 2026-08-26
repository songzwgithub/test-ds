from __future__ import annotations

from datetime import datetime
import numpy as np

YEAR_DAYS = 365.25


def time_axis(dates):
    labels = [str(x) for x in dates]
    if not labels:
        raise ValueError("dates must not be empty")
    dt = [datetime.strptime(x, "%Y%m%d") for x in labels]
    days = np.asarray([(x - dt[0]).days for x in dt], dtype=np.float64)
    years = days / YEAR_DAYS
    centered = years - years.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 0:
        raise ValueError("time axis has zero span")
    slope_weights = centered / denom
    return {
        "dates": np.asarray(labels, dtype="U8"),
        "days": days,
        "years": years,
        "centered_years": centered,
        "slope_weights": slope_weights,
    }


def compute_point_metrics(los_mm, dates):
    Y = np.asarray(los_mm, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("los_mm must be [point, acquisition]")
    tc = time_axis(dates)
    years = tc["years"]
    centered = tc["centered_years"]
    denom = float(np.sum(centered * centered))
    n = Y.shape[1]
    if n < 3:
        raise ValueError("at least three acquisitions are required")

    mean = np.mean(Y, axis=1)
    slope = ((Y - mean[:, None]) @ centered) / denom
    intercept = mean - slope * float(np.mean(years))
    fit = intercept[:, None] + slope[:, None] * years[None, :]
    residual = Y - fit
    sse = np.sum(residual * residual, axis=1)
    rms = np.sqrt(sse / n)
    slope_se = np.sqrt((sse / (n - 2)) / denom)
    cumulative = Y[:, -1] - Y[:, 0]

    return {
        "velocity_mm_per_year": slope,
        "cumulative_mm": cumulative,
        "linear_residual_rms_mm": rms,
        "velocity_slope_standard_error_mm_per_year": slope_se,
    }
