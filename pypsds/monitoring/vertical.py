from __future__ import annotations

import numpy as np


def vertical_factor(incidence_rad, positive: str = "down"):
    """
    LOS is positive toward satellite.

    Zero-horizontal-motion approximation:
      vertical_up   = LOS / cos(incidence)
      vertical_down = -vertical_up
    """
    inc = np.asarray(incidence_rad, dtype=np.float64)
    if np.any(~np.isfinite(inc)):
        raise ValueError("incidence angle contains non-finite values")
    if np.any((inc <= 0.0) | (inc >= np.pi / 2.0)):
        raise ValueError("incidence angles must be inside (0, pi/2)")
    factor = 1.0 / np.cos(inc)
    p = str(positive).strip().lower()
    if p == "up":
        return factor
    if p == "down":
        return -factor
    raise ValueError("vertical positive must be up or down")
