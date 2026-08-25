"""
Verbatim algorithm-function extraction from the validated
development implementation.

NOT YET A PRODUCTION MODULE.
Generated for pyPSDS-GAMMA v1.1 migration review.
"""

from pathlib import Path
import json
import math
import numpy as np


def read_par(path):
    d = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip().lower()] = v.strip()
    return d


def val(d, key):
    return float(d[key].split()[0])


def stamps_incidence(rg):
    x = (
        se * se
        - re * re
        - rg * rg
    ) / (
        2.0 * re * rg
    )

    return np.degrees(
        np.arccos(
            np.clip(x, -1.0, 1.0)
        )
    )

