"""
Portable SCLA K-estimation core.

The validated geometry_factors expression is unchanged. Historical
module globals describing GAMMA geometry are explicit required keyword
arguments in v1.1.

Catalog/path orchestration (`ensure_catalog`) is intentionally handled
by the production stage layer.
"""

from pathlib import Path
import re

import numpy as np


NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)

PAIR_RE = re.compile(
    r"(20\d{6})[_-](20\d{6})"
)


def read_par(path):

    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        key, rhs = line.split(
            ":",
            1,
        )

        d[
            key.strip().lower()
        ] = rhs.strip()

    return d


def scalar_from_par(
    d,
    keys,
):

    for key in keys:

        rhs = d.get(
            key.lower()
        )

        if rhs is None:
            continue

        m = NUM_RE.search(
            rhs
        )

        if m:
            return float(
                m.group(0)
            )

    raise KeyError(
        " / ".join(keys)
    )


def parse_vector(
    text,
    labels,
):

    for line in text.splitlines():

        if not any(
            label in line
            for label in labels
        ):
            continue

        rhs = (
            line.split(
                ":",
                1,
            )[1]
            if ":" in line
            else line
        )

        values = NUM_RE.findall(
            rhs
        )

        if len(values) >= 3:

            return np.asarray(
                [
                    float(x)
                    for x in values[:3]
                ],
                dtype=np.float64,
            )

    return None


def parse_base(
    path,
):

    text = path.read_text(
        errors="ignore"
    )

    B = parse_vector(
        text,
        (
            "initial_baseline(TCN)",
            "initial_baseline",
        ),
    )

    Br = parse_vector(
        text,
        (
            "initial_baseline_rate",
            "baseline_rate(TCN)",
        ),
    )

    if (
        B is None
        or
        Br is None
        or
        not np.all(
            np.isfinite(B)
        )
        or
        not np.all(
            np.isfinite(Br)
        )
    ):
        raise RuntimeError(
            f"invalid GAMMA baseline model: {path}"
        )

    return B, Br


def path_from_source_entry(
    entry,
):

    if isinstance(
        entry,
        str,
    ):
        return Path(entry)

    if isinstance(
        entry,
        dict,
    ):

        for key in (
            "path",
            "base_file",
            "file",
        ):

            value = entry.get(
                key
            )

            if value:
                return Path(value)

    return None


def pair_orientation(
    path,
    date_i,
    date_j,
):

    m = PAIR_RE.search(
        path.name
    )

    if m is None:

        m = PAIR_RE.search(
            str(path)
        )

    if m is None:
        raise RuntimeError(
            f"cannot determine .base orientation: {path}"
        )

    a = m.group(1)
    b = m.group(2)

    if (
        a == date_i
        and
        b == date_j
    ):
        return 1

    if (
        a == date_j
        and
        b == date_i
    ):
        return -1

    raise RuntimeError(
        f".base date mismatch: {path}"
    )


def geometry_factors(row_chunk, col_chunk, *, azimuth_looks, earth_radius, mean_azimuth, near_range, prf, range_looks, range_spacing, sar_to_earth):
    range_original = col_chunk * range_looks + (range_looks - 1) / 2.0
    azimuth_original = row_chunk * azimuth_looks + (azimuth_looks - 1) / 2.0
    slant_range = near_range + range_original * range_spacing
    look_arg = (sar_to_earth ** 2 + slant_range ** 2 - earth_radius ** 2) / (2.0 * sar_to_earth * slant_range)
    look_arg = np.clip(look_arg, -1.0, 1.0)
    look = np.arccos(look_arg)
    c = np.cos(look)
    s = np.sin(look)
    dt = (azimuth_original - mean_azimuth) / prf
    return (c, s, dt)

__all__ = [
    "read_par",
    "scalar_from_par",
    "parse_vector",
    "parse_base",
    "path_from_source_entry",
    "pair_orientation",
    "geometry_factors",
]
