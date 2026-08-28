"""
Portable SCLA C-estimation core.

Validated GLS and geometry expressions are retained. Historical GAMMA
geometry globals and baseline coefficients are explicit keyword
arguments in v1.1.
"""

import re

import numpy as np


NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def read_par(path):

    out = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        key, rhs = line.split(
            ":",
            1,
        )

        out[
            key.strip().lower()
        ] = rhs.strip()

    return out


def par_scalar(
    pars,
    keys,
):

    for key in keys:

        rhs = pars.get(
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


def gls_projector(
    A,
    covariance,
):
    """
    pySTAMPS-GAMMA / MATLAB lscov equivalent:

        P = (A' C^-1 A)^-1 A' C^-1

        coefficient_row = y @ P.T
    """

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    C = np.asarray(
        covariance,
        dtype=np.float64,
    )

    CiA = np.linalg.solve(
        C,
        A,
    )

    normal = (
        A.T
        @ CiA
    )

    if (
        np.linalg.matrix_rank(
            normal
        )
        !=
        normal.shape[0]
    ):
        raise RuntimeError(
            "GLS normal matrix rank deficient"
        )

    return np.linalg.solve(
        normal,
        CiA.T,
    )


def geometry_factors(rr, cc, *, azimuth_looks, earth_radius, mean_azimuth, near_range, prf, range_looks, range_spacing, sar_to_earth):
    range_original = cc * range_looks + (range_looks - 1) / 2.0
    azimuth_original = rr * azimuth_looks + (azimuth_looks - 1) / 2.0
    slant_range = near_range + range_original * range_spacing
    look_arg = (sar_to_earth ** 2 + slant_range ** 2 - earth_radius ** 2) / (2.0 * sar_to_earth * slant_range)
    look = np.arccos(np.clip(look_arg, -1.0, 1.0))
    cs = np.cos(look)
    ss = np.sin(look)
    dt = (azimuth_original - mean_azimuth) / prf
    return (cs, ss, dt)

def baseline_c_projection(rr, cc, *, CW_C, CW_CR, CW_N, CW_NR, azimuth_looks, earth_radius, mean_azimuth, near_range, prf, range_looks, range_spacing, sar_to_earth):
    cs, ss, dt = geometry_factors(rr, cc, azimuth_looks=azimuth_looks, earth_radius=earth_radius, mean_azimuth=mean_azimuth, near_range=near_range, prf=prf, range_looks=range_looks, range_spacing=range_spacing, sar_to_earth=sar_to_earth)
    return cs * CW_C - ss * CW_N + dt * cs * CW_CR - dt * ss * CW_NR

__all__ = [
    "read_par",
    "par_scalar",
    "gls_projector",
    "geometry_factors",
    "baseline_c_projection",
]
