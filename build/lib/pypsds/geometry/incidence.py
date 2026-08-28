from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numba import njit, prange

from .gamma_par import (
    gamma_par_int,
    gamma_par_scalar,
    read_gamma_par,
)


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

_NUM_RE = re.compile(
    r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
)


class IncidenceError(RuntimeError):
    """Invalid input or orbit geometry for incidence calculation."""


@dataclass(
    frozen=True,
    slots=True,
)
class RowOrbitGeometry:
    row_time_s: np.ndarray
    satellite_xyz_m: np.ndarray


def _gamma_vec3(
    values: dict[str, str],
    key: str,
) -> np.ndarray:
    if key not in values:
        raise IncidenceError(
            f"Missing GAMMA parameter: {key}"
        )

    tokens = _NUM_RE.findall(
        values[key]
    )

    if len(tokens) < 3:
        raise IncidenceError(
            f"Invalid GAMMA vector parameter {key!r}: "
            f"{values[key]!r}"
        )

    return np.array(
        [
            float(tokens[0]),
            float(tokens[1]),
            float(tokens[2]),
        ],
        dtype=np.float64,
    )


def orbit_position(
    query_time,
    *,
    sv_t0: float,
    sv_dt: float,
    position_m,
    velocity_m_s,
) -> np.ndarray:
    """
    Cubic Hermite interpolation of GAMMA state vectors.

    Numerical formulation is preserved from the validated
    authoritative incidence implementation.
    """

    query_time = np.asarray(
        query_time,
        dtype=np.float64,
    )

    position_m = np.asarray(
        position_m,
        dtype=np.float64,
    )

    velocity_m_s = np.asarray(
        velocity_m_s,
        dtype=np.float64,
    )

    if query_time.ndim != 1:
        raise IncidenceError(
            "query_time must be one-dimensional."
        )

    if (
        position_m.ndim != 2
        or position_m.shape[1] != 3
        or velocity_m_s.shape != position_m.shape
    ):
        raise IncidenceError(
            "Orbit position/velocity arrays must have shape (N, 3)."
        )

    nsv = position_m.shape[0]

    if nsv < 2:
        raise IncidenceError(
            "At least two orbit state vectors are required."
        )

    if sv_dt <= 0:
        raise IncidenceError(
            f"Invalid state-vector interval: {sv_dt}"
        )

    state_time = (
        sv_t0
        +
        np.arange(
            nsv,
            dtype=np.float64,
        )
        *
        sv_dt
    )

    idx = (
        np.searchsorted(
            state_time,
            query_time,
            side="right",
        )
        -
        1
    )

    idx = np.clip(
        idx,
        0,
        nsv - 2,
    )

    t_left = (
        sv_t0
        +
        idx
        *
        sv_dt
    )

    u = (
        query_time
        -
        t_left
    ) / sv_dt

    u2 = u * u
    u3 = u2 * u

    h00 = (
        2 * u3
        -
        3 * u2
        +
        1
    )

    h10 = (
        u3
        -
        2 * u2
        +
        u
    )

    h01 = (
        -2 * u3
        +
        3 * u2
    )

    h11 = (
        u3
        -
        u2
    )

    return (
        h00[:, None]
        *
        position_m[idx]

        +

        h10[:, None]
        *
        sv_dt
        *
        velocity_m_s[idx]

        +

        h01[:, None]
        *
        position_m[idx + 1]

        +

        h11[:, None]
        *
        sv_dt
        *
        velocity_m_s[idx + 1]
    )


def build_row_orbit_geometry(
    reference_rslc_par: str | Path,
) -> RowOrbitGeometry:
    """
    Build one interpolated satellite ECEF position per radar row.
    """

    par = read_gamma_par(
        reference_rslc_par
    )

    nlines = gamma_par_int(
        par,
        "azimuth_lines",
    )

    line_dt = gamma_par_scalar(
        par,
        "azimuth_line_time",
    )


    if "center_time" in par:

        center_time = gamma_par_scalar(
            par,
            "center_time",
        )

        row_time = (
            center_time
            +
            (
                np.arange(
                    nlines,
                    dtype=np.float64,
                )
                -
                0.5
                *
                (
                    nlines - 1
                )
            )
            *
            line_dt
        )

    else:

        start_time = gamma_par_scalar(
            par,
            "start_time",
        )

        row_time = (
            start_time
            +
            np.arange(
                nlines,
                dtype=np.float64,
            )
            *
            line_dt
        )


    nsv = gamma_par_int(
        par,
        "number_of_state_vectors",
    )

    sv_t0 = gamma_par_scalar(
        par,
        "time_of_first_state_vector",
    )

    sv_dt = gamma_par_scalar(
        par,
        "state_vector_interval",
    )


    position = np.vstack(
        [
            _gamma_vec3(
                par,
                f"state_vector_position_{i + 1}",
            )
            for i in range(nsv)
        ]
    )

    velocity = np.vstack(
        [
            _gamma_vec3(
                par,
                f"state_vector_velocity_{i + 1}",
            )
            for i in range(nsv)
        ]
    )


    sat_xyz = orbit_position(
        row_time,
        sv_t0=sv_t0,
        sv_dt=sv_dt,
        position_m=position,
        velocity_m_s=velocity,
    )


    return RowOrbitGeometry(
        row_time_s=
            row_time,

        satellite_xyz_m=
            sat_xyz,
    )


@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def _incidence_fast_kernel(
    lon_deg,
    lat_deg,
    hgt_m,
    radar_row,
    sat_xyz,
    a,
    e2,
):
    n = lon_deg.size

    out = np.empty(
        n,
        dtype=np.float32,
    )

    d2r = (
        np.pi
        /
        180.0
    )


    for i in prange(
        n
    ):

        lam = (
            lon_deg[i]
            *
            d2r
        )

        phi = (
            lat_deg[i]
            *
            d2r
        )


        sp = np.sin(
            phi
        )

        cp = np.cos(
            phi
        )

        sl = np.sin(
            lam
        )

        cl = np.cos(
            lam
        )


        rn = (
            a
            /
            np.sqrt(
                1.0
                -
                e2
                *
                sp
                *
                sp
            )
        )


        h = hgt_m[i]


        gx = (
            rn + h
        ) * cp * cl


        gy = (
            rn + h
        ) * cp * sl


        gz = (
            rn
            *
            (
                1.0
                -
                e2
            )
            +
            h
        ) * sp


        r = radar_row[i]


        lx = (
            sat_xyz[r, 0]
            -
            gx
        )

        ly = (
            sat_xyz[r, 1]
            -
            gy
        )

        lz = (
            sat_xyz[r, 2]
            -
            gz
        )


        los_norm = np.sqrt(
            lx * lx
            +
            ly * ly
            +
            lz * lz
        )


        ground_norm = np.sqrt(
            gx * gx
            +
            gy * gy
            +
            gz * gz
        )


        cosine = (
            gx * lx
            +
            gy * ly
            +
            gz * lz
        ) / (
            ground_norm
            *
            los_norm
        )


        if cosine > 1.0:
            cosine = 1.0

        elif cosine < -1.0:
            cosine = -1.0


        out[i] = np.arccos(
            cosine
        )


    return out


def compute_incidence_rad(
    *,
    longitude_deg,
    latitude_deg,
    height_m,
    radar_row,
    reference_rslc_par: str | Path,
) -> np.ndarray:
    """
    Compute GAMMA-compatible incidence angle in radians.

    Definition:
        G = WGS84 ECEF(lon, lat, height)
        radial normal = G / |G|
        LOS = satellite - G
        incidence = acos(radial_normal dot LOS_hat)

    Output dtype is float32, matching the validated production
    implementation.
    """

    lon = np.asarray(
        longitude_deg,
        dtype=np.float64,
    )

    lat = np.asarray(
        latitude_deg,
        dtype=np.float64,
    )

    hgt = np.asarray(
        height_m,
        dtype=np.float64,
    )

    row = np.asarray(
        radar_row,
        dtype=np.int32,
    )


    if not (
        lon.ndim
        ==
        lat.ndim
        ==
        hgt.ndim
        ==
        row.ndim
        ==
        1
    ):
        raise IncidenceError(
            "longitude, latitude, height and radar_row "
            "must be one-dimensional."
        )


    n = lon.size

    if not (
        lat.size
        ==
        hgt.size
        ==
        row.size
        ==
        n
    ):
        raise IncidenceError(
            "Point-array size mismatch."
        )


    if not np.all(
        np.isfinite(
            lon
        )
    ):
        raise IncidenceError(
            "Longitude contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            lat
        )
    ):
        raise IncidenceError(
            "Latitude contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            hgt
        )
    ):
        raise IncidenceError(
            "Height contains non-finite values."
        )


    orbit = build_row_orbit_geometry(
        reference_rslc_par
    )


    if row.size:

        if row.min() < 0:
            raise IncidenceError(
                "radar_row contains negative values."
            )

        if (
            row.max()
            >=
            orbit.satellite_xyz_m.shape[0]
        ):
            raise IncidenceError(
                "radar_row exceeds RSLC azimuth-line range."
            )


    incidence = _incidence_fast_kernel(
        lon,
        lat,
        hgt,
        row,
        orbit.satellite_xyz_m,
        WGS84_A,
        WGS84_E2,
    )


    valid = (
        np.isfinite(
            incidence
        )
        &
        (
            incidence > 0.0
        )
        &
        (
            incidence
            <
            np.pi / 2.0
        )
    )


    if not np.all(
        valid
    ):
        raise IncidenceError(
            "Computed incidence contains invalid values."
        )


    return incidence
