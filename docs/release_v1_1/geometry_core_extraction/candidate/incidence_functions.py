"""
Verbatim algorithm-function extraction from the validated
development implementation.

NOT YET A PRODUCTION MODULE.
Generated for pyPSDS-GAMMA v1.1 migration review.
"""

from pathlib import Path
import json
import re
import time
import numpy as np
from numba import njit, prange


def read_par(path):

    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        d[k.strip().lower()] = v.strip()

    return d


def scalar(d, key):

    return float(
        NUM_RE.findall(
            d[key]
        )[0]
    )


def vec3(d, key):

    x = NUM_RE.findall(
        d[key]
    )

    return np.array(
        [
            float(x[0]),
            float(x[1]),
            float(x[2]),
        ],
        dtype=np.float64,
    )


def orbit_position(
    query_time,
):

    idx = (
        np.searchsorted(
            sv_t0
            +
            np.arange(
                nsv,
                dtype=np.float64,
            )
            *
            sv_dt,
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
        2*u3
        -
        3*u2
        +
        1
    )

    h10 = (
        u3
        -
        2*u2
        +
        u
    )

    h01 = (
        -2*u3
        +
        3*u2
    )

    h11 = (
        u3
        -
        u2
    )


    return (
        h00[:, None]
        *
        pos[idx]

        +

        h10[:, None]
        *
        sv_dt
        *
        vel[idx]

        +

        h01[:, None]
        *
        pos[idx + 1]

        +

        h11[:, None]
        *
        sv_dt
        *
        vel[idx + 1]
    )


def incidence_fast(
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
            lx*lx
            +
            ly*ly
            +
            lz*lz
        )


        ground_norm = np.sqrt(
            gx*gx
            +
            gy*gy
            +
            gz*gz
        )


        cosine = (
            gx*lx
            +
            gy*ly
            +
            gz*lz
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

