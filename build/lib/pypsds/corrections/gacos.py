"""
Portable GACOS numerical and date-order core.

Scientific lineage
------------------
Authoritative production source:
docs/release_v1_1/authoritative_sources/production/corrections/gacos.py

The numerical/parsing functions are migrated directly from the
validated production implementation.

The only intentional portability adaptation is:

    discover_phase_dates(expected_dates)
    + global PROC

becoming:

    discover_phase_dates(proc, expected_dates)

No GACOS correction sign, interpolation, incidence convention,
reference convention, or numerical kernel is changed here.
"""

from pathlib import Path
import json
import re

import numpy as np
from numba import njit, prange


DATE_RE = re.compile(
    r"^\d{8}$"
)


def par_scalar(
    path,
    key,
):

    rx = re.compile(
        r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
    )

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        if (
            k.strip().lower()
            ==
            key.lower()
        ):

            m = rx.search(
                v
            )

            if m:

                return float(
                    m.group(0)
                )


    raise KeyError(
        key
    )


def read_rsc(
    path,
):

    d = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        p = line.strip().split()

        if (
            len(p) >= 2
            and
            not line.lstrip().startswith("#")
        ):

            d[
                p[0].upper()
            ] = p[1]


    return d


def normalize_dates(
    x,
):

    a = np.asarray(
        x
    ).reshape(
        -1
    )

    out = []


    for v in a:

        if isinstance(
            v,
            (
                bytes,
                np.bytes_,
            ),
        ):

            s = bytes(
                v
            ).decode(
                errors="ignore"
            )

        else:

            s = str(
                v
            )


        s = re.sub(
            r"[^0-9]",
            "",
            s,
        )


        if len(s) >= 8:

            s = s[:8]


        if not DATE_RE.match(
            s
        ):

            return None


        out.append(
            s
        )


    return out


def discover_phase_dates(proc, expected_dates):
    expected_set = set(expected_dates)
    hits = []
    candidates = sorted(proc.rglob('*date*.npy')) + sorted(proc.rglob('*dates*.npy'))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            vals = normalize_dates(np.load(p, allow_pickle=False))
        except Exception:
            continue
        if vals and len(vals) == len(expected_dates) and (set(vals) == expected_set):
            hits.append((str(p), vals))

    def walk(obj, path=''):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f'{path}.{k}' if path else str(k)
                if isinstance(v, list):
                    vals = normalize_dates(v)
                    if vals and len(vals) == len(expected_dates) and (set(vals) == expected_set):
                        found.append((kp, vals))
                found.extend(walk(v, kp))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(walk(v, f'{path}[{i}]'))
        return found
    for p in sorted(proc.rglob('*.json')):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        for keypath, vals in walk(obj):
            hits.append((f'{p}:{keypath}', vals))
    if not hits:
        raise RuntimeError('Cannot prove acquisition-date order. No date sequence matching the GACOS set was found.')
    orders = {}
    for src, vals in hits:
        orders.setdefault(tuple(vals), []).append(src)
    if len(orders) != 1:
        raise RuntimeError('Conflicting acquisition-date orders found.')
    order = list(next(iter(orders.keys())))
    sources = next(iter(orders.values()))
    return (order, sources)

@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def sample_dlos_block(
    ztd,
    base,
    fx,
    fy,
    sec_inc,
    width,
    ref_epoch,
    out,
):

    n = base.size

    ne = ztd.shape[1]


    for k in prange(
        n
    ):

        b = base[k]

        x = fx[k]

        y = fy[k]


        w00 = (
            (1.0 - x)
            *
            (1.0 - y)
        )

        w01 = (
            x
            *
            (1.0 - y)
        )

        w10 = (
            (1.0 - x)
            *
            y
        )

        w11 = (
            x
            *
            y
        )


        s = sec_inc[k]


        z0 = (
            w00
            *
            ztd[
                b,
                ref_epoch,
            ]

            +

            w01
            *
            ztd[
                b + 1,
                ref_epoch,
            ]

            +

            w10
            *
            ztd[
                b + width,
                ref_epoch,
            ]

            +

            w11
            *
            ztd[
                b + width + 1,
                ref_epoch,
            ]
        ) * s


        for e in range(
            ne
        ):

            ze = (
                w00
                *
                ztd[
                    b,
                    e,
                ]

                +

                w01
                *
                ztd[
                    b + 1,
                    e,
                ]

                +

                w10
                *
                ztd[
                    b + width,
                    e,
                ]

                +

                w11
                *
                ztd[
                    b + width + 1,
                    e,
                ]
            ) * s


            out[
                k,
                e,
            ] = (
                ze
                -
                z0
            )


@njit(
    parallel=True,
    fastmath=False,
    cache=True,
)
def apply_phase_block(
    src_block,
    dlos,
    dlos_ref_med,
    final_ref_phase_offset,
    phase_factor,
    out,
):

    n, ne = src_block.shape


    for k in prange(
        n
    ):

        for e in range(
            ne
        ):

            corr = (
                phase_factor
                *
                (
                    dlos[
                        k,
                        e,
                    ]
                    -
                    dlos_ref_med[e]
                )
            )


            out[
                k,
                e,
            ] = (
                src_block[
                    k,
                    e,
                ]
                +
                corr
                -
                final_ref_phase_offset[e]
            )


__all__ = [
    "par_scalar",
    "read_rsc",
    "normalize_dates",
    "discover_phase_dates",
    "sample_dlos_block",
    "apply_phase_block",
]
