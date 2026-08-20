#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit, prange


SEQDIR = Path(
    "/home/ubuntu/Downloads/psds/output/processing/sequential"
)

METRICS = (
    SEQDIR
    / "u35b_m19_fullspan_tc_metrics.npz"
)

PHASE_PATH = (
    SEQDIR
    / "u34m_beta0_sequential_phase_points.npy"
)

SEARCH_RADIUS = 11


def q(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return {}

    v = np.percentile(
        x,
        [0, 5, 25, 50, 75, 95, 99, 100],
    )

    names = (
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "p99",
        "max",
    )

    return {
        k: float(y)
        for k, y in zip(names, v)
    }


def circle_offsets(radius):
    """
    Simple audit-only circular search neighborhood.

    Equivalent intent to Dolphin spatial phase similarity:
    compare each phase history to surrounding valid pixels.
    """

    out = []

    r2 = radius * radius

    for dr in range(
        -radius,
        radius + 1,
    ):
        for dc in range(
            -radius,
            radius + 1,
        ):

            if (
                dr == 0
                and
                dc == 0
            ):
                continue

            if (
                dr * dr
                +
                dc * dc
                <= r2
            ):
                out.append(
                    (dr, dc)
                )

    return np.asarray(
        out,
        dtype=np.int16,
    )


@njit(
    cache=True,
    parallel=True,
    nogil=True,
)
def spatial_median_similarity(
    rows,
    cols,
    phase,
    point_id,
    offsets,
):
    """
    Dolphin-style phase-history similarity on the available
    sequential-route point population.

    For center p and neighboring point q:

        similarity =
            mean_t Re(
                phase[p,t] * conj(phase[q,t])
            )

    Final metric = median over spatial neighbors.
    """

    n = rows.size
    ndate = phase.shape[1]

    H, W = point_id.shape

    out = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    neighbor_count = np.zeros(
        n,
        dtype=np.int16,
    )

    noff = offsets.shape[0]

    for p in prange(n):

        r0 = rows[p]
        c0 = cols[p]

        vals = np.empty(
            noff,
            dtype=np.float32,
        )

        nv = 0

        for k in range(noff):

            rr = (
                r0
                +
                offsets[k, 0]
            )

            cc = (
                c0
                +
                offsets[k, 1]
            )

            if (
                rr < 0
                or
                rr >= H
                or
                cc < 0
                or
                cc >= W
            ):
                continue

            j = point_id[
                rr,
                cc,
            ]

            if j < 0:
                continue

            s = 0.0
            good = True

            for t in range(ndate):

                a = phase[
                    p,
                    t,
                ]

                b = phase[
                    j,
                    t,
                ]

                if not (
                    np.isfinite(a.real)
                    and
                    np.isfinite(a.imag)
                    and
                    np.isfinite(b.real)
                    and
                    np.isfinite(b.imag)
                ):
                    good = False
                    break

                # real(a * conj(b))
                s += (
                    a.real * b.real
                    +
                    a.imag * b.imag
                )

            if not good:
                continue

            vals[nv] = (
                s
                /
                ndate
            )

            nv += 1

        neighbor_count[p] = nv

        if nv > 0:
            out[p] = np.median(
                vals[:nv]
            )

    return (
        out,
        neighbor_count,
    )


def report(
    name,
    mask,
    sim,
    nnb,
    p95err,
):
    n = int(
        np.count_nonzero(
            mask
        )
    )

    print()
    print("=" * 96)
    print(name)
    print("=" * 96)

    print(
        "n                       :",
        f"{n:,}",
    )

    if n == 0:
        return

    print(
        "spatial similarity      :",
        q(
            sim[
                mask
            ]
        ),
    )

    print(
        "neighbor count          :",
        q(
            nnb[
                mask
            ]
        ),
    )

    print(
        "p95 error deg           :",
        q(
            p95err[
                mask
            ]
        ),
    )


def main():

    for p in (
        METRICS,
        PHASE_PATH,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                p
            )

    z = np.load(
        METRICS
    )

    rr = np.asarray(
        z["rows"],
        dtype=np.int32,
    )

    cc = np.asarray(
        z["cols"],
        dtype=np.int32,
    )

    full_accept = np.asarray(
        z["full_accept"],
        dtype=np.bool_,
    )

    seq_accept = np.asarray(
        z["sequential_accept"],
        dtype=np.bool_,
    )

    p95err = np.asarray(
        z["p95_abs_error_deg"],
        dtype=np.float32,
    )

    phase = np.load(
        PHASE_PATH,
        mmap_mode="r",
    )

    n = rr.size

    if phase.shape[0] != n:
        raise RuntimeError(
            "phase population mismatch"
        )

    H = int(
        rr.max()
        +
        1
    )

    W = int(
        cc.max()
        +
        1
    )

    # Current benchmark scene is known from coordinates.
    # Preserve actual required extent.
    H = max(
        H,
        600,
    )

    W = max(
        W,
        2000,
    )

    point_id = np.full(
        (H, W),
        -1,
        dtype=np.int32,
    )

    point_id[
        rr,
        cc,
    ] = np.arange(
        n,
        dtype=np.int32,
    )

    offsets = circle_offsets(
        SEARCH_RADIUS
    )

    print(
        "=" * 104
    )

    print(
        "U3.5d M19 DOLPHIN-STYLE "
        "SPATIAL PHASE-HISTORY SIMILARITY"
    )

    print(
        "=" * 104
    )

    print(
        "points                 :",
        f"{n:,}",
    )

    print(
        "dates                  :",
        phase.shape[1],
    )

    print(
        "search radius          :",
        SEARCH_RADIUS,
    )

    print(
        "candidate offsets      :",
        offsets.shape[0],
    )

    t0 = time.perf_counter()

    sim, nnb = spatial_median_similarity(
        rr,
        cc,
        phase,
        point_id,
        offsets,
    )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    both = (
        full_accept
        &
        seq_accept
    )

    seq_only = (
        seq_accept
        &
        ~full_accept
    )

    full_only = (
        full_accept
        &
        ~seq_accept
    )

    neither = (
        ~full_accept
        &
        ~seq_accept
    )

    print()
    print(
        "elapsed                :",
        f"{elapsed:.3f} s",
    )

    print(
        "finite similarity      :",
        f"{np.count_nonzero(np.isfinite(sim)):,}",
    )

    print(
        "similarity all         :",
        q(sim),
    )

    print(
        "neighbors all          :",
        q(nnb),
    )

    report(
        "BOTH ACCEPTED",
        both,
        sim,
        nnb,
        p95err,
    )

    report(
        "SEQ-ONLY / PROBLEM POPULATION",
        seq_only,
        sim,
        nnb,
        p95err,
    )

    report(
        "FULL38-ONLY",
        full_only,
        sim,
        nnb,
        p95err,
    )

    report(
        "BOTH REJECTED",
        neither,
        sim,
        nnb,
        p95err,
    )

    out = (
        SEQDIR
        /
        "u35d_m19_spatial_similarity_metrics.npz"
    )

    np.savez_compressed(
        out,
        rows=rr,
        cols=cc,
        spatial_similarity=sim,
        neighbor_count=nnb,
        full_accept=full_accept,
        sequential_accept=seq_accept,
        p95_abs_error_deg=p95err,
    )

    print()
    print(
        "metrics                :",
        out,
    )

    print()
    print(
        "U3.5d COMPUTATIONAL INTEGRITY: PASS"
    )

    print(
        "U3.5d SCIENTIFIC DECISION: "
        "PENDING OBSERVED SEPARATION"
    )


if __name__ == "__main__":
    main()
