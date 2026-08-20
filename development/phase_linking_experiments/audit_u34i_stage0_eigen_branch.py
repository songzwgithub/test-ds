#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    image_pairs,
    uncompress_coherence,
)
from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


PROC = Path(
    "/home/ubuntu/Downloads/psds/output/processing"
)
SEQ = PROC / "sequential"

MU = 0.99
BATCH = 1024


def bool_windows(x, hr, hc):

    x = np.asarray(x, dtype=np.bool_)

    p = np.pad(
        x,
        ((hr, hr), (hc, hc)),
        mode="constant",
        constant_values=False,
    )

    return np.lib.stride_tricks.sliding_window_view(
        p,
        (2 * hr + 1, 2 * hc + 1),
    )


def ref_phase_from_vec(v):

    ph = np.exp(
        1j * np.angle(v)
    ).astype(np.complex64)

    ph *= np.conj(
        ph[:, 0:1]
    )

    return ph


def similarity(a, b):

    a = np.asarray(a, dtype=np.complex64)
    b = np.asarray(b, dtype=np.complex64)

    a = (
        a
        *
        np.conj(a[:, 0:1])
    )

    b = (
        b
        *
        np.conj(b[:, 0:1])
    )

    return np.abs(
        np.mean(
            a * np.conj(b),
            axis=1,
        )
    ).astype(np.float32)


def q(x):

    x = np.asarray(x)
    x = x[np.isfinite(x)]

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
    }


# ------------------------------------------------------------------
# Point population
# ------------------------------------------------------------------

g = np.load(
    SEQ / "u34g_support_matched_full_scm.npz"
)

audit_ids = g["audit_ids"]
is_cat = g["is_catastrophic"]
is_false = g["is_false_accept"]

cat_point_ids = audit_ids[
    is_cat
]

control_point_ids = audit_ids[
    ~is_false
]

point_ids = np.concatenate(
    [
        cat_point_ids,
        control_point_ids,
    ]
)

is_cat_sel = np.zeros(
    point_ids.size,
    dtype=np.bool_,
)

is_cat_sel[
    :cat_point_ids.size
] = True

is_control_sel = ~is_cat_sel


# ------------------------------------------------------------------
# Coordinates / existing phases
# ------------------------------------------------------------------

metrics = np.load(
    SEQ / "u34c_M16_phase_metrics.npz"
)

rr_all = metrics["rows"]
cc_all = metrics["cols"]

rr = rr_all[
    point_ids
]

cc = cc_all[
    point_ids
]

seq_phase = np.load(
    SEQ / "u34a_sequential_phase_points.npy",
    mmap_mode="r",
)

stage0_actual = np.asarray(
    seq_phase[
        point_ids,
        :16,
    ]
)

full_phase = np.load(
    PROC / "linked_phase.npy",
    mmap_mode="r",
)

full16 = np.column_stack(
    [
        np.asarray(
            full_phase[
                d,
                rr,
                cc,
            ]
        )
        for d in range(16)
    ]
)


# ------------------------------------------------------------------
# Spatial support inputs
# ------------------------------------------------------------------

yxt = np.load(
    PROC / "cache/phase_corrected_yxt.npy",
    mmap_mode="r",
)

scale2 = np.load(
    PROC / "ds_statistics/rayleigh_scale2.npy",
    mmap_mode="r",
)

raw_valid = np.load(
    PROC / "ds_statistics/raw_valid.npy",
    mmap_mode="r",
)

geom = np.load(
    PROC / "cache/phase_geometry_valid.npy",
    mmap_mode="r",
)

ps = np.load(
    PROC / "ps_mask.npy",
    mmap_mode="r",
)

core = np.load(
    SEQ / "compression_state_core_K24.npy",
    mmap_mode="r",
)

stage0_estimator_map = np.load(
    SEQ / "u33b_stage0000_estimator.npy",
    mmap_mode="r",
)

actual_estimator = np.asarray(
    stage0_estimator_map[
        rr,
        cc,
    ]
)

valid = (
    np.asarray(raw_valid, dtype=np.bool_)
    &
    np.asarray(geom, dtype=np.bool_)
)

ctx = prepare_glrt_window_context(
    scale2,
    valid,
    np.asarray(ps, dtype=np.bool_),
    half_row=5,
    half_col=11,
)

core_windows = bool_windows(
    core,
    5,
    11,
)

pairs = image_pairs(16)

pi = pairs[:, 0]
pj = pairs[:, 1]


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

N = point_ids.size

d1 = np.full(N, np.nan, np.float32)
d2 = np.full(N, np.nan, np.float32)
distance_gap = np.full(N, np.nan, np.float32)

selected_sim = np.full(N, np.nan, np.float32)
second_sim = np.full(N, np.nan, np.float32)
best_sim = np.full(N, np.nan, np.float32)

best_rank = np.full(
    N,
    -1,
    np.int16,
)

evd_sim = np.full(N, np.nan, np.float32)

gamma_min = np.full(N, np.nan, np.float32)
gamma_condition = np.full(N, np.nan, np.float32)

actual_reconstructed_sim = np.full(
    N,
    np.nan,
    np.float32,
)


t0 = perf_counter()


for b0 in range(
    0,
    N,
    BATCH,
):

    b1 = min(
        N,
        b0 + BATCH,
    )

    br = rr[b0:b1]
    bc = cc[b0:b1]

    support, _ = (
        glrt_support_vectorized_exact(
            ctx,
            br,
            bc,
            alpha=0.005,
            nslc=38,
            block_size=1024,
        )
    )

    support &= np.asarray(
        core_windows[
            br,
            bc,
        ],
        dtype=np.bool_,
    )

    coh = compressed_coherence(
        yxt,
        br,
        bc,
        support,
        pi,
        pj,
    )

    C = uncompress_coherence(
        coh,
        16,
        pairs,
    ).astype(
        np.complex128,
        copy=False,
    )

    nb = C.shape[0]

    # ----------------------------------------------------------
    # Exact production EMI matrix
    # ----------------------------------------------------------

    eye = np.eye(
        16,
        dtype=np.float64,
    )

    Gamma = np.abs(C).real

    Gamma = (
        0.95 * Gamma
        +
        0.05 * eye[None, :, :]
    )

    Gamma += (
        1e-6
        *
        eye[None, :, :]
    )

    Gamma = 0.5 * (
        Gamma
        +
        np.swapaxes(
            Gamma,
            -1,
            -2,
        )
    )

    gw, gv = np.linalg.eigh(
        Gamma
    )

    gamma_min[
        b0:b1
    ] = gw[:, 0].astype(
        np.float32
    )

    gamma_condition[
        b0:b1
    ] = (
        gw[:, -1]
        /
        gw[:, 0]
    ).astype(
        np.float32
    )

    Gamma_inv = np.einsum(
        "bik,bk,bjk->bij",
        gv,
        1.0 / gw,
        gv,
        optimize=True,
    )

    A = Gamma_inv * C

    A = 0.5 * (
        A
        +
        np.swapaxes(
            A.conj(),
            -1,
            -2,
        )
    )

    ew, ev = np.linalg.eigh(
        A
    )

    dist = np.abs(
        ew.real
        -
        MU
    )

    order = np.argsort(
        dist,
        axis=1,
    )

    idx1 = order[:, 0]
    idx2 = order[:, 1]

    row = np.arange(nb)

    d1[
        b0:b1
    ] = dist[
        row,
        idx1,
    ].astype(np.float32)

    d2[
        b0:b1
    ] = dist[
        row,
        idx2,
    ].astype(np.float32)

    distance_gap[
        b0:b1
    ] = (
        dist[row, idx2]
        -
        dist[row, idx1]
    ).astype(np.float32)

    # ----------------------------------------------------------
    # Compare every EMI eigenvector to 38-date solution,
    # restricted to the first 16 acquisitions.
    # ----------------------------------------------------------

    cand = np.exp(
        1j
        *
        np.angle(ev)
    ).astype(np.complex64)

    # ev dimensions:
    # [point, acquisition, eigenvector]

    cand *= np.conj(
        cand[
            :,
            0:1,
            :
        ]
    )

    target = full16[
        b0:b1
    ].copy()

    target *= np.conj(
        target[:, 0:1]
    )

    sim_all = np.abs(
        np.mean(
            cand
            *
            np.conj(
                target[:, :, None]
            ),
            axis=1,
        )
    )

    selected_sim[
        b0:b1
    ] = sim_all[
        row,
        idx1,
    ].astype(np.float32)

    second_sim[
        b0:b1
    ] = sim_all[
        row,
        idx2,
    ].astype(np.float32)

    best_idx = np.argmax(
        sim_all,
        axis=1,
    )

    best_sim[
        b0:b1
    ] = sim_all[
        row,
        best_idx,
    ].astype(np.float32)

    # Rank of the full38-best eigenvector
    # in |lambda - mu| ordering.
    inverse_rank = np.empty_like(
        order
    )

    inverse_rank[
        row[:, None],
        order,
    ] = np.arange(
        16,
        dtype=np.int64,
    )[None, :]

    best_rank[
        b0:b1
    ] = (
        inverse_rank[
            row,
            best_idx,
        ]
        +
        1
    ).astype(np.int16)

    # ----------------------------------------------------------
    # EVD candidate
    # ----------------------------------------------------------

    B = (
        C
        *
        np.abs(C)
    )

    B = 0.5 * (
        B
        +
        np.swapaxes(
            B.conj(),
            -1,
            -2,
        )
    )

    bw, bv = np.linalg.eigh(
        B
    )

    evd_vec = bv[
        :,
        :,
        -1,
    ]

    evd_phase = ref_phase_from_vec(
        evd_vec
    )

    evd_sim[
        b0:b1
    ] = similarity(
        evd_phase,
        target,
    )

    # ----------------------------------------------------------
    # Verify eigenspectrum reconstruction against actual
    # stage0 production output.
    # ----------------------------------------------------------

    emi_vec = ev[
        row,
        :,
        idx1,
    ]

    emi_phase = ref_phase_from_vec(
        emi_vec
    )

    reconstructed = emi_phase.copy()

    evd_mask = (
        actual_estimator[
            b0:b1
        ]
        ==
        ESTIMATOR_EVD
    )

    reconstructed[
        evd_mask
    ] = evd_phase[
        evd_mask
    ]

    actual_reconstructed_sim[
        b0:b1
    ] = similarity(
        reconstructed,
        stage0_actual[
            b0:b1
        ],
    )

    elapsed = (
        perf_counter()
        -
        t0
    )

    print(
        f"{b1:7,d}/{N:7,d} "
        f"({100*b1/N:6.2f}%) "
        f"rate={b1/elapsed:,.0f} center/s"
    )


# ------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------

def report_group(
    name,
    mask,
):

    print()
    print(name)
    print("-" * 105)

    print(
        "n                         :",
        f"{mask.sum():,}",
    )

    print(
        "selected/full38 sim       :",
        q(selected_sim[mask]),
    )

    print(
        "second/full38 sim         :",
        q(second_sim[mask]),
    )

    print(
        "best eigenvector sim      :",
        q(best_sim[mask]),
    )

    print(
        "|lambda1-mu|              :",
        q(d1[mask]),
    )

    print(
        "|lambda2-mu|              :",
        q(d2[mask]),
    )

    print(
        "nearest-distance gap      :",
        q(distance_gap[mask]),
    )

    print(
        "Gamma condition           :",
        q(gamma_condition[mask]),
    )

    print(
        "EVD/full38 sim            :",
        q(evd_sim[mask]),
    )

    print(
        "best rank = 1             :",
        f"{100*np.mean(best_rank[mask] == 1):.3f}%",
    )

    print(
        "best rank = 2             :",
        f"{100*np.mean(best_rank[mask] == 2):.3f}%",
    )

    print(
        "best rank <= 3            :",
        f"{100*np.mean(best_rank[mask] <= 3):.3f}%",
    )

    print(
        "2nd better by >0.05       :",
        f"{100*np.mean(second_sim[mask] > selected_sim[mask] + 0.05):.3f}%",
    )

    print(
        "some eigenvector >=0.99   :",
        f"{100*np.mean(best_sim[mask] >= 0.99):.3f}%",
    )

    print(
        "stage0 estimator EMI      :",
        f"{np.count_nonzero(actual_estimator[mask] == ESTIMATOR_EMI):,}",
    )

    print(
        "stage0 estimator EVD      :",
        f"{np.count_nonzero(actual_estimator[mask] == ESTIMATOR_EVD):,}",
    )


print()
print("=" * 105)
print("U3.4i STAGE0 EMI EIGEN-BRANCH AUDIT")
print("=" * 105)

print(
    "actual/reconstructed parity:",
    q(
        actual_reconstructed_sim
    ),
)

if np.min(
    actual_reconstructed_sim
) < 0.9999:

    raise RuntimeError(
        "Stage0 eigenspectrum reconstruction parity failed"
    )


report_group(
    "CATASTROPHIC",
    is_cat_sel,
)

report_group(
    "STABLE CONTROL",
    is_control_sel,
)


out = (
    SEQ
    /
    "u34i_stage0_eigen_branch.npz"
)

np.savez_compressed(
    out,

    point_ids=point_ids,
    is_catastrophic=is_cat_sel,

    d1=d1,
    d2=d2,
    distance_gap=distance_gap,

    selected_similarity=selected_sim,
    second_similarity=second_sim,
    best_similarity=best_sim,

    best_rank=best_rank,

    evd_similarity=evd_sim,

    gamma_min=gamma_min,
    gamma_condition=gamma_condition,

    actual_reconstructed_similarity=
        actual_reconstructed_sim,

    estimator=actual_estimator,
)

print()
print("output:", out)
print()
print("U3.4i EIGEN-BRANCH AUDIT: PASS")
