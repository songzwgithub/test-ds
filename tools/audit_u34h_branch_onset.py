#!/usr/bin/env python3

from pathlib import Path
import numpy as np


SEQ = Path(
    "/home/ubuntu/Downloads/psds/output/"
    "processing/sequential"
)

PROC = SEQ.parent


def phase_stats(a, b):
    """
    Compare two phase vectors after removing the
    common phase of the first date in the subset.
    """

    a = np.asarray(a, dtype=np.complex64)
    b = np.asarray(b, dtype=np.complex64)

    ar = (
        a
        *
        np.conj(a[:, :1])
    )

    br = (
        b
        *
        np.conj(b[:, :1])
    )

    d = (
        ar
        *
        np.conj(br)
    )

    sim = np.abs(
        np.mean(
            d,
            axis=1,
        )
    )

    err = (
        np.abs(
            np.angle(d)
        )
        *
        180.0
        /
        np.pi
    )

    med = np.median(
        err,
        axis=1,
    )

    p95 = np.percentile(
        err,
        95,
        axis=1,
    )

    return (
        sim.astype(np.float32),
        med.astype(np.float32),
        p95.astype(np.float32),
    )


def q(x):
    return (
        float(np.median(x)),
        float(np.percentile(x, 95)),
    )


# ------------------------------------------------------------
# M16 sequential point-domain phase
# ------------------------------------------------------------

seq_phase = np.load(
    SEQ / "u34a_sequential_phase_points.npy",
    mmap_mode="r",
)

metrics = np.load(
    SEQ / "u34c_M16_phase_metrics.npz"
)

rr = metrics["rows"]
cc = metrics["cols"]

if seq_phase.shape != (
    rr.size,
    38,
):
    raise RuntimeError(
        f"Unexpected seq phase shape {seq_phase.shape}"
    )


# ------------------------------------------------------------
# Original 38-date full-SCM phase
# ------------------------------------------------------------

full = np.load(
    PROC / "linked_phase.npy",
    mmap_mode="r",
)


# ------------------------------------------------------------
# Catastrophic ids from U3.4g
# ------------------------------------------------------------

g = np.load(
    SEQ / "u34g_support_matched_full_scm.npz"
)

audit_ids = g["audit_ids"]
is_cat = g["is_catastrophic"]

cat_ids = audit_ids[
    is_cat
]

control_ids = audit_ids[
    ~g["is_false_accept"]
]


print("=" * 120)
print("U3.4h M16 SEQUENTIAL BRANCH-ONSET AUDIT")
print("=" * 120)

print(
    "catastrophic points :",
    f"{cat_ids.size:,}",
)

print(
    "control points      :",
    f"{control_ids.size:,}",
)

print()


stage_ranges = [
    ("stage0", 0, 16),
    ("stage1", 16, 32),
    ("stage2", 32, 38),
]


def run_group(name, ids):

    r = rr[ids]
    c = cc[ids]

    sp = np.asarray(
        seq_phase[ids]
    )

    fp = np.column_stack(
        [
            np.asarray(
                full[d, r, c]
            )
            for d in range(38)
        ]
    )

    print()
    print(name)
    print("-" * 120)

    print(
        "stage   dates     sim50    sim05    "
        "medErr50  medErr95   p95Err50   p95Err95"
    )

    for stage, a, b in stage_ranges:

        sim, med, p95 = phase_stats(
            sp[:, a:b],
            fp[:, a:b],
        )

        print(
            f"{stage:<7s} "
            f"{a:02d}:{b:02d} "
            f"{np.median(sim):8.4f} "
            f"{np.percentile(sim,5):8.4f} "
            f"{np.median(med):10.3f} "
            f"{np.percentile(med,95):10.3f} "
            f"{np.median(p95):11.3f} "
            f"{np.percentile(p95,95):11.3f}"
        )

    # --------------------------------------------------------
    # Cumulative chronology
    # --------------------------------------------------------

    print()
    print(
        "cumulative chronology"
    )

    print("-" * 120)

    for b in (
        16,
        32,
        38,
    ):

        sim, med, p95 = phase_stats(
            sp[:, :b],
            fp[:, :b],
        )

        print(
            f"0:{b:<2d} "
            f"sim50={np.median(sim):.4f} "
            f"medErr50={np.median(med):.3f} "
            f"p95Err50={np.median(p95):.3f}"
        )


run_group(
    "CATASTROPHIC FALSE ACCEPT",
    cat_ids,
)

run_group(
    "STABLE CONTROL",
    control_ids,
)

print()
print("=" * 120)
print("U3.4h BRANCH-ONSET AUDIT: PASS")
print("=" * 120)
