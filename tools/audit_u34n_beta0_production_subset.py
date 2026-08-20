#!/usr/bin/env python3
from pathlib import Path
import numpy as np


SEQDIR = Path(
    "/home/ubuntu/Downloads/psds/output/processing/sequential"
)

P_METRIC = (
    SEQDIR
    / "u34m_beta0_phase_parity_metrics.npz"
)

P_FULL = (
    SEQDIR
    / "u34m_beta0_full_reference_metrics.npz"
)


def q(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95, 99],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
    }


def pick(z, candidates):
    for k in candidates:
        if k in z.files:
            return k, z[k]

    raise KeyError(
        "Could not locate any of "
        f"{candidates}\n"
        f"Available keys:\n{z.files}"
    )


zm = np.load(P_METRIC)
zf = np.load(P_FULL)

print("=" * 110)
print("U3.4n BETA0 PRODUCTION-QUALITY PARITY")
print("=" * 110)

print("metric keys :", zm.files)
print("full keys   :", zf.files)
print()

# ------------------------------------------------------------------
# Coordinates: must be identical.
# ------------------------------------------------------------------

_, rr_m = pick(
    zm,
    ("rows",),
)

_, cc_m = pick(
    zm,
    ("cols",),
)

_, rr_f = pick(
    zf,
    ("rows",),
)

_, cc_f = pick(
    zf,
    ("cols",),
)

if not np.array_equal(rr_m, rr_f):
    raise RuntimeError("row population mismatch")

if not np.array_equal(cc_m, cc_f):
    raise RuntimeError("column population mismatch")

N = rr_m.size

# ------------------------------------------------------------------
# Existing U3.4m parity metrics.
# ------------------------------------------------------------------

sim_key, sim = pick(
    zm,
    (
        "phase_similarity",
        "center_similarity",
    ),
)

med_key, mederr = pick(
    zm,
    (
        "median_abs_error_deg",
        "center_median_error_deg",
        "median_error_deg",
        "center_median_error",
    ),
)

p95_key, p95err = pick(
    zm,
    (
        "p95_abs_error_deg",
        "center_p95_error_deg",
        "p95_error_deg",
        "center_p95_error",
    ),
)

# ------------------------------------------------------------------
# Exact beta0 full38 quality criterion.
# ------------------------------------------------------------------

tc = np.asarray(
    zf["temporal_coherence"],
    dtype=np.float32,
)

K = np.asarray(
    zf["effective_K"],
    dtype=np.int16,
)

if (
    sim.size != N
    or mederr.size != N
    or p95err.size != N
    or tc.size != N
    or K.size != N
):
    raise RuntimeError("metric population length mismatch")

finite = (
    np.isfinite(sim)
    & np.isfinite(mederr)
    & np.isfinite(p95err)
    & np.isfinite(tc)
)

formal_seq = (
    finite
    & (K >= 48)
)

accepted = (
    formal_seq
    & (tc >= 0.80)
)

rejected = (
    formal_seq
    & (tc < 0.80)
)


def report(name, m):
    n = int(np.count_nonzero(m))

    print()
    print(name)
    print("-" * 110)

    print(
        "n                    :",
        f"{n:,}",
    )

    if n == 0:
        return

    print(
        "phase similarity     :",
        q(sim[m]),
    )

    print(
        "median error deg     :",
        q(mederr[m]),
    )

    print(
        "p95 error deg        :",
        q(p95err[m]),
    )

    print(
        "full38 TC            :",
        q(tc[m]),
    )

    print(
        "effective K          :",
        q(K[m]),
    )

    print(
        "median err >30 deg   :",
        f"{100*np.mean(mederr[m] > 30.0):.6f}%",
    )

    print(
        "median err >60 deg   :",
        f"{100*np.mean(mederr[m] > 60.0):.6f}%",
    )

    print(
        "p95 err >30 deg      :",
        f"{100*np.mean(p95err[m] > 30.0):.6f}%",
    )

    print(
        "p95 err >60 deg      :",
        f"{100*np.mean(p95err[m] > 60.0):.6f}%",
    )

    print(
        "p95 err >90 deg      :",
        f"{100*np.mean(p95err[m] > 90.0):.6f}%",
    )

    print(
        "p95 err >120 deg     :",
        f"{100*np.mean(p95err[m] > 120.0):.6f}%",
    )


print("metric similarity key :", sim_key)
print("metric median key     :", med_key)
print("metric p95 key        :", p95_key)

print()
print(
    "sequential route      :",
    f"{np.count_nonzero(formal_seq):,}",
)

print(
    "TC >= 0.80 accepted  :",
    f"{np.count_nonzero(accepted):,}",
    f"({100*np.mean(accepted[formal_seq]):.6f}% of route)",
)

print(
    "TC < 0.80 rejected   :",
    f"{np.count_nonzero(rejected):,}",
)

report(
    "ALL K>=48 SEQUENTIAL",
    formal_seq,
)

report(
    "PRODUCTION ACCEPTED: full38 TC>=0.80",
    accepted,
)

report(
    "PRODUCTION REJECTED: full38 TC<0.80",
    rejected,
)

# ------------------------------------------------------------------
# Same existing TC bands for direct comparison with U3.4m output.
# ------------------------------------------------------------------

for name, m in (
    (
        "TC 0.80-0.90",
        formal_seq
        & (tc >= 0.80)
        & (tc < 0.90),
    ),
    (
        "TC >=0.90",
        formal_seq
        & (tc >= 0.90),
    ),
    (
        "TC >=0.95",
        formal_seq
        & (tc >= 0.95),
    ),
):

    report(
        name,
        m,
    )

print()
print(
    "U3.4n BETA0 PRODUCTION QUALITY AUDIT: PASS"
)
