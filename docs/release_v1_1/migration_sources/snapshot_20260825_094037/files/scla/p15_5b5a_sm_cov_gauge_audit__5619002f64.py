from pathlib import Path
import json
import re

import numpy as np


ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"

GMAN = (
    PROC
    / "gacos_corrected_phase"
    / "gacos_correction_manifest.json"
)

NETWORK_LOG_DIR = (
    PROC
    / "batch_unwrap_validation"
    / "logs"
)

CURRENT_COV = (
    PROC
    / "stamps_scla_final_pass"
    / "sm_cov_unit_ifg_geom_master.npy"
)

OLD_COV = (
    PSDS
    / "prototype_outputs/v09/scla_v09/"
      "pystamps_bridge/r4a_stage7_contract/"
      "bridge_sm_cov_unit_ifg.npy"
)

OUT = (
    PROC
    / "stamps_scla_final_pass"
)

REPORT = (
    OUT
    / "p15_5b5a_sm_cov_gauge_audit.json"
)

OLD_TO_GEOM_OUT = (
    OUT
    / "prototype_sm_cov_transformed_to_geom_master.npy"
)


GEOM_MASTER = "20151212"
TOL = 1e-10


# ================================================================
# Input
# ================================================================

for p in (
    GMAN,
    CURRENT_COV,
    OLD_COV,
):
    if not p.is_file():
        raise FileNotFoundError(p)


gman = json.loads(
    GMAN.read_text()
)

dates = list(
    gman["acquisition_dates"]
)


if len(dates) != 38:
    raise RuntimeError(
        f"expected 38 dates, got {len(dates)}"
    )


geom0 = dates.index(
    GEOM_MASTER
)


Ccur = np.load(
    CURRENT_COV
).astype(
    np.float64
)


Cold = np.load(
    OLD_COV
).astype(
    np.float64
)


if (
    Ccur.shape != (38, 38)
    or
    Cold.shape != (38, 38)
):
    raise RuntimeError(
        f"covariance shapes: current={Ccur.shape}, old={Cold.shape}"
    )


# ================================================================
# Current production 108-IFG graph
# ================================================================

pat = re.compile(
    r"pair(\d+)_"
    r"(20\d{6})_"
    r"(20\d{6})_"
    r"single_ifg\.log$"
)


pairs = []

for p in NETWORK_LOG_DIR.glob(
    "pair*_single_ifg.log"
):

    m = pat.match(p.name)

    if m:
        pairs.append(
            (
                int(m.group(1)),
                m.group(2),
                m.group(3),
            )
        )


pairs.sort()


if (
    len(pairs) != 108
    or [x[0] for x in pairs]
    != list(range(1, 109))
):
    raise RuntimeError(
        "108-IFG network ordering failed"
    )


dix = {
    d: i
    for i, d in enumerate(dates)
}


G = np.zeros(
    (108, 38),
    dtype=np.float64,
)


for e, (_, d1, d2) in enumerate(pairs):

    G[e, dix[d1]] = -1.0
    G[e, dix[d2]] = +1.0


if np.linalg.matrix_rank(G) != 37:
    raise RuntimeError(
        f"full graph rank={np.linalg.matrix_rank(G)}"
    )


# ================================================================
# Unit-IFG covariance for an arbitrary fixed reference
#
# y = G theta + eps
# Cov(eps) = I
#
# theta_ref = 0
# Cov(theta_nonref) = inv(G_r' G_r)
# ================================================================

def covariance_for_reference(ref0):

    keep = np.asarray(
        [
            i
            for i in range(38)
            if i != ref0
        ],
        dtype=np.int64,
    )

    Gr = G[:, keep]

    rank = int(
        np.linalg.matrix_rank(Gr)
    )

    if rank != 37:
        raise RuntimeError(
            f"reference {ref0} rank={rank}"
        )

    Cred = np.linalg.inv(
        Gr.T @ Gr
    )

    C = np.zeros(
        (38, 38),
        dtype=np.float64,
    )

    C[np.ix_(keep, keep)] = Cred

    return C


# ================================================================
# 1. Verify CURRENT covariance corresponds exactly to geom master
# ================================================================

Cgeom_direct = covariance_for_reference(
    geom0
)


current_direct_diff = (
    Ccur
    -
    Cgeom_direct
)


current_direct_max = float(
    np.max(
        np.abs(current_direct_diff)
    )
)


current_direct_rms = float(
    np.sqrt(
        np.mean(
            current_direct_diff**2
        )
    )
)


# ================================================================
# 2. Which reference/gauge does OLD prototype use?
# ================================================================

candidates = []


for ref0 in range(38):

    Ctest = covariance_for_reference(
        ref0
    )

    diff = (
        Cold
        -
        Ctest
    )

    candidates.append(
        {
            "reference_index_0based":
                ref0,

            "reference_date":
                dates[ref0],

            "max_abs_diff":
                float(
                    np.max(
                        np.abs(diff)
                    )
                ),

            "rms_diff":
                float(
                    np.sqrt(
                        np.mean(
                            diff**2
                        )
                    )
                ),
        }
    )


candidates.sort(
    key=lambda x:
        (
            x["max_abs_diff"],
            x["rms_diff"],
        )
)


best = candidates[0]

old_ref0 = int(
    best[
        "reference_index_0based"
    ]
)


# ================================================================
# 3. Inspect zero-variance row/column directly
# ================================================================

old_diag = np.diag(
    Cold
)


old_min_diag_idx = int(
    np.argmin(
        np.abs(old_diag)
    )
)


old_row_norm = np.max(
    np.abs(Cold),
    axis=1,
)


old_zero_row_idx = int(
    np.argmin(
        old_row_norm
    )
)


# ================================================================
# 4. Gauge transform OLD -> geometric master
#
# If x is represented in old gauge:
#
#       x' = x - x_m * 1
#
# T = I - 1 e_m^T
#
# Cov' = T Cov T'
# ================================================================

T_old_to_geom = (
    np.eye(
        38,
        dtype=np.float64,
    )
    -
    np.outer(
        np.ones(
            38,
            dtype=np.float64,
        ),
        np.eye(
            38,
            dtype=np.float64,
        )[geom0],
    )
)


Cold_geom = (
    T_old_to_geom
    @ Cold
    @ T_old_to_geom.T
)


old_geom_diff = (
    Cold_geom
    -
    Ccur
)


old_geom_max = float(
    np.max(
        np.abs(
            old_geom_diff
        )
    )
)


old_geom_rms = float(
    np.sqrt(
        np.mean(
            old_geom_diff**2
        )
    )
)


np.save(
    OLD_TO_GEOM_OUT,
    Cold_geom
)


# ================================================================
# 5. Reverse transformation:
#    current geom-master -> old inferred reference
# ================================================================

T_geom_to_old = (
    np.eye(
        38,
        dtype=np.float64,
    )
    -
    np.outer(
        np.ones(
            38,
            dtype=np.float64,
        ),
        np.eye(
            38,
            dtype=np.float64,
        )[old_ref0],
    )
)


Ccur_old = (
    T_geom_to_old
    @ Ccur
    @ T_geom_to_old.T
)


reverse_diff = (
    Ccur_old
    -
    Cold
)


reverse_max = float(
    np.max(
        np.abs(
            reverse_diff
        )
    )
)


reverse_rms = float(
    np.sqrt(
        np.mean(
            reverse_diff**2
        )
    )
)


# ================================================================
# 6. Structural QA
# ================================================================

current_sym = float(
    np.max(
        np.abs(
            Ccur
            -
            Ccur.T
        )
    )
)


old_sym = float(
    np.max(
        np.abs(
            Cold
            -
            Cold.T
        )
    )
)


eig_current = np.linalg.eigvalsh(
    Ccur
)


eig_old = np.linalg.eigvalsh(
    Cold
)


rank_current = int(
    np.linalg.matrix_rank(
        Ccur,
        tol=1e-12,
    )
)


rank_old = int(
    np.linalg.matrix_rank(
        Cold,
        tol=1e-12,
    )
)


# ================================================================
# Decision
# ================================================================

gauge_identified = (
    best["max_abs_diff"]
    <= TOL
)


geom_parity = (
    old_geom_max
    <= TOL
)


current_valid = (
    current_direct_max
    <= TOL
)


passed = (
    gauge_identified
    and
    geom_parity
    and
    current_valid
)


status = (
    "PASS_SM_COV_GAUGE_RECONCILED"
    if passed
    else
    "FAIL_SM_COV_GAUGE_RECONCILIATION"
)


payload = {
    "status":
        status,

    "tolerance":
        TOL,

    "network":
        {
            "ifgs": 108,
            "images": 38,
            "rank_full_G":
                int(
                    np.linalg.matrix_rank(G)
                ),
        },

    "current_covariance":
        {
            "file":
                str(CURRENT_COV),

            "reference_date":
                GEOM_MASTER,

            "reference_index_0based":
                geom0,

            "direct_formula_max_abs_diff":
                current_direct_max,

            "direct_formula_rms_diff":
                current_direct_rms,

            "rank":
                rank_current,

            "symmetry_max_abs":
                current_sym,

            "eig_min":
                float(
                    eig_current.min()
                ),

            "eig_max":
                float(
                    eig_current.max()
                ),
        },

    "old_prototype":
        {
            "file":
                str(OLD_COV),

            "best_reference_date":
                best[
                    "reference_date"
                ],

            "best_reference_index_0based":
                old_ref0,

            "best_direct_max_abs_diff":
                best[
                    "max_abs_diff"
                ],

            "best_direct_rms_diff":
                best[
                    "rms_diff"
                ],

            "minimum_abs_diagonal_index":
                old_min_diag_idx,

            "minimum_abs_diagonal_date":
                dates[
                    old_min_diag_idx
                ],

            "minimum_row_norm_index":
                old_zero_row_idx,

            "minimum_row_norm_date":
                dates[
                    old_zero_row_idx
                ],

            "rank":
                rank_old,

            "symmetry_max_abs":
                old_sym,

            "eig_min":
                float(
                    eig_old.min()
                ),

            "eig_max":
                float(
                    eig_old.max()
                ),
        },

    "gauge_transform":
        {
            "old_to_geom_max_abs_diff":
                old_geom_max,

            "old_to_geom_rms_diff":
                old_geom_rms,

            "geom_to_old_max_abs_diff":
                reverse_max,

            "geom_to_old_rms_diff":
                reverse_rms,

            "transformed_file":
                str(
                    OLD_TO_GEOM_OUT
                ),
        },

    "all_reference_candidates":
        candidates,

    "phase_modified":
        False,
}


REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    +
    "\n"
)


# ================================================================
# Print
# ================================================================

print("=" * 92)
print("P15-5B5A SM_COV REFERENCE-GAUGE AUDIT")
print("=" * 92)

print(
    "images / IFGs                  :",
    "38 / 108",
)

print(
    "rank(full G)                   :",
    np.linalg.matrix_rank(G),
)

print()

print(
    "current reference              :",
    f"{GEOM_MASTER} (0b={geom0})",
)

print(
    "current vs direct max diff     :",
    f"{current_direct_max:.12e}",
)

print(
    "current vs direct RMS          :",
    f"{current_direct_rms:.12e}",
)

print()

print(
    "old inferred reference         :",
    (
        f"{best['reference_date']} "
        f"(0b={old_ref0})"
    ),
)

print(
    "old vs that-gauge max diff     :",
    f"{best['max_abs_diff']:.12e}",
)

print(
    "old vs that-gauge RMS          :",
    f"{best['rms_diff']:.12e}",
)

print(
    "old minimum diagonal           :",
    (
        f"{dates[old_min_diag_idx]} "
        f"(0b={old_min_diag_idx})"
    ),
)

print(
    "old minimum row norm           :",
    (
        f"{dates[old_zero_row_idx]} "
        f"(0b={old_zero_row_idx})"
    ),
)

print()

print(
    "old -> geom max diff           :",
    f"{old_geom_max:.12e}",
)

print(
    "old -> geom RMS                :",
    f"{old_geom_rms:.12e}",
)

print(
    "geom -> old max diff           :",
    f"{reverse_max:.12e}",
)

print(
    "geom -> old RMS                :",
    f"{reverse_rms:.12e}",
)

print()

print(
    "current covariance rank        :",
    rank_current,
)

print(
    "old covariance rank            :",
    rank_old,
)

print(
    "current symmetry max           :",
    f"{current_sym:.12e}",
)

print(
    "old symmetry max               :",
    f"{old_sym:.12e}",
)

print()

print(
    "phase modified                 :",
    False,
)

print(
    "report                         :",
    REPORT,
)

print("=" * 92)
print(
    "P15-5B5A FINAL RESULT:",
    status,
)
print("=" * 92)


if not passed:
    raise RuntimeError(
        status
    )
