#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


BRIDGE = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "pystamps_bridge_v09"
)

R3C2 = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09/"
    "scla_v09/pystamps_bridge/"
    "r3c2_pointwise_bperp"
)


def qline(
    title,
    values,
    qs=(1, 5, 50, 95, 99),
    fmt=".6e",
):
    x = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        print(f"{title}\n  NO FINITE VALUES")
        return

    q = np.percentile(
        x,
        qs,
    )

    print(title)

    print(
        "  "
        +
        " / ".join(
            format(v, fmt)
            for v in q
        )
    )


def load_vector(
    path: Path,
    variable: str,
):
    d = loadmat(
        path,
        variable_names=[
            variable,
        ],
        squeeze_me=False,
    )

    if variable not in d:
        raise RuntimeError(
            f"{path.name} missing "
            f"{variable}"
        )

    return np.asarray(
        d[variable],
        dtype=np.float64,
    ).reshape(-1)


def schema(path: Path):

    return {
        name: (
            tuple(shape),
            dtype,
        )
        for name, shape, dtype
        in whosmat(path)
    }


def main():

    debug_path = (
        BRIDGE
        /
        "stage7_sbas_debug.json"
    )

    sb_path = (
        BRIDGE
        /
        "scla_sb2.mat"
    )

    smooth_path = (
        BRIDGE
        /
        "scla_smooth_sb2.mat"
    )

    final_path = (
        BRIDGE
        /
        "scla2.mat"
    )

    for p in (
        debug_path,
        sb_path,
        smooth_path,
        final_path,
    ):
        if not p.is_file():
            raise RuntimeError(
                f"Missing Stage7 output: {p}"
            )

    debug = json.loads(
        debug_path.read_text(
            encoding="utf-8"
        )
    )

    print("=" * 112)
    print(
        "Step 10R5a - mature pySTAMPS "
        "Stage7 output audit"
    )
    print("=" * 112)

    print(
        f"bridge                     : "
        f"{BRIDGE}"
    )

    print(
        f"debug status               : "
        f"{debug.get('status')}"
    )

    print(
        f"implementation             : "
        f"{debug.get('implementation')}"
    )

    print(
        f"duration                   : "
        f"{float(debug.get('duration_sec', np.nan)):.2f} s"
    )

    print(
        f"n_ps                       : "
        f"{debug.get('n_ps')}"
    )

    print(
        f"n_ifg                      : "
        f"{debug.get('n_ifg')}"
    )

    print(
        f"n_image                    : "
        f"{debug.get('n_image')}"
    )

    print(
        f"master_ix                  : "
        f"{debug.get('master_ix')}"
    )

    print(
        f"reference_ps               : "
        f"{debug.get('reference_ps')}"
    )

    if debug.get(
        "status"
    ) != "completed":

        raise RuntimeError(
            "Stage7 debug status is "
            "not completed"
        )

    npoint = int(
        debug["n_ps"]
    )

    nedge = int(
        debug["n_ifg"]
    )

    ndate = int(
        debug["n_image"]
    )

    master_idx0 = (
        int(
            debug["master_ix"]
        )
        -
        1
    )

    # ========================================================
    # Output schema
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Stage7 output schemas"
    )
    print("=" * 112)

    schemas = {}

    for p in (
        sb_path,
        smooth_path,
        final_path,
    ):

        s = schema(
            p
        )

        schemas[
            p.name
        ] = s

        print()
        print(
            f"[{p.name}]"
        )

        for name, (
            shape,
            dtype,
        ) in s.items():

            print(
                f"  {name:20s} "
                f"{shape!s:22s} "
                f"{dtype}"
            )

    expected_ph = {
        "scla_sb2.mat":
            (
                npoint,
                nedge,
            ),

        "scla_smooth_sb2.mat":
            (
                npoint,
                nedge,
            ),

        "scla2.mat":
            (
                npoint,
                ndate,
            ),
    }

    for filename, expected in (
        expected_ph.items()
    ):

        got = schemas[
            filename
        ].get(
            "ph_scla",
            (
                None,
                None,
            ),
        )[0]

        if got != expected:
            raise RuntimeError(
                f"{filename}.ph_scla "
                f"shape={got}, "
                f"expected={expected}"
            )

    # ========================================================
    # K / C
    # ========================================================

    K1 = load_vector(
        sb_path,
        "K_ps_uw",
    )

    C1 = load_vector(
        sb_path,
        "C_ps_uw",
    )

    K1s = load_vector(
        smooth_path,
        "K_ps_uw",
    )

    C1s = load_vector(
        smooth_path,
        "C_ps_uw",
    )

    K2 = load_vector(
        final_path,
        "K_ps_uw",
    )

    C2 = load_vector(
        final_path,
        "C_ps_uw",
    )

    for name, x in (
        ("K1", K1),
        ("C1", C1),
        ("K1s", K1s),
        ("C1s", C1s),
        ("K2", K2),
        ("C2", C2),
    ):

        if x.size != npoint:
            raise RuntimeError(
                f"{name}: {x.size} "
                f"!= {npoint}"
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                f"{name} contains "
                "non-finite values"
            )

    changed_K = int(
        np.count_nonzero(
            K1s != K1
        )
    )

    changed_C = int(
        np.count_nonzero(
            C1s != C1
        )
    )

    dbg_K = int(
        debug.get(
            "smooth",
            {},
        ).get(
            "K_changed",
            -1,
        )
    )

    dbg_C = int(
        debug.get(
            "smooth",
            {},
        ).get(
            "C_changed",
            -1,
        )
    )

    print()
    print("=" * 112)
    print(
        "Stage7 SCLA coefficients"
    )
    print("=" * 112)

    qline(
        "Pass1 K "
        "p01/p05/p50/p95/p99 [rad/m]:",
        K1,
    )

    print(
        f"Pass1 C max |.|            : "
        f"{np.max(np.abs(C1)):.9e} rad"
    )

    qline(
        "Pass2 K "
        "p01/p05/p50/p95/p99 [rad/m]:",
        K1s,
    )

    print(
        f"K changed                  : "
        f"{changed_K:,}/{npoint:,} "
        f"({100.0*changed_K/npoint:.3f}%)"
    )

    print(
        f"K changed debug            : "
        f"{dbg_K:,}"
    )

    print(
        f"K debug match              : "
        f"{changed_K == dbg_K}"
    )

    print(
        f"C changed                  : "
        f"{changed_C:,}/{npoint:,}"
    )

    print(
        f"C debug match              : "
        f"{changed_C == dbg_C}"
    )

    print()

    qline(
        "Final K "
        "p01/p05/p50/p95/p99 [rad/m]:",
        K2,
    )

    qline(
        "Final C "
        "p01/p05/p50/p95/p99 [rad]:",
        C2,
    )

    # ========================================================
    # Equivalent DEM error
    #
    # From R3c2:
    #   S_h ~= alpha * Bperp
    #
    # Stage7:
    #   ph_scla = K * Bperp
    #
    # Thus:
    #   dh_equiv ~= K / alpha
    # ========================================================

    alpha_path = (
        R3C2
        /
        "bperp_vs_Sh_acquisition_alpha_rad_per_m2.npy"
    )

    if not alpha_path.is_file():
        raise RuntimeError(
            f"Missing R3c2 alpha: "
            f"{alpha_path}"
        )

    alpha = np.asarray(
        np.load(
            alpha_path,
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    if alpha.shape != (
        npoint,
    ):
        raise RuntimeError(
            f"alpha shape={alpha.shape}"
        )

    good = (
        np.isfinite(alpha)
        &
        (
            np.abs(alpha)
            >
            1.0e-8
        )
    )

    dh = np.full(
        npoint,
        np.nan,
        dtype=np.float32,
    )

    dh[
        good
    ] = (
        K2[
            good
        ]
        /
        alpha[
            good
        ]
    ).astype(
        np.float32
    )

    np.save(
        BRIDGE
        /
        "stage7_equivalent_dem_error_m_QA.npy",
        dh,
    )

    print()
    print("=" * 112)
    print(
        "Equivalent DEM-error interpretation"
    )
    print("=" * 112)

    qline(
        "delta_h(eq) "
        "p01/p05/p50/p95/p99 [m]:",
        dh,
        fmt=".4f",
    )

    print(
        f"|delta_h| > 20 m           : "
        f"{np.count_nonzero(np.abs(dh) > 20):,} "
        f"({100*np.mean(np.abs(dh) > 20):.3f}%)"
    )

    print(
        f"|delta_h| > 50 m           : "
        f"{np.count_nonzero(np.abs(dh) > 50):,} "
        f"({100*np.mean(np.abs(dh) > 50):.3f}%)"
    )

    print(
        f"|delta_h| > 100 m          : "
        f"{np.count_nonzero(np.abs(dh) > 100):,} "
        f"({100*np.mean(np.abs(dh) > 100):.3f}%)"
    )

    # ========================================================
    # Final ph_scla
    # ========================================================

    final = loadmat(
        final_path,
        variable_names=[
            "ph_scla",
            "ifg_vcm",
        ],
        squeeze_me=False,
    )

    ph_scla = np.asarray(
        final[
            "ph_scla"
        ],
        dtype=np.float32,
    )

    if ph_scla.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            f"ph_scla shape="
            f"{ph_scla.shape}"
        )

    if not np.all(
        np.isfinite(
            ph_scla
        )
    ):
        raise RuntimeError(
            "ph_scla has non-finite values"
        )

    model_rms = np.sqrt(
        np.mean(
            ph_scla.astype(
                np.float64
            ) ** 2,
            axis=1,
        )
    )

    master_max = float(
        np.max(
            np.abs(
                ph_scla[
                    :,
                    master_idx0
                ]
            )
        )
    )

    print()
    print("=" * 112)
    print(
        "Final acquisition SCLA model"
    )
    print("=" * 112)

    print(
        f"shape                      : "
        f"{ph_scla.shape}"
    )

    print(
        f"master model max |.|       : "
        f"{master_max:.9e} rad"
    )

    qline(
        "per-point SCLA RMS "
        "p01/p05/p50/p95/p99 [rad]:",
        model_rms,
    )

    qline(
        "per-point SCLA max|.| "
        "p01/p05/p50/p95/p99 [rad]:",
        np.max(
            np.abs(
                ph_scla
            ),
            axis=1,
        ),
    )

    vcm = np.asarray(
        final[
            "ifg_vcm"
        ],
        dtype=np.float64,
    )

    print(
        f"scla2.ifg_vcm shape        : "
        f"{vcm.shape}"
    )

    if vcm.shape != (
        ndate,
        ndate,
    ):
        raise RuntimeError(
            "ifg_vcm shape mismatch"
        )

    # ========================================================
    # Mature semantic audit
    # ========================================================

    smooth = debug.get(
        "smooth",
        {}
    )

    pass1 = debug.get(
        "pass1",
        {}
    )

    pass3 = debug.get(
        "pass3",
        {}
    )

    print()
    print("=" * 112)
    print(
        "Mature Stage7 semantic audit"
    )
    print("=" * 112)

    print(
        f"Pass1 used IFGs            : "
        f"{pass1.get('used_ifgs')}"
    )

    print(
        f"Pass1 design               : "
        f"{pass1.get('design_shape')}"
    )

    print(
        f"Pass1 covariance           : "
        f"{pass1.get('covariance_shape')}"
    )

    print(
        f"Delaunay backend           : "
        f"{smooth.get('backend')}"
    )

    print(
        f"Delaunay edges             : "
        f"{smooth.get('n_edge')}"
    )

    print(
        f"Pass3 images no master     : "
        f"{pass3.get('images_without_master')}"
    )

    print(
        f"Pass3 baseline rank        : "
        f"{pass3.get('baseline_rank')}"
    )

    print(
        f"Pass3 K design             : "
        f"{pass3.get('K_design_shape')}"
    )

    print(
        f"Pass3 C design             : "
        f"{pass3.get('C_design_shape')}"
    )

    print(
        f"Pass3 C covariance         : "
        f"{pass3.get('C_covariance_shape')}"
    )

    # ========================================================
    # Computational decision only
    # ========================================================

    if changed_K != dbg_K:

        status = (
            "REVIEW_STAGE7_K_SMOOTH_MISMATCH"
        )

    elif changed_C != dbg_C:

        status = (
            "REVIEW_STAGE7_C_SMOOTH_MISMATCH"
        )

    elif master_max > 1e-6:

        status = (
            "REVIEW_STAGE7_MASTER_GAUGE"
        )

    elif int(
        pass3.get(
            "baseline_rank",
            -1,
        )
    ) != ndate - 1:

        status = (
            "REVIEW_STAGE7_BASELINE_RANK"
        )

    else:

        status = (
            "PASS_STAGE7_COMPLETED_PENDING_SCIENTIFIC_QA"
        )

    audit = {
        "format":
            "pyPSDS-GAMMA-stage7-output-audit-v09",

        "status":
            status,

        "implementation":
            debug.get(
                "implementation"
            ),

        "duration_sec":
            debug.get(
                "duration_sec"
            ),

        "n_ps":
            npoint,

        "n_ifg":
            nedge,

        "n_image":
            ndate,

        "master_ix_1based":
            master_idx0 + 1,

        "reference_ps":
            debug.get(
                "reference_ps"
            ),

        "smooth": {
            "edges":
                smooth.get(
                    "n_edge"
                ),

            "K_changed":
                changed_K,

            "K_changed_fraction":
                changed_K / npoint,
        },

        "final_K_rad_per_m": {
            "p01":
                float(
                    np.percentile(
                        K2,
                        1,
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        K2,
                        50,
                    )
                ),

            "p99":
                float(
                    np.percentile(
                        K2,
                        99,
                    )
                ),
        },

        "equivalent_dem_error_m": {
            "p05":
                float(
                    np.nanpercentile(
                        dh,
                        5,
                    )
                ),

            "p50":
                float(
                    np.nanpercentile(
                        dh,
                        50,
                    )
                ),

            "p95":
                float(
                    np.nanpercentile(
                        dh,
                        95,
                    )
                ),
        },

        "scla_model_rms_rad": {
            "p50":
                float(
                    np.percentile(
                        model_rms,
                        50,
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        model_rms,
                        95,
                    )
                ),
        },

        "master_model_max_abs_rad":
            master_max,

        "scla_applied":
            False,

        "stage8_executed":
            False,
    }

    out = (
        BRIDGE
        /
        "stage7_output_audit_v09.json"
    )

    out.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"audit                       : "
        f"{out}"
    )

    print()
    print(
        f"STEP 10R5a STATUS: "
        f"{status}"
    )

    print(
        "SCLA has NOT been applied to "
        "the canonical pyPSDS time series."
    )


if __name__ == "__main__":
    main()
