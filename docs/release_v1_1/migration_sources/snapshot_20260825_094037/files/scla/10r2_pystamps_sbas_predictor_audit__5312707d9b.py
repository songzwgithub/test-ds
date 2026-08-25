#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pypsds.prototype import open_from_config


def read_itab(path: Path, ndate: int):

    pairs = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            s = line.strip()

            if (
                not s
                or s.startswith("#")
            ):
                continue

            fields = s.split()

            ints = []

            for x in fields:
                try:
                    ints.append(
                        int(x)
                    )
                except ValueError:
                    pass

            if len(ints) < 2:
                continue

            i, j = ints[0], ints[1]

            if not (
                1 <= i <= ndate
                and
                1 <= j <= ndate
            ):
                continue

            pairs.append(
                (i, j)
            )

    if not pairs:
        raise RuntimeError(
            f"No network pairs found in {path}"
        )

    return np.asarray(
        pairs,
        dtype=np.int64,
    )


def network_matrix(
    ndate: int,
    pairs_1based: np.ndarray,
):

    nedge = pairs_1based.shape[0]

    G = np.zeros(
        (nedge, ndate),
        dtype=np.float64,
    )

    rr = np.arange(
        nedge,
        dtype=np.int64,
    )

    G[
        rr,
        pairs_1based[:, 0] - 1
    ] = -1.0

    G[
        rr,
        pairs_1based[:, 1] - 1
    ] = +1.0

    return G


def canonical_edge_set(pairs):

    return {
        tuple(
            sorted(
                (
                    int(i),
                    int(j),
                )
            )
        )
        for i, j in pairs
    }


def parity_stream(
    X,
    *,
    pairs,
    G,
    reference_idx,
    batch_size,
):

    npoint, ndate = X.shape

    keep = np.asarray(
        [
            i
            for i in range(ndate)
            if i != reference_idx
        ],
        dtype=np.int64,
    )

    Gred = G[:, keep]

    rank = int(
        np.linalg.matrix_rank(
            Gred
        )
    )

    if rank != keep.size:
        raise RuntimeError(
            f"Reduced network rank "
            f"{rank}/{keep.size}"
        )

    P = np.linalg.pinv(
        Gred
    )

    ii = (
        pairs[:, 0] - 1
    )

    jj = (
        pairs[:, 1] - 1
    )

    ss = 0.0
    nn = 0
    mx = 0.0

    point_rms = np.empty(
        npoint,
        dtype=np.float32,
    )

    for b0 in range(
        0,
        npoint,
        batch_size,
    ):

        b1 = min(
            b0 + batch_size,
            npoint,
        )

        x = np.asarray(
            X[b0:b1],
            dtype=np.float64,
        )

        # Virtual IFGs:
        # G*x = x_j - x_i
        d = (
            x[:, jj]
            -
            x[:, ii]
        )

        xr = (
            d
            @ P.T
        )

        xrec = np.zeros_like(
            x
        )

        xrec[:, keep] = xr

        # Gauge consistency:
        # input X must already be zero at reference.
        delta = (
            xrec - x
        )

        finite = np.isfinite(
            delta
        )

        if not np.all(finite):
            raise RuntimeError(
                "Non-finite parity values"
            )

        ss += float(
            np.sum(
                delta * delta
            )
        )

        nn += int(
            delta.size
        )

        mx = max(
            mx,
            float(
                np.max(
                    np.abs(delta)
                )
            ),
        )

        point_rms[
            b0:b1
        ] = np.sqrt(
            np.mean(
                delta * delta,
                axis=1,
            )
        ).astype(
            np.float32
        )

    rms = float(
        np.sqrt(
            ss / nn
        )
    )

    return {
        "rank":
            rank,

        "rms":
            rms,

        "max":
            mx,

        "point_rms":
            point_rms,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--pystamps-root",
        default="/home/ubuntu/Downloads/pystamps",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        _,
    ) = open_from_config(
        args.config
    )

    root = (
        Path(paths.output_dir)
        /
        "v09"
    )

    invdir = (
        root
        /
        "network_inversion_v09"
    )

    netdir = (
        root
        /
        "network"
    )

    sdir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r2_predictor_audit"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ndate = len(
        stack.dates
    )

    # ========================================================
    # Inputs
    # ========================================================

    phase_path = (
        invdir
        /
        "acquisition_phase_l2_candidate_rad.npy"
    )

    sensitivity_path = (
        sdir
        /
        "topographic_phase_sensitivity_rad_per_m.npy"
    )

    itab_path = (
        netdir
        /
        "network.itab"
    )

    Y = np.load(
        phase_path,
        mmap_mode="r",
    )

    Sh = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    if Y.shape != Sh.shape:
        raise RuntimeError(
            "phase / sensitivity shape mismatch"
        )

    npoint = Y.shape[0]

    if Y.shape[1] != ndate:
        raise RuntimeError(
            "acquisition count mismatch"
        )

    # ========================================================
    # Temporal reference
    # ========================================================

    inv_manifest_path = (
        invdir
        /
        "network_inversion_parity_manifest.json"
    )

    inv_manifest = json.loads(
        inv_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    reference_idx = int(
        inv_manifest[
            "reference_acquisition_index"
        ]
    )

    reference_date = str(
        inv_manifest[
            "reference_date"
        ]
    )

    # ========================================================
    # Current pyPSDS network
    # ========================================================

    pairs = read_itab(
        itab_path,
        ndate,
    )

    nedge = pairs.shape[0]

    G = network_matrix(
        ndate,
        pairs,
    )

    rankG = int(
        np.linalg.matrix_rank(
            G
        )
    )

    cycle_rank = (
        nedge
        -
        ndate
        +
        1
    )

    # ========================================================
    # Old pySTAMPS reference topology
    # ========================================================

    old_root = Path(
        args.pystamps_root
    )

    old_ps_path = (
        old_root
        /
        "ps2.mat"
    )

    old_topology_status = (
        "NO_REFERENCE_DATASET"
    )

    old_n_ifg = None
    old_n_image = None
    old_master_ix = None

    if old_ps_path.is_file():

        old = loadmat(
            old_ps_path,
            squeeze_me=False,
            struct_as_record=False,
        )

        old_n_ifg = int(
            round(
                float(
                    np.asarray(
                        old["n_ifg"]
                    ).reshape(-1)[0]
                )
            )
        )

        old_n_image = int(
            round(
                float(
                    np.asarray(
                        old["n_image"]
                    ).reshape(-1)[0]
                )
            )
        )

        old_master_ix = int(
            round(
                float(
                    np.asarray(
                        old["master_ix"]
                    ).reshape(-1)[0]
                )
            )
        )

        old_pairs = np.asarray(
            old["ifgday_ix"],
            dtype=np.int64,
        )

        if old_pairs.shape == (
            2,
            old_n_ifg,
        ):
            old_pairs = old_pairs.T

        if (
            old_pairs.shape
            ==
            pairs.shape
        ):

            if np.array_equal(
                old_pairs,
                pairs,
            ):

                old_topology_status = (
                    "EXACT_PAIR_ORDER_MATCH"
                )

            elif (
                canonical_edge_set(
                    old_pairs
                )
                ==
                canonical_edge_set(
                    pairs
                )
            ):

                old_topology_status = (
                    "SAME_EDGE_SET_DIFFERENT_ORDER"
                )

            else:

                old_topology_status = (
                    "DIFFERENT_NETWORK"
                )

        else:

            old_topology_status = (
                "DIFFERENT_NETWORK_SIZE"
            )

    # ========================================================
    # Gauge checks
    # ========================================================

    phase_ref_max = float(
        np.max(
            np.abs(
                np.asarray(
                    Y[:, reference_idx],
                    dtype=np.float64,
                )
            )
        )
    )

    sens_ref_max = float(
        np.max(
            np.abs(
                np.asarray(
                    Sh[:, reference_idx],
                    dtype=np.float64,
                )
            )
        )
    )

    # ========================================================
    # Main parity tests
    #
    # No persistent [Npoint,108] cube is created.
    # ========================================================

    print("=" * 112)
    print(
        "Step 10R2 - pySTAMPS SBAS network / "
        "topographic-predictor compatibility audit"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"strict points              : "
        f"{npoint:,}"
    )

    print(
        f"acquisitions               : "
        f"{ndate}"
    )

    print(
        f"SB network edges           : "
        f"{nedge}"
    )

    print(
        f"network rank               : "
        f"{rankG}"
    )

    print(
        f"cycle rank                 : "
        f"{cycle_rank}"
    )

    print(
        f"temporal reference         : "
        f"{reference_date} "
        f"(index {reference_idx})"
    )

    print(
        f"reference max |phase|      : "
        f"{phase_ref_max:.3e} rad"
    )

    print(
        f"reference max |S_h|        : "
        f"{sens_ref_max:.3e} rad/m"
    )

    print()
    print("=" * 112)
    print(
        "Old pySTAMPS SBAS topology reference"
    )
    print("=" * 112)

    print(
        f"reference dataset          : "
        f"{old_root}"
    )

    print(
        f"old n_ifg                  : "
        f"{old_n_ifg}"
    )

    print(
        f"old n_image                : "
        f"{old_n_image}"
    )

    print(
        f"old master_ix              : "
        f"{old_master_ix}"
    )

    print(
        f"topology comparison        : "
        f"{old_topology_status}"
    )

    print()
    print(
        "Auditing acquisition phase through "
        "virtual SB network ..."
    )

    phase_parity = parity_stream(
        Y,
        pairs=pairs,
        G=G,
        reference_idx=reference_idx,
        batch_size=args.batch_size,
    )

    print(
        "Auditing GAMMA topographic sensitivity "
        "through virtual SB network ..."
    )

    sens_parity = parity_stream(
        Sh,
        pairs=pairs,
        G=G,
        reference_idx=reference_idx,
        batch_size=args.batch_size,
    )

    np.save(
        outdir
        /
        "phase_network_reconstruction_rms_by_point_rad.npy",
        phase_parity[
            "point_rms"
        ],
    )

    np.save(
        outdir
        /
        "sensitivity_network_reconstruction_rms_by_point_rad_per_m.npy",
        sens_parity[
            "point_rms"
        ],
    )

    print()
    print("=" * 112)
    print(
        "Virtual-SB -> acquisition reconstruction"
    )
    print("=" * 112)

    print(
        f"phase reduced rank         : "
        f"{phase_parity['rank']}/"
        f"{ndate-1}"
    )

    print(
        f"phase RMS difference       : "
        f"{phase_parity['rms']:.6e} rad"
    )

    print(
        f"phase maximum difference   : "
        f"{phase_parity['max']:.6e} rad"
    )

    print()

    print(
        f"S_h reduced rank           : "
        f"{sens_parity['rank']}/"
        f"{ndate-1}"
    )

    print(
        f"S_h RMS difference         : "
        f"{sens_parity['rms']:.6e} rad/m"
    )

    print(
        f"S_h maximum difference     : "
        f"{sens_parity['max']:.6e} rad/m"
    )

    sens_rms = float(
        np.sqrt(
            np.mean(
                np.asarray(
                    Sh,
                    dtype=np.float64,
                ) ** 2
            )
        )
    )

    rel_sens = (
        sens_parity[
            "rms"
        ]
        /
        sens_rms
    )

    print(
        f"S_h relative RMS           : "
        f"{rel_sens:.6e}"
    )

    # ========================================================
    # Bridge interpretation
    # ========================================================

    virtual_ifg_bytes = (
        npoint
        *
        nedge
        *
        4
    )

    acquisition_bytes = (
        npoint
        *
        ndate
        *
        4
    )

    print()
    print("=" * 112)
    print(
        "pySTAMPS Stage-7 bridge interpretation"
    )
    print("=" * 112)

    print(
        "Proposed bridge predictor:"
    )

    print(
        "  bp2.bperp_mat[p,e] := "
        "S_h[p,j(e)] - S_h[p,i(e)]"
    )

    print(
        "  predictor units          : rad/m"
    )

    print(
        "  resulting K_ps_uw units  : m"
    )

    print(
        "  ph_scla units            : rad"
    )

    print()

    print(
        "Proposed bridge SB phase:"
    )

    print(
        "  phuw_sb2.ph_uw[p,e] := "
        "phase[p,j(e)] - phase[p,i(e)]"
    )

    print()

    print(
        f"temporary IFG matrix size  : "
        f"{virtual_ifg_bytes/1024**2:.1f} MiB each"
    )

    print(
        f"acquisition matrix size    : "
        f"{acquisition_bytes/1024**2:.1f} MiB each"
    )

    print(
        "Canonical pyPSDS policy    : "
        "virtual IFGs remain non-persistent; "
        "bridge matrices, if needed, are temporary."
    )

    # ========================================================
    # Status
    # ========================================================

    if rankG != ndate - 1:

        status = (
            "REVIEW_NETWORK_RANK"
        )

    elif (
        phase_ref_max > 1e-6
        or
        sens_ref_max > 1e-7
    ):

        status = (
            "REVIEW_REFERENCE_GAUGE"
        )

    elif phase_parity[
        "max"
    ] > 1e-4:

        status = (
            "REVIEW_PHASE_NETWORK_PARITY"
        )

    elif rel_sens > 2e-3:

        status = (
            "REVIEW_SENSITIVITY_NETWORK_PARITY"
        )

    else:

        status = (
            "PASS_BRIDGE_PREDICTOR_COMPATIBLE"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-pySTAMPS-SBAS-predictor-audit-v09",

        "status":
            status,

        "points":
            int(npoint),

        "acquisitions":
            int(ndate),

        "edges":
            int(nedge),

        "network_rank":
            int(rankG),

        "cycle_rank":
            int(cycle_rank),

        "reference": {
            "index":
                int(reference_idx),
            "date":
                reference_date,
        },

        "old_pystamps": {
            "root":
                str(old_root),
            "n_ifg":
                old_n_ifg,
            "n_image":
                old_n_image,
            "master_ix":
                old_master_ix,
            "topology_status":
                old_topology_status,
        },

        "phase_parity": {
            "rms_rad":
                phase_parity["rms"],
            "max_rad":
                phase_parity["max"],
        },

        "sensitivity_parity": {
            "rms_rad_per_m":
                sens_parity["rms"],
            "max_rad_per_m":
                sens_parity["max"],
            "relative_rms":
                rel_sens,
        },

        "proposed_stage7_geometry_predictor": {
            "field_name_required_by_pystamps":
                "bp2.bperp_mat",
            "actual_semantics":
                "virtual IFG topographic phase sensitivity",
            "definition":
                "S_h(j)-S_h(i)",
            "units":
                "rad/m",
            "K_ps_uw_units":
                "m",
            "ph_scla_units":
                "rad",
        },

        "persistent_ifg_cube_created":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "pystamps_sbas_predictor_audit_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    print()
    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R2 STATUS: {status}"
    )

    print(
        "No pySTAMPS bridge MAT files were created."
    )

    print(
        "No phase, SCLA, or APS correction was applied."
    )


if __name__ == "__main__":
    main()
