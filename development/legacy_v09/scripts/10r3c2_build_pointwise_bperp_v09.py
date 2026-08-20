#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


def resolve_pystamps_source(explicit: str | None) -> Path:

    candidates = []

    if explicit:
        candidates.append(
            Path(explicit).expanduser()
        )

    candidates.extend([
        Path("/home/ubuntu/software/pystamps-gamma"),
        Path("/home/ubuntu/software/pystamps-gamma-main"),
        Path.home() / "software" / "pystamps-gamma",
    ])

    software = Path("/home/ubuntu/software")

    if software.is_dir():
        candidates.extend(
            sorted(
                p
                for p in software.glob("*pystamps*")
                if p.is_dir()
            )
        )

    seen = set()

    for p in candidates:

        try:
            p = p.resolve()
        except Exception:
            continue

        if p in seen:
            continue

        seen.add(p)

        if (
            (p / "pystamps").is_dir()
            and
            (p / "pystamps" / "prep").is_dir()
        ):

            sys.path.insert(
                0,
                str(p),
            )

            return p

    raise RuntimeError(
        "Cannot locate pystamps-gamma. "
        "Use --pystamps-source."
    )


def read_gamma_int(
    par: Path,
    keys: tuple[str, ...],
) -> int:

    values = {}

    with par.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            left, sep, right = (
                line.partition(":")
            )

            if not sep:
                continue

            key = left.strip()

            if key not in keys:
                continue

            fields = right.split()

            if fields:
                values[key] = fields[0]

    for key in keys:

        if key in values:
            return int(
                round(
                    float(
                        values[key]
                    )
                )
            )

    raise RuntimeError(
        f"Unable to read any of {keys} "
        f"from {par}"
    )


def read_itab(
    path: Path,
    ndate: int,
) -> np.ndarray:

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

            ints = []

            for token in s.split():

                try:
                    ints.append(
                        int(token)
                    )
                except ValueError:
                    pass

            if len(ints) < 2:
                continue

            i, j = ints[:2]

            if not (
                1 <= i <= ndate
                and
                1 <= j <= ndate
            ):
                continue

            if i == j:
                raise RuntimeError(
                    f"Self-edge in network: "
                    f"{i},{j}"
                )

            pairs.append(
                (i, j)
            )

    a = np.asarray(
        pairs,
        dtype=np.int64,
    )

    if (
        a.ndim != 2
        or
        a.shape[1] != 2
    ):
        raise RuntimeError(
            f"Invalid network.itab: {path}"
        )

    return a


def build_network_matrix(
    pairs: np.ndarray,
    ndate: int,
) -> np.ndarray:

    G = np.zeros(
        (
            pairs.shape[0],
            ndate,
        ),
        dtype=np.float64,
    )

    row = np.arange(
        pairs.shape[0],
        dtype=np.int64,
    )

    G[
        row,
        pairs[:, 0] - 1,
    ] = -1.0

    G[
        row,
        pairs[:, 1] - 1,
    ] = +1.0

    return G


def row_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    ac = (
        a
        -
        np.mean(
            a,
            axis=1,
            keepdims=True,
        )
    )

    bc = (
        b
        -
        np.mean(
            b,
            axis=1,
            keepdims=True,
        )
    )

    num = np.sum(
        ac * bc,
        axis=1,
    )

    den = np.sqrt(
        np.sum(
            ac * ac,
            axis=1,
        )
        *
        np.sum(
            bc * bc,
            axis=1,
        )
    )

    return np.divide(
        num,
        den,
        out=np.full(
            a.shape[0],
            np.nan,
            dtype=np.float64,
        ),
        where=(
            den > 0
        ),
    )


def fit_scale_and_relative_residual(
    predictor: np.ndarray,
    target: np.ndarray,
):

    pp = np.sum(
        predictor * predictor,
        axis=1,
    )

    pt = np.sum(
        predictor * target,
        axis=1,
    )

    alpha = np.divide(
        pt,
        pp,
        out=np.full(
            predictor.shape[0],
            np.nan,
            dtype=np.float64,
        ),
        where=(
            pp > 0
        ),
    )

    residual = (
        target
        -
        alpha[:, None]
        *
        predictor
    )

    residual_rms = np.sqrt(
        np.mean(
            residual * residual,
            axis=1,
        )
    )

    target_rms = np.sqrt(
        np.mean(
            target * target,
            axis=1,
        )
    )

    relative = np.divide(
        residual_rms,
        target_rms,
        out=np.full(
            predictor.shape[0],
            np.nan,
            dtype=np.float64,
        ),
        where=(
            target_rms > 0
        ),
    )

    return (
        alpha,
        relative,
    )


def qprint(
    title,
    x,
    qs=(1,5,50,95,99),
    fmt=".6f",
):

    a = np.asarray(
        x,
        dtype=np.float64,
    )

    a = a[
        np.isfinite(a)
    ]

    values = np.percentile(
        a,
        qs,
    )

    print(title)

    print(
        "  "
        +
        " / ".join(
            format(v, fmt)
            for v in values
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--project-root",
        default="/home/ubuntu/Downloads",
    )

    ap.add_argument(
        "--pystamps-source",
        default=None,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16384,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = ap.parse_args()

    # ========================================================
    # pyPSDS inputs
    # ========================================================

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

    ppsdir = (
        root
        /
        "point_phase_stack"
    )

    netdir = (
        root
        /
        "network"
    )

    sensitivity_dir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
    )

    source_dir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3b_grid_adapter"
        /
        "generated_bases"
    )

    source_manifest_path = (
        source_dir
        /
        "current_network_108_baseline_sources.json"
    )

    source_status_path = (
        source_dir
        /
        "base_generation_manifest.json"
    )

    outdir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3c2_pointwise_bperp"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    strict_ids = np.asarray(
        np.load(
            invdir
            /
            "strict_point_ids.npy",
            mmap_mode="r",
        ),
        dtype=np.int64,
    )

    all_rows = np.load(
        ppsdir
        /
        "rows.npy",
        mmap_mode="r",
    )

    all_cols = np.load(
        ppsdir
        /
        "cols.npy",
        mmap_mode="r",
    )

    rows = np.asarray(
        all_rows[
            strict_ids
        ],
        dtype=np.int32,
    )

    cols = np.asarray(
        all_cols[
            strict_ids
        ],
        dtype=np.int32,
    )

    Sh = np.load(
        sensitivity_dir
        /
        "topographic_phase_sensitivity_rad_per_m.npy",
        mmap_mode="r",
    )

    npoint = strict_ids.size
    ndate = len(
        stack.dates
    )

    dates = [
        str(x)
        for x in stack.dates
    ]

    if Sh.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            f"S_h shape mismatch: "
            f"{Sh.shape}"
        )

    pairs = read_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    nedge = pairs.shape[0]

    # ========================================================
    # Require R3c1 PASS
    # ========================================================

    if not source_status_path.is_file():
        raise RuntimeError(
            f"Missing R3c1 manifest: "
            f"{source_status_path}"
        )

    source_status = json.loads(
        source_status_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        source_status.get(
            "status"
        )
        !=
        "PASS_108_BASELINE_SOURCES_READY"
    ):
        raise RuntimeError(
            "Step10R3c1 is not frozen PASS"
        )

    sources = json.loads(
        source_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if len(sources) != nedge:
        raise RuntimeError(
            f"Baseline source count "
            f"{len(sources)} != "
            f"network edge count {nedge}"
        )

    # ========================================================
    # Validate exact edge order / dates
    # ========================================================

    baseline_files = []
    orientations = np.empty(
        nedge,
        dtype=np.int8,
    )

    for e in range(nedge):

        item = sources[e]

        expected_edge = e + 1

        if int(
            item["edge"]
        ) != expected_edge:

            raise RuntimeError(
                f"Baseline source edge ordering "
                f"mismatch at {expected_edge}"
            )

        i1, j1 = (
            pairs[e]
        )

        di = dates[
            i1 - 1
        ]

        dj = dates[
            j1 - 1
        ]

        if (
            str(
                item["date_i"]
            )
            != di
            or
            str(
                item["date_j"]
            )
            != dj
        ):
            raise RuntimeError(
                f"Date mismatch edge "
                f"{expected_edge}: "
                f"network={di}->{dj}, "
                f"source="
                f"{item['date_i']}->"
                f"{item['date_j']}"
            )

        base = Path(
            item["base_file"]
        ).expanduser().resolve()

        if not base.is_file():
            raise RuntimeError(
                f"Missing baseline source: "
                f"{base}"
            )

        orientation = int(
            item.get(
                "orientation",
                1,
            )
        )

        if orientation not in (
            -1,
            1,
        ):
            raise RuntimeError(
                f"Invalid orientation "
                f"{orientation} edge "
                f"{expected_edge}"
            )

        baseline_files.append(
            base
        )

        orientations[e] = (
            orientation
        )

    # ========================================================
    # pyPSDS temporal reference
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

    if not (
        0 <= reference_idx < ndate
    ):
        raise RuntimeError(
            "Invalid temporal reference index"
        )

    if (
        dates[
            reference_idx
        ]
        != reference_date
    ):
        raise RuntimeError(
            "Temporal reference date mismatch"
        )

    # ========================================================
    # Import mature pySTAMPS geometry
    # ========================================================

    pystamps_source = (
        resolve_pystamps_source(
            args.pystamps_source
        )
    )

    from pystamps.prep.gamma_geometry import (
        build_radar_geometry,
        calculate_bperp_matrix,
    )

    project_root = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

    first_date = dates[0]

    first_par_candidates = [
        project_root
        /
        "RSLC"
        /
        f"{first_date}.rslc.par",

        project_root
        /
        "RSLC"
        /
        f"{first_date}.slc.par",
    ]

    first_par = None

    for p in first_par_candidates:
        if p.is_file():
            first_par = p.resolve()
            break

    if first_par is None:
        raise RuntimeError(
            f"Cannot locate RSLC par for "
            f"{first_date}"
        )

    raw_width = read_gamma_int(
        first_par,
        (
            "range_samples",
            "width",
        ),
    )

    raw_length = read_gamma_int(
        first_par,
        (
            "azimuth_lines",
            "nlines",
        ),
    )

    if (
        np.min(rows) < 0
        or
        np.min(cols) < 0
        or
        np.max(rows) >= raw_length
        or
        np.max(cols) >= raw_width
    ):
        raise RuntimeError(
            "Strict point coordinates do not "
            "fit RSLC 1:1 grid"
        )

    radar_geometry = (
        build_radar_geometry(
            first_par,
            multilook_width=(
                raw_width
            ),
            multilook_length=(
                raw_length
            ),
            mli_parameter_file=None,
            range_looks=1,
            azimuth_looks=1,
        )
    )

    # Mature routine controls its own column workers.
    os.environ[
        "PYSTAMPS_STAGE1_IFG_WORKERS"
    ] = str(
        max(
            1,
            int(args.workers),
        )
    )

    bperp_path = (
        outdir
        /
        "bperp_ifg_m.float32.dat"
    )

    expected_bytes = (
        npoint
        *
        nedge
        *
        np.dtype(
            np.float32
        ).itemsize
    )

    if bperp_path.exists():

        if not args.rebuild:
            raise RuntimeError(
                f"{bperp_path} already exists. "
                "Use --rebuild to replace it."
            )

        bperp_path.unlink()

    # ========================================================
    # BUILD PRODUCTION POINT-WISE BPERP
    # ========================================================

    print("=" * 112)
    print(
        "Step 10R3c2 - mature pySTAMPS "
        "point-wise Bperp generation"
    )
    print("=" * 112)

    print(
        f"config                     : "
        f"{config_path}"
    )

    print(
        f"pystamps source            : "
        f"{pystamps_source}"
    )

    print(
        f"RSLC parameter             : "
        f"{first_par}"
    )

    print(
        f"raw radar grid             : "
        f"{raw_length} x "
        f"{raw_width}"
    )

    print(
        f"range/azimuth looks        : "
        f"1:1"
    )

    print(
        f"strict points              : "
        f"{npoint:,}"
    )

    print(
        f"current network edges      : "
        f"{nedge}"
    )

    print(
        f"temporal reference         : "
        f"{reference_date} "
        f"(index {reference_idx})"
    )

    print(
        f"workers                    : "
        f"{args.workers}"
    )

    print(
        f"output backing file        : "
        f"{bperp_path}"
    )

    print(
        f"expected size              : "
        f"{expected_bytes / 1024**2:.1f} MiB"
    )

    print()

    bperp = calculate_bperp_matrix(
        baseline_files,
        rows,
        cols,
        radar_geometry,
        output_file=(
            bperp_path
        ),
    )

    if bperp.shape != (
        npoint,
        nedge,
    ):
        raise RuntimeError(
            f"Unexpected Bperp shape: "
            f"{bperp.shape}"
        )

    # Align any reversed source to frozen network
    # orientation i -> j.
    reversed_edges = np.flatnonzero(
        orientations < 0
    )

    for e in reversed_edges:
        bperp[:, e] *= -1.0

    if isinstance(
        bperp,
        np.memmap,
    ):
        bperp.flush()

    actual_bytes = (
        bperp_path.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Bperp backing-file size "
            f"{actual_bytes} != "
            f"{expected_bytes}"
        )

    # ========================================================
    # Network projector
    #
    # This reproduces the Stage7 PASS-3 geometry:
    #
    #   bperp_some =
    #       bperp_ifg @ pinv(G_without_master).T
    #
    # master/reference column is zero.
    # ========================================================

    G = build_network_matrix(
        pairs,
        ndate,
    )

    keep = np.asarray(
        [
            i
            for i in range(ndate)
            if i != reference_idx
        ],
        dtype=np.int64,
    )

    Gbase = G[
        :,
        keep,
    ]

    rank = int(
        np.linalg.matrix_rank(
            Gbase
        )
    )

    if rank != ndate - 1:
        raise RuntimeError(
            f"Reduced network rank "
            f"{rank}/{ndate-1}"
        )

    Pbase = np.linalg.pinv(
        Gbase
    )

    ii = (
        pairs[:, 0]
        - 1
    )

    jj = (
        pairs[:, 1]
        - 1
    )

    # ========================================================
    # QA arrays
    # ========================================================

    acq_corr = np.empty(
        npoint,
        dtype=np.float32,
    )

    acq_alpha = np.empty(
        npoint,
        dtype=np.float32,
    )

    acq_relres = np.empty(
        npoint,
        dtype=np.float32,
    )

    ifg_corr = np.empty(
        npoint,
        dtype=np.float32,
    )

    ifg_alpha = np.empty(
        npoint,
        dtype=np.float32,
    )

    ifg_relres = np.empty(
        npoint,
        dtype=np.float32,
    )

    network_relres = np.empty(
        npoint,
        dtype=np.float32,
    )

    edge_sum = np.zeros(
        nedge,
        dtype=np.float64,
    )

    global_b_ss = 0.0
    global_network_ss = 0.0

    bmin = np.inf
    bmax = -np.inf

    nonfinite = 0

    print()
    print("=" * 112)
    print(
        "Network + independent GAMMA S_h parity"
    )
    print("=" * 112)

    for b0 in range(
        0,
        npoint,
        args.batch_size,
    ):

        b1 = min(
            b0
            +
            args.batch_size,
            npoint,
        )

        Bifg = np.asarray(
            bperp[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        S = np.asarray(
            Sh[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        finite = (
            np.all(
                np.isfinite(
                    Bifg
                ),
                axis=1,
            )
            &
            np.all(
                np.isfinite(
                    S
                ),
                axis=1,
            )
        )

        nonfinite += int(
            np.count_nonzero(
                ~finite
            )
        )

        if not np.all(
            finite
        ):
            raise RuntimeError(
                "Non-finite Bperp or S_h"
            )

        edge_sum += np.sum(
            Bifg,
            axis=0,
            dtype=np.float64,
        )

        bmin = min(
            bmin,
            float(
                np.min(
                    Bifg
                )
            ),
        )

        bmax = max(
            bmax,
            float(
                np.max(
                    Bifg
                )
            ),
        )

        # --------------------------------------------
        # IFG -> acquisition Bperp
        # --------------------------------------------

        Bsome = (
            Bifg
            @ Pbase.T
        )

        B = np.zeros(
            (
                b1 - b0,
                ndate,
            ),
            dtype=np.float64,
        )

        B[
            :,
            keep
        ] = Bsome

        # --------------------------------------------
        # Network self-consistency
        # --------------------------------------------

        Bpred_ifg = (
            B[:, jj]
            -
            B[:, ii]
        )

        net_err = (
            Bifg
            -
            Bpred_ifg
        )

        net_rms = np.sqrt(
            np.mean(
                net_err
                *
                net_err,
                axis=1,
            )
        )

        bifg_rms = np.sqrt(
            np.mean(
                Bifg
                *
                Bifg,
                axis=1,
            )
        )

        net_rel = np.divide(
            net_rms,
            bifg_rms,
            out=np.zeros(
                b1 - b0,
                dtype=np.float64,
            ),
            where=(
                bifg_rms > 0
            ),
        )

        network_relres[
            b0:b1
        ] = net_rel.astype(
            np.float32
        )

        global_network_ss += float(
            np.sum(
                net_err
                *
                net_err
            )
        )

        global_b_ss += float(
            np.sum(
                Bifg
                *
                Bifg
            )
        )

        # --------------------------------------------
        # Independent QA in acquisition domain
        #
        # S_h is already referenced to the same
        # temporal acquisition.
        # --------------------------------------------

        corr_a = row_correlation(
            B,
            S,
        )

        (
            alpha_a,
            rel_a,
        ) = (
            fit_scale_and_relative_residual(
                B,
                S,
            )
        )

        acq_corr[
            b0:b1
        ] = corr_a.astype(
            np.float32
        )

        acq_alpha[
            b0:b1
        ] = alpha_a.astype(
            np.float32
        )

        acq_relres[
            b0:b1
        ] = rel_a.astype(
            np.float32
        )

        # --------------------------------------------
        # Independent QA directly in IFG domain
        # --------------------------------------------

        DSh = (
            S[:, jj]
            -
            S[:, ii]
        )

        corr_e = row_correlation(
            Bifg,
            DSh,
        )

        (
            alpha_e,
            rel_e,
        ) = (
            fit_scale_and_relative_residual(
                Bifg,
                DSh,
            )
        )

        ifg_corr[
            b0:b1
        ] = corr_e.astype(
            np.float32
        )

        ifg_alpha[
            b0:b1
        ] = alpha_e.astype(
            np.float32
        )

        ifg_relres[
            b0:b1
        ] = rel_e.astype(
            np.float32
        )

        print(
            f"  {b1:,}/{npoint:,}"
        )

    # ========================================================
    # Save QA
    # ========================================================

    edge_mean = (
        edge_sum
        /
        float(
            npoint
        )
    )

    np.save(
        outdir
        /
        "bperp_mean_by_ifg_m.npy",
        edge_mean,
    )

    np.save(
        outdir
        /
        "network_reconstruction_relative_rms_by_point.npy",
        network_relres,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_acquisition_signed_corr.npy",
        acq_corr,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_acquisition_alpha_rad_per_m2.npy",
        acq_alpha,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_acquisition_relative_residual.npy",
        acq_relres,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_ifg_signed_corr.npy",
        ifg_corr,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_ifg_alpha_rad_per_m2.npy",
        ifg_alpha,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_ifg_relative_residual.npy",
        ifg_relres,
    )

    # ========================================================
    # Summary
    # ========================================================

    global_network_relative = (
        np.sqrt(
            global_network_ss
            /
            global_b_ss
        )
        if global_b_ss > 0
        else np.nan
    )

    alpha_difference = (
        np.asarray(
            acq_alpha,
            dtype=np.float64,
        )
        -
        np.asarray(
            ifg_alpha,
            dtype=np.float64,
        )
    )

    print()
    print("=" * 112)
    print(
        "Production point-wise Bperp"
    )
    print("=" * 112)

    print(
        f"shape                      : "
        f"({npoint}, {nedge})"
    )

    print(
        f"dtype                      : "
        f"float32"
    )

    print(
        f"min / max                  : "
        f"{bmin:.6f} / "
        f"{bmax:.6f} m"
    )

    qprint(
        "mean Bperp by IFG "
        "p01/p05/p50/p95/p99 [m]:",
        edge_mean,
        fmt=".3f",
    )

    print()
    print("=" * 112)
    print(
        "SB-network Bperp consistency"
    )
    print("=" * 112)

    print(
        f"reduced network rank       : "
        f"{rank}/{ndate-1}"
    )

    print(
        f"global relative RMS        : "
        f"{global_network_relative:.9e}"
    )

    qprint(
        "point relative RMS "
        "p01/p05/p50/p95/p99:",
        network_relres,
        fmt=".9e",
    )

    print()
    print("=" * 112)
    print(
        "Independent GAMMA S_h parity - acquisition domain"
    )
    print("=" * 112)

    qprint(
        "signed corr "
        "p01/p05/p50/p95/p99:",
        acq_corr,
    )

    qprint(
        "|corr| "
        "p01/p05/p50/p95/p99:",
        np.abs(
            acq_corr
        ),
    )

    qprint(
        "alpha "
        "p01/p05/p50/p95/p99 "
        "[rad/m^2]:",
        acq_alpha,
        fmt=".9e",
    )

    qprint(
        "relative residual "
        "p01/p05/p50/p95/p99:",
        acq_relres,
    )

    print()
    print("=" * 112)
    print(
        "Independent GAMMA S_h parity - IFG domain"
    )
    print("=" * 112)

    qprint(
        "signed corr "
        "p01/p05/p50/p95/p99:",
        ifg_corr,
    )

    qprint(
        "|corr| "
        "p01/p05/p50/p95/p99:",
        np.abs(
            ifg_corr
        ),
    )

    qprint(
        "alpha "
        "p01/p05/p50/p95/p99 "
        "[rad/m^2]:",
        ifg_alpha,
        fmt=".9e",
    )

    qprint(
        "relative residual "
        "p01/p05/p50/p95/p99:",
        ifg_relres,
    )

    qprint(
        "|alpha_acq-alpha_ifg| "
        "p50/p90/p95/p99/max "
        "[rad/m^2]:",
        np.abs(
            alpha_difference
        ),
        qs=(
            50,
            90,
            95,
            99,
            100,
        ),
        fmt=".9e",
    )

    # ========================================================
    # Decision
    #
    # S_h is independent QA only.
    #
    # We accept either Bperp sign convention because Stage7
    # estimates the coefficient sign from phase.
    # ========================================================

    acq_abs = np.abs(
        np.asarray(
            acq_corr,
            dtype=np.float64,
        )
    )

    ifg_abs = np.abs(
        np.asarray(
            ifg_corr,
            dtype=np.float64,
        )
    )

    acq_rel = np.asarray(
        acq_relres,
        dtype=np.float64,
    )

    ifg_rel = np.asarray(
        ifg_relres,
        dtype=np.float64,
    )

    all_finite = (
        np.all(
            np.isfinite(
                acq_abs
            )
        )
        and
        np.all(
            np.isfinite(
                ifg_abs
            )
        )
        and
        np.all(
            np.isfinite(
                acq_rel
            )
        )
        and
        np.all(
            np.isfinite(
                ifg_rel
            )
        )
    )

    if not all_finite:

        status = (
            "REVIEW_NONFINITE_BPERP_QA"
        )

    elif (
        global_network_relative
        >
        5.0e-3
    ):

        status = (
            "REVIEW_BPERP_NETWORK_CONSISTENCY"
        )

    elif (
        np.percentile(
            acq_abs,
            5,
        )
        < 0.99
    ):

        status = (
            "REVIEW_BPERP_SH_ACQUISITION_PARITY"
        )

    elif (
        np.percentile(
            ifg_abs,
            5,
        )
        < 0.99
    ):

        status = (
            "REVIEW_BPERP_SH_IFG_PARITY"
        )

    elif (
        np.percentile(
            acq_rel,
            95,
        )
        > 0.05
    ):

        status = (
            "REVIEW_BPERP_SH_SCALE_MODEL"
        )

    else:

        status = (
            "PASS_PRODUCTION_POINTWISE_BPERP_READY"
        )

    layout = {
        "file":
            str(
                bperp_path
            ),

        "dtype":
            "float32",

        "byte_order":
            "native",

        "order":
            "C",

        "shape":
            [
                int(
                    npoint
                ),
                int(
                    nedge
                ),
            ],

        "units":
            "m",

        "columns":
            (
                "current pyPSDS network edge order"
            ),
    }

    layout_path = (
        outdir
        /
        "bperp_ifg_layout.json"
    )

    layout_path.write_text(
        json.dumps(
            layout,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    manifest = {
        "format":
            "pyPSDS-GAMMA-production-pointwise-bperp-v09",

        "status":
            status,

        "pystamps_source":
            str(
                pystamps_source
            ),

        "generator":
            (
                "pystamps.prep.gamma_geometry."
                "calculate_bperp_matrix"
            ),

        "RSLC_parameter":
            str(
                first_par
            ),

        "grid": {
            "rows":
                int(
                    raw_length
                ),

            "cols":
                int(
                    raw_width
                ),

            "range_looks":
                1,

            "azimuth_looks":
                1,
        },

        "points":
            int(
                npoint
            ),

        "network_edges":
            int(
                nedge
            ),

        "temporal_reference": {
            "index_zero_based":
                int(
                    reference_idx
                ),

            "date":
                reference_date,
        },

        "baseline_sources":
            str(
                source_manifest_path
            ),

        "bperp": {
            **layout,

            "minimum_m":
                float(
                    bmin
                ),

            "maximum_m":
                float(
                    bmax
                ),
        },

        "network_consistency": {
            "rank":
                int(
                    rank
                ),

            "expected_rank":
                int(
                    ndate - 1
                ),

            "global_relative_rms":
                float(
                    global_network_relative
                ),
        },

        "independent_Sh_QA": {
            "role":
                (
                    "independent geometry QA only; "
                    "not production Stage7 predictor"
                ),

            "acquisition_abs_corr_p05":
                float(
                    np.percentile(
                        acq_abs,
                        5,
                    )
                ),

            "acquisition_relative_residual_p95":
                float(
                    np.percentile(
                        acq_rel,
                        95,
                    )
                ),

            "ifg_abs_corr_p05":
                float(
                    np.percentile(
                        ifg_abs,
                        5,
                    )
                ),

            "ifg_relative_residual_p95":
                float(
                    np.percentile(
                        ifg_rel,
                        95,
                    )
                ),
        },

        "bp2_mat_generated":
            False,

        "stage7_executed":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "pointwise_bperp_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Bperp layout              : "
        f"{layout_path}"
    )

    print(
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R3c2 STATUS: "
        f"{status}"
    )

    print(
        "bp2.mat has NOT been generated."
    )

    print(
        "No Stage-7/Stage-8 correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
