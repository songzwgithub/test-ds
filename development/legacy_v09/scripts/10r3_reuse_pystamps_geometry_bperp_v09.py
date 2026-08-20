#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        "Unable to locate pystamps-gamma source tree. "
        "Use --pystamps-source /path/to/pystamps-gamma"
    )


def read_network_itab(
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

            vals = []

            for tok in s.split():
                try:
                    vals.append(int(tok))
                except ValueError:
                    continue

            if len(vals) < 2:
                continue

            i, j = vals[:2]

            if not (
                1 <= i <= ndate
                and
                1 <= j <= ndate
            ):
                continue

            if i == j:
                raise RuntimeError(
                    f"Self-edge in network: {i},{j}"
                )

            pairs.append(
                (i, j)
            )

    a = np.asarray(
        pairs,
        dtype=np.int64,
    )

    if a.ndim != 2 or a.shape[1] != 2:
        raise RuntimeError(
            f"Invalid network.itab: {path}"
        )

    return a


def network_matrix(
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

    r = np.arange(
        pairs.shape[0],
        dtype=np.int64,
    )

    G[
        r,
        pairs[:, 0] - 1
    ] = -1.0

    G[
        r,
        pairs[:, 1] - 1
    ] = +1.0

    return G


def quantile_line(
    name,
    x,
    qs=(1,5,50,95,99),
    fmt=".6f",
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    q = np.percentile(
        x,
        qs,
    )

    print(name)
    print(
        "  "
        +
        " / ".join(
            format(v, fmt)
            for v in q
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
        "--height-file",
        default=None,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32768,
    )

    args = ap.parse_args()

    # ========================================================
    # Load pyPSDS
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

    sensdir = (
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
        "r3_mature_geometry"
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
        all_rows[strict_ids],
        dtype=np.int32,
    )

    cols = np.asarray(
        all_cols[strict_ids],
        dtype=np.int32,
    )

    Sh = np.load(
        sensdir
        /
        "topographic_phase_sensitivity_rad_per_m.npy",
        mmap_mode="r",
    )

    npoint = strict_ids.size
    ndate = len(stack.dates)

    dates = [
        str(x)
        for x in stack.dates
    ]

    if Sh.shape != (
        npoint,
        ndate,
    ):
        raise RuntimeError(
            "Sensitivity shape mismatch"
        )

    pairs = read_network_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    nedge = pairs.shape[0]

    # ========================================================
    # Import mature pySTAMPS implementation
    # ========================================================

    pystamps_source = (
        resolve_pystamps_source(
            args.pystamps_source
        )
    )

    from pystamps.prep.gamma_sbas import (
        load_gamma_sbas_project,
    )

    from pystamps.prep.gamma_geometry import (
        build_radar_geometry,
        calculate_bperp_matrix,
    )

    from pystamps.prep.gamma_lonlat import (
        ensure_gamma_radar_lonlat,
    )

    from pystamps.prep.gamma_observations import (
        resolve_radar_geometry_files,
        sample_radar_geometry,
        lonlat_to_local_xy,
    )

    # ========================================================
    # Load GAMMA project using mature pySTAMPS parser
    # ========================================================

    project_root = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

    project = load_gamma_sbas_project(
        project_root
    )

    if (
        project.width is None
        or
        project.length is None
    ):
        raise RuntimeError(
            "pySTAMPS could not resolve GAMMA radar dimensions"
        )

    width = int(
        project.width
    )

    length = int(
        project.length
    )

    if width != 2000 or length != 600:
        print(
            "WARNING: resolved radar grid is "
            f"{width}x{length}; "
            "expected current pyPSDS stack 2000x600"
        )

    if (
        np.max(cols) >= width
        or
        np.max(rows) >= length
    ):
        raise RuntimeError(
            "pyPSDS coordinates exceed GAMMA project grid"
        )

    # ========================================================
    # Build mature GAMMA radar geometry
    # ========================================================

    if not project.acquisitions:
        raise RuntimeError(
            "No GAMMA acquisitions found"
        )

    first_acq = project.acquisitions[0]

    radar_geometry = build_radar_geometry(
        first_acq.par,
        multilook_width=width,
        multilook_length=length,
        mli_parameter_file=(
            first_acq.mli_par
        ),
    )

    # ========================================================
    # Resolve/generate lon-lat using mature pySTAMPS
    # ========================================================

    lonlat_result = (
        ensure_gamma_radar_lonlat(
            project_root,
            radar_width=width,
            radar_length=length,
            range_looks=(
                radar_geometry.range_looks
            ),
            azimuth_looks=(
                radar_geometry.azimuth_looks
            ),
            dem_directory=(
                project.dem_dir
            ),
            force=False,
        )
    )

    # Prefer explicitly supplied height.
    height_file = None

    if args.height_file:
        height_file = (
            Path(args.height_file)
            .expanduser()
            .resolve()
        )
    else:
        known_height = (
            project_root
            /
            "DEM_prep"
            /
            "20151212.hgt"
        )

        if known_height.is_file():
            height_file = known_height

    geometry_files = (
        resolve_radar_geometry_files(
            project,
            longitude_file=(
                lonlat_result
                .longitude_file
            ),
            latitude_file=(
                lonlat_result
                .latitude_file
            ),
            height_file=(
                height_file
            ),
        )
    )

    samples = sample_radar_geometry(
        geometry_files,
        rows,
        cols,
        width=width,
        length=length,
    )

    valid_geometry = np.asarray(
        samples.valid,
        dtype=bool,
    )

    nvalid_geometry = int(
        np.count_nonzero(
            valid_geometry
        )
    )

    if nvalid_geometry != npoint:
        bad = (
            npoint
            -
            nvalid_geometry
        )

        raise RuntimeError(
            f"{bad} strict points have invalid "
            "mature pySTAMPS lon/lat/height geometry. "
            "Do not silently drop them."
        )

    longitude = np.asarray(
        samples.longitude,
        dtype=np.float64,
    )

    latitude = np.asarray(
        samples.latitude,
        dtype=np.float64,
    )

    height = np.asarray(
        samples.height,
        dtype=np.float32,
    )

    local_xy, ll0 = (
        lonlat_to_local_xy(
            longitude,
            latitude,
            heading_degrees=(
                radar_geometry.heading
            ),
        )
    )

    local_xy = np.asarray(
        local_xy,
        dtype=np.float32,
    )

    # ========================================================
    # Match current 108-edge network to GAMMA .base files
    # ========================================================

    direct = {}
    reverse = {}

    duplicates = []

    for ifg in project.interferograms:

        a = str(
            ifg.master_date
        )

        b = str(
            ifg.slave_date
        )

        key = (
            a,
            b,
        )

        if key in direct:
            duplicates.append(
                key
            )
        else:
            direct[key] = ifg

        rkey = (
            b,
            a,
        )

        if rkey not in reverse:
            reverse[rkey] = ifg

    if duplicates:
        print(
            "WARNING: duplicate GAMMA IFG date pairs:"
        )

        for p in duplicates[:20]:
            print(
                "  ",
                p,
            )

    baseline_files = []

    orientation = np.ones(
        nedge,
        dtype=np.int8,
    )

    mapping_rows = []

    missing = []

    direct_count = 0
    reversed_count = 0

    for e, (
        i1,
        j1,
    ) in enumerate(pairs):

        di = dates[
            i1 - 1
        ]

        dj = dates[
            j1 - 1
        ]

        key = (
            di,
            dj,
        )

        if key in direct:

            ifg = direct[
                key
            ]

            sign = +1
            direct_count += 1

        elif key in reverse:

            ifg = reverse[
                key
            ]

            sign = -1
            reversed_count += 1

        else:

            missing.append(
                (
                    e + 1,
                    di,
                    dj,
                )
            )
            continue

        base = Path(
            ifg.base
        ).resolve()

        if not base.is_file():
            missing.append(
                (
                    e + 1,
                    di,
                    dj,
                    str(base),
                )
            )
            continue

        baseline_files.append(
            base
        )

        orientation[e] = sign

        mapping_rows.append(
            {
                "edge_index_1based":
                    int(e + 1),

                "network_i_1based":
                    int(i1),

                "network_j_1based":
                    int(j1),

                "date_i":
                    di,

                "date_j":
                    dj,

                "base_file":
                    str(base),

                "base_orientation":
                    (
                        "direct"
                        if sign > 0
                        else "reversed_sign"
                    ),
            }
        )

    if missing:

        print()
        print(
            "Missing current-network GAMMA baseline files:"
        )

        for x in missing[:30]:
            print(
                "  ",
                x,
            )

        raise RuntimeError(
            f"Unable to map {len(missing)} "
            "network edges to GAMMA .base files"
        )

    if len(
        baseline_files
    ) != nedge:

        raise RuntimeError(
            "Baseline list length mismatch"
        )

    # ========================================================
    # Generate mature point-wise signed Bperp
    #
    # IMPORTANT:
    # output_file is a raw float32 memmap, shape is stored
    # in the manifest.
    # ========================================================

    bperp_raw_path = (
        outdir
        /
        "bperp_ifg_m.float32.dat"
    )

    bperp_ifg = (
        calculate_bperp_matrix(
            baseline_files,
            rows,
            cols,
            radar_geometry,
            output_file=(
                bperp_raw_path
            ),
        )
    )

    if bperp_ifg.shape != (
        npoint,
        nedge,
    ):
        raise RuntimeError(
            f"Bperp shape mismatch: "
            f"{bperp_ifg.shape}"
        )

    # Align .base orientation to current network i -> j.
    reversed_ix = np.flatnonzero(
        orientation < 0
    )

    for e in reversed_ix:
        bperp_ifg[
            :,
            e
        ] *= -1.0

    if isinstance(
        bperp_ifg,
        np.memmap,
    ):
        bperp_ifg.flush()

    # ========================================================
    # Save mature coordinate products
    # ========================================================

    np.save(
        outdir
        /
        "longitude_deg.npy",
        longitude,
    )

    np.save(
        outdir
        /
        "latitude_deg.npy",
        latitude,
    )

    np.save(
        outdir
        /
        "height_m.npy",
        height,
    )

    np.save(
        outdir
        /
        "local_xy_m.npy",
        local_xy,
    )

    np.save(
        outdir
        /
        "ll0_lonlat_deg.npy",
        np.asarray(
            ll0,
            dtype=np.float64,
        ),
    )

    np.save(
        outdir
        /
        "bperp_ifg_mean_m.npy",
        np.mean(
            np.asarray(
                bperp_ifg,
                dtype=np.float64,
            ),
            axis=0,
        ).astype(
            np.float64
        ),
    )

    mapping_json = (
        outdir
        /
        "network_to_gamma_base_mapping.json"
    )

    mapping_json.write_text(
        json.dumps(
            mapping_rows,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Network reconstruction:
    #
    # physical point-wise Bperp IFGs -> acquisition Bperp
    #
    # Current pyPSDS temporal reference index = 0.
    # ========================================================

    G = network_matrix(
        pairs,
        ndate,
    )

    reference_idx = 0

    keep = np.asarray(
        [
            j
            for j in range(ndate)
            if j != reference_idx
        ],
        dtype=np.int64,
    )

    Gred = (
        G[:, keep]
    )

    rank = int(
        np.linalg.matrix_rank(
            Gred
        )
    )

    if rank != ndate - 1:
        raise RuntimeError(
            f"Bperp network rank "
            f"{rank}/{ndate-1}"
        )

    P = np.linalg.pinv(
        Gred
    )

    # ========================================================
    # Independent geometry parity:
    #
    # mature StaMPS Bperp temporal geometry
    # versus exact GAMMA finite-difference S_h.
    #
    # We do NOT require exact scale because:
    #
    #   S_h ~ alpha_p * Bperp
    #
    # and alpha depends on range/incidence geometry.
    # ========================================================

    abs_corr = np.empty(
        npoint,
        dtype=np.float32,
    )

    signed_corr = np.empty(
        npoint,
        dtype=np.float32,
    )

    alpha = np.empty(
        npoint,
        dtype=np.float32,
    )

    relative_residual = np.empty(
        npoint,
        dtype=np.float32,
    )

    for b0 in range(
        0,
        npoint,
        args.batch_size,
    ):

        b1 = min(
            b0 + args.batch_size,
            npoint,
        )

        Bifg = np.asarray(
            bperp_ifg[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        Bsome = (
            Bifg
            @ P.T
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

        S = np.asarray(
            Sh[
                b0:b1,
                :
            ],
            dtype=np.float64,
        )

        # --------------------------------------------
        # correlation after removing temporal mean
        # --------------------------------------------

        Bc = (
            B
            -
            np.mean(
                B,
                axis=1,
                keepdims=True,
            )
        )

        Sc = (
            S
            -
            np.mean(
                S,
                axis=1,
                keepdims=True,
            )
        )

        num = np.sum(
            Bc * Sc,
            axis=1,
        )

        den = np.sqrt(
            np.sum(
                Bc * Bc,
                axis=1,
            )
            *
            np.sum(
                Sc * Sc,
                axis=1,
            )
        )

        corr = np.divide(
            num,
            den,
            out=np.full(
                b1 - b0,
                np.nan,
                dtype=np.float64,
            ),
            where=(
                den > 0
            ),
        )

        # --------------------------------------------
        # best physical linear mapping through origin:
        #
        # S_h ~= alpha_p * Bperp_acq
        # --------------------------------------------

        bb = np.sum(
            B * B,
            axis=1,
        )

        bs = np.sum(
            B * S,
            axis=1,
        )

        a = np.divide(
            bs,
            bb,
            out=np.full(
                b1 - b0,
                np.nan,
                dtype=np.float64,
            ),
            where=(
                bb > 0
            ),
        )

        pred = (
            a[:, None]
            *
            B
        )

        err = (
            S - pred
        )

        err_rms = np.sqrt(
            np.mean(
                err * err,
                axis=1,
            )
        )

        s_rms = np.sqrt(
            np.mean(
                S * S,
                axis=1,
            )
        )

        rel = np.divide(
            err_rms,
            s_rms,
            out=np.full(
                b1 - b0,
                np.nan,
                dtype=np.float64,
            ),
            where=(
                s_rms > 0
            ),
        )

        signed_corr[
            b0:b1
        ] = corr.astype(
            np.float32
        )

        abs_corr[
            b0:b1
        ] = np.abs(
            corr
        ).astype(
            np.float32
        )

        alpha[
            b0:b1
        ] = a.astype(
            np.float32
        )

        relative_residual[
            b0:b1
        ] = rel.astype(
            np.float32
        )

        print(
            f"[geometry parity] "
            f"{b1:,}/{npoint:,}"
        )

    np.save(
        outdir
        /
        "bperp_vs_Sh_signed_corr.npy",
        signed_corr,
    )

    np.save(
        outdir
        /
        "bperp_vs_Sh_abs_corr.npy",
        abs_corr,
    )

    np.save(
        outdir
        /
        "bperp_to_Sh_alpha_rad_per_m2.npy",
        alpha,
    )

    np.save(
        outdir
        /
        "bperp_to_Sh_relative_residual.npy",
        relative_residual,
    )

    # ========================================================
    # Summary
    # ========================================================

    bmean = np.asarray(
        np.mean(
            np.asarray(
                bperp_ifg,
                dtype=np.float64,
            ),
            axis=0,
        ),
        dtype=np.float64,
    )

    print()
    print("=" * 112)
    print(
        "Step 10R3 - mature pySTAMPS GAMMA "
        "geometry / Bperp bridge"
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
        f"GAMMA project              : "
        f"{project_root}"
    )

    print(
        f"radar grid                 : "
        f"{length} rows x {width} cols"
    )

    print(
        f"range / azimuth looks      : "
        f"{radar_geometry.range_looks}:"
        f"{radar_geometry.azimuth_looks}"
    )

    print(
        f"strict points              : "
        f"{npoint:,}"
    )

    print(
        f"current network edges      : "
        f"{nedge}"
    )

    print()
    print("=" * 112)
    print(
        "Current-network -> GAMMA baseline mapping"
    )
    print("=" * 112)

    print(
        f"direct orientation         : "
        f"{direct_count}"
    )

    print(
        f"reversed orientation       : "
        f"{reversed_count}"
    )

    print(
        f"missing baseline files     : "
        f"{len(missing)}"
    )

    print(
        f"unique .base files         : "
        f"{len(set(map(str, baseline_files)))}"
    )

    print()
    print("=" * 112)
    print(
        "Mature pySTAMPS point geometry"
    )
    print("=" * 112)

    print(
        f"valid lon/lat/height       : "
        f"{nvalid_geometry:,}/{npoint:,}"
    )

    print(
        f"longitude range            : "
        f"{longitude.min():.8f} .. "
        f"{longitude.max():.8f}"
    )

    print(
        f"latitude range             : "
        f"{latitude.min():.8f} .. "
        f"{latitude.max():.8f}"
    )

    print(
        f"local X range              : "
        f"{local_xy[:,0].min():.3f} .. "
        f"{local_xy[:,0].max():.3f} m"
    )

    print(
        f"local Y range              : "
        f"{local_xy[:,1].min():.3f} .. "
        f"{local_xy[:,1].max():.3f} m"
    )

    print(
        f"ll0                        : "
        f"{ll0}"
    )

    print()
    print("=" * 112)
    print(
        "Mature point-wise Bperp"
    )
    print("=" * 112)

    print(
        f"shape                      : "
        f"({npoint}, {nedge})"
    )

    print(
        f"raw backing file           : "
        f"{bperp_raw_path}"
    )

    print(
        f"min / max                  : "
        f"{np.min(bperp_ifg):.6f} / "
        f"{np.max(bperp_ifg):.6f} m"
    )

    quantile_line(
        "mean Bperp by IFG "
        "p01/p05/p50/p95/p99 [m]:",
        bmean,
        fmt=".3f",
    )

    print()
    print("=" * 112)
    print(
        "Independent Bperp vs GAMMA S_h geometry parity"
    )
    print("=" * 112)

    quantile_line(
        "signed corr "
        "p01/p05/p50/p95/p99:",
        signed_corr,
    )

    quantile_line(
        "|corr| "
        "p01/p05/p50/p95/p99:",
        abs_corr,
    )

    quantile_line(
        "best-fit alpha "
        "p01/p05/p50/p95/p99 [rad/m^2]:",
        alpha,
        fmt=".9e",
    )

    quantile_line(
        "relative S_h residual "
        "p01/p05/p50/p95/p99:",
        relative_residual,
    )

    abs_corr_valid = (
        np.asarray(
            abs_corr,
            dtype=np.float64,
        )
    )

    rel_valid = (
        np.asarray(
            relative_residual,
            dtype=np.float64,
        )
    )

    finite = (
        np.isfinite(
            abs_corr_valid
        )
        &
        np.isfinite(
            rel_valid
        )
    )

    if not np.all(
        finite
    ):
        status = (
            "REVIEW_NONFINITE_GEOMETRY_PARITY"
        )

    elif np.percentile(
        abs_corr_valid,
        5,
    ) < 0.95:
        status = (
            "REVIEW_BPERP_SH_TEMPORAL_MISMATCH"
        )

    else:
        status = (
            "PASS_MATURE_PYSTAMPS_GEOMETRY_READY"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-pystamps-mature-geometry-v09",

        "status":
            status,

        "pystamps_source":
            str(pystamps_source),

        "gamma_project":
            str(project_root),

        "points":
            int(npoint),

        "acquisitions":
            int(ndate),

        "network_edges":
            int(nedge),

        "range_looks":
            int(
                radar_geometry.range_looks
            ),

        "azimuth_looks":
            int(
                radar_geometry.azimuth_looks
            ),

        "geometry": {
            "longitude_file":
                str(
                    geometry_files.longitude
                ),

            "latitude_file":
                str(
                    geometry_files.latitude
                ),

            "height_file":
                str(
                    geometry_files.height
                ),

            "valid_points":
                int(
                    nvalid_geometry
                ),

            "local_xy_units":
                "metres",

            "ll0_lonlat_deg":
                np.asarray(
                    ll0,
                    dtype=float,
                ).tolist(),
        },

        "baseline": {
            "source":
                (
                    "pystamps.prep.gamma_geometry."
                    "calculate_bperp_matrix"
                ),

            "semantics":
                (
                    "signed point-wise perpendicular "
                    "baseline for current pyPSDS network"
                ),

            "units":
                "metres",

            "shape":
                [
                    int(npoint),
                    int(nedge),
                ],

            "raw_file":
                str(
                    bperp_raw_path
                ),

            "direct_edges":
                int(
                    direct_count
                ),

            "reversed_edges":
                int(
                    reversed_count
                ),
        },

        "Sh_usage":
            (
                "independent QA only; "
                "not production bp2 predictor"
            ),

        "stage7_executed":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "mature_geometry_manifest.json"
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
        f"manifest                   : "
        f"{manifest_path}"
    )

    print()
    print(
        f"STEP 10R3 STATUS: {status}"
    )

    print(
        "No Stage-7/Stage-8 correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
