#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from pypsds.prototype import open_from_config


DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def resolve_pystamps_source(explicit):
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
        "Cannot locate pystamps-gamma source. "
        "Use --pystamps-source."
    )


def read_itab(path, ndate):

    pairs = []

    with Path(path).open(
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
                    ints.append(int(token))
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


def extract_dates_from_path(path):

    texts = [
        path.name,
        path.parent.name,
        str(path),
    ]

    for text in texts:

        ds = DATE_RE.findall(
            text
        )

        if len(ds) >= 2:

            # Preserve order, remove immediate duplicates.
            out = []

            for d in ds:
                if (
                    not out
                    or d != out[-1]
                ):
                    out.append(d)

            if len(out) >= 2:
                return (
                    out[0],
                    out[1],
                )

    return None


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

    q = np.percentile(
        a,
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

    outdir = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge"
        /
        "r3b_grid_adapter"
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

    npoint = strict_ids.size
    ndate = len(stack.dates)

    dates = [
        str(x)
        for x in stack.dates
    ]

    pairs = read_itab(
        netdir
        /
        "network.itab",
        ndate,
    )

    nedge = pairs.shape[0]

    # ========================================================
    # Import mature pySTAMPS helpers
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
        calculate_candidate_geometry,
        read_baseline_model,
    )

    from pystamps.prep.gamma_lonlat import (
        ensure_gamma_radar_lonlat,
    )

    from pystamps.prep.gamma_binary import (
        sample_gamma_raster,
    )

    from pystamps.prep.gamma_observations import (
        lonlat_to_local_xy,
    )

    project_root = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

    project = load_gamma_sbas_project(
        project_root
    )

    if not project.acquisitions:
        raise RuntimeError(
            "No GAMMA acquisitions"
        )

    first = project.acquisitions[0]

    # ========================================================
    # A. 1:1 raw-RSLC radar geometry for Bperp
    # ========================================================

    raw_width = 2000
    raw_length = 600

    raw_geometry = build_radar_geometry(
        first.par,
        multilook_width=raw_width,
        multilook_length=raw_length,
        mli_parameter_file=None,
        range_looks=1,
        azimuth_looks=1,
    )

    candidate_geometry = (
        calculate_candidate_geometry(
            rows,
            cols,
            raw_geometry,
        )
    )

    look_angle = np.asarray(
        candidate_geometry.look_angle,
        dtype=np.float64,
    )

    slant_range = np.asarray(
        candidate_geometry.slant_range,
        dtype=np.float64,
    )

    if (
        not np.all(
            np.isfinite(
                look_angle
            )
        )
        or
        not np.all(
            np.isfinite(
                slant_range
            )
        )
    ):
        raise RuntimeError(
            "Non-finite raw-RSLC candidate geometry"
        )

    # ========================================================
    # B. Mature 4:1 lon/lat source
    # ========================================================

    mli_width = int(
        project.width
    )

    mli_length = int(
        project.length
    )

    if (
        mli_width != 500
        or
        mli_length != 600
    ):
        raise RuntimeError(
            "Expected mature pySTAMPS 4:1 "
            f"grid 500x600, got "
            f"{mli_width}x{mli_length}"
        )

    lonlat_result = (
        ensure_gamma_radar_lonlat(
            project_root,
            radar_width=mli_width,
            radar_length=mli_length,
            range_looks=4,
            azimuth_looks=1,
            dem_directory=(
                project.dem_dir
            ),
            force=False,
        )
    )

    lon_file = Path(
        lonlat_result.longitude_file
    )

    lat_file = Path(
        lonlat_result.latitude_file
    )

    # ========================================================
    # C. Convert raw RSLC column -> fractional 4:1 pixel
    #
    # Mature calculate_candidate_geometry convention:
    #
    #   raw_center = c_mli * 4 + 1.5
    #
    # therefore:
    #
    #   c_mli_fractional = (raw_col - 1.5) / 4
    #
    # Azimuth looks = 1, so row is unchanged.
    # ========================================================

    u = (
        cols.astype(np.float64)
        -
        1.5
    ) / 4.0

    u = np.clip(
        u,
        0.0,
        mli_width - 1.0,
    )

    c0 = np.floor(
        u
    ).astype(
        np.int64
    )

    c1 = np.minimum(
        c0 + 1,
        mli_width - 1,
    )

    w = (
        u - c0
    ).astype(
        np.float64
    )

    row64 = rows.astype(
        np.int64
    )

    lon0 = sample_gamma_raster(
        lon_file,
        row64,
        c0,
        width=mli_width,
        length=mli_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lon1 = sample_gamma_raster(
        lon_file,
        row64,
        c1,
        width=mli_width,
        length=mli_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lat0 = sample_gamma_raster(
        lat_file,
        row64,
        c0,
        width=mli_width,
        length=mli_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lat1 = sample_gamma_raster(
        lat_file,
        row64,
        c1,
        width=mli_width,
        length=mli_length,
        dtype="float",
    ).astype(
        np.float64
    )

    longitude = (
        (1.0 - w) * lon0
        +
        w * lon1
    )

    latitude = (
        (1.0 - w) * lat0
        +
        w * lat1
    )

    valid_ll = (
        np.isfinite(longitude)
        &
        np.isfinite(latitude)
        &
        (longitude >= -180.0)
        &
        (longitude <= 180.0)
        &
        (latitude >= -90.0)
        &
        (latitude <= 90.0)
        &
        ~(
            (np.abs(longitude) < 1e-8)
            &
            (np.abs(latitude) < 1e-8)
        )
    )

    nvalid_ll = int(
        np.count_nonzero(
            valid_ll
        )
    )

    if nvalid_ll != npoint:

        bad = np.flatnonzero(
            ~valid_ll
        )

        print()
        print(
            "Invalid interpolated lon/lat "
            f"points: {bad.size}"
        )

        print(
            "first invalid point indices:",
            bad[:20].tolist(),
        )

    # Only build metric coordinates if all are valid.
    if nvalid_ll == npoint:

        local_xy, ll0 = (
            lonlat_to_local_xy(
                longitude,
                latitude,
                heading_degrees=(
                    raw_geometry.heading
                ),
            )
        )

        local_xy = np.asarray(
            local_xy,
            dtype=np.float32,
        )

    else:

        local_xy = np.full(
            (npoint, 2),
            np.nan,
            dtype=np.float32,
        )

        ll0 = np.asarray(
            [np.nan, np.nan],
            dtype=np.float64,
        )

    # ========================================================
    # D. Existing .base coverage for current 108-edge network
    # ========================================================

    base_candidates = sorted(
        {
            p.resolve()
            for p in (
                project_root
                /
                "DIFF"
            ).rglob("*.base")
            if p.is_file()
        }
    )

    by_canonical = {}

    parsed_base_count = 0

    for p in base_candidates:

        d = extract_dates_from_path(
            p
        )

        if d is None:
            continue

        a, b = d

        parsed_base_count += 1

        key = tuple(
            sorted(
                (a, b)
            )
        )

        by_canonical.setdefault(
            key,
            []
        ).append(
            (
                a,
                b,
                p,
            )
        )

    mapping = []

    missing = []

    direct_count = 0
    reversed_count = 0
    duplicate_choices = 0
    invalid_base_models = []

    for edge_idx, (
        i1,
        j1,
    ) in enumerate(
        pairs,
        start=1,
    ):

        di = dates[
            i1 - 1
        ]

        dj = dates[
            j1 - 1
        ]

        key = tuple(
            sorted(
                (di, dj)
            )
        )

        candidates = (
            by_canonical.get(
                key,
                []
            )
        )

        if not candidates:

            missing.append(
                {
                    "edge":
                        edge_idx,
                    "date_i":
                        di,
                    "date_j":
                        dj,
                }
            )

            continue

        if len(candidates) > 1:
            duplicate_choices += 1

        # Prefer exact orientation.
        candidates = sorted(
            candidates,
            key=lambda x:
                (
                    0
                    if (
                        x[0] == di
                        and
                        x[1] == dj
                    )
                    else 1,
                    len(
                        str(
                            x[2]
                        )
                    ),
                    str(
                        x[2]
                    ),
                )
        )

        a, b, base = (
            candidates[0]
        )

        if (
            a == di
            and b == dj
        ):
            orientation = +1
            direct_count += 1
        elif (
            a == dj
            and b == di
        ):
            orientation = -1
            reversed_count += 1
        else:
            raise RuntimeError(
                "Internal base orientation error"
            )

        # Mature parser validation.
        try:
            model = read_baseline_model(
                base
            )

            finite_model = (
                np.all(
                    np.isfinite(
                        model.baseline_tcn
                    )
                )
                and
                np.all(
                    np.isfinite(
                        model.baseline_rate_tcn
                    )
                )
            )

        except Exception as exc:

            finite_model = False

            invalid_base_models.append(
                {
                    "edge":
                        edge_idx,
                    "base_file":
                        str(base),
                    "error":
                        repr(exc),
                }
            )

        mapping.append(
            {
                "edge":
                    edge_idx,

                "network_i":
                    int(i1),

                "network_j":
                    int(j1),

                "date_i":
                    di,

                "date_j":
                    dj,

                "base_file":
                    str(base),

                "orientation":
                    int(
                        orientation
                    ),

                "base_model_valid":
                    bool(
                        finite_model
                    ),
            }
        )

    # ========================================================
    # Save QA geometry products
    # ========================================================

    np.save(
        outdir
        /
        "longitude_deg.npy",
        longitude.astype(
            np.float64
        ),
    )

    np.save(
        outdir
        /
        "latitude_deg.npy",
        latitude.astype(
            np.float64
        ),
    )

    np.save(
        outdir
        /
        "local_xy_m.npy",
        local_xy.astype(
            np.float32
        ),
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
        "look_angle_rad.npy",
        look_angle.astype(
            np.float32
        ),
    )

    np.save(
        outdir
        /
        "slant_range_m.npy",
        slant_range.astype(
            np.float32
        ),
    )

    (
        outdir
        /
        "network_base_mapping.json"
    ).write_text(
        json.dumps(
            mapping,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    (
        outdir
        /
        "missing_network_bases.json"
    ).write_text(
        json.dumps(
            missing,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Console
    # ========================================================

    print("=" * 112)
    print(
        "Step 10R3b - mixed-grid adapter "
        "and current-network baseline coverage audit"
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
        "Raw-RSLC point geometry"
    )
    print("=" * 112)

    print(
        f"raw grid                   : "
        f"{raw_length} x {raw_width}"
    )

    print(
        f"range/azimuth looks        : "
        f"{raw_geometry.range_looks}:"
        f"{raw_geometry.azimuth_looks}"
    )

    qprint(
        "look angle "
        "p01/p05/p50/p95/p99 [rad]:",
        look_angle,
    )

    qprint(
        "slant range "
        "p01/p05/p50/p95/p99 [m]:",
        slant_range,
        fmt=".3f",
    )

    print()
    print("=" * 112)
    print(
        "4:1 lon/lat -> 1:1 point adapter"
    )
    print("=" * 112)

    print(
        f"source radar grid          : "
        f"{mli_length} x {mli_width}"
    )

    print(
        f"longitude source           : "
        f"{lon_file}"
    )

    print(
        f"latitude source            : "
        f"{lat_file}"
    )

    print(
        f"valid interpolated points  : "
        f"{nvalid_ll:,}/{npoint:,}"
    )

    if nvalid_ll == npoint:

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
        "Current-network GAMMA .base coverage"
    )
    print("=" * 112)

    print(
        f".base files found          : "
        f"{len(base_candidates)}"
    )

    print(
        f".base files date-parsed    : "
        f"{parsed_base_count}"
    )

    print(
        f"mapped network edges       : "
        f"{len(mapping)}/{nedge}"
    )

    print(
        f"direct orientation         : "
        f"{direct_count}"
    )

    print(
        f"reversed orientation       : "
        f"{reversed_count}"
    )

    print(
        f"edges with multiple choices: "
        f"{duplicate_choices}"
    )

    print(
        f"missing network edges      : "
        f"{len(missing)}"
    )

    print(
        f"invalid baseline models    : "
        f"{len(invalid_base_models)}"
    )

    if missing:

        print()
        print(
            "First missing edges:"
        )

        for item in missing[:20]:

            print(
                f"  edge {item['edge']:3d}: "
                f"{item['date_i']} -> "
                f"{item['date_j']}"
            )

    # ========================================================
    # Status
    # ========================================================

    if nvalid_ll != npoint:

        status = (
            "REVIEW_POINT_LONLAT_ADAPTER"
        )

    elif invalid_base_models:

        status = (
            "REVIEW_INVALID_BASE_MODELS"
        )

    elif missing:

        status = (
            "MISSING_CURRENT_NETWORK_BASE_FILES"
        )

    else:

        status = (
            "PASS_RAW_GRID_ADAPTER_AND_BASE_COVERAGE"
        )

    manifest = {
        "format":
            "pyPSDS-GAMMA-mixed-grid-stage7-adapter-audit-v09",

        "status":
            status,

        "points":
            int(
                npoint
            ),

        "network_edges":
            int(
                nedge
            ),

        "phase_grid": {
            "rows":
                raw_length,
            "cols":
                raw_width,
            "range_looks":
                1,
            "azimuth_looks":
                1,
        },

        "lonlat_source_grid": {
            "rows":
                mli_length,
            "cols":
                mli_width,
            "range_looks":
                4,
            "azimuth_looks":
                1,
        },

        "point_lonlat_adapter": {
            "method":
                (
                    "linear interpolation in range "
                    "using mature 4:1 radar lon/lat"
                ),

            "fractional_coordinate":
                "(raw_col - 1.5) / 4",

            "valid_points":
                nvalid_ll,

            "height_interpolated":
                False,
        },

        "raw_radar_geometry": {
            "source":
                (
                    "pystamps.prep.gamma_geometry."
                    "build_radar_geometry + "
                    "calculate_candidate_geometry"
                ),

            "range_looks":
                1,

            "azimuth_looks":
                1,
        },

        "baseline_coverage": {
            "files_found":
                len(
                    base_candidates
                ),

            "files_date_parsed":
                parsed_base_count,

            "mapped_edges":
                len(
                    mapping
                ),

            "missing_edges":
                len(
                    missing
                ),

            "direct_orientation":
                direct_count,

            "reversed_orientation":
                reversed_count,

            "invalid_models":
                len(
                    invalid_base_models
                ),
        },

        "height_required_for_stage7":
            False,

        "bperp_matrix_generated":
            False,

        "stage7_executed":
            False,

        "phase_modified":
            False,
    }

    manifest_path = (
        outdir
        /
        "mixed_grid_adapter_manifest.json"
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
        f"STEP 10R3b STATUS: {status}"
    )

    print(
        "No Bperp matrix was generated."
    )

    print(
        "No Stage-7/Stage-8 correction was applied."
    )


if __name__ == "__main__":
    main()
