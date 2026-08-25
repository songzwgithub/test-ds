#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(
    "/home/ubuntu/Downloads/psds/"
    "prototype_outputs/v09"
)

R3B = (
    ROOT
    / "scla_v09"
    / "pystamps_bridge"
    / "r3b_grid_adapter"
)

R4A = (
    ROOT
    / "scla_v09"
    / "pystamps_bridge"
    / "r4a_stage7_contract"
)

BRIDGE = (
    ROOT
    / "pystamps_bridge_v09"
)

LON_FILE = Path(
    "/home/ubuntu/Downloads/"
    "DEM_prep/20151212_4_1.rdc.lon"
)

LAT_FILE = Path(
    "/home/ubuntu/Downloads/"
    "DEM_prep/20151212_4_1.rdc.lat"
)

RSLC_PAR = Path(
    "/home/ubuntu/Downloads/"
    "RSLC/20141006.rslc.par"
)


def resolve_pystamps(explicit=None):

    candidates = []

    if explicit:
        candidates.append(
            Path(explicit).expanduser()
        )

    candidates.extend([
        Path(
            "/home/ubuntu/software/pystamps-gamma"
        ),
        Path.home()
        / "software"
        / "pystamps-gamma",
    ])

    for p in candidates:

        try:
            p = p.resolve()
        except Exception:
            continue

        if (
            p.is_dir()
            and
            (p / "pystamps").is_dir()
        ):

            sys.path.insert(
                0,
                str(p),
            )

            return p

    raise RuntimeError(
        "Cannot locate pystamps-gamma"
    )


def duplicate_stats(xy):

    u, counts = np.unique(
        xy,
        axis=0,
        return_counts=True,
    )

    groups = int(
        np.count_nonzero(
            counts > 1
        )
    )

    members = int(
        np.sum(
            counts[
                counts > 1
            ]
        )
    )

    return (
        groups,
        members,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pystamps-source",
        default=None,
    )

    args = ap.parse_args()

    source = resolve_pystamps(
        args.pystamps_source
    )

    from pystamps.io.mat import (
        read_mat,
        write_mat,
    )

    from pystamps.prep.gamma_binary import (
        sample_gamma_raster,
    )

    from pystamps.prep.gamma_geometry import (
        build_radar_geometry,
    )

    from pystamps.prep.gamma_observations import (
        lonlat_to_local_xy,
    )

    from pystamps.pipeline.stage6_sbas import (
        _stage6_reference_indices,
    )

    from pystamps.pipeline.stage7_sbas import (
        _build_edges,
        _preflight,
    )

    # ========================================================
    # Input validation
    # ========================================================

    for p in (
        BRIDGE / "ps2.mat",
        BRIDGE / "parms.mat",
        LON_FILE,
        LAT_FILE,
        RSLC_PAR,
    ):
        if not p.is_file():
            raise RuntimeError(
                f"Missing required file: {p}"
            )

    ps = read_mat(
        BRIDGE / "ps2.mat"
    )

    parms = read_mat(
        BRIDGE / "parms.mat"
    )

    xy_old = np.asarray(
        ps["xy"],
        dtype=np.float64,
    )

    ij = np.asarray(
        ps["ij"],
        dtype=np.float64,
    )

    nps = int(
        round(
            float(
                np.asarray(
                    ps["n_ps"]
                ).reshape(-1)[0]
            )
        )
    )

    if xy_old.shape != (
        nps,
        3,
    ):
        raise RuntimeError(
            f"Unexpected xy shape: "
            f"{xy_old.shape}"
        )

    rows = (
        np.rint(
            ij[:, 1]
        ).astype(
            np.int64
        )
        -
        1
    )

    cols = (
        np.rint(
            ij[:, 2]
        ).astype(
            np.int64
        )
        -
        1
    )

    # ========================================================
    # Preserve current bugged geometry + failed Stage7
    # ========================================================

    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = (
        BRIDGE
        /
        (
            "_failed_stage7_duplicate_xy_"
            +
            stamp
        )
    )

    backup.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Existing bridge ps2 is important provenance.
    shutil.copy2(
        BRIDGE / "ps2.mat",
        backup / "ps2_clipped_xy.mat",
    )

    for name in (
        "scla_sb2.mat",
        "scla_smooth_sb2.mat",
        "scla2.mat",
        "stage7_sbas_debug.json",
        "stage7_run_v09.log",
        "stage7_delaunay_isolate_audit_v09.json",
    ):

        p = BRIDGE / name

        if p.exists():
            shutil.move(
                str(p),
                str(
                    backup
                    /
                    p.name
                ),
            )

    tri_old = (
        BRIDGE
        /
        "_stage7_triangle_work"
    )

    if tri_old.exists():

        shutil.move(
            str(tri_old),
            str(
                backup
                /
                "_stage7_triangle_work"
            ),
        )

    # Preserve R3b coordinate arrays before replacing.
    r3_backup = (
        R3B
        /
        "_pre_r5a1_clipped_geometry"
    )

    r3_backup.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in (
        "longitude_deg.npy",
        "latitude_deg.npy",
        "local_xy_m.npy",
        "ll0_lonlat_deg.npy",
    ):

        src = (
            R3B
            /
            name
        )

        dst = (
            r3_backup
            /
            name
        )

        if (
            src.is_file()
            and
            not dst.exists()
        ):
            shutil.copy2(
                src,
                dst,
            )

    old_lon = np.asarray(
        np.load(
            r3_backup
            /
            "longitude_deg.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    old_lat = np.asarray(
        np.load(
            r3_backup
            /
            "latitude_deg.npy",
            mmap_mode="r",
        ),
        dtype=np.float64,
    )

    old_xy = np.asarray(
        xy_old[
            :,
            1:3
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Rebuild 4:1 -> 1:1 point lon/lat
    #
    # CRITICAL FIX:
    #
    # old:
    #   u = clip(u, 0, 499)
    #
    # new:
    #   DO NOT clip u.
    #
    # Clamp only interpolation SEGMENT:
    #
    #   c0 in [0, 498]
    #   c1=c0+1
    #   w=u-c0
    #
    # Thus w may be <0 or >1 at the radar boundary,
    # giving linear extrapolation rather than duplicate
    # coordinates.
    # ========================================================

    source_width = 500
    source_length = 600
    range_looks = 4

    u = (
        cols.astype(
            np.float64
        )
        -
        1.5
    ) / float(
        range_looks
    )

    c0 = np.floor(
        u
    ).astype(
        np.int64
    )

    c0 = np.clip(
        c0,
        0,
        source_width - 2,
    )

    c1 = c0 + 1

    w = (
        u
        -
        c0.astype(
            np.float64
        )
    )

    lon0 = sample_gamma_raster(
        LON_FILE,
        rows,
        c0,
        width=source_width,
        length=source_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lon1 = sample_gamma_raster(
        LON_FILE,
        rows,
        c1,
        width=source_width,
        length=source_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lat0 = sample_gamma_raster(
        LAT_FILE,
        rows,
        c0,
        width=source_width,
        length=source_length,
        dtype="float",
    ).astype(
        np.float64
    )

    lat1 = sample_gamma_raster(
        LAT_FILE,
        rows,
        c1,
        width=source_width,
        length=source_length,
        dtype="float",
    ).astype(
        np.float64
    )

    longitude = (
        (1.0 - w)
        *
        lon0
        +
        w
        *
        lon1
    )

    latitude = (
        (1.0 - w)
        *
        lat0
        +
        w
        *
        lat1
    )

    valid = (
        np.isfinite(
            longitude
        )
        &
        np.isfinite(
            latitude
        )
        &
        (
            longitude >= -180
        )
        &
        (
            longitude <= 180
        )
        &
        (
            latitude >= -90
        )
        &
        (
            latitude <= 90
        )
    )

    if not np.all(
        valid
    ):
        raise RuntimeError(
            "Non-finite/invalid extrapolated lon/lat"
        )

    # ========================================================
    # Verify that interior points are mathematically unchanged
    # ========================================================

    interior = (
        (cols >= 2)
        &
        (cols <= 1997)
    )

    max_lon_interior = float(
        np.max(
            np.abs(
                longitude[
                    interior
                ]
                -
                old_lon[
                    interior
                ]
            )
        )
    )

    max_lat_interior = float(
        np.max(
            np.abs(
                latitude[
                    interior
                ]
                -
                old_lat[
                    interior
                ]
            )
        )
    )

    # Should be exactly or essentially identical.
    if (
        max_lon_interior
        >
        1e-10
        or
        max_lat_interior
        >
        1e-10
    ):
        raise RuntimeError(
            "Interior lon/lat changed unexpectedly: "
            f"lon={max_lon_interior:.3e}, "
            f"lat={max_lat_interior:.3e}"
        )

    # ========================================================
    # Rebuild mature metric xy
    # ========================================================

    radar_geometry = (
        build_radar_geometry(
            RSLC_PAR,
            multilook_width=2000,
            multilook_length=600,
            mli_parameter_file=None,
            range_looks=1,
            azimuth_looks=1,
        )
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

    ll0 = np.asarray(
        ll0,
        dtype=np.float64,
    ).reshape(-1)[:2]

    # ========================================================
    # Duplicate audit BEFORE touching ps2
    # ========================================================

    old_exact_groups, old_exact_members = (
        duplicate_stats(
            old_xy
        )
    )

    old_6_groups, old_6_members = (
        duplicate_stats(
            np.round(
                old_xy,
                6,
            )
        )
    )

    new_exact_groups, new_exact_members = (
        duplicate_stats(
            local_xy.astype(
                np.float64
            )
        )
    )

    new_6_groups, new_6_members = (
        duplicate_stats(
            np.round(
                local_xy.astype(
                    np.float64
                ),
                6,
            )
        )
    )

    print("=" * 112)
    print(
        "Step 10R5a1 - repair Stage7 "
        "boundary coordinate adapter"
    )
    print("=" * 112)

    print(
        f"pystamps source            : "
        f"{source}"
    )

    print(
        f"PS                         : "
        f"{nps:,}"
    )

    print()
    print("=" * 112)
    print(
        "Boundary interpolation repair"
    )
    print("=" * 112)

    print(
        f"raw col range              : "
        f"{cols.min()} .. "
        f"{cols.max()}"
    )

    print(
        f"fractional source u range  : "
        f"{u.min():.6f} .. "
        f"{u.max():.6f}"
    )

    print(
        f"u < 0 points               : "
        f"{np.count_nonzero(u < 0):,}"
    )

    print(
        f"u > 499 points             : "
        f"{np.count_nonzero(u > 499):,}"
    )

    print(
        f"interior max Δlon          : "
        f"{max_lon_interior:.3e} deg"
    )

    print(
        f"interior max Δlat          : "
        f"{max_lat_interior:.3e} deg"
    )

    print()
    print("=" * 112)
    print(
        "Coordinate uniqueness"
    )
    print("=" * 112)

    print(
        f"OLD exact dup groups       : "
        f"{old_exact_groups}"
    )

    print(
        f"OLD exact dup members      : "
        f"{old_exact_members}"
    )

    print(
        f"OLD 6-dec dup groups       : "
        f"{old_6_groups}"
    )

    print(
        f"OLD 6-dec dup members      : "
        f"{old_6_members}"
    )

    print()
    print(
        f"NEW exact dup groups       : "
        f"{new_exact_groups}"
    )

    print(
        f"NEW exact dup members      : "
        f"{new_exact_members}"
    )

    print(
        f"NEW 6-dec dup groups       : "
        f"{new_6_groups}"
    )

    print(
        f"NEW 6-dec dup members      : "
        f"{new_6_members}"
    )

    if (
        new_exact_groups != 0
        or
        new_6_groups != 0
    ):
        raise RuntimeError(
            "Duplicate coordinates remain "
            "after boundary extrapolation"
        )

    # ========================================================
    # Build corrected ps2 in memory
    # ========================================================

    lonlat = np.column_stack(
        (
            longitude,
            latitude,
        )
    )

    xy_new = np.asarray(
        xy_old,
        dtype=np.float32,
    ).copy()

    xy_new[
        :,
        1:3
    ] = local_xy

    ps_new = dict(
        ps
    )

    ps_new[
        "lonlat"
    ] = lonlat

    ps_new[
        "xy"
    ] = xy_new

    ps_new[
        "ll0"
    ] = ll0.reshape(
        1,
        2,
    )

    # ========================================================
    # Reference identity must remain EXACT
    # ========================================================

    ref_expected = np.asarray(
        np.load(
            R4A
            /
            "stage7_reference_point_indices.npy"
        ),
        dtype=np.int64,
    )

    ref_new = (
        _stage6_reference_indices(
            ps_new,
            parms,
            nps,
        )
    )

    print()
    print("=" * 112)
    print(
        "Stage7 reference invariance"
    )
    print("=" * 112)

    print(
        f"reference PS before        : "
        f"{ref_expected.size}"
    )

    print(
        f"reference PS after         : "
        f"{ref_new.size}"
    )

    print(
        f"reference set exact        : "
        f"{np.array_equal(ref_new, ref_expected)}"
    )

    if not np.array_equal(
        ref_new,
        ref_expected,
    ):
        raise RuntimeError(
            "Spatial coordinate repair changed "
            "the frozen Stage7 reference set"
        )

    # ========================================================
    # Exact mature Triangle path pre-audit
    #
    # This writes with %.6f exactly like production Stage7.
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Mature Triangle topology pre-audit"
    )
    print("=" * 112)

    edges, backend = (
        _build_edges(
            local_xy.astype(
                np.float64
            ),
            BRIDGE,
            "/usr/bin/triangle",
        )
    )

    edge0 = np.asarray(
        edges,
        dtype=np.int64,
    ) - 1

    degree = np.zeros(
        nps,
        dtype=np.int64,
    )

    np.add.at(
        degree,
        edge0[:, 0],
        1,
    )

    np.add.at(
        degree,
        edge0[:, 1],
        1,
    )

    isolated = np.flatnonzero(
        degree == 0
    )

    print(
        f"backend                    : "
        f"{backend}"
    )

    print(
        f"Delaunay edges             : "
        f"{edges.shape[0]:,}"
    )

    print(
        f"degree min/median/max      : "
        f"{degree.min()} / "
        f"{np.median(degree):.0f} / "
        f"{degree.max()}"
    )

    print(
        f"isolated PS                : "
        f"{isolated.size}"
    )

    if isolated.size != 0:
        raise RuntimeError(
            f"{isolated.size} isolated PS remain"
        )

    # Remove pre-audit work.
    shutil.rmtree(
        BRIDGE
        /
        "_stage7_triangle_work",
        ignore_errors=True,
    )

    # ========================================================
    # Commit corrected coordinate products
    # ========================================================

    np.save(
        R3B
        /
        "longitude_deg.npy",
        longitude.astype(
            np.float64
        ),
    )

    np.save(
        R3B
        /
        "latitude_deg.npy",
        latitude.astype(
            np.float64
        ),
    )

    np.save(
        R3B
        /
        "local_xy_m.npy",
        local_xy,
    )

    np.save(
        R3B
        /
        "ll0_lonlat_deg.npy",
        ll0,
    )

    # Atomic-ish MAT replacement:
    temp_ps = (
        BRIDGE
        /
        "ps2.r5a1_tmp.mat"
    )

    write_mat(
        temp_ps,
        ps_new,
    )

    temp_check = read_mat(
        temp_ps
    )

    ref_check = (
        _stage6_reference_indices(
            temp_check,
            parms,
            nps,
        )
    )

    if not np.array_equal(
        ref_check,
        ref_expected,
    ):
        raise RuntimeError(
            "Reference changed after MAT serialization"
        )

    shutil.move(
        str(temp_ps),
        str(
            BRIDGE
            /
            "ps2.mat"
        ),
    )

    # ========================================================
    # Native Stage7 preflight again
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Native Stage7 preflight after repair"
    )
    print("=" * 112)

    _preflight(
        BRIDGE,
        triangle_path="/usr/bin/triangle",
        phase_file="phuw2.mat",
    )

    # ========================================================
    # Provenance
    # ========================================================

    repair_manifest = {
        "format":
            "pyPSDS-GAMMA-stage7-boundary-coordinate-repair-v09",

        "status":
            "PASS_STAGE7_COORDINATE_REPAIR_READY",

        "root_cause":
            (
                "R3b clipped fractional 4:1 radar "
                "coordinate u to [0,499], causing "
                "raw cols 0/1 at the same row to "
                "receive identical lon/lat/xy."
            ),

        "old_duplicates": {
            "exact_groups":
                old_exact_groups,

            "exact_members":
                old_exact_members,

            "rounded6_groups":
                old_6_groups,

            "rounded6_members":
                old_6_members,
        },

        "new_duplicates": {
            "exact_groups":
                new_exact_groups,

            "exact_members":
                new_exact_members,

            "rounded6_groups":
                new_6_groups,

            "rounded6_members":
                new_6_members,
        },

        "adapter": {
            "fractional_coordinate":
                "(raw_col - 1.5) / 4",

            "old_boundary_policy":
                "clip fractional coordinate",

            "new_boundary_policy":
                (
                    "linear extrapolation using nearest "
                    "two 4:1 radar-coordinate samples"
                ),
        },

        "reference": {
            "count":
                int(
                    ref_new.size
                ),

            "unchanged":
                True,
        },

        "triangle": {
            "backend":
                backend,

            "edges":
                int(
                    edges.shape[0]
                ),

            "isolated_points":
                int(
                    isolated.size
                ),
        },

        "unmodified_products": [
            "bp2.mat",
            "phuw2.mat",
            "phuw_sb2.mat",
            "phuw_sb_res2.mat",
            "canonical Step09 acquisition phase",
            "R3c2 point-wise Bperp",
        ],

        "failed_stage7_archive":
            str(
                backup
            ),
    }

    repair_path = (
        BRIDGE
        /
        "stage7_coordinate_repair_v09.json"
    )

    repair_path.write_text(
        json.dumps(
            repair_manifest,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # Annotate bridge manifest but preserve its prior content.
    bridge_manifest_path = (
        BRIDGE
        /
        "bridge_manifest.json"
    )

    if bridge_manifest_path.is_file():

        bridge_manifest = json.loads(
            bridge_manifest_path.read_text(
                encoding="utf-8"
            )
        )

        bridge_manifest[
            "spatial_coordinate_repair"
        ] = {
            "status":
                "PASS",

            "manifest":
                str(
                    repair_path
                ),

            "boundary_policy":
                "linear_extrapolation_no_clip",

            "duplicate_xy_after":
                0,
        }

        bridge_manifest_path.write_text(
            json.dumps(
                bridge_manifest,
                indent=2,
                ensure_ascii=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

    print()
    print("=" * 112)

    print(
        f"failed run archive         : "
        f"{backup}"
    )

    print(
        f"repair manifest            : "
        f"{repair_path}"
    )

    print()
    print(
        "STEP 10R5a1 STATUS: "
        "PASS_STAGE7_COORDINATE_REPAIR_READY"
    )

    print(
        "Only lon/lat/xy/ll0 and ps2.mat "
        "spatial coordinates were repaired."
    )

    print(
        "Phase, point-wise Bperp and "
        "covariance products were not modified."
    )


if __name__ == "__main__":
    main()
