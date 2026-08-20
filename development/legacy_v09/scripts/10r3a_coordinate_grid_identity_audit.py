#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np

ROOT = Path("/home/ubuntu/Downloads")
RSLC_TAB = ROOT / "RSLC_tab"
DEM = ROOT / "DEM_prep"
HGT = DEM / "20151212.hgt"

POINT_ROOT = (
    ROOT
    / "psds"
    / "prototype_outputs"
    / "v09"
)

STRICT = (
    POINT_ROOT
    / "network_inversion_v09"
    / "strict_point_ids.npy"
)

ROWS = (
    POINT_ROOT
    / "point_phase_stack"
    / "rows.npy"
)

COLS = (
    POINT_ROOT
    / "point_phase_stack"
    / "cols.npy"
)


def read_par(path, keys):
    out = {}

    with Path(path).open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            left, sep, right = line.partition(":")

            if not sep:
                continue

            key = left.strip()

            if key not in keys:
                continue

            vals = right.strip().split()

            if vals:
                out[key] = vals[0]

    return out


def find_rslc_par():

    # First try RSLC_tab.
    lines = RSLC_TAB.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    for line in lines:

        if not line.strip():
            continue

        fields = line.split()

        # Usually second column is .par.
        for token in fields:

            p = Path(token)

            if not p.is_absolute():
                p = ROOT / token

            if (
                p.is_file()
                and
                (
                    p.name.endswith(".rslc.par")
                    or
                    p.name.endswith(".slc.par")
                )
            ):
                return p.resolve()

    # Fallback search.
    candidates = sorted(
        (ROOT / "RSLC").glob("*.rslc.par")
    )

    if not candidates:
        candidates = sorted(
            (ROOT / "RSLC").glob("*.slc.par")
        )

    if not candidates:
        raise RuntimeError(
            "No RSLC/SLC parameter file found"
        )

    return candidates[0].resolve()


strict = np.load(
    STRICT,
    mmap_mode="r",
).astype(np.int64)

all_rows = np.load(
    ROWS,
    mmap_mode="r",
)

all_cols = np.load(
    COLS,
    mmap_mode="r",
)

rows = np.asarray(
    all_rows[strict],
    dtype=np.int64,
)

cols = np.asarray(
    all_cols[strict],
    dtype=np.int64,
)

rslc_par = find_rslc_par()

par = read_par(
    rslc_par,
    {
        "range_samples",
        "azimuth_lines",
        "width",
        "nlines",
        "range_looks",
        "azimuth_looks",
    },
)

width = int(
    float(
        par.get(
            "range_samples",
            par.get("width", "0"),
        )
    )
)

length = int(
    float(
        par.get(
            "azimuth_lines",
            par.get("nlines", "0"),
        )
    )
)

hgt_bytes = (
    HGT.stat().st_size
    if HGT.is_file()
    else None
)

expected_f32 = (
    width * length * 4
    if width > 0 and length > 0
    else None
)

print("=" * 100)
print("Step 10R3a - pyPSDS coordinate-grid identity audit")
print("=" * 100)

print(f"RSLC parameter file       : {rslc_par}")
print(f"RSLC width                : {width}")
print(f"RSLC length               : {length}")

print()
print(f"strict points             : {strict.size:,}")
print(f"row range                 : {rows.min()} .. {rows.max()}")
print(f"col range                 : {cols.min()} .. {cols.max()}")

print()
print(f"height file               : {HGT}")
print(f"height exists             : {HGT.is_file()}")
print(f"height bytes              : {hgt_bytes}")
print(f"RSLC-grid float32 bytes   : {expected_f32}")

point_grid_matches = (
    rows.min() >= 0
    and cols.min() >= 0
    and rows.max() < length
    and cols.max() < width
)

hgt_matches = (
    hgt_bytes is not None
    and expected_f32 is not None
    and hgt_bytes == expected_f32
)

print()
print("=" * 100)
print("Grid identity")
print("=" * 100)

print(
    "points fit RSLC grid      :",
    point_grid_matches,
)

print(
    "height matches RSLC grid  :",
    hgt_matches,
)

# Also show the known 4:1 MLI parameter if present.
mli_candidates = sorted(
    DEM.glob("*_4_1*.mli.par")
)

if mli_candidates:

    mp = mli_candidates[0]

    mpar = read_par(
        mp,
        {
            "range_samples",
            "azimuth_lines",
            "width",
            "nlines",
            "range_looks",
            "azimuth_looks",
        },
    )

    mw = int(
        float(
            mpar.get(
                "range_samples",
                mpar.get("width", "0"),
            )
        )
    )

    ml = int(
        float(
            mpar.get(
                "azimuth_lines",
                mpar.get("nlines", "0"),
            )
        )
    )

    print()
    print("=" * 100)
    print("Existing 4:1 MLI reference")
    print("=" * 100)

    print(f"MLI par                   : {mp}")
    print(f"MLI width                 : {mw}")
    print(f"MLI length                : {ml}")

    if (
        width > 0
        and mw > 0
    ):
        print(
            f"RSLC width / MLI width    : "
            f"{width / mw:.6f}"
        )

if (
    width == 2000
    and
    length == 600
    and
    point_grid_matches
    and
    hgt_matches
):

    status = "PASS_RSLCCOORD_1TO1"

else:

    status = "REVIEW_GRID_IDENTITY"

print()
print(
    f"STEP 10R3a STATUS: {status}"
)

if status == "PASS_RSLCCOORD_1TO1":
    print(
        "pyPSDS points are on the 2000x600 RSLC radar grid."
    )
    print(
        "For the pySTAMPS geometry adapter use "
        "range_looks=1, azimuth_looks=1."
    )
