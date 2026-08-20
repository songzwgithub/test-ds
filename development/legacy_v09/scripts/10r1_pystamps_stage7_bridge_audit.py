#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    import h5py
except Exception:
    h5py = None


def inspect_mat(path: Path):
    out = {
        "path": str(path),
        "exists": path.is_file(),
        "format": None,
        "variables": {},
    }

    if not path.is_file():
        return out

    # --------------------------------------------------------
    # Try MATLAB v5 / scipy first
    # --------------------------------------------------------

    if loadmat is not None:
        try:
            m = loadmat(
                path,
                squeeze_me=False,
                struct_as_record=False,
            )

            out["format"] = "mat_v5"

            for k, v in m.items():
                if k.startswith("__"):
                    continue

                a = np.asarray(v)

                info = {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                }

                if np.issubdtype(
                    a.dtype,
                    np.number,
                ) and a.size:

                    af = np.asarray(
                        a,
                        dtype=np.float64,
                    )

                    finite = np.isfinite(af)

                    info["finite_fraction"] = float(
                        np.mean(finite)
                    )

                    if np.any(finite):
                        info["min"] = float(
                            np.min(af[finite])
                        )

                        info["max"] = float(
                            np.max(af[finite])
                        )

                        info["median"] = float(
                            np.median(af[finite])
                        )

                out["variables"][k] = info

            return out

        except Exception:
            pass

    # --------------------------------------------------------
    # MATLAB v7.3 / HDF5
    # --------------------------------------------------------

    if h5py is not None:
        try:
            with h5py.File(
                path,
                "r",
            ) as f:

                out["format"] = "hdf5"

                def visit(name, obj):
                    if not isinstance(
                        obj,
                        h5py.Dataset,
                    ):
                        return

                    info = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }

                    try:
                        if (
                            np.issubdtype(
                                obj.dtype,
                                np.number,
                            )
                            and
                            obj.size
                            and
                            obj.size <= 20_000_000
                        ):

                            a = np.asarray(
                                obj[...],
                                dtype=np.float64,
                            )

                            finite = np.isfinite(
                                a
                            )

                            info["finite_fraction"] = float(
                                np.mean(finite)
                            )

                            if np.any(finite):
                                info["min"] = float(
                                    np.min(a[finite])
                                )

                                info["max"] = float(
                                    np.max(a[finite])
                                )

                                info["median"] = float(
                                    np.median(a[finite])
                                )

                    except Exception:
                        pass

                    out["variables"][name] = info

                f.visititems(visit)

            return out

        except Exception as exc:
            out["read_error"] = repr(exc)

    return out


def find_files(
    roots,
    names,
    max_results=100,
):

    results = []

    for root in roots:

        root = Path(root)

        if not root.exists():
            continue

        for name in names:

            try:
                for p in root.rglob(name):

                    results.append(
                        p.resolve()
                    )

                    if len(results) >= max_results:
                        return results

            except PermissionError:
                continue

    # stable unique
    unique = []

    seen = set()

    for p in results:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)

    return unique


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default=(
            "/home/ubuntu/Downloads/psds/"
            "prototype_outputs/v09"
        ),
    )

    ap.add_argument(
        "--search-root",
        action="append",
        default=[],
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    # ========================================================
    # Current pyPSDS products
    # ========================================================

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

    sdir = (
        root
        /
        "scla_v09"
        /
        "production_sensitivity"
    )

    phase_path = (
        invdir
        /
        "acquisition_phase_l2_candidate_rad.npy"
    )

    strict_path = (
        invdir
        /
        "strict_point_ids.npy"
    )

    rows_path = (
        ppsdir
        /
        "rows.npy"
    )

    cols_path = (
        ppsdir
        /
        "cols.npy"
    )

    sensitivity_path = (
        sdir
        /
        "topographic_phase_sensitivity_rad_per_m.npy"
    )

    print("=" * 112)
    print(
        "Step 10R1 - pyPSDS -> pySTAMPS "
        "Stage-7 bridge audit"
    )
    print("=" * 112)

    required = {
        "phase":
            phase_path,

        "strict_ids":
            strict_path,

        "rows":
            rows_path,

        "cols":
            cols_path,

        "topographic_sensitivity":
            sensitivity_path,
    }

    missing = []

    for name, path in required.items():

        print(
            f"{name:28s}: "
            f"{path}"
        )

        print(
            f"{'':28s}  "
            f"exists={path.is_file()}"
        )

        if not path.is_file():
            missing.append(
                str(path)
            )

    if missing:
        raise RuntimeError(
            "Missing pyPSDS inputs:\n"
            +
            "\n".join(
                missing
            )
        )

    Y = np.load(
        phase_path,
        mmap_mode="r",
    )

    strict = np.load(
        strict_path,
        mmap_mode="r",
    )

    all_rows = np.load(
        rows_path,
        mmap_mode="r",
    )

    all_cols = np.load(
        cols_path,
        mmap_mode="r",
    )

    Sh = np.load(
        sensitivity_path,
        mmap_mode="r",
    )

    npoint, ndate = Y.shape

    print()
    print("=" * 112)
    print(
        "Current pyPSDS domain"
    )
    print("=" * 112)

    print(
        f"phase shape                : "
        f"{Y.shape}"
    )

    print(
        f"phase dtype                : "
        f"{Y.dtype}"
    )

    print(
        f"strict IDs                 : "
        f"{strict.shape}"
    )

    print(
        f"all point rows             : "
        f"{all_rows.shape}"
    )

    print(
        f"all point cols             : "
        f"{all_cols.shape}"
    )

    print(
        f"topographic sensitivity    : "
        f"{Sh.shape}"
    )

    if Sh.shape != Y.shape:
        raise RuntimeError(
            "Sensitivity and phase shape mismatch"
        )

    if strict.shape != (
        npoint,
    ):
        raise RuntimeError(
            "strict_point_ids shape mismatch"
        )

    strict_i = np.asarray(
        strict,
        dtype=np.int64,
    )

    rows = np.asarray(
        all_rows[
            strict_i
        ],
        dtype=np.int32,
    )

    cols = np.asarray(
        all_cols[
            strict_i
        ],
        dtype=np.int32,
    )

    print(
        f"strict row range           : "
        f"{rows.min()} .. {rows.max()}"
    )

    print(
        f"strict col range           : "
        f"{cols.min()} .. {cols.max()}"
    )

    print(
        f"first phase max |.|        : "
        f"{np.max(np.abs(Y[:,0])):.3e} rad"
    )

    print(
        f"first sensitivity max |.|  : "
        f"{np.max(np.abs(Sh[:,0])):.3e} rad/m"
    )

    # ========================================================
    # What Stage-7 fields are already constructible?
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Stage-7 field mapping"
    )
    print("=" * 112)

    mapping = [
        (
            "phuw2.ph_uw",
            "YES",
            "Step09a acquisition phase"
        ),
        (
            "ps2.n_ps",
            "YES",
            str(npoint)
        ),
        (
            "ps2.day",
            "YES",
            f"{ndate} acquisition dates"
        ),
        (
            "ps2.master_ix",
            "YES",
            "temporal reference acquisition"
        ),
        (
            "ps2.xy",
            "YES",
            "radar col/row available"
        ),
        (
            "ps2.lonlat",
            "PENDING",
            "not production-geocoded yet"
        ),
        (
            "bp2.bperp_mat",
            "BLOCKER",
            "must be signed point-wise acquisition baseline"
        ),
        (
            "ifgstd2.ifg_std",
            "PENDING",
            "need pyPSDS->StaMPS QA mapping"
        ),
        (
            "small_baseline_flag",
            "YES",
            "use y"
        ),
    ]

    for field, status, note in mapping:

        print(
            f"{field:24s} "
            f"{status:10s} "
            f"{note}"
        )

    # ========================================================
    # Search existing pySTAMPS products.
    # ========================================================

    search_roots = [
        Path(
            "/home/ubuntu/Downloads"
        ),
        Path(
            "/home/ubuntu/software"
        ),
    ]

    for x in args.search_root:
        search_roots.append(
            Path(x)
        )

    names = [
        "bp2.mat",
        "ps2.mat",
        "phuw2.mat",
        "ifgstd2.mat",
        "scla2.mat",
        "scla_smooth2.mat",
    ]

    print()
    print("=" * 112)
    print(
        "Searching for existing pySTAMPS "
        "schema/reference products"
    )
    print("=" * 112)

    found = find_files(
        search_roots,
        names,
        max_results=100,
    )

    by_name = {}

    for p in found:
        by_name.setdefault(
            p.name,
            []
        ).append(
            p
        )

    for name in names:

        files = by_name.get(
            name,
            []
        )

        print()
        print(
            f"{name}: {len(files)} found"
        )

        for p in files[:10]:
            print(
                f"  {p}"
            )

    # ========================================================
    # Inspect representative MAT files.
    # ========================================================

    inspected = {}

    print()
    print("=" * 112)
    print(
        "Representative MAT schemas"
    )
    print("=" * 112)

    for name in names:

        files = by_name.get(
            name,
            []
        )

        if not files:
            continue

        # Prefer newest modified file as likely current.
        files = sorted(
            files,
            key=lambda p:
                p.stat().st_mtime,
            reverse=True,
        )

        p = files[0]

        info = inspect_mat(
            p
        )

        inspected[
            name
        ] = info

        print()
        print(
            f"[{name}]"
        )

        print(
            f"path   : {p}"
        )

        print(
            f"format : "
            f"{info.get('format')}"
        )

        for var, meta in (
            info.get(
                "variables",
                {}
            ).items()
        ):

            print(
                f"  {var:25s} "
                f"shape={meta.get('shape')} "
                f"dtype={meta.get('dtype')}"
            )

    # ========================================================
    # Specific bp2 decision
    # ========================================================

    print()
    print("=" * 112)
    print(
        "Critical baseline decision"
    )
    print("=" * 112)

    bp_files = by_name.get(
        "bp2.mat",
        []
    )

    if bp_files:

        print(
            "Existing bp2.mat found."
        )

        print(
            "IMPORTANT: its numerical values CANNOT "
            "be copied directly to pyPSDS points unless "
            "the point coordinates are identical."
        )

        print(
            "Use it only to audit schema and to trace "
            "the pySTAMPS point-wise baseline generator."
        )

        baseline_status = (
            "SCHEMA_REFERENCE_FOUND"
        )

    else:

        print(
            "No existing bp2.mat found."
        )

        print(
            "Next action must be to recover/reuse the "
            "pySTAMPS baseline-generation implementation."
        )

        baseline_status = (
            "BASELINE_GENERATOR_REQUIRED"
        )

    # ========================================================
    # Final audit decision
    # ========================================================

    manifest = {
        "format":
            "pyPSDS-GAMMA-pySTAMPS-stage7-bridge-audit-v09",

        "pypsds": {
            "points":
                int(npoint),

            "acquisitions":
                int(ndate),

            "phase_shape":
                list(Y.shape),

            "sensitivity_shape":
                list(Sh.shape),
        },

        "mapping": {
            field: {
                "status":
                    status,
                "note":
                    note,
            }
            for field, status, note
            in mapping
        },

        "found_pystamps_files": {
            k: [
                str(x)
                for x in v
            ]
            for k, v in by_name.items()
        },

        "representative_schemas":
            inspected,

        "baseline_status":
            baseline_status,

        "production_phase_modified":
            False,

        "stage7_executed":
            False,
    }

    manifest_path = (
        root
        /
        "scla_v09"
        /
        "pystamps_bridge_audit.json"
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
        "STEP 10R1 STATUS: "
        f"{baseline_status}"
    )

    print(
        "No pySTAMPS input was generated and "
        "no phase/SCLA product was modified."
    )


if __name__ == "__main__":
    main()
