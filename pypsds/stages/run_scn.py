from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from pypsds.stages._v11_common import (
    cfg_get,
    ensure_geometry_compat,
    load_context,
    run_runtime,
    write_json,
)


def _locate_support(root, name):
    target = root / "support" / name
    if target.is_file():
        return target
    hits = [p for p in root.rglob(name) if p != target]
    if len(hits) != 1:
        raise RuntimeError(f"SCN support {name} not uniquely produced: {hits}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hits[0], target)
    return target


def main():
    ap = argparse.ArgumentParser(description="Exact StaMPS-compatible SCN filtering.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ctx = load_context(args.config)
    mode = str(cfg_get(ctx["cfg"], "corrections.scn.mode", "disabled")).strip().lower()

    pre = ctx["proc"] / "scla" / "acquisition_phase_pre_scn_rad.npy"
    if not pre.is_file():
        raise FileNotFoundError(pre)
    source = np.load(pre, mmap_mode="r")
    npoint, ndate = source.shape

    outdir = ctx["proc"] / "scn"
    outdir.mkdir(parents=True, exist_ok=True)
    scn_path = outdir / "ph_scn_slave_rad.npy"
    manifest = outdir / "scn_manifest.json"

    if mode == "disabled":
        out = np.lib.format.open_memmap(
            scn_path,
            mode="w+",
            dtype=np.float64,
            shape=source.shape,
        )
        out[:] = 0.0
        out.flush()
        del out

        master0 = ctx["dates"].index(ctx["master_date"])
        write_json(manifest, {
            "status": "PASS_STAMPS_STAGE8_SCN",
            "mode": "disabled",
            "scientific_contract": {
                "geometric_master_date": ctx["master_date"],
                "geometric_master_index_0based": master0,
                "scn_time_win_days": float(cfg_get(ctx["cfg"], "corrections.scn.temporal_window_days", 365.0)),
                "scn_wavelength_m": float(cfg_get(ctx["cfg"], "corrections.scn.wavelength_m", 100.0)),
                "spatial_radius_m": 0.0,
            },
            "outputs": {"ph_scn_slave": str(scn_path)},
        })

    elif mode == "stamps":
        compat = ensure_geometry_compat(ctx)
        time_win = float(cfg_get(ctx["cfg"], "corrections.scn.temporal_window_days", 365.0))
        wavelength = float(cfg_get(ctx["cfg"], "corrections.scn.wavelength_m", 100.0))
        radius_factor = float(cfg_get(ctx["cfg"], "corrections.scn.radius_factor", 4.0))
        cell_size = float(cfg_get(ctx["cfg"], "corrections.scn.cell_size_m", 200.0))
        radius = radius_factor * wavelength

        # --------------------------------------------------------------
        # SCN support lineage expects ROOT/RSLC/<master>.rslc.par.
        # Public projects do not require paths.rslc_dir to live below
        # paths.data_dir or even to be literally named "RSLC".
        # Build a symlink-only compatibility root and override only the
        # runtime environment. Scientific arrays/kernels are unchanged.
        # --------------------------------------------------------------

        scn_support_root = (
            outdir
            /
            "_support"
        )

        scn_support_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        scn_data_root = (
            scn_support_root
            /
            "data_root_compat"
        )

        scn_data_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        rslc_owner = Path(
            ctx["paths"].rslc_dir
        ).resolve()

        if not rslc_owner.is_dir():
            raise FileNotFoundError(
                rslc_owner
            )

        rslc_link = (
            scn_data_root
            /
            "RSLC"
        )

        if (
            rslc_link.is_symlink()
            or
            rslc_link.exists()
        ):
            if not (
                rslc_link.is_symlink()
                and
                rslc_link.resolve()
                ==
                rslc_owner
            ):
                if (
                    rslc_link.is_dir()
                    and
                    not rslc_link.is_symlink()
                ):
                    shutil.rmtree(
                        rslc_link
                    )
                else:
                    rslc_link.unlink()

        if not rslc_link.exists():
            rslc_link.symlink_to(
                rslc_owner,
                target_is_directory=True,
            )

        master_par = (
            rslc_link
            /
            f"{ctx['master_date']}.rslc.par"
        )

        if not master_par.is_file():
            raise FileNotFoundError(
                master_par
            )

        print(
            "SCN compatibility data root:",
            scn_data_root,
        )

        print(
            "SCN RSLC owner             :",
            rslc_owner,
        )

        print(
            "SCN master RSLC par        :",
            master_par,
        )

        extra = {
            "PYPSDS_PUBLIC_DATA_ROOT": scn_data_root.resolve(),
            "PYPSDS_PUBLIC_GEOM_COMPAT": compat,
            "PYPSDS_PUBLIC_SCN_TIME_WIN": time_win,
            "PYPSDS_PUBLIC_SCN_WAVELENGTH": wavelength,
            "PYPSDS_PUBLIC_SCN_RADIUS": radius,
            "PYPSDS_PUBLIC_SCN_CELL_SIZE": cell_size,
        }

        run_runtime("scn_support_runtime.py", ctx, extra)

        support = outdir / "support"
        _locate_support(outdir, "stamps_xy_exact_float32_m.npy")
        _locate_support(outdir, "stamps_sort_index.npy")
        # Historical producer name -> public semantic name.
        old_count = _locate_support(outdir, "stage8_neighbor_count_r400m.npy") \
            if any(outdir.rglob("stage8_neighbor_count_r400m.npy")) else None
        public_count = support / "neighbor_count_r400m.npy"
        if old_count is not None and not public_count.is_file():
            shutil.copyfile(old_count, public_count)
        if not public_count.is_file():
            _locate_support(outdir, "neighbor_count_r400m.npy")

        run_runtime("scn_runtime.py", ctx, extra)

        if not scn_path.is_file():
            raise RuntimeError("SCN runtime did not create ph_scn_slave_rad.npy")

    else:
        raise ValueError(f"Unsupported corrections.scn.mode={mode!r}")

    scn = np.load(scn_path, mmap_mode="r")
    if scn.shape != source.shape or scn.dtype != np.float64:
        raise RuntimeError(f"SCN output contract failed: {scn.shape}/{scn.dtype}")

    print("=" * 88)
    print("SCN STATUS: PASS")
    print("mode   :", mode)
    print("output :", scn_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
