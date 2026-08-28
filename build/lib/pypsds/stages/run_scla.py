from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from pypsds.stages._stage_common import (
    atomic_copy,
    cfg_get,
    load_context,
    derive_multilook_factors,
    public_env,
    runtime_path,
    write_json,
)
from pypsds.runtime_backend.scla_support import prepare


def _count_contract(payload, token):
    token = token.lower()
    found = []
    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                kk = str(k).lower()
                if token in kk:
                    if isinstance(v, list):
                        found.append(len(v))
                    elif isinstance(v, int):
                        found.append(int(v))
                walk(v, kk)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
    walk(payload)
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser(description="Dynamic StaMPS-compatible SCLA correction.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ctx = load_context(args.config)
    mode = str(cfg_get(ctx["cfg"], "corrections.scla.mode", "disabled")).strip().lower()

    src = ctx["proc"] / "atmosphere_correction" / "acquisition_phase_corrected_rad.npy"
    outdir = ctx["proc"] / "scla"
    outdir.mkdir(parents=True, exist_ok=True)
    pre = outdir / "acquisition_phase_pre_scn_rad.npy"
    manifest = outdir / "scla_manifest.json"

    if mode == "disabled":
        atomic_copy(src, pre)
        write_json(manifest, {
            "status": "PASS_SCLA_CANONICAL_PRE_SCN",
            "mode": "disabled",
            "source_phase": str(src),
            "pre_scn_phase": str(pre),
            "acquisition_dates": ctx["dates"],
            "point_count": int(np.load(src, mmap_mode="r").shape[0]),
            "scientific_operation": "identity_passthrough",
        })
    elif mode == "stamps":
        support = outdir / "_support"
        support_info = prepare(
            Path(ctx["paths"].work_dir),
            ctx["data_root"],
            ctx["proc"],
            support,
        )
        contract_path = Path(support_info["contract_path"])
        contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))

        original = _count_contract(contract_payload, "original")
        missing = _count_contract(contract_payload, "missing")
        if original is None or missing is None:
            # Counts are QA only; derive from network total if the helper schema
            # intentionally does not expose both.
            missing = 0 if missing is None else missing
            original = ctx["nifg"] - missing

        # --------------------------------------------------------------
        # Compatibility data-root view for the already validated dynamic
        # SCLA runners.
        #
        # Their frozen path contract expects:
        #
        #     <DATA_ROOT>/RSLC/<date>.rslc.par
        #
        # Public pyPSDS does not require paths.data_dir to be the parent of
        # paths.rslc_dir, so using ctx["data_root"] directly is incorrect.
        # Build a tiny symlink-only compatibility view instead.  This changes
        # path ownership only; no numerical SCLA code or input bytes change.
        # --------------------------------------------------------------

        compat_data_root = (
            support
            /
            "data_root_compat"
        )

        compat_data_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        def _compat_dir(
            alias,
            target,
        ):
            if target is None:
                return

            target = Path(
                target
            ).resolve()

            if not target.is_dir():
                raise FileNotFoundError(
                    target
                )

            link = (
                compat_data_root
                /
                alias
            )

            if (
                link.is_symlink()
                or
                link.exists()
            ):
                if (
                    link.is_symlink()
                    and
                    link.resolve()
                    ==
                    target
                ):
                    return

                if (
                    link.is_dir()
                    and
                    not link.is_symlink()
                ):
                    shutil.rmtree(
                        link
                    )
                else:
                    link.unlink()

            link.symlink_to(
                target,
                target_is_directory=True,
            )

        _compat_dir(
            "RSLC",
            ctx["paths"].rslc_dir,
        )

        # Preserve the common historical auxiliary aliases used by the
        # validated SCLA runtime family when those public paths exist.
        if ctx["paths"].dem_dir is not None:
            _compat_dir(
                "DEM_prep",
                ctx["paths"].dem_dir,
            )

        if ctx["paths"].gacos_dir is not None:
            _compat_dir(
                "GACOS",
                ctx["paths"].gacos_dir,
            )

        # Fail early with a dynamic acquisition-set check rather than
        # allowing one missing RSLC parameter file to surface deep inside
        # the baseline runner.
        missing_rslc_par = [
            str(
                compat_data_root
                /
                "RSLC"
                /
                f"{date}.rslc.par"
            )
            for date in ctx["dates"]
            if not (
                compat_data_root
                /
                "RSLC"
                /
                f"{date}.rslc.par"
            ).is_file()
        ]

        if missing_rslc_par:
            raise RuntimeError(
                "SCLA RSLC parameter contract failed; "
                f"missing={missing_rslc_par[:5]}"
            )

        print(
            "SCLA compatibility data root:",
            compat_data_root,
        )

        print(
            "SCLA RSLC owner             :",
            Path(
                ctx["paths"].rslc_dir
            ).resolve(),
        )

        range_looks, azimuth_looks = derive_multilook_factors(
            ctx["rslc_par"], ctx["geometry_par"]
        )
        print("SCLA multilook factors       :", f"{range_looks}:{azimuth_looks}")

        env = public_env(ctx)
        env.update({
            "PYPSDS_SCLA_PROJECT": str(Path(ctx["paths"].work_dir).resolve()),
            "PYPSDS_SCLA_DATA_ROOT": str(compat_data_root.resolve()),
            "PYPSDS_SCLA_PROC": str(ctx["proc"]),
            "PYPSDS_SCLA_NETWORK_LOG_DIR": str(support_info["network_log_dir"]),
            "PYPSDS_SCLA_BASELINE_CONTRACT": str(contract_path),
            "PYPSDS_SCLA_NIMAGE": str(ctx["nimage"]),
            "PYPSDS_SCLA_NIFG": str(ctx["nifg"]),
            "PYPSDS_SCLA_NSLAVE": str(ctx["nslave"]),
            "PYPSDS_SCLA_REFERENCE_COUNT": str(ctx["nref"]),
            "PYPSDS_SCLA_MISSING_COUNT": str(missing),
            "PYPSDS_SCLA_ORIGINAL_COUNT": str(original),
            "PYPSDS_SCLA_GEOMETRIC_MASTER": str(ctx["master_date"]),
            "PYPSDS_SCLA_RSLC_PAR": str(ctx["rslc_par"]),
            "PYPSDS_SCLA_RANGE_LOOKS": str(range_looks),
            "PYPSDS_SCLA_AZIMUTH_LOOKS": str(azimuth_looks),
        })

        for name in (
            "scla_baseline_runtime.py",
            "scla_k_runtime.py",
            "scla_c_runtime.py",
            "scla_pre_scn_runtime.py",
        ):
            subprocess.run(
                [sys.executable, str(runtime_path(name))],
                env=env,
                check=True,
            )

        if not pre.is_file():
            raise RuntimeError("SCLA runtime did not create acquisition_phase_pre_scn_rad.npy")

        write_json(manifest, {
            "status": "PASS_SCLA_CANONICAL_PRE_SCN",
            "mode": "stamps",
            "source_phase": str(src),
            "pre_scn_phase": str(pre),
            "acquisition_dates": ctx["dates"],
            "images": ctx["nimage"],
            "ifgs": ctx["nifg"],
            "reference_points": ctx["nref"],
            "geometric_master": ctx["master_date"],
            "baseline_source_contract": str(contract_path),
        })
    else:
        raise ValueError(f"Unsupported corrections.scla.mode={mode!r}")

    arr = np.load(pre, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != ctx["nimage"] or arr.dtype != np.float32:
        raise RuntimeError(f"SCLA pre-SCN contract failed: {arr.shape}/{arr.dtype}")

    print("=" * 88)
    print("SCLA STATUS: PASS")
    print("mode   :", mode)
    print("output :", pre)
    print("=" * 88)


if __name__ == "__main__":
    main()
