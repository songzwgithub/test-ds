from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from pypsds.stages._v11_common import (
    atomic_copy,
    cfg_get,
    ensure_gacos_cache,
    ensure_geometry_compat,
    load_context,
    public_env,
    run_runtime,
    write_json,
)


def main():
    ap = argparse.ArgumentParser(description="Optional atmospheric correction with canonical phase output.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ctx = load_context(args.config)
    mode = str(cfg_get(ctx["cfg"], "corrections.atmosphere.mode", "disabled")).strip().lower()

    src = ctx["proc"] / "referenced_timeseries" / "acquisition_phase_referenced_rad.npy"
    outdir = ctx["proc"] / "atmosphere_correction"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / "acquisition_phase_corrected_rad.npy"
    manifest = outdir / "atmosphere_correction_manifest.json"

    if mode == "disabled":
        atomic_copy(src, dst)
        write_json(manifest, {
            "status": "PASS_ATMOSPHERE_CANONICAL_PHASE",
            "mode": "disabled",
            "backend": None,
            "source_phase": str(src),
            "output_phase": str(dst),
            "acquisition_dates": ctx["dates"],
            "reference_date": ctx["temporal_reference_date"],
            "reference_points": ctx["nref"],
            "scientific_operation": "identity_passthrough",
        })
    elif mode == "gacos":
        if ctx["gacos_dir"] is None or not ctx["gacos_dir"].is_dir():
            raise RuntimeError(
                "corrections.atmosphere.mode=gacos requires paths.gacos_dir"
            )
        compat = ensure_geometry_compat(ctx)
        extra = {
            "PYPSDS_PUBLIC_GEOM_COMPAT": compat,
            "PYPSDS_PUBLIC_ATM_CACHE": outdir / "mapping_cache",
        }
        # Validated mapping producer -> canonical mapping cache.
        run_runtime("gacos_mapping_runtime.py", ctx, extra)
        ensure_gacos_cache(ctx)
        run_runtime("gacos_runtime.py", ctx, extra)
        if not dst.is_file():
            raise RuntimeError("GACOS runtime did not create canonical corrected phase")
        # authoritative runtime writes the canonical manifest after transformation
    else:
        raise ValueError(f"Unsupported corrections.atmosphere.mode={mode!r}")

    phase = np.load(dst, mmap_mode="r")
    if phase.ndim != 2 or phase.shape[1] != ctx["nimage"] or phase.dtype != np.float32:
        raise RuntimeError(f"canonical atmosphere phase contract failed: {phase.shape}/{phase.dtype}")

    print("=" * 88)
    print("ATMOSPHERE CORRECTION STATUS: PASS")
    print("mode   :", mode)
    print("output :", dst)
    print("=" * 88)


if __name__ == "__main__":
    main()
