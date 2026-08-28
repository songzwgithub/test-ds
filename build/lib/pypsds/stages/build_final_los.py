from __future__ import annotations

import argparse
import numpy as np

from pypsds.stages._stage_common import load_context, run_runtime


def main():
    ap = argparse.ArgumentParser(description="Build final referenced LOS time series.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ctx = load_context(args.config)
    run_runtime("final_los_runtime.py", ctx)

    outdir = ctx["proc"] / "final_los"
    phase = outdir / "acquisition_phase_final_rad.npy"
    los_m = outdir / "los_displacement_toward_satellite_m.npy"
    los_mm = outdir / "los_displacement_toward_satellite_mm.npy"
    manifest = outdir / "final_los_manifest.json"

    for p in (phase, los_m, los_mm, manifest):
        if not p.is_file():
            raise FileNotFoundError(p)

    p = np.load(phase, mmap_mode="r")
    l = np.load(los_mm, mmap_mode="r")
    if p.shape != l.shape or p.shape[1] != ctx["nimage"]:
        raise RuntimeError("final LOS shape contract failed")
    if p.dtype != np.float32 or l.dtype != np.float32:
        raise RuntimeError("final LOS dtype contract failed")

    print("=" * 88)
    print("FINAL LOS STATUS: PASS")
    print("LOS positive : toward satellite")
    print("output       :", los_mm)
    print("=" * 88)


if __name__ == "__main__":
    main()
