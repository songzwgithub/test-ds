from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from pypsds.prototype import open_from_config


def main():
    ap = argparse.ArgumentParser(
        description="Create an explicit final-DS mask from v0.9 quality layers. "
                    "No universal threshold is implied; thresholds are user-controlled."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--tc-min", type=float, required=True)
    ap.add_argument("--pair-min", type=float, default=0.0)
    ap.add_argument("--accept-evd", action="store_true")
    ap.add_argument("--emi-only", action="store_true")
    args = ap.parse_args()

    if args.accept_evd and args.emi_only:
        raise SystemExit("Choose either --accept-evd or --emi-only, not both")

    cfg, config_path, paths, stack, (_, _, H, W) = open_from_config(args.config)
    d = Path(paths.output_dir) / "v09"
    ps = np.load(d / "ps_mask.npy")
    pl = np.load(d / "pl_valid.npy")
    tc = np.load(d / "temporal_coherence.npy")
    pc = np.load(d / "median_pair_coherence.npy")
    est = np.load(d / "estimator_code.npy")

    ds = pl & ~ps & np.isfinite(tc) & np.isfinite(pc)
    ds &= tc >= args.tc_min
    ds &= pc >= args.pair_min
    if args.emi_only:
        ds &= est == 1
    elif not args.accept_evd:
        # Safe default: keep only EMI unless EVD is explicitly accepted.
        ds &= est == 1

    tag = f"tc{args.tc_min:.3f}_pc{args.pair_min:.3f}_" + ("evd" if args.accept_evd else "emi")
    out = d / f"final_ds_{tag}.npy"
    np.save(out, ds)

    print("=" * 80)
    print("v0.9 final DS selection")
    print("=" * 80)
    print(f"TC min       : {args.tc_min}")
    print(f"pair min     : {args.pair_min}")
    print(f"accept EVD   : {args.accept_evd}")
    print(f"selected DS  : {ds.sum()} ({100*ds.mean():.3f}%)")
    print(f"saved        : {out}")


if __name__ == "__main__":
    main()
