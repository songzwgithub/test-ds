"""Read-only Whirlwind/SNAPHU A/B on *phase-linked PS/DS coarse nodes*.

Input is PointPhaseStack-derived, complex-mean coarse acquisition phase, NOT
GAMMA's original interferograms. Only observed cells are passed to the
unwrappers. The correlation array here is explicitly a diagnostic *surrogate*,
not independently estimated interferometric coherence; an assumed nlooks does
not validate probabilistic costs. Results MUST NOT go into production before
independent PS and temporal-network validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from pypsds.unwrap.whirlwind_ab import (
    compare_same_input,
    unwrap_snaphu,
    unwrap_whirlwind,
)

TAU = 2.0 * np.pi


def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def read_network(path: Path, ndate: int):
    """Read GAMMA 1-based itab, consistent with stamps3d_backend.load_itab."""
    edges = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 2:
            raise ValueError(f"Invalid itab line {lineno}")
        try:
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
        except ValueError as exc:
            raise ValueError(f"Invalid itab index on line {lineno}") from exc
        if not (0 <= i < ndate and 0 <= j < ndate and i != j):
            raise ValueError(f"Out-of-bounds itab edge on line {lineno}")
        edges.append((i, j))
    if not edges:
        raise ValueError("No temporal edges in network.itab")
    return edges


def build_point_ifg(node_phase, node_grid, index_i: int, index_j: int,
                    *, quality_mode: str = "uniform", uniform_quality: float = 0.80,
                    resultant_length=None):
    """Rasterize wrapped node phase difference without nearest-hole filling.

    One grid cell corresponds to one complex-mean PS/DS node; invalid pixels
    remain masked. Input point phases are *not* interpolated across gaps.
    """
    ph = np.asarray(node_phase)
    grid = np.asarray(node_grid)
    if ph.ndim != 2 or grid.ndim != 2 or ph.shape[0] < 1:
        raise ValueError("Expected node phase [node,acq] and node grid [row,col]")
    if not np.issubdtype(grid.dtype, np.integer) or np.any(grid < -1):
        raise ValueError("Invalid node grid identifiers")
    if not (0 <= index_i < ph.shape[1] and 0 <= index_j < ph.shape[1] and index_i != index_j):
        raise ValueError("Invalid acquisition pair indices")
    occupied = grid >= 0
    ids = np.asarray(grid[occupied], np.int64)
    if not np.array_equal(np.sort(ids), np.arange(ph.shape[0])):
        raise ValueError("Each occupied cell must map to exactly one contiguous node ID")
    if not (np.isfinite(ph[:, index_i]).all() and np.isfinite(ph[:, index_j]).all()):
        raise ValueError("Nonfinite phase at a requested acquisition")
    if quality_mode not in {"uniform", "resultant"}:
        raise ValueError("quality_mode must be uniform or resultant")
    if not (np.isfinite(uniform_quality) and 0 < uniform_quality <= 1):
        raise ValueError("uniform_quality must lie in (0, 1]")

    if quality_mode == "uniform":
        reliability = np.full(ph.shape[0], uniform_quality, dtype=np.float32)
    else:
        if resultant_length is None:
            raise ValueError("resultant quality requires coarse_phase_resultant_length.npy")
        res = np.asarray(resultant_length)
        if res.shape != ph.shape:
            raise ValueError("resultant quality and phase shapes differ")
        q0 = np.asarray(res[:, index_i], np.float64)
        q1 = np.asarray(res[:, index_j], np.float64)
        if not (np.isfinite(q0).all() and np.isfinite(q1).all()):
            raise ValueError("Nonfinite resultant length")
        if np.any((q0 < 0) | (q0 > 1) | (q1 < 0) | (q1 > 1)):
            raise ValueError("Resultant length outside [0,1]")
        # This is a phasor-concentration proxy, NOT sample coherence.
        # Avoid zero correlation only to meet solver's numerical contract.
        reliability = np.clip(np.sqrt(q0 * q1), 0.03, 0.999).astype(np.float32)

    phase_diff = _wrap(np.asarray(ph[:, index_j], np.float64) -
                       np.asarray(ph[:, index_i], np.float64))
    ifg = np.zeros(grid.shape, dtype=np.complex64)
    corr = np.zeros(grid.shape, dtype=np.float32)
    ifg[occupied] = np.exp(1j * phase_diff[ids]).astype(np.complex64)
    corr[occupied] = reliability[ids]
    return ifg, corr, np.ascontiguousarray(occupied)


def _topology(mask):
    labels, ncomp = ndimage.label(mask, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    sizes = np.bincount(labels[mask], minlength=ncomp + 1)[1:]
    return {
        "four_connected_components": int(ncomp),
        "largest_component_nodes": int(sizes.max()) if sizes.size else 0,
        "largest_component_fraction": float(sizes.max() / mask.sum()) if sizes.size else 0.,
    }


def _baseline_comparison(*, baseline_dir, node_phase, node_grid,
                         index_i, index_j, solver_uw, solver_cc, mask):
    path = Path(baseline_dir) / "coarse_acquisition_phase_unwrapped_rad.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find baseline phase: {path}")
    base = np.load(path, mmap_mode="r", allow_pickle=False)
    if base.shape != node_phase.shape:
        raise ValueError("Baseline phase has different node/acquisition shape")
    i, j = int(index_i), int(index_j)
    # Verify identity of the source nodes before comparing the integer branch.
    worst = 0.0
    for s in range(0, len(node_phase), 4096):
        stop = min(len(node_phase), s + 4096)
        p0 = np.asarray(node_phase[s:stop, i], np.float64)
        p1 = np.asarray(node_phase[s:stop, j], np.float64)
        b0 = np.asarray(base[s:stop, i], np.float64)
        b1 = np.asarray(base[s:stop, j], np.float64)
        worst = max(worst, float(np.max(np.abs(_wrap(b0 - p0)))),
                    float(np.max(np.abs(_wrap(b1 - p1)))))
    if not np.isfinite(worst) or worst > 1e-3:
        raise ValueError(f"Baseline source nodes fail wrapped parity: {worst:g} rad")
    diff = (np.asarray(base[:, j], np.float64) -
            np.asarray(base[:, i], np.float64)).astype(np.float32)
    base_raster = np.zeros(node_grid.shape, np.float32)
    occ = node_grid >= 0
    base_raster[occ] = diff[node_grid[occ]]
    # Each occupied component has a separate unobservable spatial gauge.
    base_cc, _ = ndimage.label(mask, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    return {
        "source": str(path),
        "node_phase_max_wrapped_difference_rad": worst,
        "solver_vs_statcost": compare_same_input(
            solver_uw, base_raster, solver_cc, base_cc, mask),
        "note": "IFG differential phase gauge-aligned per component intersection; baseline is not truth",
    }


def main(argv=None, *, whirlwind_func=None, snaphu_func=None):
    p = argparse.ArgumentParser(description="Whirlwind/SNAPHU A/B on phase-linked PS/DS coarse nodes")
    p.add_argument("--node-phase", type=Path, required=True)
    p.add_argument("--node-grid", type=Path, required=True)
    p.add_argument("--dates", type=Path, required=True)
    p.add_argument("--itab", type=Path, required=True)
    p.add_argument("--pair-index", type=int, default=0,
                   help="0-based index of one IFG in network.itab")
    p.add_argument("--quality-mode", choices=("uniform", "resultant"), default="uniform")
    p.add_argument("--uniform-quality", type=float, default=0.8)
    p.add_argument("--resultant-length", type=Path)
    p.add_argument("--nlooks", type=float)
    p.add_argument("--wavelength-m", type=float)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--snaphu-exe", default="snaphu")
    args = p.parse_args(argv)
    dates = args.dates.read_text(encoding="utf-8").split()
    ph = np.load(args.node_phase, mmap_mode="r", allow_pickle=False)
    grid = np.load(args.node_grid, mmap_mode="r", allow_pickle=False)
    if ph.ndim != 2 or ph.shape[1] != len(dates):
        raise ValueError("Node phase and dates count disagree")
    edges = read_network(args.itab, len(dates))
    if not (0 <= args.pair_index < len(edges)):
        p.error("pair-index is outside the network.itab range")
    if not args.audit_only and not (args.nlooks is not None and
        math.isfinite(args.nlooks) and args.nlooks >= 1 and
        args.wavelength_m is not None and math.isfinite(args.wavelength_m) and args.wavelength_m > 0):
        p.error("Solver A/B requires --nlooks >=1 and --wavelength-m >0; these are uncalibrated benchmark assumptions")
    i, j = edges[args.pair_index]
    res = None
    if args.quality_mode == "resultant":
        if args.resultant_length is None:
            p.error("--resultant-length required for resultant quality mode")
        res = np.load(args.resultant_length, mmap_mode="r", allow_pickle=False)
    igram, quality, mask = build_point_ifg(
        ph, grid, i, j, quality_mode=args.quality_mode,
        uniform_quality=args.uniform_quality, resultant_length=res)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "INPUT_AUDIT" if args.audit_only else "BENCHMARK_ONLY_NOT_PRODUCTION",
        "input_origin": "PHASE_LINKED_PS_DS_COARSE_NODES_NOT_GAMMA_IFG",
        "point_data": str(args.node_phase),
        "node_grid": str(args.node_grid),
        "date_i": dates[i], "date_j": dates[j],
        "acquisition_index_0based": [i, j],
        "pair_index_0based": int(args.pair_index),
        "grid_shape": list(grid.shape),
        "valid_nodes": int(mask.sum()),
        "valid_fraction": float(mask.mean()),
        "quality_mode": args.quality_mode,
        "quality_is_true_interferometric_coherence": False,
        "assumed_nlooks": None if args.nlooks is None else float(args.nlooks),
        "cost_model_calibrated_to_data": False,
        "empty_cells_interpolated": False,
        "synthetic_bridges_enabled": False,
        "quality_p05_p50_p95": [float(x) for x in np.percentile(quality[mask], [5, 50, 95])],
        "production_phase_modified": False,
        **_topology(mask),
        "scientific_caveat": (
            "This comparison uses phase-linked PS/DS node IFGs and a surrogate "
            "correlation field. A lower cost or smoother map does not establish "
            "physical correctness. Disconnected components have independent gauges."
        ),
    }
    (out / "input_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    wuw, wcc, ws = unwrap_whirlwind(
        igram, quality, mask, args.nlooks, unwrap_func=whirlwind_func)
    if snaphu_func is None:
        snaphu_func = unwrap_snaphu
    suw, scc, ss = snaphu_func(
        igram, quality, mask, args.nlooks, args.wavelength_m,
        scratch=out / "snaphu_scratch", executable=args.snaphu_exe)
    for label, uw, cc in (("whirlwind", wuw, wcc), ("snaphu", suw, scc)):
        if uw.shape != grid.shape or cc.shape != grid.shape:
            raise ValueError(f"{label} output does not match node grid shape")
        np.save(out / (label + "_node_ifg_unwrapped_rad.npy"),
                np.asarray(uw[mask], dtype=np.float32)[np.argsort(grid[mask])])
        np.save(out / (label + "_node_conncomp.npy"),
                np.asarray(cc[mask])[np.argsort(grid[mask])])
    summary.update({
        "whirlwind": ws, "snaphu": ss,
        "whirlwind_labeled_nodes": int(np.count_nonzero(wcc[mask])),
        "snaphu_labeled_nodes": int(np.count_nonzero(scc[mask])),
        "gauge_invariant_solver_comparison": compare_same_input(wuw, suw, wcc, scc, mask),
    })
    if args.baseline_dir is not None:
        summary["whirlwind_vs_statcost"] = _baseline_comparison(
            baseline_dir=args.baseline_dir, node_phase=ph, node_grid=grid,
            index_i=i, index_j=j, solver_uw=wuw, solver_cc=wcc, mask=mask)
        summary["snaphu_vs_statcost"] = _baseline_comparison(
            baseline_dir=args.baseline_dir, node_phase=ph, node_grid=grid,
            index_i=i, index_j=j, solver_uw=suw, solver_cc=scc, mask=mask)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
