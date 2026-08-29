
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pypsds.config import cfg_get
from pypsds.context import open_from_config
from pypsds.monitoring.stage_visualization_v7 import render_override


STAGES = [
    "ds_statistics","phase_cache","exact_support_cache","phase_linking",
    "ds_selection","ps_finalize","point_stack","network_prepare",
    "network_build","network_cycle_quality","network_finalize",
    "virtual_ifg_quality","spatial_graph_quality","spatial_bridge_quality",
    "spatial_component_quality","spatial_anchor_quality",
    "spatial_anchor_summary","spatial_local_graph_quality","spatial_graph",
    "spatial_gradient_quality","unwrap_policy","unwrap",
    "unwrap_severity_quality","unwrap_conflict_quality",
    "unwrap_acquisition_quality","temporal_closure",
    "temporal_integer_candidate","temporal_candidate_spatial_quality",
    "unwrap_signature_quality","unwrap_finalize","timeseries_inversion",
    "point_geometry","residual_ramp","reference","atmosphere_correction",
    "scla","scn","final_los","point_products",
]
NUM = {s: i for i, s in enumerate(STAGES, 1)}
STAGE_NAMES = STAGES
STAGE_NUMBER = NUM

ALIASES = {
    "exact_support_cache": ["exact_support_cache", "shp_cache"],
    "network_cycle_quality": ["network_cycle_quality", "network_cycle"],
    "virtual_ifg_quality": ["virtual_ifg_quality", "virtual_ifg"],
    "spatial_component_quality": ["spatial_component_quality", "spatial_components"],
    "spatial_local_graph_quality": ["spatial_local_graph_quality", "local_spatial_graph"],
    "spatial_gradient_quality": ["spatial_gradient_quality", "spatial_phase_gradient"],
    "unwrap_policy": ["unwrap_policy", "unwrap_component_policy"],
    "unwrap_conflict_quality": ["unwrap_conflict_quality", "safe_conflict_acquisition_quality"],
    "unwrap_acquisition_quality": ["unwrap_acquisition_quality", "safe_conflict_acquisition_quality"],
    "temporal_closure": ["temporal_closure", "temporal_integer_closure_quality"],
    "temporal_integer_candidate": ["temporal_integer_candidate", "temporal_integer_closure_quality"],
    "unwrap_signature_quality": ["unwrap_signature_quality", "fragment_signature_feasibility"],
    "unwrap_finalize": ["final_unwrap"],
}

def _date_objects(stack):
    out = []
    for d in stack.dates:
        s = str(d)
        if len(s) >= 8 and s[:8].isdigit():
            out.append(datetime.strptime(s[:8], "%Y%m%d"))
        else:
            out.append(datetime(1970,1,1))
    return out

def _safe_load(path: Path):
    return np.load(path, mmap_mode="r", allow_pickle=False)

def _finite_sample(x, max_values=250000):
    a = np.asarray(x).reshape(-1)
    if a.size > max_values:
        a = a[::max(1, a.size // max_values)]
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    if np.iscomplexobj(a):
        a = np.angle(a)
    if np.issubdtype(a.dtype, np.number):
        a = a.astype(np.float64, copy=False)
        a = a[np.isfinite(a)]
    return a

def _robust_limits(x, pct=(2.0, 98.0)):
    s = _finite_sample(x, 200000)
    if s.size == 0:
        return None, None
    lo, hi = np.percentile(s, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    return float(lo), float(hi)

def _raster(ax, a, title, *, categorical=False):
    x = np.asarray(a)
    rs = max(1, int(np.ceil(x.shape[0] / 900))) if x.ndim >= 2 else 1
    cs = max(1, int(np.ceil(x.shape[1] / 900))) if x.ndim >= 2 else 1
    b = np.asarray(x[::rs, ::cs]) if x.ndim == 2 else np.asarray(x)

    kwargs = {}
    if not categorical:
        lo, hi = _robust_limits(b)
        if lo is not None:
            kwargs["vmin"] = lo
            kwargs["vmax"] = hi

    im = ax.imshow(b, aspect="auto", **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    if not categorical:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

def _full_point_xy(output: Path):
    pps = output / "processing" / "point_phase_stack"
    rp = pps / "rows.npy"
    cp = pps / "cols.npy"
    if rp.is_file() and cp.is_file():
        return (
            np.asarray(_safe_load(cp), dtype=np.float64),
            np.asarray(_safe_load(rp), dtype=np.float64),
            "Radar column",
            "Radar row",
        )
    return None

def _strict_point_xy(output: Path):
    geom = output / "processing" / "point_geometry"
    lonp = geom / "longitude_deg.npy"
    latp = geom / "latitude_deg.npy"
    if lonp.is_file() and latp.is_file():
        return (
            np.asarray(_safe_load(lonp), dtype=np.float64),
            np.asarray(_safe_load(latp), dtype=np.float64),
            "Longitude",
            "Latitude",
        )
    return None

def _point_xy_for_length(output: Path, n: int):
    strict = _strict_point_xy(output)
    if strict is not None and strict[0].size == n:
        return strict
    full = _full_point_xy(output)
    if full is not None and full[0].size == n:
        return full
    sid = output / "processing" / "network_inversion" / "strict_point_ids.npy"
    if full is not None and sid.is_file():
        ids = np.asarray(np.load(sid), dtype=np.int64)
        if ids.size == n and ids.max(initial=-1) < full[0].size:
            return (full[0][ids], full[1][ids], full[2], full[3])
    return None

def _point_map(ax, output: Path, values, title, max_points=300000, categorical=False):
    v = np.asarray(values).reshape(-1)
    xy = _point_xy_for_length(output, v.size)
    if xy is None:
        s = _finite_sample(v, max_points)
        ax.plot(np.arange(s.size), s, ".", markersize=1)
        ax.set_title(title + " [index view]", fontsize=10)
        return

    x, y, xl, yl = xy
    ids = np.arange(v.size)
    if ids.size > max_points:
        ids = ids[::max(1, ids.size // max_points)]
    z = np.asarray(v[ids])

    if z.dtype == np.bool_:
        z = z.astype(np.uint8)
        categorical = True
    if np.iscomplexobj(z):
        z = np.angle(z)

    good = np.isfinite(z.astype(np.float64, copy=False))
    ids = ids[good]
    z = z[good]

    kwargs = {}
    if not categorical:
        lo, hi = _robust_limits(z)
        if lo is not None:
            kwargs["vmin"] = lo
            kwargs["vmax"] = hi

    sc = ax.scatter(x[ids], y[ids], c=z, s=1.0, linewidths=0, rasterized=True, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if not categorical and z.size:
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

def _mask_map(ax, output: Path, mask, title, max_points=300000):
    m = np.asarray(mask, dtype=bool).reshape(-1)
    xy = _point_xy_for_length(output, m.size)
    if xy is None:
        ax.plot(m.astype(np.uint8), linewidth=0.5)
        ax.set_title(title, fontsize=10)
        return

    x, y, xl, yl = xy
    all_ids = np.arange(m.size)
    if all_ids.size > max_points:
        all_ids = all_ids[::max(1, all_ids.size // max_points)]
    ids = np.flatnonzero(m)
    if ids.size > max_points:
        ids = ids[::max(1, ids.size // max_points)]

    ax.scatter(x[all_ids], y[all_ids], s=0.15, alpha=0.10, linewidths=0)
    if ids.size:
        ax.scatter(x[ids], y[ids], s=1.0, linewidths=0)
    ax.set_title(f"{title} — {np.count_nonzero(m):,}", fontsize=10)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

def _hist(ax, x, title, xlabel=None, bins=60):
    s = _finite_sample(x)
    if s.size == 0:
        ax.set_axis_off()
        ax.text(0.05, 0.95, "No finite values", va="top", transform=ax.transAxes)
        return
    ax.hist(s, bins=bins)
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)

def _bar_counts(ax, labels, values, title):
    ids = np.arange(len(labels))
    ax.bar(ids, values)
    ax.set_xticks(ids, labels, rotation=25, ha="right")
    ax.set_title(title, fontsize=10)

def _find_bperp(output: Path, ndate: int):
    p = output / "processing"
    candidates = [
        p / "network" / "acquisition_bperp_m.npy",
        p / "network_prepare" / "acquisition_bperp_m.npy",
        p / "acquisition_bperp_m.npy",
        p / "network" / "bperp_m.npy",
    ]
    for c in candidates:
        if c.is_file():
            a = np.asarray(np.load(c), dtype=np.float64).reshape(-1)
            if a.size == ndate and np.all(np.isfinite(a)):
                return a, c
    for c in p.rglob("*bperp*.npy"):
        try:
            a = np.asarray(np.load(c), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if a.size == ndate and np.all(np.isfinite(a)):
            return a, c
    return None, None

def _load_itab(path: Path, ndate: int):
    edges = []
    if not path.is_file():
        return edges
    for raw in path.read_text(errors="ignore").splitlines():
        x = raw.split()
        if len(x) < 2:
            continue
        try:
            i = int(x[0]) - 1
            j = int(x[1]) - 1
        except Exception:
            continue
        if 0 <= i < ndate and 0 <= j < ndate:
            edges.append((i, j))
    return edges

def _network_itab(output: Path):
    p = output / "processing"
    candidates = [
        p / "network" / "network.itab",
        p / "network" / "network_final.itab",
        p / "network_prepare" / "network.itab",
    ]
    return next((x for x in candidates if x.is_file()), None)

def _network_dashboard(ax, output: Path, stack, title_prefix="Network"):
    dates = _date_objects(stack)
    ndate = len(dates)
    bperp, bpath = _find_bperp(output, ndate)
    itabp = _network_itab(output)
    edges = _load_itab(itabp, ndate) if itabp else []

    if bperp is None:
        ax[0].set_axis_off()
        ax[0].text(
            0.03, 0.97,
            "Perpendicular baseline unavailable.\n"
            "Cannot draw a valid time–B⊥ network.",
            va="top", transform=ax[0].transAxes, family="monospace"
        )
    else:
        for i, j in edges:
            ax[0].plot([dates[i], dates[j]], [bperp[i], bperp[j]], linewidth=0.7, alpha=0.45)
        ax[0].scatter(dates, bperp, s=20)
        ax[0].axhline(0.0, linewidth=0.8)
        ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
        ax[0].set_ylabel("B⊥ [m]")
        ax[0].set_xlabel("Acquisition date")
        ax[0].set_title(f"{title_prefix}: time–perpendicular-baseline network", fontsize=10)

    if edges:
        dt = np.array([abs((dates[j] - dates[i]).days) for i, j in edges], dtype=np.float64)
        ax[1].hist(dt, bins=min(30, max(8, len(edges)//4)))
        ax[1].set_xlabel("|Δt| [days]")
        ax[1].set_title("Temporal-baseline distribution", fontsize=10)

        if bperp is not None:
            db = np.array([abs(bperp[j] - bperp[i]) for i, j in edges], dtype=np.float64)
            ax[2].hist(db, bins=min(30, max(8, len(edges)//4)))
            ax[2].set_xlabel("|ΔB⊥| [m]")
            ax[2].set_title("Perpendicular-baseline distribution", fontsize=10)
        else:
            ax[2].set_axis_off()
    else:
        ax[1].set_axis_off()
        ax[2].set_axis_off()

    return {"edges": len(edges), "bperp_available": bperp is not None, "bperp_source": str(bpath) if bpath else None}

def _log_tail(output: Path, stage: str, n=12):
    p = output / "logs" / f"{stage}.log"
    if not p.is_file():
        return []
    return p.read_text(errors="ignore").splitlines()[-n:]

def _summary_panel(ax, stage, status, messages):
    ax.set_axis_off()
    lines = [f"Stage {NUM[stage]:02d}: {stage}", f"Visualization status: {status}", ""] + [str(x) for x in messages]
    ax.text(0.01, 0.99, "\n".join(lines), va="top", transform=ax.transAxes, family="monospace", fontsize=7.5)

def _artifact_inventory_panel(ax, output: Path, stage: str):
    ax.set_axis_off()
    lines = [f"Artifact inventory: {stage}", ""]
    dirs = _stage_dirs(output, stage)
    if not dirs:
        lines.append("No stage directory found under output/processing/")
    else:
        for d in dirs:
            lines.append(str(d.relative_to(output)))
            for f in sorted(d.glob("*"))[:20]:
                suffix = "/" if f.is_dir() else ""
                lines.append(f"  - {f.name}{suffix}")
            if len(list(d.glob("*"))) > 20:
                lines.append("  - ...")
            lines.append("")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", transform=ax.transAxes, family="monospace", fontsize=7.3)

def _log_panel(ax, output: Path, stage: str):
    ax.set_axis_off()
    tail = _log_tail(output, stage, 18)
    txt = "\n".join(tail) if tail else "No stage log found."
    ax.text(0.02, 0.98, txt, va="top", transform=ax.transAxes, family="monospace", fontsize=7.2)

def _mode(cfg, stage):
    key = {"residual_ramp": "residual_ramp", "atmosphere_correction": "atmosphere", "scla": "scla", "scn": "scn"}.get(stage)
    if key is None:
        return None
    return str(cfg_get(cfg, f"corrections.{key}.mode", "disabled")).strip().lower()

def _stage_dirs(output: Path, stage: str):
    p = output / "processing"
    names = ALIASES.get(stage, [stage])
    dirs = []
    for n in names:
        d = p / n
        if d.is_dir():
            dirs.append(d)
    return dirs

def _candidate_arrays(output: Path, stage: str, max_arrays=20):
    p = output / "processing"
    result = []
    seen = set()
    for d in _stage_dirs(output, stage):
        for f in sorted(d.glob("*.npy")):
            k = str(f.resolve())
            if k not in seen:
                seen.add(k)
                result.append(f)
    return result[:max_arrays]

def _artifact_score(path: Path):
    n = path.name.lower()
    score = 0
    for tok, w in (
        ("mask", 30), ("quality", 28), ("residual", 26), ("count", 24),
        ("coherence", 24), ("cov", 23), ("corr", 22), ("matrix", 21),
        ("confidence", 22), ("degree", 20), ("component", 18), ("integer", 18),
        ("severity", 18), ("status", 16), ("gradient", 16), ("anchor", 15),
        ("edge", 12),
    ):
        if tok in n:
            score += w
    if "phase" in n:
        score -= 4
    return score


def _find_square_matrix_candidate(output: Path, stage: str, ndate: int):
    for path in _candidate_arrays(output, stage, max_arrays=80):
        try:
            a = _safe_load(path)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[0] == a.shape[1] and 4 <= a.shape[0] <= 256:
            return path, np.asarray(a)
        if a.ndim == 3 and a.shape[-1] == a.shape[-2] and 4 <= a.shape[-1] <= 256:
            return path, np.asarray(a[0])
    return None, None


def _find_phase_link_covariance(output: Path, ndate: int):
    root = output / "processing"
    candidates = []

    for name in (
        "sm_cov_unit_ifg_geom_master.npy",
        "covariance_matrix.npy",
        "temporal_covariance_matrix.npy",
        "coherence_matrix.npy",
    ):
        candidates.extend(root.rglob(name))

    for p in root.rglob("*.npy"):
        low = p.name.lower()
        if any(k in low for k in ("cov", "coh", "corr")):
            candidates.append(p)

    seen = set()

    for path in candidates:
        k = str(path.resolve())
        if k in seen:
            continue
        seen.add(k)

        try:
            a = _safe_load(path)
        except Exception:
            continue

        if a.ndim == 2 and a.shape == (ndate, ndate):
            return path, np.asarray(a)

        if a.ndim == 3 and a.shape[-2:] == (ndate, ndate):
            return path, np.asarray(a[0])

    return None, None


def _covariance_to_coherence(matrix):
    c = np.asarray(matrix)

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"not a square matrix: {c.shape}")

    mag = np.abs(c).astype(np.float64, copy=False)
    diag = np.abs(np.diag(c)).astype(np.float64, copy=False)
    denom = np.sqrt(np.outer(diag, diag))

    coh = np.full(mag.shape, np.nan, dtype=np.float64)
    good = np.isfinite(denom) & (denom > 0)

    coh[good] = mag[good] / denom[good]
    coh = np.clip(coh, 0.0, 1.0)

    ii = np.flatnonzero(np.isfinite(diag) & (diag > 0))
    coh[ii, ii] = 1.0

    return coh


def _plot_coherence_matrix(ax, matrix, title):
    coh = _covariance_to_coherence(matrix)

    im = ax.imshow(
        coh,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Acquisition index")
    ax.set_ylabel("Acquisition index")

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.04,
        pad=0.02,
        label="Normalized coherence",
    )

    return coh


def _plot_matrix(ax, m, title):
    x = np.asarray(m)
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = np.asarray(x, dtype=np.float64)
    im = ax.imshow(x, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Acquisition index")
    ax.set_ylabel("Acquisition index")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def _generic_semantic_panel(ax, output: Path, stack, stage: str, max_points: int):
    arrays = sorted(
        _candidate_arrays(output, stage),
        key=lambda p: (-_artifact_score(p), p.name),
    )

    ndate = len(stack.dates)
    dates = _date_objects(stack)
    itabp = _network_itab(output)
    nedge = len(_load_itab(itabp, ndate)) if itabp else 0

    used = 0
    messages = []

    for path in arrays:
        if used >= 3:
            break

        try:
            a = _safe_load(path)
        except Exception as exc:
            messages.append(f"{path.name}: load failed: {exc}")
            continue

        title = path.stem

        try:
            if a.ndim == 2:
                fullxy = _full_point_xy(output)
                strictxy = _strict_point_xy(output)

                point_time = (
                    (fullxy is not None and a.shape[0] == fullxy[0].size and a.shape[1] == ndate)
                    or
                    (strictxy is not None and a.shape[0] == strictxy[0].size and a.shape[1] == ndate)
                )

                if point_time:
                    n = a.shape[0]
                    v = np.empty(n, dtype=np.float64)

                    for p0 in range(0, n, 100000):
                        p1 = min(p0 + 100000, n)
                        q = np.asarray(a[p0:p1], dtype=np.float64)
                        v[p0:p1] = np.sqrt(np.nanmean(q*q, axis=1))

                    _point_map(ax[used], output, v, title + " — temporal RMS", max_points)
                    messages.append(f"{path.name}: point×acquisition {a.shape}")
                    used += 1
                    continue

                if nedge > 0 and a.shape[1] == nedge:
                    b = np.asarray(a)
                    if b.dtype == np.bool_:
                        b = b.astype(np.uint8)

                    im = ax[used].imshow(b, aspect="auto")
                    ax[used].set_title(title + " — cycle × IFG", fontsize=10)
                    ax[used].set_xlabel("IFG index")
                    ax[used].set_ylabel("Cycle / constraint index")
                    plt.colorbar(im, ax=ax[used], fraction=0.04, pad=0.02)

                    messages.append(f"{path.name}: cycle×IFG {a.shape}")
                    used += 1
                    continue

                if a.shape[0] > 8 and a.shape[1] > 8:
                    _raster(
                        ax[used],
                        a,
                        title,
                        categorical=(a.dtype == np.bool_ or np.issubdtype(a.dtype, np.integer)),
                    )
                    messages.append(f"{path.name}: raster {a.shape}")
                    used += 1
                    continue

            if a.ndim == 1:
                if a.dtype == np.bool_:
                    if _point_xy_for_length(output, a.size) is not None:
                        _mask_map(ax[used], output, a, title, max_points)
                    else:
                        vals, counts = np.unique(np.asarray(a, dtype=np.uint8), return_counts=True)
                        ax[used].bar(vals.astype(str), counts)
                        ax[used].set_title(title, fontsize=10)

                elif a.size == ndate:
                    ax[used].plot(dates, np.asarray(a, dtype=np.float64), ".-")
                    ax[used].xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax[used].xaxis.set_major_formatter(
                        mdates.ConciseDateFormatter(ax[used].xaxis.get_major_locator())
                    )
                    ax[used].set_title(title + " — acquisition series", fontsize=10)

                elif nedge > 0 and a.size == nedge:
                    ax[used].plot(np.arange(1, nedge + 1), np.asarray(a), ".")
                    ax[used].set_xlabel("IFG index")
                    ax[used].set_title(title + " — IFG metric", fontsize=10)

                elif _point_xy_for_length(output, a.size) is not None:
                    _point_map(
                        ax[used],
                        output,
                        a,
                        title,
                        max_points,
                        categorical=np.issubdtype(a.dtype, np.integer),
                    )

                else:
                    _hist(ax[used], a, title)

                messages.append(f"{path.name}: vector {a.shape}")
                used += 1

        except Exception as exc:
            messages.append(f"{path.name}: render skipped: {exc}")

    if used == 0:
        _artifact_inventory_panel(ax[0], output, stage)
        _log_panel(ax[1], output, stage)
        ax[2].set_axis_off()
        ax[2].text(
            0.03,
            0.97,
            "No persistent numeric artifact with a unique scientific visualization contract.\n"
            "See artifact inventory and stage log.",
            va="top",
            transform=ax[2].transAxes,
            family="monospace",
            fontsize=8,
        )
        return "INFO", messages

    while used < 3:
        if used == 1:
            _log_panel(ax[used], output, stage)
        elif used == 2:
            _artifact_inventory_panel(ax[used], output, stage)
        used += 1

    return "PASS", messages[:10]

def _ds_statistics(ax, output):
    d = output / "processing" / "ds_statistics"
    validp = d / "raw_valid.npy"
    psp = d / "ps_mask.npy"
    adip = d / "amplitude_dispersion_index.npy"
    s2p = d / "rayleigh_scale2.npy"

    if validp.is_file():
        _raster(ax[0], _safe_load(validp), f"Raw-valid mask — {np.count_nonzero(_safe_load(validp)):,}", categorical=True)
    else:
        _artifact_inventory_panel(ax[0], output, "ds_statistics")

    if psp.is_file():
        _raster(ax[1], _safe_load(psp), f"Raw ADI PS — {np.count_nonzero(_safe_load(psp)):,}", categorical=True)
    else:
        _log_panel(ax[1], output, "ds_statistics")

    if adip.is_file():
        x = _safe_load(adip)
        _hist(ax[2], x, "Amplitude-dispersion index", "ADI")
        ax[2].axvline(0.25, linestyle="--")
    elif s2p.is_file():
        x = _finite_sample(_safe_load(s2p))
        if x.size:
            lo, hi = np.percentile(x, [1, 99])
            x = x[(x >= lo) & (x <= hi)]
        _hist(ax[2], x, "Rayleigh scale² (p01–p99)", bins=60)
    else:
        ax[2].set_axis_off()

    return "PASS", ["PS threshold ADI ≤ 0.25."]

def _phase_cache(ax, output):
    cache = output / "processing" / "cache"
    yxt = cache / "phase_corrected_yxt.npy"
    gvalid = cache / "phase_geometry_valid.npy"
    if yxt.is_file():
        ax[0].set_axis_off()
        a = _safe_load(yxt)
        ax[0].text(0.03, 0.97, f"phase_corrected_yxt.npy\nshape={a.shape}\ndtype={a.dtype}", va="top", transform=ax[0].transAxes, family="monospace")
    else:
        ax[0].set_axis_off()
        ax[0].text(0.03, 0.97, "PHASE CACHE NOT MATERIALIZED\nSequential/date-aware streaming mode is active.", va="top", transform=ax[0].transAxes, family="monospace")

    if gvalid.is_file():
        _raster(ax[1], _safe_load(gvalid), f"{gvalid.stem} — {np.count_nonzero(_safe_load(gvalid)):,}", categorical=True)
    else:
        _artifact_inventory_panel(ax[1], output, "phase_cache")

    _log_panel(ax[2], output, "phase_cache")
    return "PASS", ["No fake cache raster is drawn when cache is intentionally skipped."]


def _exact_support_cache(ax, output, stack, max_points):
    dirs = _stage_dirs(output, "exact_support_cache")
    files = []

    for d in dirs:
        files.extend(sorted(d.glob("*.npy")))

    count_candidates = [
        p for p in files
        if ("shp_count" in p.name.lower() or "support_count" in p.name.lower())
    ]

    countp = count_candidates[0] if count_candidates else None

    if countp is not None:
        count = _safe_load(countp)

        if count.ndim == 2:
            _raster(ax[0], count, f"{countp.stem} — SHP/support count")
            _hist(
                ax[1],
                count,
                "SHP/support-count distribution",
                "support pixels",
                bins=60,
            )
        else:
            _hist(ax[0], count, countp.stem)
            _artifact_inventory_panel(ax[1], output, "exact_support_cache")
    else:
        _artifact_inventory_panel(ax[0], output, "exact_support_cache")
        _log_panel(ax[1], output, "exact_support_cache")

    ax[2].set_axis_off()

    support_files = [
        p.name
        for p in files
        if "support" in p.name.lower()
    ]

    ax[2].text(
        0.03,
        0.97,
        "Exact-support cache QA\n\n"
        + "\n".join(f"- {x}" for x in support_files[:15]),
        va="top",
        transform=ax[2].transAxes,
        family="monospace",
        fontsize=7.5,
    )

    return "PASS", [
        f"support count source: {countp}"
        if countp else
        "support-count raster not persisted"
    ]


def _phase_linking(ax, output, stack, max_points):
    ndate = len(stack.dates)

    cov_path, covariance = _find_phase_link_covariance(output, ndate)
    messages = []

    if cov_path is not None:
        coherence = _plot_coherence_matrix(
            ax[0],
            covariance,
            "Representative normalized temporal coherence matrix",
        )

        offdiag = coherence[~np.eye(coherence.shape[0], dtype=bool)]
        finite = offdiag[np.isfinite(offdiag)]

        if finite.size:
            messages.append(
                "coherence offdiag p05/p50/p95 = "
                + str(np.percentile(finite, [5, 50, 95]).tolist())
            )

        messages.append(f"covariance source: {cov_path}")
    else:
        ax[0].set_axis_off()
        ax[0].text(
            0.03,
            0.97,
            "No Ndate×Ndate covariance/coherence matrix found.\n"
            "The QA panel will not invent one.",
            va="top",
            transform=ax[0].transAxes,
            family="monospace",
        )
        messages.append("No representative covariance matrix found.")

    tc_candidates = [
        output / "processing" / "temporal_coherence.npy",
        output / "processing" / "phase_linking" / "temporal_coherence.npy",
    ]

    tcp = next((p for p in tc_candidates if p.is_file()), None)

    if tcp is not None:
        tc = _safe_load(tcp)

        if tc.ndim == 2:
            _raster(ax[1], tc, "Temporal coherence spatial distribution")
            _hist(
                ax[2],
                tc,
                "Temporal coherence distribution",
                "Temporal coherence",
                bins=60,
            )
        elif tc.ndim == 1:
            _point_map(ax[1], output, tc, "Temporal coherence", max_points)
            _hist(
                ax[2],
                tc,
                "Temporal coherence distribution",
                "Temporal coherence",
                bins=60,
            )
        else:
            _artifact_inventory_panel(ax[1], output, "phase_linking")
            _log_panel(ax[2], output, "phase_linking")

        messages.append(f"temporal coherence source: {tcp}")
    else:
        _artifact_inventory_panel(ax[1], output, "phase_linking")
        _log_panel(ax[2], output, "phase_linking")
        messages.append("temporal_coherence.npy not persisted")

    return "PASS", messages

def _ds_selection(ax, output):
    p = output / "processing"
    masks = sorted(p.glob("final_ds_*.npy"))
    if not masks:
        _artifact_inventory_panel(ax[0], output, "ds_selection")
        _log_panel(ax[1], output, "ds_selection")
        ax[2].set_axis_off()
        return "REVIEW", ["final_ds_*.npy missing"]

    ds = _safe_load(masks[-1])
    _raster(ax[0], ds, f"Final DS mask — {np.count_nonzero(ds):,}", categorical=True)

    tcp = p / "temporal_coherence.npy"
    if tcp.is_file():
        tc = _safe_load(tcp)
        if tc.shape == ds.shape:
            z = np.asarray(tc, dtype=np.float64)
            z = np.where(np.asarray(ds, dtype=bool), z, np.nan)
            _raster(ax[1], z, "Selected-DS temporal coherence")
            _hist(ax[2], z, "Selected-DS coherence distribution", "Temporal coherence")
            return "PASS", [f"Final DS: {np.count_nonzero(ds):,}"]

    _bar_counts(ax[2], ["selected DS", "other pixels"], [np.count_nonzero(ds), ds.size - np.count_nonzero(ds)], "DS accounting")
    return "PASS", [f"Final DS: {np.count_nonzero(ds):,}"]

def _ps_finalize(ax, output):
    p = output / "processing"
    rawp = p / "ps_mask.npy"
    finalp = p / "final_ps_mask.npy"
    if not rawp.is_file() or not finalp.is_file():
        _artifact_inventory_panel(ax[0], output, "ps_finalize")
        _log_panel(ax[1], output, "ps_finalize")
        ax[2].set_axis_off()
        return "REVIEW", ["PS mask output missing"]

    raw = _safe_load(rawp)
    final = _safe_load(finalp)
    rejected = np.asarray(raw, dtype=bool) & ~np.asarray(final, dtype=bool)

    _raster(ax[0], raw, f"Raw PS — {np.count_nonzero(raw):,}", categorical=True)
    _raster(ax[1], final, f"Final usable PS — {np.count_nonzero(final):,}", categorical=True)
    _bar_counts(ax[2], ["raw PS", "final PS", "rejected"], [np.count_nonzero(raw), np.count_nonzero(final), np.count_nonzero(rejected)], "PS finalize accounting")
    return "PASS", []

def _point_stack(ax, output, max_points):
    d = output / "processing" / "point_phase_stack"
    tp = d / "point_type.npy"
    tcp = d / "temporal_coherence.npy"

    if not tp.is_file():
        _artifact_inventory_panel(ax[0], output, "point_stack")
        _log_panel(ax[1], output, "point_stack")
        ax[2].set_axis_off()
        return "REVIEW", ["point_type.npy missing"]

    typ = _safe_load(tp)
    _point_map(ax[0], output, typ, "Point type (PS/DS)", max_points, categorical=True)
    vals, counts = np.unique(np.asarray(typ), return_counts=True)
    ax[1].bar(vals.astype(str), counts)
    ax[1].set_title("Point-type accounting", fontsize=10)

    if tcp.is_file():
        _point_map(ax[2], output, _safe_load(tcp), "Point temporal coherence", max_points)

    return "PASS", [f"Point stack: {typ.size:,} points"]

def _unwrap(ax, output):
    d = output / "processing" / "single_ifg_robust_solution"
    fs = sorted(d.glob("*_unwrapped_phase_rad.npy")) if d.is_dir() else []
    if not fs:
        _artifact_inventory_panel(ax[0], output, "unwrap")
        _log_panel(ax[1], output, "unwrap")
        ax[2].set_axis_off()
        return "REVIEW", ["No unwrapped IFG files"]

    rms = []
    reg = []
    for f in fs:
        a = _safe_load(f)
        x = _finite_sample(a, 50000)
        rms.append(float(np.sqrt(np.mean(x*x))) if x.size else np.nan)
        rp = f.with_name(f.name.replace("_unwrapped_phase_rad.npy", "_registered_mask.npy"))
        if rp.is_file():
            r = _safe_load(rp)
            reg.append(100.0 * float(np.mean(np.asarray(r, dtype=bool))))
        else:
            reg.append(np.nan)

    x = np.arange(1, len(fs)+1)
    ax[0].plot(x, rms, ".-")
    ax[0].set_title("Unwrapped phase RMS by IFG", fontsize=10)
    ax[0].set_xlabel("IFG index")
    ax[0].set_ylabel("RMS [rad]")

    ax[1].plot(x, reg, ".-")
    ax[1].set_title("Registration ratio by IFG", fontsize=10)
    ax[1].set_xlabel("IFG index")
    ax[1].set_ylabel("Registered [%]")

    _hist(ax[2], np.asarray(rms), "IFG RMS distribution", "RMS [rad]", bins=30)
    return "PASS", [f"Unwrapped IFGs: {len(fs)}"]

def _unwrap_finalize(ax, output, max_points):
    d = output / "processing" / "final_unwrap"
    strictp = d / "strict_unwrap_valid_mask.npy"
    temporalp = d / "temporal_valid_mask.npy"
    gaugep = d / "global_ifg_integer_delta.npy"

    if strictp.is_file():
        _mask_map(ax[0], output, _safe_load(strictp), "STRICT-valid unwrap points", max_points)
    else:
        _artifact_inventory_panel(ax[0], output, "unwrap_finalize")

    if temporalp.is_file():
        _mask_map(ax[1], output, _safe_load(temporalp), "Temporal-valid points", max_points)
    else:
        _log_panel(ax[1], output, "unwrap_finalize")

    if gaugep.is_file():
        g = np.asarray(np.load(gaugep), dtype=np.int64)
        vals, counts = np.unique(g, return_counts=True)
        ax[2].bar(vals.astype(str), counts)
        ax[2].set_title("Global IFG integer gauge", fontsize=10)
        ax[2].set_xlabel("2π integer correction")
    else:
        ax[2].set_axis_off()

    return "PASS", []

def _timeseries(ax, output, max_points):
    d = output / "processing" / "network_inversion"
    rp = d / "l2_network_residual_rms_rad.npy"
    tp = d / "tree_l2_max_abs_diff_rad.npy"

    if rp.is_file():
        _point_map(ax[0], output, _safe_load(rp), "L2 network residual RMS [rad]", max_points)
        _hist(ax[1], _safe_load(rp), "L2 residual RMS distribution", "RMS [rad]")
    else:
        _artifact_inventory_panel(ax[0], output, "timeseries_inversion")
        _log_panel(ax[1], output, "timeseries_inversion")

    if tp.is_file():
        _hist(ax[2], _safe_load(tp), "Tree–L2 max difference", "rad")
    else:
        ax[2].set_axis_off()

    return "PASS", []

def _geometry(ax, output, max_points):
    d = output / "processing" / "point_geometry"
    hp = d / "height_m.npy"
    ip = d / "incidence_rad.npy"
    if hp.is_file():
        _point_map(ax[0], output, _safe_load(hp), "Point height [m]", max_points)
        _hist(ax[1], _safe_load(hp), "Height distribution", "m")
    else:
        _artifact_inventory_panel(ax[0], output, "point_geometry")
        _log_panel(ax[1], output, "point_geometry")
    if ip.is_file():
        inc = np.degrees(np.asarray(_safe_load(ip), dtype=np.float64))
        _point_map(ax[2], output, inc, "Incidence angle [deg]", max_points)
    else:
        ax[2].set_axis_off()
    return "PASS", []

def _residual_ramp(ax, output, max_points):
    d = output / "processing" / "residual_ramp"
    cp = d / "ramp_coefficients_rad_per_km.npy"
    ap = d / "anchor_strict_indices.npy"

    if cp.is_file():
        c = np.asarray(np.load(cp), dtype=np.float64)
        ax[0].plot(c[:,0], ".-", label="ax")
        ax[0].plot(c[:,1], ".-", label="by")
        ax[0].set_title("Degree-1 ramp slope by acquisition", fontsize=10)
        ax[0].set_xlabel("Acquisition index")
        ax[0].set_ylabel("rad/km")
        ax[0].legend(fontsize=8)
        _hist(ax[1], np.hypot(c[:,0], c[:,1]), "Ramp slope magnitude", "rad/km", bins=25)
    else:
        _artifact_inventory_panel(ax[0], output, "residual_ramp")
        _log_panel(ax[1], output, "residual_ramp")

    if ap.is_file():
        ids = np.asarray(np.load(ap), dtype=np.int64)
        xy = _strict_point_xy(output)
        if xy is not None:
            x, y, xl, yl = xy
            all_ids = np.arange(x.size)
            if all_ids.size > max_points:
                all_ids = all_ids[::max(1, all_ids.size // max_points)]
            ax[2].scatter(x[all_ids], y[all_ids], s=0.15, alpha=0.08, linewidths=0)
            ax[2].scatter(x[ids], y[ids], s=8, linewidths=0)
            ax[2].set_xlabel(xl)
            ax[2].set_ylabel(yl)
            ax[2].set_title(f"ADI-ranked ramp anchors — {ids.size}", fontsize=10)
        else:
            ax[2].set_axis_off()
    else:
        ax[2].set_axis_off()

    return "PASS", []

def _reference(ax, output, max_points):
    d = output / "processing" / "referenced_timeseries"
    ratep = d / "preliminary_phase_rate_rad_per_year.npy"
    if ratep.is_file():
        rate = _safe_load(ratep)
        _point_map(ax[0], output, rate, "Referenced preliminary phase rate [rad/yr]", max_points)
        _hist(ax[1], rate, "Referenced rate distribution", "rad/yr")
    else:
        _artifact_inventory_panel(ax[0], output, "reference")
        _log_panel(ax[1], output, "reference")

    rp = d / "auto_reference_point_ids.npy"
    if rp.is_file():
        ids = np.asarray(np.load(rp), dtype=np.int64)
        xy = _strict_point_xy(output)
        if xy is not None:
            x, y, xl, yl = xy
            ax[2].scatter(x, y, s=0.15, alpha=0.08, linewidths=0)
            ax[2].scatter(x[ids], y[ids], s=5, linewidths=0)
            ax[2].set_xlabel(xl)
            ax[2].set_ylabel(yl)
            ax[2].set_title(f"Reference points — {ids.size}", fontsize=10)
        else:
            ax[2].set_axis_off()
    else:
        ax[2].set_axis_off()
    return "PASS", []

def _phase_correction(ax, output, afterp, beforep, label, max_points):
    if not afterp.is_file() or not beforep.is_file():
        _artifact_inventory_panel(ax[0], output, label.lower() if label.lower() != "gacos" else "atmosphere_correction")
        _log_panel(ax[1], output, label.lower() if label.lower() != "gacos" else "atmosphere_correction")
        ax[2].set_axis_off()
        return "REVIEW", [f"Missing before/after arrays for {label}"]

    a = _safe_load(afterp)
    b = _safe_load(beforep)
    if a.ndim != 2 or a.shape != b.shape:
        _artifact_inventory_panel(ax[0], output, label.lower() if label.lower() != "gacos" else "atmosphere_correction")
        _log_panel(ax[1], output, label.lower() if label.lower() != "gacos" else "atmosphere_correction")
        ax[2].set_axis_off()
        return "REVIEW", [f"Unexpected shape for {label}: {a.shape} vs {b.shape}"]

    n = a.shape[0]
    point_rms = np.empty(n, dtype=np.float64)
    for p0 in range(0, n, 100000):
        p1 = min(p0+100000, n)
        d = np.asarray(a[p0:p1], dtype=np.float64) - np.asarray(b[p0:p1], dtype=np.float64)
        point_rms[p0:p1] = np.sqrt(np.mean(d*d, axis=1))
    _point_map(ax[0], output, point_rms, f"{label} correction RMS [rad]", max_points)
    _hist(ax[1], point_rms, f"{label} point-RMS distribution", "rad")
    erms = []
    for e in range(a.shape[1]):
        d = np.asarray(a[:,e], dtype=np.float64) - np.asarray(b[:,e], dtype=np.float64)
        erms.append(float(np.sqrt(np.mean(d*d))))
    ax[2].plot(np.arange(len(erms)), erms, ".-")
    ax[2].set_title(f"{label} correction RMS by acquisition", fontsize=10)
    ax[2].set_xlabel("Acquisition index")
    ax[2].set_ylabel("RMS [rad]")
    return "PASS", []

def _scn(ax, cfg, output, max_points):
    mode = _mode(cfg, "scn")
    if mode in ("disabled", "none", "off", "false", "0"):
        for a in ax[:3]:
            a.set_axis_off()
        ax[0].text(0.04, 0.96, "SCN DISABLED IN CURRENT PROJECT\n\nNo SCN correction is applied.\nStale SCN files are ignored.", va="top", transform=ax[0].transAxes, family="monospace")
        return "DISABLED", ["Current project intentionally keeps SCN off."]

    p = output / "processing" / "scn" / "ph_scn_slave_rad.npy"
    if not p.is_file():
        _artifact_inventory_panel(ax[0], output, "scn")
        _log_panel(ax[1], output, "scn")
        ax[2].set_axis_off()
        return "REVIEW", ["SCN enabled but ph_scn_slave_rad.npy missing"]

    x = _safe_load(p)
    n = x.shape[0]
    rr = np.empty(n, dtype=np.float64)
    for p0 in range(0, n, 100000):
        p1 = min(p0+100000, n)
        q = np.asarray(x[p0:p1], dtype=np.float64)
        rr[p0:p1] = np.sqrt(np.mean(q*q, axis=1))
    _point_map(ax[0], output, rr, "SCN correction temporal RMS [rad]", max_points)
    _hist(ax[1], rr, "SCN point-RMS distribution", "rad")
    er = [float(np.sqrt(np.mean(np.asarray(x[:,e], dtype=np.float64)**2))) for e in range(x.shape[1])]
    ax[2].plot(er, ".-")
    ax[2].set_title("SCN RMS by acquisition", fontsize=10)
    ax[2].set_xlabel("Acquisition index")
    return "PASS", []

def _final_los(ax, output, stack, max_points):
    p = output / "processing" / "final_los" / "los_displacement_toward_satellite_mm.npy"
    if not p.is_file():
        _artifact_inventory_panel(ax[0], output, "final_los")
        _log_panel(ax[1], output, "final_los")
        ax[2].set_axis_off()
        return "REVIEW", ["final LOS displacement missing"]

    x = _safe_load(p)
    last = np.asarray(x[:,-1], dtype=np.float64)
    _point_map(ax[0], output, last, "Final LOS displacement — last acquisition [mm]", max_points)

    dates = _date_objects(stack)
    p05, p50, p95 = [], [], []
    for e in range(x.shape[1]):
        z = _finite_sample(x[:,e], 250000)
        q = np.percentile(z, [5,50,95])
        p05.append(q[0]); p50.append(q[1]); p95.append(q[2])
    ax[1].plot(dates, p50, label="p50")
    ax[1].plot(dates, p05, label="p05")
    ax[1].plot(dates, p95, label="p95")
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[1].xaxis.get_major_locator()))
    ax[1].set_title("Scene LOS displacement percentiles", fontsize=10)
    ax[1].set_ylabel("mm")
    ax[1].legend(fontsize=8)
    _hist(ax[2], last, "Last-acquisition LOS distribution", "mm")
    return "PASS", []

def _products(ax, output, max_points):
    d = output / "products"
    vp = d / "los_velocity_toward_satellite_mm_per_year.npy"
    sep = d / "velocity_slope_standard_error_mm_per_year.npy"
    rp = d / "linear_residual_rms_mm.npy"

    if not vp.is_file():
        _artifact_inventory_panel(ax[0], output, "point_products")
        _log_panel(ax[1], output, "point_products")
        ax[2].set_axis_off()
        return "REVIEW", ["LOS velocity product missing"]

    v = _safe_load(vp)
    _point_map(ax[0], output, v, "LOS velocity [mm/yr]", max_points)
    _hist(ax[1], v, "Velocity distribution", "mm/yr")

    if sep.is_file():
        _point_map(ax[2], output, _safe_load(sep), "Velocity standard error [mm/yr]", max_points)
    elif rp.is_file():
        _point_map(ax[2], output, _safe_load(rp), "Linear residual RMS [mm]", max_points)
    else:
        ax[2].set_axis_off()
    return "PASS", []

def _render_stage(stage, ax, cfg, output, stack, max_points):
    override = render_override(
        stage,
        ax,
        cfg,
        output,
        stack,
        max_points,
        globals(),
    )
    if override is not None:
        return override
    if stage == "ds_statistics":
        return _ds_statistics(ax, output)
    if stage == "phase_cache":
        return _phase_cache(ax, output)
    if stage == "exact_support_cache":
        return _exact_support_cache(ax, output, stack, max_points)
    if stage == "phase_linking":
        return _phase_linking(ax, output, stack, max_points)
    if stage == "ds_selection":
        return _ds_selection(ax, output)
    if stage == "ps_finalize":
        return _ps_finalize(ax, output)
    if stage == "point_stack":
        return _point_stack(ax, output, max_points)

    if stage in ("network_prepare", "network_build", "network_cycle_quality", "network_finalize"):
        info = _network_dashboard(ax, output, stack, stage.replace("_", " ").title())
        return ("PASS" if info["bperp_available"] else "REVIEW", [f"IFGs: {info['edges']}", f"Bperp source: {info['bperp_source']}"])

    if stage == "spatial_anchor_summary":
        _log_panel(ax[0], output, stage)
        _artifact_inventory_panel(ax[1], output, stage)
        ax[2].set_axis_off()
        return "INFO", ["Summary/non-persistent stage."]

    if stage == "unwrap":
        return _unwrap(ax, output)
    if stage == "unwrap_finalize":
        return _unwrap_finalize(ax, output, max_points)
    if stage == "timeseries_inversion":
        return _timeseries(ax, output, max_points)
    if stage == "point_geometry":
        return _geometry(ax, output, max_points)

    if stage == "residual_ramp":
        mode = _mode(cfg, stage)
        if mode in ("disabled", "none", "off", "false", "0"):
            for a in ax[:3]: a.set_axis_off()
            ax[0].text(0.05, 0.95, "Residual ramp correction DISABLED", va="top", transform=ax[0].transAxes)
            return "DISABLED", []
        return _residual_ramp(ax, output, max_points)

    if stage == "reference":
        return _reference(ax, output, max_points)

    if stage == "atmosphere_correction":
        mode = _mode(cfg, stage)
        if mode in ("disabled", "none", "off", "false", "0"):
            for a in ax[:3]: a.set_axis_off()
            ax[0].text(0.05,0.95,"Atmospheric correction DISABLED", va="top", transform=ax[0].transAxes)
            return "DISABLED", []
        return _phase_correction(ax, output, output/"processing/atmosphere_correction/acquisition_phase_corrected_rad.npy", output/"processing/referenced_timeseries/acquisition_phase_referenced_rad.npy", "GACOS", max_points)

    if stage == "scla":
        mode = _mode(cfg, stage)
        if mode in ("disabled", "none", "off", "false", "0"):
            for a in ax[:3]: a.set_axis_off()
            ax[0].text(0.05,0.95,"SCLA correction DISABLED", va="top", transform=ax[0].transAxes)
            return "DISABLED", []
        return _phase_correction(ax, output, output/"processing/scla/acquisition_phase_pre_scn_rad.npy", output/"processing/atmosphere_correction/acquisition_phase_corrected_rad.npy", "SCLA", max_points)

    if stage == "scn":
        return _scn(ax, cfg, output, max_points)

    if stage == "final_los":
        return _final_los(ax, output, stack, max_points)
    if stage == "point_products":
        return _products(ax, output, max_points)

    return _generic_semantic_panel(ax, output, stack, stage, max_points)

def generate(config, stage):
    if stage not in NUM:
        raise ValueError(stage)

    cfg, config_path, paths, stack, _ = open_from_config(config)
    if not bool(cfg_get(cfg, "visualization.enabled", True)):
        return {"stage": stage, "status": "DISABLED"}

    output = Path(paths.output_dir).resolve()
    qdir = output / "qa" / "stages" / f"{NUM[stage]:02d}_{stage}"
    qdir.mkdir(parents=True, exist_ok=True)

    figp = qdir / f"{stage}_qa.png"
    jsonp = qdir / f"{stage}_qa.json"

    dpi = int(cfg_get(cfg, "visualization.dpi", 160))
    max_points = int(cfg_get(cfg, "visualization.max_points", 300000))

    fig, aa = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    ax = aa.ravel()
    fig.suptitle(f"pyPSDS-GAMMA QA — {NUM[stage]:02d} {stage}", fontsize=13)

    try:
        status, messages = _render_stage(stage, ax, cfg, output, stack, max_points)
    except Exception as exc:
        status = "REVIEW"
        messages = [f"Visualization error: {type(exc).__name__}: {exc}"]
        for a in ax[:3]:
            a.clear()
            a.set_axis_off()
        ax[0].text(0.02, 0.98, messages[0], va="top", transform=ax[0].transAxes, family="monospace")

    tail = _log_tail(output, stage, 8)
    summary = list(messages)
    if tail:
        summary += ["", "Stage log tail:"] + tail[-6:]

    _summary_panel(ax[3], stage, status, summary)

    fig.savefig(figp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "stage_index": NUM[stage],
        "stage": stage,
        "status": status,
        "figure": str(figp),
        "diagnostic_only": True,
        "scientific_outputs_modified": False,
        "messages": messages,
    }
    jsonp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[VISUALIZATION-V3] {stage}: {status} -> {figp}", flush=True)
    return payload

def generate_stage_qa(config, stage, force=True):
    return generate(config, stage)

def maybe_generate_stage_qa(config, stage):
    cfg, _, _, _, _ = open_from_config(config)
    if not bool(cfg_get(cfg, "visualization.enabled", True)):
        return None
    fail = bool(cfg_get(cfg, "visualization.fail_on_error", False))
    try:
        return generate(config, stage)
    except Exception as exc:
        if fail:
            raise
        print(f"[VISUALIZATION-V3] {stage}: WARNING: {exc}", flush=True)
        return {"stage": stage, "status": "WARNING", "error": str(exc)}

def _status_index(config, results):
    cfg, _, paths, _, _ = open_from_config(config)
    root = Path(paths.output_dir).resolve() / "qa" / "stages"
    root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 14))
    ax.set_axis_off()
    lines = ["pyPSDS-GAMMA — 39-stage QA status index", ""]
    for r in results:
        lines.append(f"{NUM[r['stage']]:02d}  {r['stage']:<40s}  {r['status']}")
    ax.text(0.03, 0.98, "\n".join(lines), va="top", transform=ax.transAxes, family="monospace", fontsize=9.0)

    out = root / "stage_qa_index.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _contact_pages(config):
    cfg, _, paths, _, _ = open_from_config(config)

    root = Path(paths.output_dir).resolve() / "qa" / "stages"
    pages = root / "pages"

    if pages.exists():
        for p in pages.glob("stage_qa_page_*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    pages.mkdir(parents=True, exist_ok=True)

    def trim_white(img):
        x = np.asarray(img)

        if x.ndim == 2:
            mask = x < 0.985
        else:
            mask = np.min(x[..., :3], axis=2) < 0.985

        rows = np.flatnonzero(np.any(mask, axis=1))
        cols = np.flatnonzero(np.any(mask, axis=0))

        if rows.size == 0 or cols.size == 0:
            return x

        pad = 8
        r0 = max(0, int(rows[0]) - pad)
        r1 = min(x.shape[0], int(rows[-1]) + pad + 1)
        c0 = max(0, int(cols[0]) - pad)
        c1 = min(x.shape[1], int(cols[-1]) + pad + 1)

        return x[r0:r1, c0:c1]

    per_page = 6
    outputs = []

    for page, start in enumerate(range(0, len(STAGES), per_page), 1):
        subset = STAGES[start:start + per_page]
        n = len(subset)

        if n == 1:
            rows, cols = 1, 1
            figsize = (10, 8)
        elif n == 2:
            rows, cols = 1, 2
            figsize = (18, 8)
        elif n == 3:
            rows, cols = 1, 3
            figsize = (24, 8)
        elif n == 4:
            rows, cols = 2, 2
            figsize = (18, 14)
        else:
            rows, cols = 3, 2
            figsize = (18, 21)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        flat = axes.ravel()

        for a in flat:
            a.set_axis_off()

        for k, stage in enumerate(subset):
            p = root / f"{NUM[stage]:02d}_{stage}" / f"{stage}_qa.png"
            a = flat[k]
            a.set_title(f"{NUM[stage]:02d} {stage}", fontsize=11, pad=3)

            if p.is_file():
                try:
                    a.imshow(trim_white(plt.imread(p)), aspect="equal")
                except Exception as exc:
                    a.text(0.5, 0.5, f"preview failed\\n{exc}",
                           ha="center", va="center")
            else:
                a.text(0.5, 0.5, "missing", ha="center", va="center")

        fig.suptitle(
            "pyPSDS-GAMMA QA dashboards — "
            f"stages {NUM[subset[0]]:02d}–{NUM[subset[-1]]:02d}",
            fontsize=15,
            y=0.995,
        )

        fig.subplots_adjust(
            left=0.008, right=0.992, bottom=0.01,
            top=0.945 if rows == 1 else 0.965,
            wspace=0.015, hspace=0.05,
        )

        out = pages / (
            f"stage_qa_page_{page:02d}_"
            f"{NUM[subset[0]]:02d}-{NUM[subset[-1]]:02d}.png"
        )

        fig.savefig(out, dpi=125, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        outputs.append(str(out))

    return outputs


def all_stages(config):
    results = []
    for stage in STAGES:
        results.append(generate(config, stage))

    index = _status_index(config, results)
    pages = _contact_pages(config)

    cfg, _, paths, _, _ = open_from_config(config)
    root = Path(paths.output_dir).resolve() / "qa" / "stages"

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    audit = {
        "stage_count": len(STAGES),
        "status_counts": counts,
        "review_count": counts.get("REVIEW", 0),
        "index": str(index),
        "contact_pages": pages,
        "results": results,
    }

    report = root / "stage_qa_audit.json"
    report.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 96)
    print("39-STAGE SCIENTIFIC QA VISUALIZATION V3")
    print("=" * 96)
    print("status counts :", counts)
    print("index         :", index)
    for p in pages:
        print("page          :", p)
    print("report        :", report)
    if counts.get("REVIEW", 0):
        print(f"QA VISUALIZATION V3: REVIEW REQUIRED — {counts['REVIEW']} stage(s)")
    else:
        print("QA VISUALIZATION V3: PASS")
    return results

def audit_all(config):
    return all_stages(config)

def build_contact_sheet(config):
    return _contact_pages(config)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=STAGES)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        all_stages(args.config)
    else:
        generate(args.config, args.stage)

if __name__ == "__main__":
    main()
