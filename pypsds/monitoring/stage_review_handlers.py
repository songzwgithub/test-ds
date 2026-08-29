from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TARGETS = {
    "ds_statistics",
    "phase_cache",
    "ds_selection",
    "ps_finalize",
    "unwrap_finalize",
}

def _image(ax, arr, title, cbar=False):
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{title}: expected 2-D, got {a.shape}")
    rs = max(1, int(np.ceil(a.shape[0] / 900)))
    cs = max(1, int(np.ceil(a.shape[1] / 900)))
    b = np.asarray(a[::rs, ::cs])
    if b.dtype == np.bool_:
        b = b.astype(np.uint8)
    im = ax.imshow(b, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

def _finite(x, n=250000):
    a = np.asarray(x).reshape(-1)
    if a.size > n:
        a = a[::max(1, a.size // n)]
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    if np.iscomplexobj(a):
        a = np.angle(a)
    a = a.astype(np.float64, copy=False)
    return a[np.isfinite(a)]

def _point_xy(output: Path, n: int):
    pps = output / "processing" / "point_phase_stack"
    rp = pps / "rows.npy"
    cp = pps / "cols.npy"
    if rp.is_file() and cp.is_file():
        r = np.load(rp, mmap_mode="r")
        c = np.load(cp, mmap_mode="r")
        if r.size == n and c.size == n:
            return np.asarray(c), np.asarray(r)
    return None

def _point_mask(ax, output: Path, mask, title, max_points):
    m = np.asarray(mask, dtype=bool).reshape(-1)
    xy = _point_xy(output, m.size)
    if xy is None:
        ax.plot(m.astype(np.uint8), linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Point index")
        return
    x, y = xy
    ids = np.flatnonzero(m)
    if ids.size > max_points:
        ids = ids[::max(1, ids.size // max_points)]
    ax.scatter(x, y, s=0.15, alpha=0.08, linewidths=0)
    if ids.size:
        ax.scatter(x[ids], y[ids], s=1.0, linewidths=0)
    ax.set_title(f"{title} — {np.count_nonzero(m):,}")
    ax.set_xlabel("Radar column")
    ax.set_ylabel("Radar row")

def _bar_counts(ax, labels, values, title):
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)

def _ds_statistics(ax, output):
    d = output / "processing" / "ds_statistics"
    validp = d / "raw_valid.npy"
    psp = d / "ps_mask.npy"
    s2p = d / "rayleigh_scale2.npy"
    adip = d / "amplitude_dispersion_index.npy"

    if validp.is_file():
        valid = np.load(validp, mmap_mode="r")
        _image(ax[0], valid, f"Raw valid mask — {np.count_nonzero(valid):,}")
    else:
        ax[0].text(0.05, 0.95, "raw_valid.npy not found", va="top", transform=ax[0].transAxes)
        ax[0].set_axis_off()

    if psp.is_file():
        ps = np.load(psp, mmap_mode="r")
        _image(ax[1], ps, f"Raw ADI PS mask — {np.count_nonzero(ps):,}")
    else:
        ax[1].text(0.05, 0.95, "ps_mask.npy not found", va="top", transform=ax[1].transAxes)
        ax[1].set_axis_off()

    if adip.is_file():
        x = _finite(np.load(adip, mmap_mode="r"))
        ax[2].hist(x, bins=70)
        ax[2].axvline(0.25, linestyle="--")
        ax[2].set_title("Amplitude dispersion index")
        ax[2].set_xlabel("ADI")
    elif s2p.is_file():
        x = _finite(np.load(s2p, mmap_mode="r"))
        if x.size:
            lo, hi = np.percentile(x, [1, 99])
            x = x[(x >= lo) & (x <= hi)]
        ax[2].hist(x, bins=70)
        ax[2].set_title("Rayleigh scale² (p01–p99)")
    else:
        ax[2].text(0.05, 0.95, "ADI not persisted and rayleigh_scale2.npy not found",
                   va="top", transform=ax[2].transAxes)
        ax[2].set_axis_off()

def _phase_cache(ax, output):
    logp = output / "logs" / "phase_cache.log"
    text = logp.read_text(errors="ignore") if logp.is_file() else ""
    cache = output / "processing" / "cache"
    yxt = cache / "phase_corrected_yxt.npy"

    if "SKIPPED" in text.upper() or not yxt.is_file():
        ax[0].set_axis_off()
        ax[0].text(
            0.03, 0.97,
            "PHASE CACHE: SKIPPED / NOT MATERIALIZED\n\n"
            "Sequential production can stream canonical GAMMA phase directly.\n"
            "This is an expected execution mode, not a failure.",
            va="top", transform=ax[0].transAxes, family="monospace",
        )
    else:
        a = np.load(yxt, mmap_mode="r")
        ax[0].set_axis_off()
        ax[0].text(
            0.03, 0.97,
            f"phase_corrected_yxt.npy\nshape={a.shape}\ndtype={a.dtype}",
            va="top", transform=ax[0].transAxes, family="monospace",
        )

    valid_candidates = [
        cache / "phase_geometry_valid.npy",
        cache / "geometry_valid.npy",
        output / "processing" / "phase_geometry_valid.npy",
        output / "processing" / "geometry_valid.npy",
    ]
    vp = next((p for p in valid_candidates if p.is_file()), None)
    if vp is not None:
        v = np.load(vp, mmap_mode="r")
        _image(ax[1], v, f"Geometry-valid mask — {np.count_nonzero(v):,}")
    else:
        ax[1].text(0.05, 0.95, "No geometry-valid cache mask found",
                   va="top", transform=ax[1].transAxes)
        ax[1].set_axis_off()

    lines = text.splitlines()
    keep = [x for x in lines if any(k in x.lower() for k in
            ("phase source", "temporal mode", "reason", "full yxt", "skip"))]
    ax[2].set_axis_off()
    ax[2].text(
        0.02, 0.98,
        "\n".join(keep[-12:]) if keep else "\n".join(lines[-12:]),
        va="top", transform=ax[2].transAxes, family="monospace", fontsize=8,
    )
    ax[2].set_title("Phase-cache execution policy")

def _ds_selection(ax, output):
    p = output / "processing"
    masks = sorted(p.glob("final_ds_*.npy"))
    if not masks:
        raise FileNotFoundError("final_ds_*.npy not found")
    ds = np.load(masks[-1], mmap_mode="r")
    _image(ax[0], ds, f"Final DS mask — {np.count_nonzero(ds):,}")

    tcp = p / "temporal_coherence.npy"
    if tcp.is_file():
        tc = np.load(tcp, mmap_mode="r")
        if tc.shape == ds.shape:
            z = np.asarray(tc, dtype=np.float64)
            z = np.where(np.asarray(ds, dtype=bool), z, np.nan)
            _image(ax[1], z, "Selected-DS temporal coherence", cbar=True)
            x = _finite(z)
            ax[2].hist(x, bins=60)
            ax[2].set_title("Selected-DS temporal coherence")
            ax[2].set_xlabel("Temporal coherence")
            return

    _bar_counts(
        ax[2],
        ["selected", "not selected"],
        [np.count_nonzero(ds), ds.size - np.count_nonzero(ds)],
        "DS selection count",
    )

def _ps_finalize(ax, output):
    p = output / "processing"
    rawp = p / "ps_mask.npy"
    finalp = p / "final_ps_mask.npy"
    rejectp = p / "step06_ps_finalize" / "rejected_ps_mask.npy"

    if not rawp.is_file() or not finalp.is_file():
        raise FileNotFoundError("ps_mask.npy/final_ps_mask.npy missing")

    raw = np.load(rawp, mmap_mode="r")
    final = np.load(finalp, mmap_mode="r")
    _image(ax[0], raw, f"Raw PS — {np.count_nonzero(raw):,}")
    _image(ax[1], final, f"Final usable PS — {np.count_nonzero(final):,}")

    rejected = np.load(rejectp, mmap_mode="r") if rejectp.is_file() else (
        np.asarray(raw, dtype=bool) & ~np.asarray(final, dtype=bool)
    )
    _bar_counts(
        ax[2],
        ["raw PS", "final PS", "rejected"],
        [np.count_nonzero(raw), np.count_nonzero(final), np.count_nonzero(rejected)],
        "PS finalize accounting",
    )

def _unwrap_finalize(ax, output, max_points):
    d = output / "processing" / "final_unwrap"
    if not d.is_dir():
        raise FileNotFoundError(d)

    arrays = []
    for p in sorted(d.glob("*.npy")):
        try:
            arrays.append((p, np.load(p, mmap_mode="r")))
        except Exception:
            pass

    point_masks = [(p, a) for p, a in arrays if a.ndim == 1 and a.dtype == np.bool_]
    for k, (p, a) in enumerate(point_masks[:2]):
        _point_mask(ax[k], output, a, p.stem, max_points)

    gauge = None
    for p, a in arrays:
        low = p.name.lower()
        if a.ndim == 1 and np.issubdtype(a.dtype, np.integer) and (
            "delta" in low or "gauge" in low or "integer" in low
        ):
            gauge = (p, a)
            break

    if gauge is not None:
        p, a = gauge
        x = np.asarray(a, dtype=np.int64)
        vals, counts = np.unique(x, return_counts=True)
        ax[2].bar(vals.astype(str), counts)
        ax[2].set_title(p.stem)
        ax[2].set_xlabel("Integer correction")
        ax[2].set_ylabel("IFG count")
    elif len(point_masks) >= 3:
        p, a = point_masks[2]
        _point_mask(ax[2], output, a, p.stem, max_points)
    else:
        ax[2].set_axis_off()
        names = [f"{p.name}: {a.shape} {a.dtype}" for p, a in arrays[:15]]
        ax[2].text(0.02, 0.98, "\n".join(names), va="top",
                   transform=ax[2].transAxes, family="monospace", fontsize=7)
        ax[2].set_title("final_unwrap products")

def render_review_stage(stage, ax, output: Path, stack, max_points: int) -> bool:
    if stage not in TARGETS:
        return False
    if stage == "ds_statistics":
        _ds_statistics(ax, output)
    elif stage == "phase_cache":
        _phase_cache(ax, output)
    elif stage == "ds_selection":
        _ds_selection(ax, output)
    elif stage == "ps_finalize":
        _ps_finalize(ax, output)
    elif stage == "unwrap_finalize":
        _unwrap_finalize(ax, output, max_points)
    return True
