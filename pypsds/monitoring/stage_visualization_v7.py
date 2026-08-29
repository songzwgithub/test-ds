
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.dates as mdates
from pypsds.monitoring.stage_visualization_v6 import render_override as render_v6
from pypsds.monitoring.stage_visualization_v6 import _find_files
from pypsds.monitoring.stage_visualization_v5 import _read_numeric_csv

def _virtual_ifg_v7(ax, output, stack, base):
    files = _find_files(
        output,
        (
            "triangle_closure.csv",
            "*triangle*closure*.csv",
            "*virtual*ifg*.csv",
        ),
    )

    if not files:
        return render_v6(
            "virtual_ifg_quality",
            ax,
            None,
            output,
            stack,
            300000,
            base,
        )

    p = files[0]
    _, numeric, _ = _read_numeric_csv(p)

    closure_cols = []

    for name, arr in numeric.items():
        low = name.lower()

        if (
            (
                "closure" in low
                or "residual" in low
                or "rms" in low
            )
            and not any(
                tok in low
                for tok in ("id", "index")
            )
        ):
            finite = np.asarray(arr, dtype=np.float64)
            finite = finite[np.isfinite(finite)]

            if finite.size >= 2:
                closure_cols.append((name, finite))

    def score(item):
        low = item[0].lower()
        s = 0
        if "mean" in low:
            s += 40
        if "rms" in low:
            s += 35
        if "max" in low:
            s += 30
        if "abs" in low:
            s += 10
        return s

    closure_cols.sort(key=score, reverse=True)

    messages = [f"virtual-IFG table: {p}"]

    for k in range(min(2, len(closure_cols))):
        name, x = closure_cols[k]

        ax[k].hist(
            x,
            bins=min(50, max(8, int(np.sqrt(x.size)))),
        )
        ax[k].set_title(
            f"{name} distribution",
            fontsize=10,
        )
        ax[k].set_xlabel(name)
        ax[k].set_ylabel("count")

        messages.append(
            f"{name}: n={x.size}"
        )

    for k in range(len(closure_cols), 2):
        ax[k].set_axis_off()

    # Use acquisition participation only when a complete triplet is persisted.
    vertex_groups = [
        ("a1", "a2", "a3"),
        ("acq1", "acq2", "acq3"),
        ("i", "j", "k"),
    ]

    lower_map = {
        name.lower(): (name, arr)
        for name, arr in numeric.items()
    }

    vertex_cols = None

    for group in vertex_groups:
        if all(g in lower_map for g in group):
            vertex_cols = [lower_map[g] for g in group]
            break

    if vertex_cols is not None:
        allv = np.concatenate(
            [
                np.asarray(arr, dtype=np.float64)[
                    np.isfinite(np.asarray(arr, dtype=np.float64))
                ].astype(np.int64)
                for _, arr in vertex_cols
            ]
        )

        ndate = len(stack.dates)

        if allv.size:
            if allv.min() >= 1 and allv.max() <= ndate:
                idx = allv - 1
                indexing = "1-based"
            else:
                idx = allv[
                    (allv >= 0)
                    & (allv < ndate)
                ]
                indexing = "0-based/filtered"

            counts = np.bincount(
                idx,
                minlength=ndate,
            )[:ndate]

            dates = base["_date_objects"](stack)

            ax[2].plot(
                dates,
                counts,
                ".-",
            )

            ax[2].xaxis.set_major_locator(
                mdates.AutoDateLocator()
            )
            ax[2].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(
                    ax[2].xaxis.get_major_locator()
                )
            )
            ax[2].set_title(
                "Triangle-closure participation by acquisition",
                fontsize=10,
            )
            ax[2].set_ylabel("triangle count")

            messages += [
                "stage12 third panel: triangle participation",
                f"triangle vertex columns: {[x[0] for x in vertex_cols]}",
                f"vertex indexing: {indexing}",
            ]

            return "PASS", messages

    # No complete vertex triplet: never infer topology from an identifier.
    # Use the row-aligned relation between the two actual closure metrics.
    if len(closure_cols) >= 2:
        name_x = closure_cols[0][0]
        name_y = closure_cols[1][0]

        raw_x = np.asarray(
            numeric[name_x],
            dtype=np.float64,
        )
        raw_y = np.asarray(
            numeric[name_y],
            dtype=np.float64,
        )

        good = np.isfinite(raw_x) & np.isfinite(raw_y)

        ax[2].scatter(
            raw_x[good],
            raw_y[good],
            s=14,
            alpha=0.7,
        )
        ax[2].set_xlabel(name_x)
        ax[2].set_ylabel(name_y)
        ax[2].set_title(
            "Triangle closure: mean–maximum relation",
            fontsize=10,
        )

        if np.count_nonzero(good) >= 3:
            corr = float(
                np.corrcoef(
                    raw_x[good],
                    raw_y[good],
                )[0, 1]
            )
        else:
            corr = float("nan")

        messages += [
            "stage12 third panel: closure mean-vs-max",
            "No complete triangle vertex triplet was persisted; "
            "acquisition participation was intentionally not inferred.",
            f"closure metric correlation: {corr}",
        ]

        return "PASS", messages

    ax[2].set_axis_off()
    ax[2].text(
        0.03,
        0.97,
        "No complete triangle-vertex triplet was persisted.\n"
        "Only grounded closure metrics are shown.",
        va="top",
        transform=ax[2].transAxes,
        family="monospace",
    )

    messages += [
        "stage12 third panel: unavailable",
        "No complete triangle vertex triplet was persisted.",
    ]

    return "PASS", messages


def _gradient_matrix(output, stage, base):
    arrays = base["_candidate_arrays"](output, stage, max_arrays=80)
    candidates = [p for p in arrays if "suspicious_edge_acquisition_gradient" in p.name.lower()]
    if not candidates:
        candidates = _find_files(output, (
            "suspicious_edge_acquisition_gradient.npy",
            "*edge*acquisition*gradient*.npy",
        ))
    if not candidates:
        return None, None
    p = candidates[0]
    g = np.asarray(np.load(p), dtype=np.float64)
    if g.ndim != 2:
        return p, None
    return p, g

def _active_mask(g):
    a = np.abs(g)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(g, dtype=bool), 0.0
    p99 = float(np.percentile(finite, 99))
    tol = max(1e-12, 1e-9 * p99)
    return np.isfinite(g) & (a > tol), tol

def _unwrap_conflict_v7(ax, output, stack, base):
    import matplotlib.pyplot as plt

    p, g = _gradient_matrix(
        output,
        "unwrap_conflict_quality",
        base,
    )

    if p is None or g is None:
        return render_v6(
            "unwrap_conflict_quality",
            ax,
            None,
            output,
            stack,
            300000,
            base,
        )

    im = ax[0].imshow(
        g,
        aspect="auto",
    )
    ax[0].set_title(
        "Suspicious-edge acquisition gradient",
        fontsize=10,
    )
    ax[0].set_xlabel("Acquisition index")
    ax[0].set_ylabel("Suspicious-edge row")
    plt.colorbar(
        im,
        ax=ax[0],
        fraction=0.04,
        pad=0.02,
    )

    a = np.abs(g)

    edge_max = np.nanmax(
        a,
        axis=1,
    )
    finite_max = edge_max[
        np.isfinite(edge_max)
    ]

    ax[1].hist(
        finite_max,
        bins=min(
            40,
            max(
                8,
                int(np.sqrt(finite_max.size)),
            ),
        ),
    )
    ax[1].set_title(
        "Maximum |gradient| per suspicious edge",
        fontsize=10,
    )
    ax[1].set_xlabel("max |gradient| [rad]")
    ax[1].set_ylabel("edge count")

    with np.errstate(invalid="ignore"):
        edge_rms = np.sqrt(
            np.nanmean(
                g * g,
                axis=1,
            )
        )

    finite_rms = edge_rms[
        np.isfinite(edge_rms)
    ]

    ax[2].hist(
        finite_rms,
        bins=min(
            40,
            max(
                8,
                int(np.sqrt(finite_rms.size)),
            ),
        ),
    )
    ax[2].set_title(
        "RMS |gradient| per suspicious edge",
        fontsize=10,
    )
    ax[2].set_xlabel("RMS gradient [rad]")
    ax[2].set_ylabel("edge count")

    max_stats = (
        np.percentile(
            finite_max,
            [50, 95],
        ).tolist()
        + [float(np.max(finite_max))]
        if finite_max.size
        else []
    )

    rms_stats = (
        np.percentile(
            finite_rms,
            [50, 95],
        ).tolist()
        + [float(np.max(finite_rms))]
        if finite_rms.size
        else []
    )

    return "PASS", [
        f"gradient source: {p}",
        f"gradient shape: {list(g.shape)}",
        f"edge max |gradient| p50/p95/max: {max_stats}",
        f"edge RMS |gradient| p50/p95/max: {rms_stats}",
        "stage24 third panel: edge RMS gradient distribution",
    ]


def _unwrap_acquisition_v7(ax, output, stack, base):
    import matplotlib.pyplot as plt

    p, g = _gradient_matrix(
        output,
        "unwrap_acquisition_quality",
        base,
    )

    if p is None or g is None:
        return render_v6(
            "unwrap_acquisition_quality",
            ax,
            None,
            output,
            stack,
            300000,
            base,
        )

    a = np.abs(g)

    im = ax[0].imshow(
        a,
        aspect="auto",
    )
    ax[0].set_title(
        "|Suspicious-edge acquisition gradient|",
        fontsize=10,
    )
    ax[0].set_xlabel("Acquisition index")
    ax[0].set_ylabel("Suspicious-edge row")
    plt.colorbar(
        im,
        ax=ax[0],
        fraction=0.04,
        pad=0.02,
    )

    n_acq = g.shape[1]

    p95 = np.full(
        n_acq,
        np.nan,
        dtype=np.float64,
    )
    vmax = np.full(
        n_acq,
        np.nan,
        dtype=np.float64,
    )
    med = np.full(
        n_acq,
        np.nan,
        dtype=np.float64,
    )
    rms = np.full(
        n_acq,
        np.nan,
        dtype=np.float64,
    )

    for j in range(n_acq):
        x = a[:, j]
        x = x[
            np.isfinite(x)
        ]

        if x.size:
            p95[j] = np.percentile(x, 95)
            vmax[j] = np.max(x)
            med[j] = np.median(x)
            rms[j] = np.sqrt(
                np.mean(x * x)
            )

    dates = base[
        "_date_objects"
    ](
        stack
    )

    n = min(
        len(dates),
        n_acq,
    )

    ax[1].plot(
        dates[:n],
        p95[:n],
        ".-",
        label="p95",
    )
    ax[1].plot(
        dates[:n],
        vmax[:n],
        ".-",
        label="max",
    )
    ax[1].xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )
    ax[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax[1].xaxis.get_major_locator()
        )
    )
    ax[1].set_title(
        "Suspicious-gradient severity by acquisition",
        fontsize=10,
    )
    ax[1].set_ylabel("|gradient| [rad]")
    ax[1].legend(fontsize=8)

    ax[2].plot(
        dates[:n],
        med[:n],
        ".-",
        label="median",
    )
    ax[2].plot(
        dates[:n],
        rms[:n],
        ".-",
        label="RMS",
    )
    ax[2].xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )
    ax[2].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax[2].xaxis.get_major_locator()
        )
    )
    ax[2].set_title(
        "Median / RMS |gradient| by acquisition",
        fontsize=10,
    )
    ax[2].set_ylabel("|gradient| [rad]")
    ax[2].legend(fontsize=8)

    worst = (
        int(np.nanargmax(p95))
        if np.any(np.isfinite(p95))
        else -1
    )

    messages = [
        f"gradient source: {p}",
        f"gradient shape: {list(g.shape)}",
        "stage25 third panel: acquisition median/RMS gradient",
    ]

    if (
        worst >= 0
        and worst < len(stack.dates)
    ):
        messages += [
            f"worst acquisition index by p95: {worst}",
            f"worst acquisition date: {stack.dates[worst]}",
            f"worst p95 |gradient|: {p95[worst]:.6g}",
            f"worst RMS |gradient|: {rms[worst]:.6g}",
        ]

    return "PASS", messages


def render_override(stage, ax, cfg, output, stack, max_points, base):
    if stage == "virtual_ifg_quality":
        return _virtual_ifg_v7(ax, output, stack, base)
    if stage == "unwrap_conflict_quality":
        return _unwrap_conflict_v7(ax, output, stack, base)
    if stage == "unwrap_acquisition_quality":
        return _unwrap_acquisition_v7(ax, output, stack, base)
    return render_v6(stage, ax, cfg, output, stack, max_points, base)
