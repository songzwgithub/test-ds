
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pypsds.monitoring.visualization.base import render_override as render_base
from pypsds.monitoring.visualization.base import (
    _read_numeric_csv,
    _best_numeric_column,
    _network_state,
    _plot_network,
)


def _find_files(output: Path, patterns):
    root = output / "processing"
    out = []
    seen = set()

    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                out.append(p)

    return out


def _plot_numeric_columns(axs, csv_path, preferred=()):
    headers, numeric, _ = _read_numeric_csv(csv_path)

    if not numeric:
        for ax in axs:
            ax.set_axis_off()
        axs[0].text(
            0.03, 0.97,
            f"No numeric columns:\n{csv_path}",
            va="top",
            transform=axs[0].transAxes,
            family="monospace",
        )
        return []

    scored = []

    for name, arr in numeric.items():
        low = name.lower()
        score = 0

        for i, tok in enumerate(preferred):
            if tok.lower() in low:
                score += 100 - i

        finite = arr[np.isfinite(arr)]

        if finite.size >= 2:
            score += 10

        if finite.size and np.nanstd(finite) > 0:
            score += 10

        # Avoid identifier-only columns where possible.
        if any(tok in low for tok in ("id", "index", "hash")):
            score -= 30

        scored.append((score, name, arr))

    scored.sort(reverse=True, key=lambda x: x[0])
    selected = scored[:3]

    messages = []

    for ax, (_, name, arr) in zip(axs, selected):
        good = np.isfinite(arr)
        x = arr[good]

        if x.size == 0:
            ax.set_axis_off()
            continue

        uniq = np.unique(x)

        if uniq.size <= 20 and np.allclose(uniq, np.round(uniq)):
            vals, counts = np.unique(x, return_counts=True)
            ax.bar(vals.astype(str), counts)
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.set_title(f"{name} distribution", fontsize=10)
        else:
            ax.hist(x, bins=min(50, max(8, int(np.sqrt(x.size)))))
            ax.set_xlabel(name)
            ax.set_ylabel("count")
            ax.set_title(f"{name} distribution", fontsize=10)

        messages.append(
            f"{csv_path.name}: {name}, n={x.size}"
        )

    for ax in axs[len(selected):]:
        ax.set_axis_off()

    return messages


def _network_cycle_v6(ax, output, stack, base):
    dates, bperp, bpath, itabp, edges = _network_state(
        output,
        stack,
        base,
    )

    files = _find_files(
        output,
        (
            "edge_cycle_coverage.csv",
            "*cycle*coverage*.csv",
            "*cycle*quality*.csv",
        ),
    )

    csv_path = files[0] if files else None

    if csv_path is None:
        return render_base(
            "network_cycle_quality",
            ax,
            None,
            output,
            stack,
            300000,
            base,
        )

    headers, numeric, _ = _read_numeric_csv(
        csv_path
    )

    metric_name, metric = _best_numeric_column(
        numeric,
        (
            "cycle_count",
            "coverage_count",
            "cycle_coverage",
            "coverage",
            "cycles",
            "count",
        ),
    )

    if metric is not None:
        metric = np.asarray(
            metric,
            dtype=np.float64,
        )

        finite_metric = metric[
            np.isfinite(metric)
        ]
    else:
        finite_metric = np.asarray([])

    if (
        metric is not None
        and metric.size == len(edges)
        and np.all(np.isfinite(metric))
    ):
        _plot_network(
            ax[0],
            dates,
            bperp,
            edges,
            "IFG network colored by cycle support",
            metric,
        )

        ax[1].hist(
            finite_metric,
            bins=min(
                30,
                max(
                    8,
                    int(np.sqrt(finite_metric.size)),
                ),
            ),
        )
        ax[1].set_title(
            "Cycle support per IFG",
            fontsize=10,
        )
        ax[1].set_xlabel(
            metric_name,
        )

        ax[2].plot(
            np.arange(1, metric.size + 1),
            metric,
            ".",
        )
        ax[2].set_title(
            "Cycle support by IFG",
            fontsize=10,
        )
        ax[2].set_xlabel(
            "IFG index",
        )
        ax[2].set_ylabel(
            metric_name,
        )

        return "PASS", [
            f"cycle table: {csv_path}",
            f"edge metric: {metric_name}",
            f"edge metric rows: {metric.size}",
            f"IFGs: {len(edges)}",
        ]

    # Table exists but not a one-row-per-IFG table.
    _plot_network(
        ax[0],
        dates,
        bperp,
        edges,
        "Cycle-quality network",
    )

    msgs = _plot_numeric_columns(
        ax[1:3],
        csv_path,
        (
            "cycle",
            "coverage",
            "count",
        ),
    )

    return "PASS", [
        f"cycle table: {csv_path}",
        f"IFGs: {len(edges)}",
    ] + msgs


def _virtual_ifg_v6(ax, output, base):
    files = _find_files(
        output,
        (
            "triangle_closure.csv",
            "*triangle*closure*.csv",
            "*virtual*ifg*.csv",
            "*closure*.csv",
        ),
    )

    if not files:
        base["_artifact_inventory_panel"](
            ax[0],
            output,
            "virtual_ifg_quality",
        )
        base["_log_panel"](
            ax[1],
            output,
            "virtual_ifg_quality",
        )
        ax[2].set_axis_off()

        return "INFO", [
            "No virtual-IFG closure table found."
        ]

    p = files[0]

    messages = _plot_numeric_columns(
        ax[:3],
        p,
        (
            "closure",
            "residual",
            "rms",
            "error",
            "abs",
        ),
    )

    return "PASS", [
        f"virtual-IFG QA table: {p}"
    ] + messages


def _spatial_bridge_v6(ax, output, base):
    files = _find_files(
        output,
        (
            "spatial_radius_quality.csv",
            "*radius*quality*.csv",
            "*bridge*quality*.csv",
        ),
    )

    if not files:
        return None

    p = files[0]
    messages = _plot_numeric_columns(
        ax[:3],
        p,
        (
            "radius",
            "ratio",
            "fraction",
            "count",
            "milestone",
        ),
    )

    return "PASS", [
        f"bridge/radius table: {p}"
    ] + messages


def _spatial_component_v6(ax, output, max_points, base):
    labels = _find_files(
        output,
        (
            "component_label_r4.npy",
            "component_label*.npy",
        ),
    )

    csvs = _find_files(
        output,
        (
            "residual_components_to_main.csv",
            "*components*main*.csv",
            "*component*.csv",
        ),
    )

    messages = []

    if labels:
        p = labels[0]
        lab = np.asarray(
            np.load(p),
        ).reshape(-1)

        if base["_point_xy_for_length"](
            output,
            lab.size,
        ) is not None:
            base["_point_map"](
                ax[0],
                output,
                lab,
                "Residual component labels",
                max_points,
                categorical=True,
            )
        else:
            vals, counts = np.unique(
                lab,
                return_counts=True,
            )
            order = np.argsort(counts)[::-1]
            top = order[:min(30, order.size)]

            ax[0].bar(
                np.arange(top.size),
                counts[top],
            )
            ax[0].set_title(
                "Largest component sizes",
                fontsize=10,
            )
            ax[0].set_xlabel(
                "component rank",
            )
            ax[0].set_ylabel(
                "points",
            )

        vals, counts = np.unique(
            lab,
            return_counts=True,
        )

        counts = counts[
            vals != 0
        ] if np.any(vals == 0) else counts

        if counts.size:
            ax[1].hist(
                counts,
                bins=min(
                    50,
                    max(
                        8,
                        int(np.sqrt(counts.size)),
                    ),
                ),
            )
            ax[1].set_title(
                "Component-size distribution",
                fontsize=10,
            )
            ax[1].set_xlabel(
                "points per component",
            )

        messages += [
            f"component labels: {p}",
            f"unique labels: {vals.size}",
        ]
    else:
        base["_artifact_inventory_panel"](
            ax[0],
            output,
            "spatial_component_quality",
        )
        ax[1].set_axis_off()

    if csvs:
        m = _plot_numeric_columns(
            ax[2:3],
            csvs[0],
            (
                "size",
                "distance",
                "bridge",
                "component",
            ),
        )
        messages += [
            f"component table: {csvs[0]}"
        ] + m
    else:
        base["_log_panel"](
            ax[2],
            output,
            "spatial_component_quality",
        )

    return "PASS", messages


def _spatial_anchor_v6(ax, output, base):
    files = _find_files(
        output,
        (
            "residual_two_anchor_quality.csv",
            "*two_anchor*quality*.csv",
            "*anchor*quality*.csv",
        ),
    )

    if not files:
        return None

    p = files[0]

    messages = _plot_numeric_columns(
        ax[:3],
        p,
        (
            "radius",
            "anchor",
            "count",
            "ratio",
            "fraction",
        ),
    )

    return "PASS", [
        f"two-anchor table: {p}"
    ] + messages


def _anchor_summary_from_log_v6(ax, output):
    logp = (
        output
        / "logs"
        / "spatial_anchor_summary.log"
    )

    if not logp.is_file():
        return None

    text = logp.read_text(
        errors="ignore"
    )

    rows = []

    for line in text.splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$",
            line,
        )

        if m:
            rows.append(
                tuple(
                    int(x)
                    for x in m.groups()
                )
            )

    if not rows:
        return None

    arr = np.asarray(
        rows,
        dtype=np.int64,
    )

    labels = arr[:, 0]
    sizes = arr[:, 1]
    radius = arr[:, 2]

    ax[0].scatter(
        sizes,
        radius,
        s=30,
    )
    ax[0].set_title(
        "Residual components: size vs two-anchor radius",
        fontsize=10,
    )
    ax[0].set_xlabel(
        "component size",
    )
    ax[0].set_ylabel(
        "two-anchor radius",
    )

    vals, counts = np.unique(
        radius,
        return_counts=True,
    )

    ax[1].bar(
        vals.astype(str),
        counts,
    )
    ax[1].set_title(
        "Components by required radius",
        fontsize=10,
    )
    ax[1].set_xlabel(
        "two-anchor radius",
    )
    ax[1].set_ylabel(
        "component count",
    )

    ax[2].bar(
        np.arange(labels.size),
        sizes,
    )
    ax[2].set_xticks(
        np.arange(labels.size),
        labels.astype(str),
        rotation=45,
        ha="right",
    )
    ax[2].set_title(
        "Residual component sizes",
        fontsize=10,
    )
    ax[2].set_xlabel(
        "component label",
    )
    ax[2].set_ylabel(
        "points",
    )

    return "PASS", [
        f"parsed component rows: {len(rows)}",
        f"source log: {logp}",
    ]


def _local_graph_v6(ax, output, base):
    files = _find_files(
        output,
        (
            "sparse_local_graph_sweep.csv",
            "*local*graph*sweep*.csv",
            "*sparse*graph*.csv",
        ),
    )

    if not files:
        return None

    p = files[0]

    messages = _plot_numeric_columns(
        ax[:3],
        p,
        (
            "largest",
            "component",
            "edge",
            "degree",
            "candidate",
            "k",
        ),
    )

    return "PASS", [
        f"local graph sweep: {p}"
    ] + messages


def _suspicious_edges_v6(ax, output, stack, stage, base):
    arrays = base["_candidate_arrays"](
        output,
        stage,
        max_arrays=50,
    )

    gradp = next(
        (
            p for p in arrays
            if "gradient" in p.name.lower()
        ),
        None,
    )

    idp = next(
        (
            p for p in arrays
            if (
                "edge_ids" in p.name.lower()
                or "edge_id" in p.name.lower()
            )
        ),
        None,
    )

    messages = []

    if gradp is not None:
        g = base["_safe_load"](
            gradp
        )

        if g.ndim == 2:
            base["_raster"](
                ax[0],
                g,
                gradp.stem,
            )
        else:
            base["_hist"](
                ax[0],
                g,
                gradp.stem,
            )

        messages.append(
            f"gradient: {gradp}"
        )
    else:
        ax[0].set_axis_off()

    dates, bperp, bpath, itabp, edges = _network_state(
        output,
        stack,
        base,
    )

    if idp is None:
        base["_artifact_inventory_panel"](
            ax[1],
            output,
            stage,
        )
        base["_log_panel"](
            ax[2],
            output,
            stage,
        )
        return "INFO", messages + [
            "No suspicious_edge_ids array found."
        ]

    ids = np.asarray(
        np.load(idp),
    ).reshape(-1)

    try:
        ids = ids[
            np.isfinite(
                ids.astype(
                    np.float64,
                    copy=False,
                )
            )
        ].astype(
            np.int64,
        )
    except Exception:
        ids = ids.astype(
            np.int64,
        )

    ids = np.unique(
        ids,
    )

    if (
        ids.size
        and ids.min() >= 1
        and ids.max() <= len(edges)
    ):
        edge_idx = ids - 1
        indexing = "1-based"
    else:
        edge_idx = ids[
            (ids >= 0)
            & (ids < len(edges))
        ]
        indexing = "0-based/filtered"

    # Full network faint + suspicious edges thicker.
    if bperp is not None:
        for i, j in edges:
            ax[1].plot(
                [dates[i], dates[j]],
                [bperp[i], bperp[j]],
                linewidth=0.45,
                alpha=0.18,
            )

        for k in edge_idx:
            i, j = edges[int(k)]
            ax[1].plot(
                [dates[i], dates[j]],
                [bperp[i], bperp[j]],
                linewidth=2.0,
                alpha=0.9,
            )

        ax[1].scatter(
            dates,
            bperp,
            s=16,
            zorder=5,
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
            "Suspicious IFGs in time–B⊥ network",
            fontsize=10,
        )
        ax[1].set_ylabel(
            "B⊥ [m]",
        )
    else:
        ax[1].set_axis_off()

    counts = np.zeros(
        len(dates),
        dtype=np.int32,
    )

    for k in edge_idx:
        i, j = edges[int(k)]
        counts[i] += 1
        counts[j] += 1

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
        "Suspicious-edge incidence by acquisition",
        fontsize=10,
    )
    ax[2].set_ylabel(
        "incident suspicious IFGs",
    )

    messages += [
        f"suspicious edge IDs: {ids.size}",
        f"valid network edges: {edge_idx.size}",
        f"edge-ID indexing: {indexing}",
        f"ITAB: {itabp}",
    ]

    return "PASS", messages


def _signature_v6(ax, output, base):
    stage = "unwrap_signature_quality"

    arrays = base["_candidate_arrays"](
        output,
        stage,
        max_arrays=50,
    )

    groupp = next(
        (
            p for p in arrays
            if "signature_group" in p.name.lower()
        ),
        None,
    )

    if groupp is None:
        return None

    groups = np.asarray(
        np.load(groupp),
    ).reshape(-1)

    vals, counts = np.unique(
        groups,
        return_counts=True,
    )

    order = np.argsort(
        counts,
    )[::-1]

    top = order[
        :min(
            30,
            order.size,
        )
    ]

    ax[0].bar(
        np.arange(top.size),
        counts[top],
    )
    ax[0].set_yscale(
        "log",
    )
    ax[0].set_title(
        "Largest signature-group sizes (log scale)",
        fontsize=10,
    )
    ax[0].set_xlabel(
        "group rank",
    )
    ax[0].set_ylabel(
        "points",
    )

    positive = counts[
        counts > 0
    ]

    log_size = np.log10(
        positive.astype(
            np.float64,
        )
    )

    ax[1].hist(
        log_size,
        bins=min(
            50,
            max(
                8,
                int(np.sqrt(log_size.size)),
            ),
        ),
    )
    ax[1].set_title(
        "Signature-group size distribution",
        fontsize=10,
    )
    ax[1].set_xlabel(
        "log10(points per group)",
    )

    s = np.sort(
        positive.astype(
            np.float64,
        )
    )

    ccdf = (
        np.arange(
            s.size,
            0,
            -1,
        )
        / s.size
    )

    ax[2].loglog(
        s,
        ccdf,
        ".",
    )
    ax[2].set_title(
        "Signature-group size CCDF",
        fontsize=10,
    )
    ax[2].set_xlabel(
        "group size [points]",
    )
    ax[2].set_ylabel(
        "fraction of groups ≥ size",
    )

    largest = int(
        counts.max(
            initial=0,
        )
    )

    return "PASS", [
        f"unique signature groups: {vals.size}",
        f"largest group: {largest}",
        f"largest group fraction: {largest/max(1,groups.size):.6f}",
        f"group source: {groupp}",
    ]


def render_override(stage, ax, cfg, output, stack, max_points, base):
    if stage == "network_cycle_quality":
        return _network_cycle_v6(
            ax,
            output,
            stack,
            base,
        )

    if stage == "virtual_ifg_quality":
        return _virtual_ifg_v6(
            ax,
            output,
            base,
        )

    if stage == "spatial_bridge_quality":
        out = _spatial_bridge_v6(
            ax,
            output,
            base,
        )
        if out is not None:
            return out

    if stage == "spatial_component_quality":
        return _spatial_component_v6(
            ax,
            output,
            max_points,
            base,
        )

    if stage == "spatial_anchor_quality":
        out = _spatial_anchor_v6(
            ax,
            output,
            base,
        )
        if out is not None:
            return out

    if stage == "spatial_anchor_summary":
        out = _anchor_summary_from_log_v6(
            ax,
            output,
        )
        if out is not None:
            return out

    if stage == "spatial_local_graph_quality":
        out = _local_graph_v6(
            ax,
            output,
            base,
        )
        if out is not None:
            return out

    if stage in (
        "unwrap_conflict_quality",
        "unwrap_acquisition_quality",
    ):
        return _suspicious_edges_v6(
            ax,
            output,
            stack,
            stage,
            base,
        )

    if stage == "unwrap_signature_quality":
        out = _signature_v6(
            ax,
            output,
            base,
        )
        if out is not None:
            return out

    return render_base(
        stage,
        ax,
        cfg,
        output,
        stack,
        max_points,
        base,
    )
