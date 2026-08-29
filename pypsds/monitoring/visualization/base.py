
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _read_numeric_csv(path: Path):
    text = path.read_text(errors="ignore")
    if not text.strip():
        return [], {}, []

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        delim = dialect.delimiter
    except Exception:
        delim = ","

    rows = list(csv.reader(text.splitlines(), delimiter=delim))
    rows = [r for r in rows if r and any(str(x).strip() for x in r)]

    if not rows:
        return [], {}, []

    headers = [str(x).strip() for x in rows[0]]
    data_rows = rows[1:]
    numeric = {}

    for j, h in enumerate(headers):
        vals = []
        for r in data_rows:
            if j >= len(r):
                vals.append(np.nan)
                continue
            try:
                vals.append(float(str(r[j]).strip()))
            except Exception:
                vals.append(np.nan)

        a = np.asarray(vals, dtype=np.float64)
        if np.count_nonzero(np.isfinite(a)) >= max(2, int(0.25 * max(1, len(a)))):
            numeric[h] = a

    return headers, numeric, data_rows


def _best_numeric_column(numeric, tokens=()):
    ranked = []
    for name, arr in numeric.items():
        low = name.lower()
        score = 0
        for i, tok in enumerate(tokens):
            if tok.lower() in low:
                score += 100 - i
        if np.count_nonzero(np.isfinite(arr)) >= 2:
            score += 10
        if np.nanstd(arr) > 0:
            score += 5
        ranked.append((score, name, arr))

    if not ranked:
        return None, None

    ranked.sort(reverse=True, key=lambda x: x[0])
    _, name, arr = ranked[0]
    return name, arr


def _stage_csvs(output: Path, stage: str, base):
    result = []
    seen = set()

    for d in base["_stage_dirs"](output, stage):
        for p in sorted(d.glob("*.csv")):
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                result.append(p)

    tokens = {stage.lower(), stage.replace("_quality", "").lower()}
    root = output / "processing"

    for p in root.rglob("*.csv"):
        low = str(p.relative_to(root)).lower()
        if any(t and t in low for t in tokens):
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                result.append(p)

    return result


def _plot_csv_hist(ax, path, tokens, title=None):
    _, numeric, _ = _read_numeric_csv(path)
    name, arr = _best_numeric_column(numeric, tokens)

    if name is None:
        ax.set_axis_off()
        ax.text(0.03, 0.97, f"No numeric column in\n{path.name}",
                va="top", transform=ax.transAxes, family="monospace")
        return None

    x = arr[np.isfinite(arr)]
    if x.size:
        ax.hist(x, bins=min(50, max(8, int(np.sqrt(x.size)))))

    ax.set_title(title or f"{path.name}: {name}", fontsize=10)
    ax.set_xlabel(name)
    return name, x


def _plot_csv_series(ax, path, tokens, title=None):
    _, numeric, _ = _read_numeric_csv(path)
    name, arr = _best_numeric_column(numeric, tokens)

    if name is None:
        ax.set_axis_off()
        ax.text(0.03, 0.97, f"No numeric column in\n{path.name}",
                va="top", transform=ax.transAxes, family="monospace")
        return None

    good = np.isfinite(arr)
    ax.plot(np.flatnonzero(good), arr[good], ".-", linewidth=0.8, markersize=3)
    ax.set_title(title or f"{path.name}: {name}", fontsize=10)
    ax.set_xlabel("table row")
    ax.set_ylabel(name)
    return name, arr[good]


def _network_state(output, stack, base):
    dates = base["_date_objects"](stack)
    ndate = len(dates)
    bperp, bpath = base["_find_bperp"](output, ndate)
    itabp = base["_network_itab"](output)
    edges = base["_load_itab"](itabp, ndate) if itabp else []
    return dates, bperp, bpath, itabp, edges


def _plot_network(ax, dates, bperp, edges, title, edge_values=None):
    if bperp is None:
        ax.set_axis_off()
        ax.text(0.03, 0.97, "Bperp unavailable",
                va="top", transform=ax.transAxes)
        return

    if edge_values is None:
        for i, j in edges:
            ax.plot([dates[i], dates[j]], [bperp[i], bperp[j]],
                    linewidth=0.7, alpha=0.45)
    else:
        ev = np.asarray(edge_values, dtype=np.float64)
        finite = ev[np.isfinite(ev)]
        if finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            norm = plt.Normalize(lo, hi)
            cmap = plt.get_cmap("viridis")

            for k, (i, j) in enumerate(edges):
                val = ev[k] if k < ev.size else np.nan
                color = cmap(norm(val)) if np.isfinite(val) else None
                ax.plot([dates[i], dates[j]], [bperp[i], bperp[j]],
                        linewidth=0.9, alpha=0.65, color=color)

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="edge metric")

    ax.scatter(dates, bperp, s=22, zorder=5)
    ax.axhline(0.0, linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    ax.set_ylabel("B⊥ [m]")
    ax.set_xlabel("Acquisition date")
    ax.set_title(title, fontsize=10)


def _network_prepare(ax, output, stack, base):
    dates, bperp, bpath, itabp, edges = _network_state(output, stack, base)
    _plot_network(ax[0], dates, bperp, edges, "Prepared time–B⊥ network")

    if edges:
        dt = np.asarray([abs((dates[j] - dates[i]).days) for i, j in edges], float)
        base["_hist"](ax[1], dt, "Prepared temporal baselines", "|Δt| [days]",
                      bins=min(30, max(8, len(edges)//4)))

        if bperp is not None:
            db = np.asarray([abs(bperp[j] - bperp[i]) for i, j in edges], float)
            base["_hist"](ax[2], db, "Prepared perpendicular baselines", "|ΔB⊥| [m]",
                          bins=min(30, max(8, len(edges)//4)))
        else:
            ax[2].set_axis_off()
    else:
        ax[1].set_axis_off()
        ax[2].set_axis_off()

    return ("PASS" if bperp is not None else "REVIEW",
            [f"IFGs: {len(edges)}", f"Bperp source: {bpath}", f"ITAB: {itabp}"])


def _network_build(ax, output, stack, base):
    dates, bperp, bpath, itabp, edges = _network_state(output, stack, base)
    _plot_network(ax[0], dates, bperp, edges, "Selected interferogram network")

    degree = np.zeros(len(dates), dtype=np.int32)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    ax[1].plot(dates, degree, ".-")
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax[1].xaxis.get_major_locator())
    )
    ax[1].set_title("Acquisition node degree", fontsize=10)
    ax[1].set_ylabel("degree")

    base["_hist"](ax[2], degree, "Node-degree distribution", "degree",
                  bins=max(5, min(20, len(np.unique(degree)) + 2)))

    return ("PASS" if bperp is not None else "REVIEW",
            [f"IFGs: {len(edges)}",
             f"min/median/max degree: {degree.min(initial=0)} / "
             f"{float(np.median(degree)):.1f} / {degree.max(initial=0)}",
             f"Bperp source: {bpath}"])


def _network_cycle(ax, output, stack, base):
    dates, bperp, bpath, itabp, edges = _network_state(output, stack, base)
    csvs = _stage_csvs(output, "network_cycle_quality", base)

    cycle_csv = next(
        (p for p in csvs
         if "edge" in p.name.lower()
         and ("cycle" in p.name.lower() or "coverage" in p.name.lower())),
        csvs[0] if csvs else None
    )

    metric = None
    metric_name = None

    if cycle_csv is not None:
        _, numeric, _ = _read_numeric_csv(cycle_csv)
        metric_name, metric = _best_numeric_column(
            numeric, ("cycle_count", "cycle_coverage", "coverage", "count", "cycles")
        )
        if metric is not None:
            metric = metric[np.isfinite(metric)]

    if metric is not None and metric.size == len(edges):
        _plot_network(ax[0], dates, bperp, edges,
                      "Network edges colored by cycle support", metric)
        base["_hist"](ax[1], metric, "Cycle support per IFG", metric_name,
                      bins=min(30, max(8, int(np.sqrt(metric.size)))))
        ax[2].plot(np.arange(1, metric.size + 1), metric, ".")
        ax[2].set_title("Cycle support by IFG index", fontsize=10)
        ax[2].set_xlabel("IFG index")
        ax[2].set_ylabel(metric_name)
        return "PASS", [f"cycle table: {cycle_csv}", f"edge metric: {metric_name}"]

    _plot_network(ax[0], dates, bperp, edges, "Cycle-quality network")

    if cycle_csv is not None:
        _plot_csv_hist(ax[1], cycle_csv, ("coverage", "cycle", "count"),
                       "Cycle-quality distribution")
        _plot_csv_series(ax[2], cycle_csv, ("coverage", "cycle", "count"),
                         "Cycle-quality metric by row")
    else:
        base["_artifact_inventory_panel"](ax[1], output, "network_cycle_quality")
        base["_log_panel"](ax[2], output, "network_cycle_quality")

    return "INFO", [f"cycle CSV: {cycle_csv}",
                    "No one-to-one edge cycle metric found."]


def _network_finalize(ax, output, stack, base):
    dates, bperp, bpath, itabp, edges = _network_state(output, stack, base)
    _plot_network(ax[0], dates, bperp, edges, "Final connected time–B⊥ network")

    ndate = len(dates)
    degree = np.zeros(ndate, dtype=np.int32)
    adj = [[] for _ in range(ndate)]

    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
        adj[i].append(j)
        adj[j].append(i)

    visited = np.zeros(ndate, dtype=bool)
    sizes = []

    for root in range(ndate):
        if visited[root]:
            continue
        todo = [root]
        visited[root] = True
        size = 0

        while todo:
            u = todo.pop()
            size += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    todo.append(v)

        sizes.append(size)

    ax[1].bar(np.arange(len(sizes)) + 1, sizes)
    ax[1].set_title("Temporal-network component sizes", fontsize=10)
    ax[1].set_xlabel("component")
    ax[1].set_ylabel("acquisitions")

    ax[2].plot(dates, degree, ".-")
    ax[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[2].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax[2].xaxis.get_major_locator())
    )
    ax[2].set_title("Final acquisition degree", fontsize=10)
    ax[2].set_ylabel("degree")

    return ("PASS" if len(sizes) == 1 else "REVIEW",
            [f"IFGs: {len(edges)}", f"components: {len(sizes)}",
             f"component sizes: {sizes}",
             f"isolated acquisitions: {int(np.count_nonzero(degree == 0))}"])


def _csv_stage(ax, output, stage, tokens, base):
    csvs = _stage_csvs(output, stage, base)

    if not csvs:
        base["_artifact_inventory_panel"](ax[0], output, stage)
        base["_log_panel"](ax[1], output, stage)
        ax[2].set_axis_off()
        ax[2].text(0.03, 0.97, "No stage CSV discovered.",
                   va="top", transform=ax[2].transAxes, family="monospace")
        return "INFO", ["No stage CSV discovered."]

    _plot_csv_hist(ax[0], csvs[0], tokens)
    _plot_csv_series(ax[1], csvs[0], tokens)

    if len(csvs) > 1:
        _plot_csv_hist(ax[2], csvs[1], tokens)
    else:
        base["_artifact_inventory_panel"](ax[2], output, stage)

    return "PASS", [f"CSV: {p}" for p in csvs[:4]]


def _suspicious_edge_stage(ax, output, stage, base):
    arrays = base["_candidate_arrays"](output, stage, max_arrays=40)
    gradp = next((p for p in arrays if "gradient" in p.name.lower()), None)
    idp = next((p for p in arrays if "edge_ids" in p.name.lower()
                or "edge_id" in p.name.lower()), None)

    messages = []
    used = 0

    if gradp is not None:
        g = base["_safe_load"](gradp)
        if g.ndim == 2:
            base["_raster"](ax[used], g, gradp.stem)
        else:
            base["_hist"](ax[used], g, gradp.stem)
        used += 1
        messages.append(f"gradient: {gradp}")

    if idp is not None and used < 3:
        ids = np.asarray(np.load(idp)).reshape(-1)
        try:
            ids = ids[np.isfinite(ids.astype(np.float64, copy=False))]
        except Exception:
            pass
        unique_ids = np.unique(ids)

        ax[used].bar(["all IDs", "unique IDs"], [ids.size, unique_ids.size])
        ax[used].set_title("Suspicious-edge ID accounting", fontsize=10)
        used += 1

        messages.append(f"suspicious edge IDs: {ids.size}, unique={unique_ids.size}")

        if used < 3:
            ax[used].scatter(np.arange(unique_ids.size),
                             np.ones(unique_ids.size), s=8)
            ax[used].set_yticks([])
            ax[used].set_xlabel("unique suspicious-edge rank")
            ax[used].set_title("Suspicious-edge occupancy", fontsize=10)
            used += 1

    while used < 3:
        if used == 0:
            base["_artifact_inventory_panel"](ax[used], output, stage)
        elif used == 1:
            base["_log_panel"](ax[used], output, stage)
        else:
            ax[used].set_axis_off()
        used += 1

    return "PASS", messages


def _signature_stage(ax, output, max_points, base):
    stage = "unwrap_signature_quality"
    arrays = base["_candidate_arrays"](output, stage, max_arrays=50)

    groupp = next((p for p in arrays if "signature_group" in p.name.lower()), None)
    hash_paths = [p for p in arrays if "hash" in p.name.lower()]

    if groupp is None:
        for a in ax[:3]:
            a.set_axis_off()

        ax[0].text(
            0.03, 0.97,
            "Signature hashes are identifiers, not physical magnitudes.\n"
            "No hash-value histogram is drawn.",
            va="top", transform=ax[0].transAxes, family="monospace"
        )
        base["_artifact_inventory_panel"](ax[1], output, stage)
        base["_log_panel"](ax[2], output, stage)
        return "INFO", [f"hash files: {[str(p) for p in hash_paths]}",
                        "No signature_group vector found."]

    groups = np.asarray(np.load(groupp)).reshape(-1)

    if base["_point_xy_for_length"](output, groups.size) is not None:
        base["_point_map"](ax[0], output, groups,
                           "Signature-group spatial distribution",
                           max_points, categorical=True)
    else:
        ax[0].set_axis_off()

    vals, counts = np.unique(groups, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = order[:min(30, order.size)]

    ax[1].bar(np.arange(top.size), counts[top])
    ax[1].set_title("Largest signature-group sizes", fontsize=10)
    ax[1].set_xlabel("group rank")
    ax[1].set_ylabel("points")

    base["_hist"](ax[2], counts, "Signature-group size distribution",
                  "points per group",
                  bins=min(50, max(8, int(np.sqrt(counts.size)))))

    return "PASS", [
        f"unique signature groups: {vals.size}",
        f"largest group: {int(counts.max(initial=0))}",
        f"largest group fraction: "
        f"{float(counts.max(initial=0))/max(1, groups.size):.6f}",
    ]


def _phase_linking(ax, output, stack, base):
    ndate = len(stack.dates)
    dates = base["_date_objects"](stack)

    cov_path, covariance = base["_find_phase_link_covariance"](output, ndate)

    if cov_path is None:
        for a in ax[:3]:
            a.set_axis_off()
        ax[0].text(0.03, 0.97, "No Ndate×Ndate covariance matrix found.",
                   va="top", transform=ax[0].transAxes)
        return "REVIEW", ["No phase-link covariance matrix found."]

    coherence = base["_plot_coherence_matrix"](
        ax[0], covariance, "Normalized temporal coherence matrix"
    )

    med = np.full(ndate, np.nan, dtype=np.float64)
    p10 = np.full(ndate, np.nan, dtype=np.float64)

    for i in range(ndate):
        row = np.asarray(coherence[i], dtype=np.float64)
        mask = np.isfinite(row)
        mask[i] = False
        x = row[mask]

        if x.size:
            med[i] = np.median(x)
            p10[i] = np.percentile(x, 10)

    ax[1].plot(dates, med, ".-", label="median off-diagonal")
    ax[1].plot(dates, p10, ".-", label="p10 off-diagonal")
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax[1].xaxis.get_major_locator())
    )
    ax[1].set_ylim(0, 1.02)
    ax[1].set_ylabel("coherence")
    ax[1].set_title("Per-acquisition coherence quality", fontsize=10)
    ax[1].legend(fontsize=8)

    worst = int(np.nanargmin(med))
    ax[1].scatter([dates[worst]], [med[worst]], s=60, zorder=10)
    ax[1].annotate(
        f"worst: {stack.dates[worst]}\nmedian={med[worst]:.3f}",
        xy=(dates[worst], med[worst]),
        xytext=(10, -35),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->"),
    )

    tcp = next(
        (p for p in [
            output / "processing" / "temporal_coherence.npy",
            output / "processing" / "phase_linking" / "temporal_coherence.npy",
        ] if p.is_file()),
        None
    )

    if tcp is not None:
        base["_hist"](ax[2], base["_safe_load"](tcp),
                      "Temporal coherence distribution",
                      "temporal coherence", bins=60)
    else:
        offdiag = coherence[~np.eye(ndate, dtype=bool)]
        base["_hist"](ax[2], offdiag,
                      "Off-diagonal coherence distribution",
                      "normalized coherence", bins=50)

    finite_med = med[np.isfinite(med)]

    return "PASS", [
        f"covariance source: {cov_path}",
        f"worst acquisition index: {worst}",
        f"worst acquisition date: {stack.dates[worst]}",
        f"worst median offdiag coherence: {med[worst]:.6f}",
        f"median acquisition coherence p05/p50/p95: "
        f"{np.percentile(finite_med, [5,50,95]).tolist()}",
    ]


def render_override(stage, ax, cfg, output, stack, max_points, base):
    if stage == "phase_linking":
        return _phase_linking(ax, output, stack, base)
    if stage == "network_prepare":
        return _network_prepare(ax, output, stack, base)
    if stage == "network_build":
        return _network_build(ax, output, stack, base)
    if stage == "network_cycle_quality":
        return _network_cycle(ax, output, stack, base)
    if stage == "network_finalize":
        return _network_finalize(ax, output, stack, base)

    csv_tokens = {
        "spatial_bridge_quality": ("radius", "bridge", "count", "ratio"),
        "spatial_component_quality": ("component", "size", "count", "points"),
        "spatial_anchor_quality": ("anchor", "radius", "count", "distance"),
        "spatial_anchor_summary": ("radius", "component", "count", "points"),
        "spatial_local_graph_quality": ("edge", "degree", "component", "count"),
        "spatial_gradient_quality": ("gradient", "p95", "median", "rms"),
        "unwrap_severity_quality": ("severity", "fraction", "count", "rms"),
        "temporal_candidate_spatial_quality": ("candidate", "valid", "count", "ratio", "support"),
    }

    if stage in csv_tokens:
        return _csv_stage(ax, output, stage, csv_tokens[stage], base)

    if stage in ("unwrap_conflict_quality", "unwrap_acquisition_quality"):
        return _suspicious_edge_stage(ax, output, stage, base)

    if stage == "unwrap_signature_quality":
        return _signature_stage(ax, output, max_points, base)

    return None
