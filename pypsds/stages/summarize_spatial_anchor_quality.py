#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pypsds.context import open_from_config


def main():
    ap = argparse.ArgumentParser(
        description="Summarize hierarchical component-forest bridge coverage."
    )
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg, config_path, paths, stack, _ = open_from_config(args.config)
    outroot = Path(paths.output_dir) / "processing"
    qdir = outroot / "spatial_graph_two_anchor_quality"
    csv_path = qdir / "residual_two_anchor_quality.csv"
    json_path = qdir / "residual_two_anchor_quality.json"

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "component": int(r["component_label"]),
                "parent": int(r["parent_component_label"]),
                "forest_root": int(r.get("forest_root_component", -1)),
                "size": int(r["component_size"]),
                "depth": int(r.get("component_depth", 0)),
                "r1": int(r["anchor1_radius"]),
                "r2": int(r["anchor2_radius"]),
                "duplicate": int(r.get("duplicate_anchor_pair", 0) or 0),
            })

    sizes = np.asarray([x["size"] for x in rows], dtype=np.int64)
    maxr = np.asarray([max(x["r1"], x["r2"]) for x in rows], dtype=np.int32)
    depth = np.asarray([x["depth"] for x in rows], dtype=np.int32)
    duplicate = np.asarray([x["duplicate"] for x in rows], dtype=np.uint8)

    print("=" * 96)
    print("Hierarchical component bridge summary")
    print("=" * 96)
    print(f"config                    : {config_path}")
    print(f"components                : {int(summary['components']):,}")
    print(f"forest edges              : {int(summary['forest_edges']):,}")
    print(f"forest count              : {int(summary['forest_count']):,}")
    gg = summary["global_gauge_forest"]
    print(
        f"global-gauge forest       : {int(gg['components']):,} components, "
        f"{int(gg['points']):,} points ({100*float(gg['point_fraction']):.3f}%)"
    )
    print(f"maximum depth             : {int(summary['maximum_depth'])}")
    print(
        f"duplicate anchor edges    : {int(np.count_nonzero(duplicate)):,}/"
        f"{len(rows):,}"
    )

    if maxr.size:
        q = np.quantile(maxr, [0, .25, .5, .75, .9, .95, .99, 1])
        print("bridge R min/p25/p50/p75/p90/p95/p99/max:")
        print("  " + " / ".join(f"{x:.1f}" for x in q))
        print()
        print("Cumulative selected bridge coverage")
        print(" Radius | edges                    | child points")
        print("-" * 80)
        for R in (5, 6, 7, 8, 10, 12, 15, 20, 25, 30):
            m = maxr <= R
            ne = int(np.count_nonzero(m))
            npnt = int(sizes[m].sum())
            print(
                f" R<={R:2d} | {ne:7,d}/{len(rows):7,d} "
                f"({100*ne/max(1,len(rows)):7.3f}%) | {npnt:12,d}"
            )

    detached = int(summary["forest_count"]) - 1
    print()
    print(f"detached forests          : {detached:,}")
    if detached:
        print(
            "NOTE: detached forests are intentionally NOT given an unsupported "
            "global 2pi gauge. Their local solutions remain available for QA."
        )

    print()
    print("STEP spatial_anchor_summary STATUS: PASS")


if __name__ == "__main__":
    main()
