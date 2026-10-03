#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build OGR VRT union layers for the split Ningbo TP-LA GeoPackages.

No data are copied or merged. Each VRT is a tiny XML file that makes QGIS/GDAL
see all GPKG shards of one theme as one logical vector layer.

Usage:
    python build_qgis_union_vrt.py \
        --gis-root /mnt/ningbo_process/ds/output/gis_tp_la
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path


THEMES = {
    "geometry": {
        "dir": "00_geometry",
        "glob": "geometry_*.gpkg",
        "src_layer": "geometry",
        "vrt_name": "geometry_all.vrt",
        "union_name": "geometry_all",
    },
    "velocity": {
        "dir": "01_velocity",
        "glob": "velocity_*.gpkg",
        "src_layer": "velocity",
        "vrt_name": "velocity_all.vrt",
        "union_name": "velocity_all",
    },
    "seasonal": {
        "dir": "02_seasonal",
        "glob": "seasonal_*.gpkg",
        "src_layer": "seasonal",
        "vrt_name": "seasonal_all.vrt",
        "union_name": "seasonal_all",
    },
    "cumulative": {
        "dir": "03_cumulative",
        "glob": "cumulative_*.gpkg",
        "src_layer": "cumulative",
        "vrt_name": "cumulative_all.vrt",
        "union_name": "cumulative_all",
    },
    "quality": {
        "dir": "04_quality",
        "glob": "quality_*.gpkg",
        "src_layer": "quality",
        "vrt_name": "quality_all.vrt",
        "union_name": "quality_all",
    },
}


def build_one(root: Path, spec: dict) -> Path:
    src_dir = root / spec["dir"]
    files = sorted(src_dir.glob(spec["glob"]))
    if not files:
        raise FileNotFoundError(
            f"No files matched {src_dir / spec['glob']}"
        )

    vrt_dir = root / "vrt"
    vrt_dir.mkdir(parents=True, exist_ok=True)
    out = vrt_dir / spec["vrt_name"]

    lines = [
        "<OGRVRTDataSource>",
        f'  <OGRVRTUnionLayer name="{html.escape(spec["union_name"])}">',
        '    <FieldStrategy>FirstLayer</FieldStrategy>',
    ]

    for i, gpkg in enumerate(files, start=1):
        rel = gpkg.relative_to(root)
        rel_from_vrt = Path("..") / rel

        lines.extend([
            f'    <OGRVRTLayer name="src_{i:03d}">',
            f'      <SrcDataSource relativeToVRT="1">{html.escape(rel_from_vrt.as_posix())}</SrcDataSource>',
            f'      <SrcLayer>{html.escape(spec["src_layer"])}</SrcLayer>',
            "    </OGRVRTLayer>",
        ])

    lines.extend([
        "  </OGRVRTUnionLayer>",
        "</OGRVRTDataSource>",
        "",
    ])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{spec['union_name']:<16} : "
        f"{len(files):2d} shards -> {out}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gis-root",
        required=True,
        help="e.g. /mnt/ningbo_process/ds/output/gis_tp_la",
    )
    args = ap.parse_args()

    root = Path(args.gis_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    print("=" * 88)
    print("BUILD QGIS OGR-VRT UNION LAYERS")
    print("=" * 88)

    outputs = []
    for spec in THEMES.values():
        outputs.append(build_one(root, spec))

    print("=" * 88)
    print("PASS")
    print("Open these VRT files in QGIS as Vector layers:")
    for p in outputs:
        print(" ", p)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
