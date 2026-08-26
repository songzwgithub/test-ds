from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from pypsds.config import load_config
from pypsds.project import resolve_project_paths
from pypsds.geometry.inputs import resolve_geometry_inputs


def cfg_get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_context(config):
    cfg, config_path = load_config(Path(config))
    paths = resolve_project_paths(cfg, config_path)
    proc = Path(paths.output_dir) / "processing"
    proc.mkdir(parents=True, exist_ok=True)

    nman = proc / "network" / "network_manifest.json"
    if not nman.is_file():
        raise FileNotFoundError(nman)

    payload = json.loads(nman.read_text(encoding="utf-8"))
    dates = [str(x) for x in payload.get("dates", [])]
    if not dates:
        raise RuntimeError("network_manifest.json has no acquisition dates")

    itab = proc / "network" / "network.itab"
    if not itab.is_file():
        raise FileNotFoundError(itab)

    edges = []
    for raw in itab.read_text(encoding="utf-8").splitlines():
        f = raw.split()
        if len(f) < 2:
            continue
        i = int(f[0]) - 1
        j = int(f[1]) - 1
        if i < 0 or j < 0 or i >= len(dates) or j >= len(dates) or i == j:
            raise RuntimeError(f"invalid network edge: {raw}")
        edges.append((i, j))

    if not edges:
        raise RuntimeError("network.itab is empty")

    ref_file = proc / "referenced_timeseries" / "reference_strict_indices.npy"
    if not ref_file.is_file():
        raise FileNotFoundError(ref_file)
    ref_idx = np.load(ref_file, allow_pickle=False)
    if ref_idx.ndim != 1 or ref_idx.size == 0:
        raise RuntimeError("reference_strict_indices.npy must be non-empty 1-D")

    geometry = resolve_geometry_inputs(cfg, paths)
    master_date = str(geometry.reference_date)
    rslc_par = Path(geometry.reference_rslc_par)

    data_root = Path(paths.data_dir) if paths.data_dir is not None else Path(paths.work_dir)
    gacos_dir = Path(paths.gacos_dir) if paths.gacos_dir is not None else None
    products_dir = Path(paths.products_dir)
    products_dir.mkdir(parents=True, exist_ok=True)

    return {
        "cfg": cfg,
        "config_path": Path(config_path),
        "paths": paths,
        "proc": proc,
        "dates": dates,
        "edges": edges,
        "nimage": len(dates),
        "nifg": len(edges),
        "nslave": len(dates) - 1,
        "ref_file": ref_file,
        "ref_idx": np.asarray(ref_idx, dtype=np.int32),
        "nref": int(ref_idx.size),
        "master_date": master_date,
        "temporal_reference_date": dates[0],
        "rslc_par": rslc_par,
        "data_root": data_root.resolve(),
        "gacos_dir": None if gacos_dir is None else gacos_dir.resolve(),
        "products_dir": products_dir.resolve(),
        "point_geometry": proc / "point_geometry",
    }


def public_env(ctx):
    env = dict(os.environ)
    env.update({
        "PYPSDS_PUBLIC_PROJECT": str(Path(ctx["paths"].work_dir).resolve()),
        "PYPSDS_PUBLIC_DATA_ROOT": str(ctx["data_root"]),
        "PYPSDS_PUBLIC_OUTPUT": str(Path(ctx["paths"].output_dir).resolve()),
        "PYPSDS_PUBLIC_PROC": str(ctx["proc"].resolve()),
        "PYPSDS_PUBLIC_PRODUCTS": str(ctx["products_dir"]),
        "PYPSDS_PUBLIC_REF_FILE": str(ctx["ref_file"].resolve()),
        "PYPSDS_PUBLIC_RSLC_PAR": str(ctx["rslc_par"].resolve()),
        "PYPSDS_PUBLIC_NDATE": str(ctx["nimage"]),
        "PYPSDS_PUBLIC_NIFG": str(ctx["nifg"]),
        "PYPSDS_PUBLIC_NSLAVE": str(ctx["nslave"]),
        "PYPSDS_PUBLIC_NREF": str(ctx["nref"]),
        "PYPSDS_PUBLIC_REF_DATE": str(ctx["temporal_reference_date"]),
        "PYPSDS_PUBLIC_MASTER_DATE": str(ctx["master_date"]),
        "PYPSDS_PUBLIC_POINT_GEOMETRY": str(ctx["point_geometry"].resolve()),
    })
    if ctx["gacos_dir"] is not None:
        env["PYPSDS_PUBLIC_GACOS"] = str(ctx["gacos_dir"])
    return env


def runtime_path(name):
    return Path(__file__).resolve().parents[1] / "runtime_v11" / name


def run_runtime(name, ctx, extra_env=None):
    env = public_env(ctx)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    path = runtime_path(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    subprocess.run([sys.executable, str(path)], env=env, check=True)


def atomic_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("." + dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ensure_geometry_compat(ctx):
    src = ctx["point_geometry"]
    dst = ctx["proc"] / "atmosphere_correction" / "_geometry_compat"
    dst.mkdir(parents=True, exist_ok=True)

    link_map = {
        "longitude_deg.npy": "longitude_deg.npy",
        "latitude_deg.npy": "latitude_deg.npy",
        "radar_row.npy": "radar_row.npy",
        "radar_col.npy": "radar_col.npy",
        "height_m.npy": "height_m.npy",
        "strict_points.plist": "strict_points.plist",
        "incidence_gamma_compatible_fast_rad.npy": "incidence_rad.npy",
        "incidence_ellipsoid_fast_rad.npy": "incidence_rad.npy",
        "incidence_ellipsoid_zd_fast_rad.npy": "incidence_rad.npy",
        "incidence_angle_stamps_deg.npy": "incidence_rad.npy",
    }

    for dst_name, src_name in link_map.items():
        s = src / src_name
        if not s.is_file():
            continue
        d = dst / dst_name
        if d.exists() or d.is_symlink():
            d.unlink()
        try:
            d.symlink_to(s.resolve())
        except OSError:
            shutil.copyfile(s, d)

    strict_ids = ctx["proc"] / "network_inversion" / "strict_point_ids.npy"
    if strict_ids.is_file():
        d = dst / "strict_point_ids.npy"
        if d.exists() or d.is_symlink():
            d.unlink()
        try:
            d.symlink_to(strict_ids.resolve())
        except OSError:
            shutil.copyfile(strict_ids, d)

    valid = dst / "valid_gacos_geometry_mask.npy"
    if not valid.is_file():
        lon = np.load(src / "longitude_deg.npy", mmap_mode="r")
        np.save(valid, np.ones(lon.shape[0], dtype=bool))

    return dst


def ensure_gacos_cache(ctx):
    out = ctx["proc"] / "atmosphere_correction"
    cache = out / "mapping_cache"
    cache.mkdir(parents=True, exist_ok=True)

    required = ("base.npy", "fx.npy", "fy.npy", "sec_inc.npy")
    if all((cache / x).is_file() for x in required):
        np.save(cache / "ref_idx.npy", ctx["ref_idx"].astype(np.int32))
        return cache

    # Search only inside the canonical atmosphere workspace after the validated
    # mapping runtime has executed.
    for name in required:
        if (cache / name).is_file():
            continue
        hits = [
            p for p in out.rglob(name)
            if p.parent != cache
        ]
        if len(hits) != 1:
            raise RuntimeError(
                f"validated GACOS mapping runtime did not produce a unique {name}; "
                f"found={hits}"
            )
        shutil.copyfile(hits[0], cache / name)

    np.save(cache / "ref_idx.npy", ctx["ref_idx"].astype(np.int32))
    return cache
