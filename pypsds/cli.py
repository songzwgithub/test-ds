from __future__ import annotations

import argparse
from importlib import resources
import json
from pathlib import Path
import shutil

from . import __version__
from .config import cfg_get, load_config
from .project import resolve_project_paths
from .geometry import (
    resolve_data2pt,
    resolve_geometry_inputs,
    resolve_height_raster,
)
from .runtime import build_runtime_plan


def _requested_cpu(cfg):
    raw = cfg_get(cfg, "runtime.cpu", None)
    return None if raw in (None, "", "auto") else int(raw)


def _runtime_solver_size(cfg, ndate):
    strategy = str(
        cfg_get(
            cfg,
            "phase_linking.temporal.strategy",
            "full_scm",
        )
    ).strip().lower()

    if strategy != "sequential":
        return int(ndate)

    return min(
        int(ndate),
        int(
            cfg_get(
                cfg,
                "phase_linking.temporal.ministack_size",
                19,
            )
        )
        +
        int(
            cfg_get(
                cfg,
                "phase_linking.temporal.max_num_compressed",
                5,
            )
        ),
    )


def _resolve_geometry_preflight(
    cfg,
    paths,
):
    """
    Resolve the static input contract required by point_geometry.

    No scientific processing is performed here.
    """

    geometry = resolve_geometry_inputs(
        cfg,
        paths,
    )

    height_raster = resolve_height_raster(
        cfg,
        paths,
        geometry,
    )

    return geometry, height_raster


def _open_stack(cfg, path):
    paths = resolve_project_paths(cfg, path)
    from .gamma.stack import GammaStack
    stack = GammaStack.from_rslc_tab(
        paths.rslc_tab,
        rslc_dir=paths.rslc_dir,
        dtype=str(cfg_get(cfg, "gamma.rslc_dtype", "auto")),
        byte_order=str(cfg_get(cfg, "gamma.byte_order", "big")),
        io_workers=1,
    )
    return paths, stack


def cmd_init(args):
    project = Path(args.project_dir).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    target = project / "pypsds.yaml"
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists; use --force to overwrite.")
    template = resources.files("pypsds.resources").joinpath(
        "default_config.yaml"
    ).read_text(encoding="utf-8")
    target.write_text(template, encoding="utf-8")
    print("pyPSDS-GAMMA project initialized")
    print(f"config : {target}")
    print("Next: configure paths.dem_dir, the Geometry/GAMMA reference date, and the study-area reference window, then run doctor.")


def cmd_config_check(args):
    cfg, path = load_config(args.config)
    paths = resolve_project_paths(cfg, path)
    geometry, height_raster = _resolve_geometry_preflight(
        cfg,
        paths,
    )
    print("pyPSDS-GAMMA configuration: PASS")
    print(f"config     : {path}")
    print(f"RSLC dir   : {paths.rslc_dir}")
    print(f"RSLC tab   : {paths.rslc_tab}")
    print(f"output dir : {paths.output_dir}")
    print(f"DEM dir    : {paths.dem_dir}")
    print(f"geometry ref: {geometry.reference_date}")
    print(f"geometry par: {geometry.geometry_par}")
    print(f"longitude  : {geometry.longitude_raster}")
    print(f"latitude   : {geometry.latitude_raster}")
    print(f"height     : {height_raster}")


def cmd_plan(args):
    cfg, path = load_config(args.config)
    if args.ndate is not None:
        ndate = int(args.ndate)
    else:
        raw = cfg_get(cfg, "runtime.planning_ndate", None)
        if raw not in (None, "", "auto"):
            ndate = int(raw)
        else:
            _, stack = _open_stack(cfg, path)
            ndate = len(stack.dates)
    plan = build_runtime_plan(
        ndate=ndate,
        memory_fraction=float(cfg_get(cfg, "runtime.memory_fraction", 0.85)),
        requested_cpu=_requested_cpu(cfg),
        max_solver_size=_runtime_solver_size(cfg, ndate),
    )
    payload = plan.as_dict()
    payload["ndate"] = int(ndate)
    print(json.dumps(payload, indent=2))


def _validate_project_choices(cfg, stack):
    errors = []
    geometric_ref = cfg_get(cfg, "phase_correction.geometric_reference_date", None)
    if geometric_ref in (None, ""):
        errors.append("phase_correction.geometric_reference_date is not set")
    elif str(geometric_ref).lower() != "auto" and str(geometric_ref) not in stack.dates:
        errors.append("phase_correction.geometric_reference_date is not present in RSLC_tab")
    if str(cfg_get(cfg, "reference.method", "radar_window")) == "radar_window":
        row = cfg_get(cfg, "reference.radar_window.center_row", None)
        col = cfg_get(cfg, "reference.radar_window.center_col", None)
        if row is None or col is None:
            errors.append("reference.radar_window.center_row/center_col must be set")
    if errors:
        raise RuntimeError("Project-specific configuration is incomplete:\n  - " + "\n  - ".join(errors))


def cmd_doctor(args):
    cfg, path = load_config(args.config)
    paths, stack = _open_stack(cfg, path)
    _validate_project_choices(cfg, stack)
    geometry, height_raster = _resolve_geometry_preflight(
        cfg,
        paths,
    )

    geometry_data2pt = resolve_data2pt()
    plan = build_runtime_plan(
        ndate=len(stack.dates),
        memory_fraction=float(cfg_get(cfg, "runtime.memory_fraction", 0.85)),
        requested_cpu=_requested_cpu(cfg),
        max_solver_size=_runtime_solver_size(
            cfg,
            len(stack.dates),
        ),
    )
    print("=" * 80)
    print("pyPSDS-GAMMA production doctor")
    print("=" * 80)
    print(f"version          : {__version__}")
    print(f"config           : {path}")
    print(f"RSLC stack       : {stack.shape}")
    print(f"acquisitions     : {len(stack.dates)}")
    print(f"Geometry ref     : {geometry.reference_date}")
    print(f"Geometry par     : {geometry.geometry_par}")
    print(f"Geometry lon     : {geometry.longitude_raster}")
    print(f"Geometry lat     : {geometry.latitude_raster}")
    print(f"Geometry height  : {height_raster}")
    print(f"Geometry data2pt : {geometry_data2pt}")
    print(f"first / last     : {stack.dates[0]} / {stack.dates[-1]}")
    print(f"CPU              : {plan.cpu_count}")
    print(f"RAM available    : {plan.available_memory_bytes/1024**3:.2f} GiB")
    print(f"RAM usable       : {plan.usable_memory_bytes/1024**3:.2f} GiB")
    print(f"PL workers       : {plan.phase_link_workers}")
    print(f"PL chunk         : {plan.phase_link_chunk_size}")
    print(f"PL batch         : {plan.phase_link_batch_size}")
    print(f"PL solver dim    : {plan.phase_link_solver_size}")
    print(
        f"PL tile          : "
        f"{plan.phase_link_tile_rows} x "
        f"{plan.phase_link_tile_cols}"
    )
    print(
        f"SHP cache tile   : "
        f"{plan.support_cache_tile_rows} x "
        f"{plan.support_cache_tile_cols}"
    )
    print(
        f"SHP cache batch  : "
        f"{plan.support_cache_batch_size}"
    )
    missing = []
    for name in ("SLC2pt", "data2pt", "phase_sim_orb_pt", "base_calc", "base_orbit"):
        found = shutil.which(name)
        print(f"{name:16s}: {found or 'NOT FOUND'}")
        if found is None:
            missing.append(name)
    if missing:
        raise RuntimeError("Required GAMMA command(s) not found in PATH: " + ", ".join(missing))
    print("doctor           : PASS")


def cmd_run(args):
    from .pipeline import run_pipeline
    return run_pipeline(
        config=args.config,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        dry_run=args.dry_run,
        force=args.force,
        list_stages=args.list_stages,
    )


def build_parser():
    p = argparse.ArgumentParser(prog="pypsds", description="pyPSDS-GAMMA production CLI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init", help="Create a portable project configuration.")
    q.add_argument("project_dir")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_init)
    q = sub.add_parser("config-check")
    q.add_argument("--config", required=True)
    q.set_defaults(func=cmd_config_check)
    q = sub.add_parser("plan")
    q.add_argument("--config", required=True)
    q.add_argument("--ndate", type=int, default=None)
    q.set_defaults(func=cmd_plan)
    q = sub.add_parser("doctor")
    q.add_argument("--config", required=True)
    q.set_defaults(func=cmd_doctor)
    q = sub.add_parser("run", help="Run the production InSAR processing pipeline.")
    q.add_argument("--config", required=True)
    q.add_argument("--from-stage", default=None)
    q.add_argument("--to-stage", default=None)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--list-stages", action="store_true")
    q.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
