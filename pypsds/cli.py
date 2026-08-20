from __future__ import annotations

import argparse
import json
import shutil

from . import __version__
from .config import cfg_get, load_config
from .project import resolve_project_paths
from .runtime import build_runtime_plan


def cmd_config_check(args):

    cfg, path = load_config(
        args.config
    )

    paths = resolve_project_paths(
        cfg,
        path,
    )

    print(
        "pyPSDS-GAMMA configuration: PASS"
    )
    print(
        f"config     : {path}"
    )
    print(
        f"RSLC dir   : {paths.rslc_dir}"
    )
    print(
        f"RSLC tab   : {paths.rslc_tab}"
    )
    print(
        f"output dir : {paths.output_dir}"
    )


def cmd_plan(args):

    cfg, path = load_config(
        args.config
    )

    ndate = int(
        args.ndate
        or
        cfg_get(
            cfg,
            "runtime.planning_ndate",
            38,
        )
    )

    plan = build_runtime_plan(
        ndate=ndate,
        memory_fraction=float(
            cfg_get(
                cfg,
                "runtime.memory_fraction",
                0.85,
            )
        ),
    )

    print(
        json.dumps(
            plan.as_dict(),
            indent=2,
        )
    )


def cmd_doctor(args):

    cfg, path = load_config(
        args.config
    )

    paths = resolve_project_paths(
        cfg,
        path,
    )

    from .gamma.stack import GammaStack

    stack = GammaStack.from_rslc_tab(
        paths.rslc_tab,
        rslc_dir=paths.rslc_dir,
        dtype=str(
            cfg_get(
                cfg,
                "gamma.rslc_dtype",
                "auto",
            )
        ),
        byte_order=str(
            cfg_get(
                cfg,
                "gamma.byte_order",
                "big",
            )
        ),
        io_workers=1,
    )

    plan = build_runtime_plan(
        ndate=len(stack.dates),
        memory_fraction=float(
            cfg_get(
                cfg,
                "runtime.memory_fraction",
                0.85,
            )
        ),
    )

    print("=" * 80)
    print(
        "pyPSDS-GAMMA v1.0 doctor"
    )
    print("=" * 80)

    print(
        f"version          : {__version__}"
    )

    print(
        f"config           : {path}"
    )

    print(
        f"RSLC stack       : "
        f"{stack.shape}"
    )

    print(
        f"acquisitions     : "
        f"{len(stack.dates)}"
    )

    print(
        f"first / last     : "
        f"{stack.dates[0]} / "
        f"{stack.dates[-1]}"
    )

    print(
        f"CPU              : "
        f"{plan.cpu_count}"
    )

    print(
        f"RAM available    : "
        f"{plan.available_memory_bytes/1024**3:.2f} GiB"
    )

    print(
        f"RAM usable       : "
        f"{plan.usable_memory_bytes/1024**3:.2f} GiB"
    )

    print(
        f"PL workers       : "
        f"{plan.phase_link_workers}"
    )

    print(
        f"PL chunk         : "
        f"{plan.phase_link_chunk_size}"
    )

    print(
        f"PL batch         : "
        f"{plan.phase_link_batch_size}"
    )

    commands = [
        "SLC2pt",
        "data2pt",
        "phase_sim_orb_pt",
        "base_calc",
        "base_orbit",
    ]

    for name in commands:
        print(
            f"{name:16s}: "
            f"{shutil.which(name) or 'NOT FOUND'}"
        )



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

    p = argparse.ArgumentParser(
        prog="pypsds",
        description=(
            "pyPSDS-GAMMA production CLI"
        ),
    )

    p.add_argument(
        "--version",
        action="version",
        version=(
            f"%(prog)s {__version__}"
        ),
    )

    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    q = sub.add_parser(
        "config-check",
    )
    q.add_argument(
        "--config",
        required=True,
    )
    q.set_defaults(
        func=cmd_config_check,
    )

    q = sub.add_parser(
        "plan",
    )
    q.add_argument(
        "--config",
        required=True,
    )
    q.add_argument(
        "--ndate",
        type=int,
        default=None,
    )
    q.set_defaults(
        func=cmd_plan,
    )

    q = sub.add_parser(
        "doctor",
    )
    q.add_argument(
        "--config",
        required=True,
    )
    q.set_defaults(
        func=cmd_doctor,
    )


    q = sub.add_parser(
        "run",
        help=(
            "Run the production InSAR "
            "processing pipeline."
        ),
    )

    q.add_argument(
        "--config",
        required=True,
    )

    q.add_argument(
        "--from-stage",
        default=None,
    )

    q.add_argument(
        "--to-stage",
        default=None,
    )

    q.add_argument(
        "--dry-run",
        action="store_true",
    )

    q.add_argument(
        "--force",
        action="store_true",
    )

    q.add_argument(
        "--list-stages",
        action="store_true",
    )

    q.set_defaults(
        func=cmd_run,
    )

    return p


def main(argv=None):

    args = build_parser().parse_args(
        argv
    )

    return args.func(args)


if __name__ == "__main__":
    main()
