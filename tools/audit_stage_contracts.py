#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pypsds.config import load_config
from pypsds.pipeline import (
    STAGE_CONTRACTS,
    validate_stage_contract_registry,
)
from pypsds.project import resolve_project_paths


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Audit declared pyPSDS-GAMMA stage "
            "input/output contracts."
        )
    )

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--stage",
        default=None,
        help=(
            "One stage name. "
            "Default: all validated contracts."
        ),
    )

    ap.add_argument(
        "--mode",
        choices=(
            "inputs",
            "outputs",
            "all",
        ),
        default="all",
    )

    args = ap.parse_args()

    validate_stage_contract_registry()

    cfg, config_path = load_config(
        args.config
    )

    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    output_root = Path(
        paths.output_dir
    )

    if args.stage is not None:

        if args.stage not in STAGE_CONTRACTS:
            raise SystemExit(
                f"Unknown stage: {args.stage}"
            )

        names = [
            args.stage
        ]

    else:

        names = [
            name
            for name, contract
            in STAGE_CONTRACTS.items()
            if contract.validated
        ]

    failures = []

    print(
        "=" * 96
    )

    print(
        "pyPSDS-GAMMA stage contract audit"
    )

    print(
        "=" * 96
    )

    print(
        "config      :",
        config_path,
    )

    print(
        "output root :",
        output_root,
    )

    print(
        "stages      :",
        len(names),
    )

    for name in names:

        contract = (
            STAGE_CONTRACTS[
                name
            ]
        )

        print()
        print(
            "-" * 96
        )

        print(
            f"stage      : {name}"
        )

        print(
            f"validated  : "
            f"{contract.validated}"
        )

        print(
            f"cacheable  : "
            f"{contract.cacheable}"
        )

        groups = []

        if args.mode in (
            "inputs",
            "all",
        ):
            groups.append(
                (
                    "INPUT",
                    contract.required_inputs,
                )
            )

        if args.mode in (
            "outputs",
            "all",
        ):
            groups.append(
                (
                    "OUTPUT",
                    contract.required_outputs,
                )
            )

        for kind, relpaths in groups:

            for rel in relpaths:

                p = (
                    output_root
                    / rel
                )

                ok = p.is_file()

                size = (
                    p.stat().st_size
                    if ok
                    else None
                )

                print(
                    f"{'PASS' if ok else 'FAIL'}  "
                    f"{kind:6s}  "
                    f"{rel}"
                    +
                    (
                        f"  [{size:,} bytes]"
                        if size is not None
                        else ""
                    )
                )

                if not ok:
                    failures.append(
                        (
                            name,
                            kind,
                            rel,
                        )
                    )

    print()
    print(
        "=" * 96
    )

    if failures:

        print(
            f"CONTRACT AUDIT: FAIL "
            f"({len(failures)} missing)"
        )

        for item in failures:
            print(
                "  ",
                item,
            )

        raise SystemExit(1)

    print(
        "CONTRACT AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
