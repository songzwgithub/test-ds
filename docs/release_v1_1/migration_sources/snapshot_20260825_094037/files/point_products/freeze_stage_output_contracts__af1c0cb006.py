#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypsds.config import load_config
from pypsds.pipeline import STAGES
from pypsds.project import resolve_project_paths


REFERENCE_OUTPUTS = (
    "processing/referenced_timeseries/"
    "acquisition_phase_referenced_rad.npy",

    "processing/referenced_timeseries/"
    "preliminary_phase_rate_rad_per_year.npy",

    "processing/referenced_timeseries/"
    "preliminary_linear_residual_rms_rad.npy",

    "processing/referenced_timeseries/"
    "reference_region_mask.npy",

    "processing/referenced_timeseries/"
    "reference_strict_indices.npy",

    "processing/referenced_timeseries/"
    "reference_point_ids.npy",

    "processing/referenced_timeseries/"
    "reference_phase_median_rad.npy",

    "processing/referenced_timeseries/"
    "reference_phase_mad_sigma_rad.npy",

    "processing/referenced_timeseries/"
    "reference_epoch_qa.csv",

    "processing/referenced_timeseries/"
    "referenced_timeseries_manifest.json",
)


UNWRAP_FILESETS = (
    (
        "unwrapped_phase",
        "processing/single_ifg_robust_solution/"
        "pair*_*_*_unwrapped_phase_rad.npy",
    ),
    (
        "registered_mask",
        "processing/single_ifg_robust_solution/"
        "pair*_*_*_registered_mask.npy",
    ),
    (
        "single_ifg_manifest",
        "processing/single_ifg_robust_solution/"
        "pair*_*_*_manifest.json",
    ),
    (
        "safe_fragment",
        "processing/safe_fragment_integer_quality/"
        "pair*_*_*_safe_fragment.npy",
    ),
    (
        "safe_fragment_manifest",
        "processing/safe_fragment_integer_quality/"
        "pair*_*_*_manifest.json",
    ),
)


# Canonical persistent products of scripts/build_exact_support_cache.py.
# The stage script is a thin runpy wrapper around tools/build_exact_support_cache.py,
# so the generic AST inventory cannot discover these writes from the wrapper.
EXACT_SUPPORT_CACHE_OUTPUTS = (
    "processing/exact_support_cache/manifest.json",
    "processing/exact_support_cache/static_support_bits.npy",
    "processing/exact_support_cache/static_shp_count.npy",
    "processing/exact_support_cache/tile_done.npy",
)

# Per-output optionality.  ADI is diagnostic and is emitted only with --save-adi.
OPTIONAL_OUTPUTS_BY_STAGE = {
    "ds_statistics": frozenset({
        "processing/ds_statistics/amplitude_dispersion_index.npy",
    }),
}


def count_network_edges(path: Path) -> int:

    n = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for raw in f:

            s = raw.strip()

            if (
                not s
                or s.startswith("#")
            ):
                continue

            n += 1

    return n


def unique_sorted(items):

    return sorted(
        set(items)
    )


def freeze(
    *,
    inventory_path: Path,
    output_root: Path,
    snapshot_path: Path,
):

    inventory = json.loads(
        inventory_path.read_text(
            encoding="utf-8"
        )
    )

    network_itab = (
        output_root
        / "processing"
        / "network"
        / "network.itab"
    )

    if not network_itab.is_file():

        raise FileNotFoundError(
            network_itab
        )

    edge_count = count_network_edges(
        network_itab
    )

    print(
        "network edges :",
        edge_count,
    )

    stages_out = {}

    failures = []

    for stage in STAGES:

        name = stage.name

        info = inventory[
            "stages"
        ][
            name
        ]

        outputs = list(
            info.get(
                "exact_output_outputs",
                [],
            )
        )

        # ----------------------------------------------------
        # Expand dynamic OUTPUT patterns to their current exact
        # full-scene file set.
        # ----------------------------------------------------

        dynamic_sets = []

        for ref in info.get(
            "dynamic_refs",
            [],
        ):

            if (
                ref.get("kind")
                !=
                "output"
            ):
                continue

            rel = ref.get(
                "output_relative"
            )

            if not rel:
                continue

            matches = sorted(
                p.relative_to(
                    output_root
                ).as_posix()
                for p in output_root.glob(
                    rel
                )
                if p.is_file()
            )

            dynamic_sets.append({
                "pattern":
                    rel,

                "frozen_count":
                    len(matches),

                "files":
                    matches,
            })

            outputs.extend(
                matches
            )

        # ----------------------------------------------------
        # Reference contract was manually/source validated.
        # Do not let AST incompleteness weaken it.
        # ----------------------------------------------------

        if name == "reference":

            outputs = list(
                REFERENCE_OUTPUTS
            )

        # exact_support_cache is a runpy wrapper; freeze its
        # source-validated canonical persistent outputs explicitly.
        if name == "exact_support_cache":

            outputs = list(
                EXACT_SUPPORT_CACHE_OUTPUTS
            )

        outputs = unique_sorted(
            outputs
        )

        # ----------------------------------------------------
        # One stage is intentionally console-summary-only.
        # It has no persistent product.
        # ----------------------------------------------------

        non_persistent = (
            name
            ==
            "spatial_anchor_summary"
        )

        # Sequential v1 production may intentionally skip the
        # full corrected-YXT phase_cache stage and use canonical
        # Gamma streaming instead. Keep its output contract in the
        # snapshot, but do not require those files to exist.
        optional = (
            name
            ==
            "phase_cache"
        )

        witness_inputs = []

        if non_persistent:

            witness_inputs = [
                "processing/"
                "spatial_graph_two_anchor_quality/"
                "residual_two_anchor_quality.csv"
            ]

        # ----------------------------------------------------
        # unwrap is an orchestrator. Its important outputs are
        # created by subprocesses, so AST of 08p alone cannot
        # see them.
        #
        # Require one canonical product per temporal-network IFG.
        # ----------------------------------------------------

        filesets = []

        if name == "unwrap":

            for label, pattern in (
                UNWRAP_FILESETS
            ):

                matches = sorted(
                    p.relative_to(
                        output_root
                    ).as_posix()
                    for p in output_root.glob(
                        pattern
                    )
                    if p.is_file()
                )

                filesets.append({
                    "label":
                        label,

                    "pattern":
                        pattern,

                    "expected_count":
                        edge_count,

                    "frozen_count":
                        len(matches),
                })

        # ----------------------------------------------------
        # Exact output existence
        # ----------------------------------------------------

        optional_outputs = OPTIONAL_OUTPUTS_BY_STAGE.get(
            name,
            frozenset(),
        )

        records = []

        for rel in outputs:

            p = (
                output_root
                / rel
            )

            exists = p.is_file()

            required = (
                rel
                not in
                optional_outputs
            )

            records.append({
                "path":
                    rel,

                "size_bytes":
                    (
                        p.stat().st_size
                        if exists
                        else None
                    ),

                "required":
                    required,
            })

            if (
                not exists
                and
                required
                and
                not optional
            ):

                failures.append(
                    f"{name}: missing output {rel}"
                )

        # ----------------------------------------------------
        # Non-persistent witness
        # ----------------------------------------------------

        for rel in witness_inputs:

            if not (
                output_root
                / rel
            ).is_file():

                failures.append(
                    f"{name}: missing witness {rel}"
                )

        # ----------------------------------------------------
        # Fileset count validation
        # ----------------------------------------------------

        for fs in filesets:

            if (
                fs["frozen_count"]
                !=
                fs["expected_count"]
            ):

                failures.append(
                    f"{name}: fileset "
                    f"{fs['label']} "
                    f"count={fs['frozen_count']} "
                    f"expected={fs['expected_count']}"
                )

        if (
            not outputs
            and
            not non_persistent
        ):

            failures.append(
                f"{name}: no persistent output contract"
            )

        stages_out[
            name
        ] = {
            "script":
                stage.script,

            "persistent":
                not non_persistent,

            "optional":
                optional,

            "outputs":
                records,

            "dynamic_output_sets":
                dynamic_sets,

            "required_filesets":
                filesets,

            "witness_inputs":
                witness_inputs,
        }

    snapshot = {
        "format":
            "pyPSDS-GAMMA-stage-output-contract-snapshot-v1",

        "stage_count":
            len(STAGES),

        "network_edge_count":
            edge_count,

        "output_root":
            str(
                output_root
            ),

        "stages":
            stages_out,
    }

    snapshot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    if failures:

        print()
        print(
            "=" * 96
        )

        print(
            "FREEZE FAILED"
        )

        print(
            "=" * 96
        )

        for x in failures:
            print(
                "FAIL",
                x,
            )

        raise SystemExit(1)

    print()
    print(
        "snapshot :",
        snapshot_path,
    )

    print(
        "stages   :",
        len(STAGES),
    )

    print(
        "STAGE OUTPUT CONTRACT FREEZE: PASS"
    )


def audit(
    *,
    snapshot_path: Path,
    output_root: Path,
):

    x = json.loads(
        snapshot_path.read_text(
            encoding="utf-8"
        )
    )

    network_itab = (
        output_root
        / "processing"
        / "network"
        / "network.itab"
    )

    current_edges = count_network_edges(
        network_itab
    )

    failures = []

    covered = 0

    print(
        "=" * 110
    )

    print(
        f"pyPSDS-GAMMA {len(STAGES)}-stage output contract audit"
    )

    print(
        "=" * 110
    )

    print(
        "snapshot edges :",
        x["network_edge_count"],
    )

    print(
        "current edges  :",
        current_edges,
    )

    if (
        current_edges
        !=
        x["network_edge_count"]
    ):

        failures.append(
            "temporal network edge count changed"
        )

    for i, stage in enumerate(
        STAGES,
        start=1,
    ):

        c = x[
            "stages"
        ][
            stage.name
        ]

        optional = bool(
            c.get(
                "optional",
                False,
            )
        )

        stage_ok = True

        for rec in c[
            "outputs"
        ]:

            p = (
                output_root
                / rec["path"]
            )

            required = bool(
                rec.get(
                    "required",
                    True,
                )
            )

            if (
                not p.is_file()
                and
                required
                and
                not optional
            ):

                stage_ok = False

                failures.append(
                    f"{stage.name}: missing "
                    f"{rec['path']}"
                )

        for rel in c[
            "witness_inputs"
        ]:

            if not (
                output_root
                / rel
            ).is_file():

                stage_ok = False

                failures.append(
                    f"{stage.name}: missing witness "
                    f"{rel}"
                )

        for fs in c[
            "required_filesets"
        ]:

            matches = [
                p
                for p in output_root.glob(
                    fs["pattern"]
                )
                if p.is_file()
            ]

            expected = (
                current_edges
            )

            if len(
                matches
            ) != expected:

                stage_ok = False

                failures.append(
                    f"{stage.name}: "
                    f"{fs['label']} "
                    f"count={len(matches)} "
                    f"expected={expected}"
                )

        if stage_ok:
            covered += 1

        print(
            f"{i:02d} "
            f"{stage.name:38s} "
            f"{'PASS' if stage_ok else 'FAIL':4s} "
            f"OUT={len(c['outputs']):3d} "
            f"SETS={len(c['required_filesets']):2d}"
            +
            (
                " NON_PERSISTENT"
                if not c["persistent"]
                else ""
            )
        )

    print()
    print(
        "=" * 110
    )

    print(
        "covered stages :",
        f"{covered}/{len(STAGES)}",
    )

    if failures:

        print(
            "CONTRACT AUDIT: FAIL"
        )

        for x in failures:
            print(
                "FAIL",
                x,
            )

        raise SystemExit(1)

    if covered != len(STAGES):

        raise SystemExit(
            "Not all stages covered."
        )

    print(
        f"{len(STAGES)}-STAGE OUTPUT CONTRACT AUDIT: PASS"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--inventory",
        default=(
            "docs/release/"
            "stage_contract_inventory.json"
        ),
    )

    ap.add_argument(
        "--snapshot",
        default=(
            "docs/release/"
            "stage_output_contract_snapshot.json"
        ),
    )

    ap.add_argument(
        "--audit-only",
        action="store_true",
    )

    args = ap.parse_args()

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

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

    inventory = (
        root
        / args.inventory
    )

    snapshot = (
        root
        / args.snapshot
    )

    if not args.audit_only:

        freeze(
            inventory_path=inventory,
            output_root=output_root,
            snapshot_path=snapshot,
        )

    audit(
        snapshot_path=snapshot,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
