from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import cfg_get, load_config
from .manifest import build_stage_signature
from .project import resolve_project_paths
from .runtime import build_runtime_plan


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SCRIPT_DIR = (
    ROOT
    / "scripts"
)


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    script: str


@dataclass(frozen=True, slots=True)
class StageContract:
    """
    Formal production contract metadata for one pipeline stage.

    Production-contract layer:
      - contracts are centrally declared;
      - automatic cache reuse remains disabled;
      - output completeness rules are added incrementally
        and validated before cache activation.
    """

    name: str

    # Paths are relative to output_dir.
    required_inputs: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()

    # True only after the input/output contract has been
    # checked against the actual production implementation.
    validated: bool = False

    # Cache reuse remains disabled until the separate cache
    # correctness phase is completed.
    cacheable: bool = False


STAGES = [
    Stage(
        "ds_statistics",
        "build_ds_statistics.py",
    ),
    Stage(
        "phase_cache",
        "build_phase_cache.py",
    ),Stage(
    "exact_support_cache",
    "build_exact_support_cache.py",
),

    Stage(
        "phase_linking",
        "run_phase_linking.py",
    ),
    Stage(
        "ds_selection",
        "select_ds.py",
    ),
    Stage(
        "ps_finalize",
        "finalize_ps_geometry.py",
    ),
    Stage(
        "point_stack",
        "build_point_phase_stack.py",
    ),

    Stage(
        "network_prepare",
        "prepare_temporal_network.py",
    ),
    Stage(
        "network_build",
        "build_temporal_network.py",
    ),
    Stage(
        "network_cycle_quality",
        "assess_network_cycle_quality.py",
    ),
    Stage(
        "network_finalize",
        "finalize_temporal_network.py",
    ),

    Stage(
        "virtual_ifg_quality",
        "assess_virtual_ifg_quality.py",
    ),
    Stage(
        "spatial_graph_quality",
        "assess_spatial_graph_quality.py",
    ),
    Stage(
        "spatial_bridge_quality",
        "assess_spatial_bridge_quality.py",
    ),
    Stage(
        "spatial_component_quality",
        "assess_spatial_components.py",
    ),
    Stage(
        "spatial_anchor_quality",
        "assess_spatial_anchor_quality.py",
    ),
    Stage(
        "spatial_anchor_summary",
        "summarize_spatial_anchor_quality.py",
    ),
    Stage(
        "spatial_local_graph_quality",
        "assess_local_spatial_graph.py",
    ),
    Stage(
        "spatial_graph",
        "build_spatial_graph.py",
    ),
    Stage(
        "spatial_gradient_quality",
        "assess_spatial_phase_gradient.py",
    ),
    Stage(
        "unwrap_policy",
        "finalize_unwrap_policy.py",
    ),

    Stage(
        "unwrap",
        "unwrap_all_ifgs.py",
    ),
    Stage(
        "unwrap_severity_quality",
        "assess_unwrap_severity.py",
    ),
    Stage(
        "unwrap_conflict_quality",
        "assess_unwrap_conflicts.py",
    ),
    Stage(
        "unwrap_acquisition_quality",
        "assess_unwrap_acquisition_quality.py",
    ),
    Stage(
        "temporal_closure",
        "assess_temporal_integer_closure.py",
    ),

    # Sparse integer candidates required by the
    # downstream fragment-signature feasibility stage.
    Stage(
        "temporal_integer_candidate",
        "build_temporal_integer_candidates.py",
    ),

    # Spatial counterfactual validation of the temporal
    # integer candidates. Quality-only; it does not
    # overwrite the accepted unwrap solution.
    Stage(
        "temporal_candidate_spatial_quality",
        "validate_temporal_integer_candidates.py",
    ),

    Stage(
        "unwrap_signature_quality",
        "assess_unwrap_signature_feasibility.py",
    ),
    Stage(
        "unwrap_finalize",
        "finalize_unwrap_solution.py",
    ),

    Stage(
        "timeseries_inversion",
        "invert_timeseries.py",
    ),
    Stage(
        "reference",
        "apply_reference.py",
    ),
]


STAGE_INDEX = {
    s.name: i
    for i, s in enumerate(
        STAGES
    )
}


# ============================================================
# Formal stage contracts
#
# Production-contract registry:
#   every production stage must have exactly one contract.
#
# IMPORTANT:
#   cacheable remains False until required_inputs and
#   required_outputs have been explicitly validated.
# ============================================================

STAGE_CONTRACTS = {
    stage.name:
        StageContract(
            name=stage.name,
        )
    for stage in STAGES
}


# ------------------------------------------------------------
# Validated production contracts
#
# Add stages here only after checking the actual implementation
# and a completed full-scene production result.
#
# Cache remains disabled independently of contract validation.
# ------------------------------------------------------------

STAGE_CONTRACTS["reference"] = StageContract(
    name="reference",

    required_inputs=(
        "processing/network_inversion/strict_point_ids.npy",
        "processing/network_inversion/"
        "acquisition_phase_l2_candidate_rad.npy",
        "processing/point_phase_stack/rows.npy",
        "processing/point_phase_stack/cols.npy",
    ),

    required_outputs=(
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
    ),

    validated=True,
    cacheable=False,
)


def validate_stage_contract_registry() -> None:
    """
    Fail early if pipeline stages and contracts diverge.
    """

    stage_names = [
        x.name
        for x in STAGES
    ]

    if len(
        stage_names
    ) != len(
        set(
            stage_names
        )
    ):
        raise RuntimeError(
            "Duplicate production stage names."
        )

    contract_names = set(
        STAGE_CONTRACTS
    )

    expected_names = set(
        stage_names
    )

    missing = sorted(
        expected_names
        -
        contract_names
    )

    extra = sorted(
        contract_names
        -
        expected_names
    )

    if missing or extra:
        raise RuntimeError(
            "Stage contract registry mismatch: "
            f"missing={missing}, extra={extra}"
        )

    for name in stage_names:

        contract = (
            STAGE_CONTRACTS[
                name
            ]
        )

        if contract.name != name:
            raise RuntimeError(
                "Stage contract name mismatch: "
                f"{name} != {contract.name}"
            )


def _sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file."""

    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def _new_run_id() -> str:
    """
    UTC run identifier suitable for filenames.

    Microseconds prevent collisions between short partial runs.
    """

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


def _build_stack_identity(
    *,
    stack,
    paths,
) -> str:
    """
    Build a lightweight identity signature for the input stack.

    This is deliberately a provenance identity, not yet a
    content-addressed cache key for every large RSLC byte.

    Automatic stage skipping remains disabled until the formal
    output contracts are introduced.
    """

    return build_stage_signature(
        algorithm=(
            "gamma-rslc-stack-identity-v1"
        ),
        parameters={
            "shape":
                list(
                    stack.shape
                ),

            "dates":
                [
                    str(x)
                    for x in stack.dates
                ],

            "rslc_dir":
                str(
                    Path(
                        paths.rslc_dir
                    ).resolve()
                ),
        },
        inputs={
            "rslc_tab":
                str(
                    Path(
                        paths.rslc_tab
                    ).resolve()
                ),

            "rslc_tab_sha256":
                _sha256_file(
                    paths.rslc_tab
                ),
        },
    )


def _build_run_stage_signature(
    stage,
    *,
    cfg,
    config_path,
    paths,
    runtime,
    force,
    config_sha256,
    stack_identity,
    parent_signature,
    upstream_resolved,
) -> tuple[str, str]:
    """
    Build the provenance signature for one pipeline stage.

    IMPORTANT:
    This is record-only in provenance-signature layer.
    It does NOT currently permit automatic stage skipping.
    """

    script = (
        SCRIPT_DIR
        / stage.script
    )

    if not script.is_file():
        raise FileNotFoundError(
            script
        )

    script_sha256 = (
        _sha256_file(
            script
        )
    )

    argv = _stage_args(
        stage,
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=force,
    )

    signature_value = (
        build_stage_signature(
            algorithm=(
                "pyPSDS-GAMMA-stage-v1:"
                + stage.name
            ),

            parameters={
                "script":
                    stage.script,

                "script_sha256":
                    script_sha256,

                "argv":
                    [
                        str(x)
                        for x in argv
                    ],

                "force":
                    bool(
                        force
                    ),
            },

            inputs={
                "config_sha256":
                    config_sha256,

                "stack_identity":
                    stack_identity,

                "parent_stage_signature":
                    parent_signature,

                "upstream_resolved":
                    bool(
                        upstream_resolved
                    ),
            },
        )
    )

    return (
        signature_value,
        script_sha256,
    )


def list_stage_names():
    return [
        x.name
        for x in STAGES
    ]


def _fmt(x):
    return str(x)


def _stage_args(
    stage: Stage,
    *,
    cfg,
    config_path: Path,
    paths,
    runtime,
    force: bool,
):

    args = [
        "--config",
        str(config_path),
    ]

    if stage.name == "ds_statistics":

        args += [
            "--adi-max",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.ps.amplitude_dispersion_max",
                    0.25,
                )
            ),

            "--tile-rows",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.ds_statistics_tile_rows",
                    0,
                )
            ),

            "--tile-cols",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.ds_statistics_tile_cols",
                    0,
                )
            ),
        ]

        if not force:
            args.append(
                "--resume"
            )

    elif stage.name == "phase_cache":

        args += [
            "--tile-rows",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_cache_tile_rows",
                    runtime.tile_rows,
                )
            ),
            "--tile-cols",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_cache_tile_cols",
                    runtime.tile_cols,
                )
            ),
            "--workers",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_cache_workers",
                    runtime.io_workers,
                )
            ),
        ]

        if force:
            args.append(
                "--overwrite"
            )

    elif stage.name == "exact_support_cache":

        args += [
            "--half-row",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.half_row",
                    5,
                )
            ),

            "--half-col",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.half_col",
                    11,
                )
            ),

            "--alpha",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.alpha",
                    0.005,
                )
            ),

            "--tile-rows",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_tile_rows",
                    512,
                )
            ),

            "--tile-cols",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_tile_cols",
                    1024,
                )
            ),

            "--batch",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_batch_size",
                    32000,
                )
            ),

            "--support-block",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_support_block",
                    1024,
                )
            ),
        ]

        if not force:
            args.append(
                "--resume"
            )

    elif stage.name == "phase_linking":

        # Preserve validated numerical execution settings for first-release parity
        # for validated production parity. They are benchmarked and
        # auto-tuned only after parity is frozen.
        args += [
            "--center-mode",
            str(
                cfg_get(
                    cfg,
                    "selection.ds.center_mode",
                    "all",
                )
            ),

            "--half-row",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.half_row",
                    5,
                )
            ),

            "--half-col",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.half_col",
                    11,
                )
            ),

            "--alpha",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.alpha",
                    0.005,
                )
            ),

            "--min-shp",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.shp.min_count",
                    48,
                )
            ),

            "--adi-max",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.ps.amplitude_dispersion_max",
                    0.25,
                )
            ),

            "--beta",
            _fmt(
                cfg_get(
                    cfg,
                    "phase_linking.beta",
                    0.0,
                )
            ),

            "--gamma-jitter",
            _fmt(
                cfg_get(
                    cfg,
                    "phase_linking.gamma_jitter",
                    1e-6,
                )
            ),

            "--emi-mu",
            _fmt(
                cfg_get(
                    cfg,
                    "phase_linking.target_eigenvalue",
                    0.99,
                )
            ),

            "--batch-size",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_batch_size",
                    16000,
                )
            ),

            "--pl-workers",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_workers",
                    16,
                )
            ),

            "--pl-chunk-size",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_chunk_size",
                    512,
                )
            ),
        ]

        temporal_strategy = str(
            cfg_get(
                cfg,
                "phase_linking.temporal.strategy",
                "full_scm",
            )
        ).lower()

        if (
            not force
            and
            temporal_strategy == "full_scm"
        ):
            args.append(
                "--resume"
            )

    elif stage.name == "ds_selection":

        args += [
            "--tc-min",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.ds.temporal_coherence_min",
                    0.80,
                )
            ),

            "--pair-min",
            _fmt(
                cfg_get(
                    cfg,
                    "selection.ds.pair_coherence_min",
                    0.0,
                )
            ),
        ]

        if bool(
            cfg_get(
                cfg,
                "selection.ds.accept_evd",
                True,
            )
        ):
            args.append(
                "--accept-evd"
            )

    elif stage.name == "point_stack":

        tc = float(
            cfg_get(
                cfg,
                "selection.ds.temporal_coherence_min",
                0.80,
            )
        )

        pc = float(
            cfg_get(
                cfg,
                "selection.ds.pair_coherence_min",
                0.0,
            )
        )

        evd = bool(
            cfg_get(
                cfg,
                "selection.ds.accept_evd",
                True,
            )
        )

        tag = (
            f"tc{tc:.3f}_"
            f"pc{pc:.3f}_"
            + (
                "evd"
                if evd
                else "emi"
            )
        )

        ds_mask = (
            Path(paths.output_dir)
            / "processing"
            / f"final_ds_{tag}.npy"
        )

        args += [
            "--ds-mask",
            str(ds_mask),
        ]

    elif stage.name == "unwrap":

        if force:
            args.append(
                "--force"
            )

    elif stage.name == "reference":

        args += [
            "--center-row",
            _fmt(
                cfg_get(
                    cfg,
                    "reference.radar_window.center_row",
                    538,
                )
            ),

            "--center-col",
            _fmt(
                cfg_get(
                    cfg,
                    "reference.radar_window.center_col",
                    337,
                )
            ),

            "--half-row",
            _fmt(
                cfg_get(
                    cfg,
                    "reference.radar_window.half_row",
                    10,
                )
            ),

            "--half-col",
            _fmt(
                cfg_get(
                    cfg,
                    "reference.radar_window.half_col",
                    15,
                )
            ),
        ]

    return args


def _run_stage(
    stage,
    *,
    cfg,
    config_path,
    paths,
    runtime,
    force,
    dry_run,
):


    # ------------------------------------------------------------------
    # P8G2 PIPELINE PHASE-CACHE POLICY
    #
    # The full corrected-YXT cache is no longer mandatory for validated
    # sequential production.
    #
    # PYPSDS_PHASE_SOURCE:
    #
    #   gamma
    #       -> always skip phase_cache
    #
    #   cache
    #       -> preserve the original phase_cache stage
    #
    #   auto
    #       -> use an already existing standard corrected-YXT cache;
    #          otherwise skip its construction and let Step04 use the
    #          validated canonical GAMMA streaming source.
    #
    # full_scm remains unchanged because that legacy production path
    # still requires the full corrected-YXT array.
    # ------------------------------------------------------------------

    if stage.name == "phase_cache":

        phase_source_policy = (
            os.environ.get(
                "PYPSDS_PHASE_SOURCE",
                "auto",
            )
            .strip()
            .lower()
        )


        if phase_source_policy not in {
            "auto",
            "cache",
            "gamma",
        }:

            raise ValueError(
                "PYPSDS_PHASE_SOURCE must be "
                "auto/cache/gamma"
            )


        temporal_strategy = str(
            cfg_get(
                cfg,
                "phase_linking.temporal.strategy",
                "full_scm",
            )
        ).strip().lower()


        standard_yxt_path = (
            Path(
                paths.output_dir
            )
            /
            "processing"
            /
            "cache"
            /
            "phase_corrected_yxt.npy"
        )


        skip_phase_cache = False
        skip_reason = None


        if temporal_strategy == "sequential":

            if (
                phase_source_policy
                ==
                "gamma"
            ):

                skip_phase_cache = True

                skip_reason = (
                    "explicit gamma canonical streaming"
                )


            elif (
                phase_source_policy
                ==
                "auto"
                and
                not standard_yxt_path.is_file()
            ):

                skip_phase_cache = True

                skip_reason = (
                    "auto policy with no existing "
                    "corrected-YXT cache"
                )


        if skip_phase_cache:

            print()
            print(
                "=" * 96
            )

            print(
                "STAGE: phase_cache"
            )

            print(
                "=" * 96
            )

            print(
                "action          : SKIP"
            )

            print(
                "phase source    :",
                phase_source_policy,
            )

            print(
                "temporal mode   :",
                temporal_strategy,
            )

            print(
                "reason          :",
                skip_reason,
            )

            print(
                "full YXT build  : not scheduled"
            )

            print(
                "[phase_cache] SKIPPED "
                "(Gamma streaming production)"
            )


            return {
                "status":
                    "SKIPPED",

                "seconds":
                    0.0,

                "reason":
                    skip_reason,

                "phase_source_policy":
                    phase_source_policy,

                "standard_yxt_exists":
                    bool(
                        standard_yxt_path.is_file()
                    ),
            }

    script = (
        SCRIPT_DIR
        / stage.script
    )

    if not script.is_file():
        raise FileNotFoundError(
            script
        )

    args = _stage_args(
        stage,
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=force,
    )

    cmd = [
        sys.executable,
        str(script),
        *args,
    ]

    print()
    print("=" * 96)
    print(
        f"STAGE: {stage.name}"
    )
    print("=" * 96)
    print(
        " ".join(cmd)
    )

    if dry_run:
        return {
            "status": "DRY_RUN",
            "seconds": 0.0,
        }

    log_dir = (
        Path(paths.output_dir)
        / "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"{stage.name}.log"
    )

    env = os.environ.copy()

    env[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    env[
        "MKL_NUM_THREADS"
    ] = "1"

    env[
        "OMP_NUM_THREADS"
    ] = "1"

    env[
        "NUMBA_NUM_THREADS"
    ] = str(
        runtime.numba_threads
    )

    t0 = time.perf_counter()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None

        for line in proc.stdout:

            print(
                line,
                end="",
                flush=True,
            )

            log.write(
                line
            )

        rc = proc.wait()

    elapsed = (
        time.perf_counter()
        - t0
    )

    if rc != 0:
        raise RuntimeError(
            f"Stage '{stage.name}' "
            f"failed with exit status {rc}. "
            f"Log: {log_path}"
        )

    print(
        f"[{stage.name}] "
        f"PASS "
        f"({elapsed:.2f} s)"
    )

    return {
        "status": "PASS",
        "seconds": elapsed,
        "log": str(log_path),
    }


def run_pipeline(
    *,
    config,
    from_stage=None,
    to_stage=None,
    dry_run=False,
    force=False,
    list_stages=False,
):

    validate_stage_contract_registry()

    if list_stages:

        for i, stage in enumerate(
            STAGES,
            start=1,
        ):
            print(
                f"{i:02d}  "
                f"{stage.name}"
            )

        return 0

    cfg, config_path = load_config(
        config
    )

    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    # Determine acquisition count without loading RSLC data.
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

    runtime = build_runtime_plan(
        ndate=len(
            stack.dates
        ),
        memory_fraction=float(
            cfg_get(
                cfg,
                "runtime.memory_fraction",
                0.85,
            )
        ),
    )

    # --------------------------------------------------------
    # Immutable provenance identities for this execution.
    # --------------------------------------------------------

    config_sha256 = (
        _sha256_file(
            config_path
        )
    )

    stack_identity = (
        _build_stack_identity(
            stack=stack,
            paths=paths,
        )
    )

    first = (
        0
        if from_stage is None
        else STAGE_INDEX.get(
            from_stage,
            -1,
        )
    )

    last = (
        len(STAGES) - 1
        if to_stage is None
        else STAGE_INDEX.get(
            to_stage,
            -1,
        )
    )

    if first < 0:
        raise ValueError(
            f"Unknown from-stage: "
            f"{from_stage}"
        )

    if last < 0:
        raise ValueError(
            f"Unknown to-stage: "
            f"{to_stage}"
        )

    if first > last:
        raise ValueError(
            "from-stage occurs after "
            "to-stage"
        )

    selected = (
        STAGES[
            first:last + 1
        ]
    )

    print("=" * 96)
    print(
        "pyPSDS-GAMMA 1.0 "
        "production processing"
    )
    print("=" * 96)

    print(
        f"config        : "
        f"{config_path}"
    )

    print(
        f"output        : "
        f"{paths.output_dir}"
    )

    print(
        f"acquisitions  : "
        f"{len(stack.dates)}"
    )

    print(
        f"CPU           : "
        f"{runtime.cpu_count}"
    )

    print(
        f"usable RAM    : "
        f"{runtime.usable_memory_bytes/1024**3:.2f} GiB"
    )

    print(
        f"stages        : "
        f"{len(selected)}"
    )

    print(
        f"from          : "
        f"{selected[0].name}"
    )

    print(
        f"to            : "
        f"{selected[-1].name}"
    )

    print(
        f"dry run       : "
        f"{dry_run}"
    )

    print(
        f"force         : "
        f"{force}"
    )

    results = {}

    overall_start = (
        time.perf_counter()
    )

    # A full run has a completely resolved upstream signature
    # chain. A partial run starts with unresolved upstream
    # provenance until formal stage-output contracts are added.
    upstream_resolved = (
        first == 0
    )

    parent_signature = (
        None
        if upstream_resolved
        else
        "PARTIAL_RUN_UPSTREAM_UNRESOLVED"
    )

    for stage in selected:

        (
            stage_signature,
            script_sha256,
        ) = _build_run_stage_signature(
            stage,
            cfg=cfg,
            config_path=config_path,
            paths=paths,
            runtime=runtime,
            force=force,
            config_sha256=config_sha256,
            stack_identity=stack_identity,
            parent_signature=parent_signature,
            upstream_resolved=upstream_resolved,
        )

        result = _run_stage(
            stage,
            cfg=cfg,
            config_path=config_path,
            paths=paths,
            runtime=runtime,
            force=force,
            dry_run=dry_run,
        )

        result[
            "stage_signature"
        ] = stage_signature

        result[
            "script_sha256"
        ] = script_sha256

        result[
            "signature_scope"
        ] = (
            "full-chain"
            if upstream_resolved
            else
            "partial-run-upstream-unresolved"
        )

        # Deliberately false in signature phase A.
        #
        # Cache reuse is enabled only after every production
        # stage receives an explicit output contract.
        result[
            "cache_reuse_enabled"
        ] = False

        contract = (
            STAGE_CONTRACTS[
                stage.name
            ]
        )

        result[
            "contract"
        ] = {
            "required_inputs":
                list(
                    contract.required_inputs
                ),

            "required_outputs":
                list(
                    contract.required_outputs
                ),

            "validated":
                bool(
                    contract.validated
                ),

            "cacheable":
                bool(
                    contract.cacheable
                ),
        }

        results[
            stage.name
        ] = result

        parent_signature = (
            stage_signature
        )

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    if not dry_run:

        # ----------------------------------------------------
        # Every successful execution receives its own immutable
        # run manifest.
        #
        # Only a complete production traversal is allowed to
        # update output/manifest.json.
        #
        # This prevents a later partial/resume execution such as
        #
        #   --from-stage reference --to-stage reference
        #
        # from destroying the provenance of the most recent
        # complete production run.
        # ----------------------------------------------------

        run_id = _new_run_id()

        is_full_run = (
            first == 0
            and
            last == len(STAGES) - 1
        )

        created_utc = datetime.now(
            timezone.utc
        ).isoformat()

        run_manifest = {
            "format":
                "pyPSDS-GAMMA-run-manifest-v1",

            "software":
                "pyPSDS-GAMMA",

            "version":
                "1.0.0",

            "run_id":
                run_id,

            "created_utc":
                created_utc,

            "config":
                str(config_path),

            "config_sha256":
                config_sha256,

            "stack_identity":
                stack_identity,

            "stage_signature_scheme":
                "record-only-v1",

            "automatic_stage_cache":
                False,

            "full_run":
                bool(
                    is_full_run
                ),

            "from_stage":
                selected[0].name,

            "to_stage":
                selected[-1].name,

            "selected_stage_count":
                len(selected),

            "total_stage_count":
                len(STAGES),

            "elapsed_seconds":
                elapsed,

            "runtime":
                runtime.as_dict(),

            "results":
                results,
        }

        run_dir = (
            Path(paths.output_dir)
            / "logs"
            / "runs"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_manifest_path = (
            run_dir
            / f"run_{run_id}.json"
        )

        run_manifest_path.write_text(
            json.dumps(
                run_manifest,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            f"run manifest  : "
            f"{run_manifest_path}"
        )

        if is_full_run:

            full_manifest = dict(
                run_manifest
            )

            full_manifest[
                "format"
            ] = (
                "pyPSDS-GAMMA-full-run-manifest-v1"
            )

            manifest_path = (
                Path(paths.output_dir)
                / "manifest.json"
            )

            manifest_path.write_text(
                json.dumps(
                    full_manifest,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            print(
                f"full manifest : "
                f"{manifest_path}"
            )

        else:

            manifest_path = (
                Path(paths.output_dir)
                / "manifest.json"
            )

            if manifest_path.exists():

                print(
                    "full manifest : preserved"
                )

                print(
                    f"                "
                    f"{manifest_path}"
                )

            else:

                print(
                    "full manifest : not present "
                    "(partial run does not create it)"
                )

    print()
    print("=" * 96)
    print(
        "pyPSDS-GAMMA processing complete"
    )
    print("=" * 96)

    return 0


__all__ = [
    "STAGES",
    "list_stage_names",
    "run_pipeline",
]
