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
from .modules import (
    MODULES,
    resolve_module_bounds,
    selected_modules,
    validate_module_registry,
)
from .project import resolve_project_paths
from . import __version__
from .runtime_tuning import (
    ensure_runtime_profile,
    resolve_runtime_plan,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "stages"
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


STAGES = [Stage('ds_statistics', 'build_ds_statistics.py'), Stage('phase_cache', 'build_phase_cache.py'), Stage('exact_support_cache', 'build_exact_support_cache.py'), Stage('phase_linking', 'run_phase_linking.py'), Stage('ds_selection', 'select_ds.py'), Stage('ps_finalize', 'finalize_ps_geometry.py'), Stage('point_stack', 'build_point_phase_stack.py'), Stage('network_prepare', 'prepare_temporal_network.py'), Stage('network_build', 'build_temporal_network.py'), Stage('network_cycle_quality', 'assess_network_cycle_quality.py'), Stage('network_finalize', 'finalize_temporal_network.py'), Stage('virtual_ifg_quality', 'assess_virtual_ifg_quality.py'), Stage('spatial_graph_quality', 'assess_spatial_graph_quality.py'), Stage('spatial_bridge_quality', 'assess_spatial_bridge_quality.py'), Stage('spatial_component_quality', 'assess_spatial_components.py'), Stage('spatial_anchor_quality', 'assess_spatial_anchor_quality.py'), Stage('spatial_anchor_summary', 'summarize_spatial_anchor_quality.py'), Stage('spatial_local_graph_quality', 'assess_local_spatial_graph.py'), Stage('spatial_graph', 'build_spatial_graph.py'), Stage('spatial_gradient_quality', 'assess_spatial_phase_gradient.py'), Stage('unwrap_policy', 'finalize_unwrap_policy.py'), Stage('unwrap', 'unwrap_all_ifgs.py'), Stage('unwrap_severity_quality', 'assess_unwrap_severity.py'), Stage('unwrap_conflict_quality', 'assess_unwrap_conflicts.py'), Stage('unwrap_acquisition_quality', 'assess_unwrap_acquisition_quality.py'), Stage('temporal_closure', 'assess_temporal_integer_closure.py'), Stage('temporal_integer_candidate', 'build_temporal_integer_candidates.py'), Stage('temporal_candidate_spatial_quality', 'validate_temporal_integer_candidates.py'), Stage('unwrap_signature_quality', 'assess_unwrap_signature_feasibility.py'), Stage('unwrap_finalize', 'finalize_unwrap_solution.py'), Stage('timeseries_inversion', 'invert_timeseries.py'), Stage('point_geometry', 'build_point_geometry.py'), Stage('residual_ramp', 'run_residual_ramp.py'), Stage('reference', 'apply_reference.py'), Stage('atmosphere_correction', 'apply_atmosphere_correction.py'), Stage('scla', 'run_scla.py'), Stage('scn', 'run_scn.py'), Stage('final_los', 'build_final_los.py'), Stage('point_products', 'build_point_products.py')]


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

STAGE_CONTRACTS["point_geometry"] = StageContract(
    name="point_geometry",

    required_inputs=(
        "processing/network_inversion/strict_point_ids.npy",
        "processing/point_phase_stack/rows.npy",
        "processing/point_phase_stack/cols.npy",
    ),

    required_outputs=(
        "processing/point_geometry/radar_row.npy",
        "processing/point_geometry/radar_col.npy",
        "processing/point_geometry/longitude_deg.npy",
        "processing/point_geometry/latitude_deg.npy",
        "processing/point_geometry/height_m.npy",
        "processing/point_geometry/incidence_rad.npy",
        "processing/point_geometry/point_geometry_manifest.json",
    ),

    validated=True,
    cacheable=False,
)


STAGE_CONTRACTS["residual_ramp"] = StageContract(
    name="residual_ramp",

    required_inputs=(
        "processing/network_inversion/strict_point_ids.npy",
        "processing/network_inversion/"
        "acquisition_phase_l2_candidate_rad.npy",
        "processing/point_geometry/longitude_deg.npy",
        "processing/point_geometry/latitude_deg.npy",
        "processing/point_phase_stack/point_type.npy",
        "processing/point_phase_stack/temporal_coherence.npy",
    ),

    required_outputs=(
        "processing/residual_ramp/"
        "acquisition_phase_deramped_rad.npy",

        "processing/residual_ramp/"
        "ramp_coefficients_rad_per_km.npy",

        "processing/residual_ramp/"
        "anchor_strict_indices.npy",

        "processing/residual_ramp/"
        "anchor_point_ids.npy",

        "processing/residual_ramp/"
        "residual_ramp_epoch_stats.csv",

        "processing/residual_ramp/"
        "residual_ramp_manifest.json",
    ),

    validated=True,
    cacheable=False,
)


STAGE_CONTRACTS["reference"] = StageContract(
    name="reference",

    required_inputs=(
        "processing/network_inversion/strict_point_ids.npy",
        "processing/residual_ramp/"
        "acquisition_phase_deramped_rad.npy",
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


# ------------------------------------------------------------
# Validated production tail contracts
# ------------------------------------------------------------

STAGE_CONTRACTS['atmosphere_correction'] = StageContract(
    name='atmosphere_correction',
    required_inputs=(
        'processing/referenced_timeseries/acquisition_phase_referenced_rad.npy',
        'processing/referenced_timeseries/reference_strict_indices.npy',
        'processing/point_geometry/longitude_deg.npy',
        'processing/point_geometry/latitude_deg.npy',
        'processing/point_geometry/incidence_rad.npy',
    ),
    required_outputs=(
        'processing/atmosphere_correction/acquisition_phase_corrected_rad.npy',
        'processing/atmosphere_correction/atmosphere_correction_manifest.json',
    ),
    validated=True,
    cacheable=False,
)

STAGE_CONTRACTS['scla'] = StageContract(
    name='scla',
    required_inputs=(
        'processing/atmosphere_correction/acquisition_phase_corrected_rad.npy',
        'processing/network/network_manifest.json',
        'processing/network/network.itab',
        'processing/referenced_timeseries/reference_strict_indices.npy',
        'processing/point_geometry/strict_points.plist',
    ),
    required_outputs=(
        'processing/scla/acquisition_phase_pre_scn_rad.npy',
        'processing/scla/scla_manifest.json',
    ),
    validated=True,
    cacheable=False,
)

STAGE_CONTRACTS['scn'] = StageContract(
    name='scn',
    required_inputs=(
        'processing/scla/acquisition_phase_pre_scn_rad.npy',
        'processing/network/network_manifest.json',
        'processing/point_geometry/longitude_deg.npy',
        'processing/point_geometry/latitude_deg.npy',
    ),
    required_outputs=(
        'processing/scn/ph_scn_slave_rad.npy',
        'processing/scn/scn_manifest.json',
    ),
    validated=True,
    cacheable=False,
)

STAGE_CONTRACTS['final_los'] = StageContract(
    name='final_los',
    required_inputs=(
        'processing/scla/acquisition_phase_pre_scn_rad.npy',
        'processing/scn/ph_scn_slave_rad.npy',
        'processing/referenced_timeseries/reference_strict_indices.npy',
        'processing/network/network_manifest.json',
    ),
    required_outputs=(
        'processing/final_los/acquisition_phase_final_rad.npy',
        'processing/final_los/los_displacement_toward_satellite_m.npy',
        'processing/final_los/los_displacement_toward_satellite_mm.npy',
        'processing/final_los/final_los_manifest.json',
    ),
    validated=True,
    cacheable=False,
)

STAGE_CONTRACTS['point_products'] = StageContract(
    name='point_products',
    required_inputs=(
        'processing/final_los/los_displacement_toward_satellite_mm.npy',
        'processing/point_geometry/longitude_deg.npy',
        'processing/point_geometry/latitude_deg.npy',
    ),
    required_outputs=(
        'products/los_velocity_toward_satellite_mm_per_year.npy',
        'products/los_cumulative_toward_satellite_mm.npy',
        'products/linear_residual_rms_mm.npy',
        'products/velocity_slope_standard_error_mm_per_year.npy',
        'products/time_axis_contract.npz',
        'products/point_products_manifest.json',
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

def _required_cfg_value(cfg, key):
    value = cfg_get(cfg, key, None)
    if value in (None, ""):
        raise ValueError(f"Required project setting is missing: {key}")
    return value



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
                    runtime.support_cache_tile_rows,
                )
            ),

            "--tile-cols",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_tile_cols",
                    runtime.support_cache_tile_cols,
                )
            ),

            "--batch",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_batch_size",
                    runtime.support_cache_batch_size,
                )
            ),

            "--support-block",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.support_cache_support_block",
                    runtime.support_cache_support_block,
                )
            ),
        ]

        if not force:
            args.append(
                "--resume"
            )

    elif stage.name == "phase_linking":

        # Runtime scheduling is independent from scientific settings.
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
                    runtime.phase_link_batch_size,
                )
            ),

            "--pl-workers",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_workers",
                    runtime.phase_link_workers,
                )
            ),

            "--pl-chunk-size",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_chunk_size",
                    runtime.phase_link_chunk_size,
                )
            ),

            "--tile-rows",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_tile_rows",
                    runtime.phase_link_tile_rows,
                )
            ),

            "--tile-cols",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_tile_cols",
                    runtime.phase_link_tile_cols,
                )
            ),

            "--support-block",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_support_block",
                    runtime.support_cache_support_block,
                )
            ),

            "--prefetch-tiles",
            _fmt(
                cfg_get(
                    cfg,
                    "runtime.phase_link_prefetch_tiles",
                    1,
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

        method = str(
            cfg_get(
                cfg,
                "reference.method",
                "auto",
            )
        ).strip().lower()

        min_points = int(
            cfg_get(
                cfg,
                "reference.min_points",
                100,
            )
        )

        if method == "point_ids":
            value = cfg_get(
                cfg,
                "reference.point_ids_path",
                None,
            )

            if value in (
                None,
                "",
            ):
                raise ValueError(
                    "reference.method=point_ids requires "
                    "reference.point_ids_path"
                )

            point_ids_path = Path(
                value
            ).expanduser()

            if not point_ids_path.is_absolute():
                point_ids_path = (
                    Path(paths.work_dir)
                    /
                    point_ids_path
                ).resolve()

            args += [
                "--point-ids-file",
                str(
                    point_ids_path
                ),
                "--min-points",
                _fmt(
                    min_points
                ),
            ]

        elif method == "radar_window":
            center_row = cfg_get(
                cfg,
                "reference.radar_window.center_row",
                None,
            )

            center_col = cfg_get(
                cfg,
                "reference.radar_window.center_col",
                None,
            )

            if (
                center_row is None
                or
                center_col is None
            ):
                raise ValueError(
                    "reference.method=radar_window requires "
                    "reference.radar_window.center_row and center_col"
                )

            args += [
                "--center-row",
                _fmt(
                    center_row
                ),
                "--center-col",
                _fmt(
                    center_col
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
                "--min-points",
                _fmt(
                    min_points
                ),
            ]

        # PYPSDS_REFERENCE_V130_AUTO
        elif method in {"auto", "auto_stable"}:
            # No CLI reference coordinates.  The reference stage
            # performs automatic stable-region selection.
            pass

        else:
            raise ValueError(
                f"Unsupported reference.method: {method!r}"
            )

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
    # production PIPELINE PHASE-CACHE POLICY
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
    #          otherwise skip its construction and let Phase linking use the
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

    # PYPSDS_39_STAGE_VISUALIZATION_HOOK
    # Diagnostic only; never modifies scientific outputs.
    from .monitoring.stage_visualization import maybe_generate_stage_qa
    visualization_result = maybe_generate_stage_qa(
        config_path,
        stage.name,
    )

    return {
        "status": "PASS",
        "seconds": elapsed,
        "log": str(log_path),
        "visualization": visualization_result,
    }


def run_pipeline(
    *,
    config,
    module=None,
    from_module=None,
    to_module=None,
    from_stage=None,
    to_stage=None,
    dry_run=False,
    force=False,
    list_modules=False,
    list_stages=False,
):

    validate_stage_contract_registry()

    validate_module_registry(
        [stage.name for stage in STAGES]
    )

    if list_modules:
        for i, module_def in enumerate(MODULES, start=1):
            print(
                f"{i:02d}  "
                f"{module_def.name:16s} "
                f"[{len(module_def.stage_names):2d} internal stages]  "
                f"{module_def.title}"
            )
        return 0

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

    requested_cpu_raw = cfg_get(cfg, "runtime.cpu", None)
    requested_cpu = (
        None
        if requested_cpu_raw in (None, "", "auto")
        else int(requested_cpu_raw)
    )

    _runtime_strategy = str(
        cfg_get(
            cfg,
            "phase_linking.temporal.strategy",
            "full_scm",
        )
    ).strip().lower()

    if _runtime_strategy == "sequential":

        _runtime_solver_size = min(
            len(stack.dates),
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

    else:

        _runtime_solver_size = len(
            stack.dates
        )

    runtime, runtime_profile_info = resolve_runtime_plan(
        cfg,
        paths,
        ndate=len(stack.dates),
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

    module_selector_active = (
        module is not None
        or from_module is not None
        or to_module is not None
    )

    stage_selector_active = (
        from_stage is not None
        or to_stage is not None
    )

    if module is not None and (
        from_module is not None
        or to_module is not None
    ):
        raise ValueError(
            "--module cannot be combined with --from-module/--to-module"
        )

    if module_selector_active and stage_selector_active:
        raise ValueError(
            "module selectors cannot be combined with internal stage selectors"
        )

    if module is not None:
        from_module = module
        to_module = module

    if module_selector_active:
        if from_module is None:
            from_module = MODULES[0].name
        if to_module is None:
            to_module = MODULES[-1].name

        first, last = resolve_module_bounds(
            from_module=from_module,
            to_module=to_module,
            stage_index=STAGE_INDEX,
        )
    else:
        first = (
            0
            if from_stage is None
            else STAGE_INDEX.get(from_stage, -1)
        )
        last = (
            len(STAGES) - 1
            if to_stage is None
            else STAGE_INDEX.get(to_stage, -1)
        )

        if first < 0:
            raise ValueError(f"Unknown from-stage: {from_stage}")
        if last < 0:
            raise ValueError(f"Unknown to-stage: {to_stage}")
        if first > last:
            raise ValueError("from-stage occurs after to-stage")

    selected = STAGES[first:last + 1]

    selected_module_names = selected_modules(
        [stage.name for stage in selected]
    )

    print("=" * 96)
    print(
        f"pyPSDS-GAMMA {__version__} "
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
        f"PL solver dim : "
        f"{runtime.phase_link_solver_size}"
    )

    print(
        f"PL tile       : "
        f"{runtime.phase_link_tile_rows} x "
        f"{runtime.phase_link_tile_cols}"
    )

    print(
        f"PL workers    : "
        f"{runtime.phase_link_workers}"
    )

    print(
        f"PL chunk      : "
        f"{runtime.phase_link_chunk_size}"
    )

    print(
        f"PL batch      : "
        f"{runtime.phase_link_batch_size}"
    )

    print(
        "Runtime profile:",
        runtime_profile_info["status"],
    )

    print(
        f"SHP tile      : "
        f"{runtime.support_cache_tile_rows} x "
        f"{runtime.support_cache_tile_cols}"
    )

    print(
        f"SHP batch     : "
        f"{runtime.support_cache_batch_size}"
    )

    print(
        f"modules       : "
        f"{len(selected_module_names)}"
    )

    print(
        f"module from   : "
        f"{selected_module_names[0]}"
    )

    print(
        f"module to     : "
        f"{selected_module_names[-1]}"
    )

    print(
        f"internal stage: "
        f"{len(selected)}"
    )

    print(
        f"stage from    : "
        f"{selected[0].name}"
    )

    print(
        f"stage to      : "
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

        if (
            stage.name
            ==
            "phase_linking"
            and
            not dry_run
        ):
            (
                runtime,
                runtime_profile_info,
            ) = ensure_runtime_profile(
                cfg,
                config_path,
                paths,
                ndate=len(
                    stack.dates
                ),
            )

            print(
                "Phase Linking runtime profile:",
                runtime_profile_info[
                    "status"
                ],
            )

            print(
                "Phase Linking schedule:",
                {
                    "workers":
                        runtime.phase_link_workers,

                    "chunk_size":
                        runtime.phase_link_chunk_size,

                    "batch_size":
                        runtime.phase_link_batch_size,
                },
            )

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
                __version__,

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

            "selected_module_count":
                len(selected_module_names),

            "selected_modules":
                list(selected_module_names),

            "from_module":
                selected_module_names[0],

            "to_module":
                selected_module_names[-1],

            "selected_stage_count":
                len(selected),

            "total_stage_count":
                len(STAGES),

            "elapsed_seconds":
                elapsed,

            "runtime":
                runtime.as_dict(),

            "runtime_profile":
                runtime_profile_info,

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
