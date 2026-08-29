from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineModule:
    """Public production-processing module."""

    name: str
    title: str
    description: str
    stage_names: tuple[str, ...]


MODULES = (
    PipelineModule(
        name="data_ps",
        title="Data preparation and PS statistics",
        description=(
            "Read the registered stack, establish the phase source, "
            "compute amplitude statistics, validity masks and PS candidates."
        ),
        stage_names=(
            "ds_statistics",
            "phase_cache",
        ),
    ),
    PipelineModule(
        name="shp",
        title="Statistically homogeneous pixels",
        description=(
            "Identify and cache exact statistically homogeneous "
            "pixel support for DS processing."
        ),
        stage_names=(
            "exact_support_cache",
        ),
    ),
    PipelineModule(
        name="phase_linking",
        title="Covariance estimation and Phase Linking",
        description=(
            "Build coherence/covariance workspaces in bounded memory "
            "and perform sequential robust Phase Linking. Covariance "
            "estimation is fused with Phase Linking instead of being "
            "materialized as a full-scene intermediate product."
        ),
        stage_names=(
            "phase_linking",
        ),
    ),
    PipelineModule(
        name="ps_ds",
        title="PS/DS selection and merge",
        description=(
            "Select reliable DS points, finalize PS geometry and "
            "construct the unified PS/DS point phase stack."
        ),
        stage_names=(
            "ds_selection",
            "ps_finalize",
            "point_stack",
        ),
    ),
    PipelineModule(
        name="network_qc",
        title="Network, interferogram and quality control",
        description=(
            "Build and validate temporal/spatial networks, connectivity, "
            "bridges, components, anchors and phase-gradient quality before unwrapping."
        ),
        stage_names=(
            "network_prepare",
            "network_build",
            "network_cycle_quality",
            "network_finalize",
            "virtual_ifg_quality",
            "spatial_graph_quality",
            "spatial_bridge_quality",
            "spatial_component_quality",
            "spatial_anchor_quality",
            "spatial_anchor_summary",
            "spatial_local_graph_quality",
            "spatial_graph",
            "spatial_gradient_quality",
            "unwrap_policy",
        ),
    ),
    PipelineModule(
        name="unwrap",
        title="Phase unwrapping",
        description=(
            "Perform spatial/temporal unwrapping and validate severity, conflicts, "
            "acquisition consistency, integer candidates and final unwrap feasibility."
        ),
        stage_names=(
            "unwrap",
            "unwrap_severity_quality",
            "unwrap_conflict_quality",
            "unwrap_acquisition_quality",
            "temporal_closure",
            "temporal_integer_candidate",
            "temporal_candidate_spatial_quality",
            "unwrap_signature_quality",
            "unwrap_finalize",
        ),
    ),
    PipelineModule(
        name="timeseries",
        title="Time-series inversion",
        description=(
            "Invert the validated unwrapped network into acquisition-domain time series."
        ),
        stage_names=(
            "timeseries_inversion",
        ),
    ),
    PipelineModule(
        name="corrections",
        title="Reference and error corrections",
        description=(
            "Attach point geometry, remove residual spatial ramp, apply the spatial reference, optional atmospheric "
            "correction, SCLA and SCN corrections."
        ),
        stage_names=(
            "point_geometry",
            "residual_ramp",
            "reference",
            "atmosphere_correction",
            "scla",
            "scn",
        ),
    ),
    PipelineModule(
        name="products",
        title="Final LOS and products",
        description=(
            "Build the final LOS displacement solution and export point products."
        ),
        stage_names=(
            "final_los",
            "point_products",
        ),
    ),
)

MODULE_BY_NAME = {m.name: m for m in MODULES}
MODULE_INDEX = {m.name: i for i, m in enumerate(MODULES)}
STAGE_TO_MODULE = {
    stage_name: module.name
    for module in MODULES
    for stage_name in module.stage_names
}


def module_names() -> tuple[str, ...]:
    return tuple(m.name for m in MODULES)


def module_for_stage(stage_name: str) -> str:
    try:
        return STAGE_TO_MODULE[stage_name]
    except KeyError as exc:
        raise KeyError(
            f"No pipeline module owns stage: {stage_name}"
        ) from exc


def validate_module_registry(stage_names) -> None:
    expected = tuple(str(x) for x in stage_names)
    flattened = tuple(
        stage_name
        for module in MODULES
        for stage_name in module.stage_names
    )

    if flattened != expected:
        missing = [x for x in expected if x not in flattened]
        extra = [x for x in flattened if x not in expected]
        duplicates = sorted({
            x for x in flattened if flattened.count(x) > 1
        })
        raise RuntimeError(
            "Pipeline module registry does not exactly match the production stage sequence.\n"
            f"missing={missing}\n"
            f"extra={extra}\n"
            f"duplicates={duplicates}\n"
            f"expected={expected}\n"
            f"actual={flattened}"
        )


def resolve_module_bounds(
    *,
    from_module: str,
    to_module: str,
    stage_index,
) -> tuple[int, int]:
    if from_module not in MODULE_BY_NAME:
        raise ValueError(f"Unknown from-module: {from_module}")
    if to_module not in MODULE_BY_NAME:
        raise ValueError(f"Unknown to-module: {to_module}")
    if MODULE_INDEX[from_module] > MODULE_INDEX[to_module]:
        raise ValueError("from-module occurs after to-module")

    first_module = MODULE_BY_NAME[from_module]
    last_module = MODULE_BY_NAME[to_module]

    return (
        int(stage_index[first_module.stage_names[0]]),
        int(stage_index[last_module.stage_names[-1]]),
    )


def selected_modules(stage_names) -> tuple[str, ...]:
    result = []
    for stage_name in stage_names:
        module_name = module_for_stage(stage_name)
        if not result or result[-1] != module_name:
            result.append(module_name)
    return tuple(result)


__all__ = [
    "MODULES",
    "MODULE_BY_NAME",
    "MODULE_INDEX",
    "PipelineModule",
    "STAGE_TO_MODULE",
    "module_for_stage",
    "module_names",
    "resolve_module_bounds",
    "selected_modules",
    "validate_module_registry",
]
