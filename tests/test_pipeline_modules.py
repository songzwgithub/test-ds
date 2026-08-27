from pypsds.cli import build_parser
from pypsds.modules import (
    MODULES,
    MODULE_BY_NAME,
    STAGE_TO_MODULE,
    module_for_stage,
    resolve_module_bounds,
    selected_modules,
    validate_module_registry,
)
from pypsds.pipeline import STAGES, STAGE_INDEX

EXPECTED_MODULES = (
    "data_ps",
    "shp",
    "phase_linking",
    "ps_ds",
    "network_qc",
    "unwrap",
    "timeseries",
    "corrections",
    "products",
)


def test_module_registry_exactly_covers_pipeline():
    stage_names = tuple(stage.name for stage in STAGES)
    validate_module_registry(stage_names)
    flattened = tuple(
        stage_name
        for module in MODULES
        for stage_name in module.stage_names
    )
    assert flattened == stage_names
    assert len(flattened) == len(set(flattened))


def test_public_module_sequence():
    assert tuple(m.name for m in MODULES) == EXPECTED_MODULES
    assert len(MODULES) == 9


def test_every_internal_stage_has_one_module():
    assert len(STAGE_TO_MODULE) == len(STAGES)
    for stage in STAGES:
        assert module_for_stage(stage.name) in MODULE_BY_NAME


def test_phase_linking_module_fuses_covariance_compute():
    module = MODULE_BY_NAME["phase_linking"]
    assert module.stage_names == ("phase_linking",)
    text = module.description.lower()
    assert "covariance" in text
    assert "materialized" in text


def test_module_range_resolution():
    first, last = resolve_module_bounds(
        from_module="shp",
        to_module="phase_linking",
        stage_index=STAGE_INDEX,
    )
    assert STAGES[first].name == "exact_support_cache"
    assert STAGES[last].name == "phase_linking"


def test_selected_modules_for_internal_stage_range():
    names = selected_modules((
        "network_finalize",
        "virtual_ifg_quality",
        "spatial_graph_quality",
        "unwrap_policy",
        "unwrap",
    ))
    assert names == ("network_qc", "unwrap")


def test_cli_accepts_single_module():
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--config",
        "project.yaml",
        "--module",
        "phase_linking",
        "--dry-run",
    ])
    assert args.module == "phase_linking"
    assert args.dry_run is True


def test_cli_accepts_module_range():
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--config",
        "project.yaml",
        "--from-module",
        "shp",
        "--to-module",
        "unwrap",
        "--dry-run",
    ])
    assert args.from_module == "shp"
    assert args.to_module == "unwrap"
