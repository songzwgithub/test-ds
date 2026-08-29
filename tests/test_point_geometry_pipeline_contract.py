from pypsds.pipeline import (
    STAGES,
    STAGE_CONTRACTS,
    validate_stage_contract_registry,
)


def test_point_geometry_pipeline_order():
    names = [stage.name for stage in STAGES]

    assert len(names) == 39

    i = names.index("point_geometry")

    assert names[i - 1] == "unwrap_finalize"
    assert names[i + 1] == "residual_ramp"
    assert names[i + 2] == "timeseries_inversion"
    assert names[i + 3] == "reference"


def test_point_geometry_stage_contract():
    validate_stage_contract_registry()

    contract = STAGE_CONTRACTS["point_geometry"]

    assert contract.validated is True
    assert contract.cacheable is False

    assert contract.required_inputs == (
        "processing/final_unwrap/strict_point_ids.npy",
        "processing/point_phase_stack/rows.npy",
        "processing/point_phase_stack/cols.npy",
    )
