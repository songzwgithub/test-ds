from pypsds.pipeline import (
    STAGES,
    STAGE_CONTRACTS,
    validate_stage_contract_registry,
)


def test_point_geometry_pipeline_order():

    names = [
        stage.name
        for stage in STAGES
    ]

    assert len(names) == 33

    i = names.index(
        "point_geometry"
    )

    assert (
        names[i - 1]
        ==
        "timeseries_inversion"
    )

    assert (
        names[i + 1]
        ==
        "reference"
    )


def test_point_geometry_stage_contract():

    validate_stage_contract_registry()

    contract = STAGE_CONTRACTS[
        "point_geometry"
    ]

    assert contract.validated is True
    assert contract.cacheable is False

    assert contract.required_inputs == (
        "processing/network_inversion/strict_point_ids.npy",
        "processing/point_phase_stack/rows.npy",
        "processing/point_phase_stack/cols.npy",
    )

    required = set(
        contract.required_outputs
    )

    expected = {
        "processing/point_geometry/radar_row.npy",
        "processing/point_geometry/radar_col.npy",
        "processing/point_geometry/longitude_deg.npy",
        "processing/point_geometry/latitude_deg.npy",
        "processing/point_geometry/height_m.npy",
        "processing/point_geometry/incidence_rad.npy",
        "processing/point_geometry/point_geometry_manifest.json",
    }

    assert required == expected
