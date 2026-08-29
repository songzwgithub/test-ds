from pathlib import Path

from pypsds.pipeline import (
    STAGE_CONTRACTS,
)


def test_residual_ramp_contract_contains_files_not_directory():
    contract = STAGE_CONTRACTS[
        "residual_ramp"
    ]

    assert (
        "processing/residual_ramp/ifgs"
        not in
        contract.required_outputs
    )

    required = set(
        contract.required_outputs
    )

    assert (
        "processing/residual_ramp/"
        "residual_ramp_manifest.json"
        in required
    )

    assert (
        "processing/residual_ramp/"
        "ifg_ramp_projected_coefficients_rad_per_km.npy"
        in required
    )

    assert (
        "processing/residual_ramp/"
        "acquisition_ramp_coefficients_rad_per_km.npy"
        in required
    )


def test_freeze_has_residual_ramp_network_fileset():
    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    source = (
        root
        / "tools"
        / "freeze_stage_output_contracts.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "RESIDUAL_RAMP_FILESETS" in source
    assert "processing/residual_ramp/ifgs/" in source
    assert "pair*_*_*_unwrapped_phase_rad.npy" in source
    assert 'if name == "residual_ramp"' in source
    assert '"expected_count":' in source
    assert "edge_count" in source
