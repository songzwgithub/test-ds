from pathlib import Path

from pypsds.pipeline import (
    STAGES,
    STAGE_CONTRACTS,
    validate_stage_contract_registry,
)


EXPECTED_TAIL = [
    "atmosphere_correction",
    "scla",
    "scn",
    "final_los",
    "point_products",
]


def test_public_tail_is_packaged_and_validated():
    validate_stage_contract_registry()

    names = [stage.name for stage in STAGES]

    assert len(names) == 39
    assert names[-5:] == EXPECTED_TAIL

    stage_by_name = {
        stage.name: stage
        for stage in STAGES
    }

    stage_root = (
        Path(__file__).resolve().parents[1]
        / "pypsds"
        / "stages"
    )

    for name in EXPECTED_TAIL:
        contract = STAGE_CONTRACTS[name]

        assert contract.validated is True
        assert contract.cacheable is False
        assert contract.required_inputs
        assert contract.required_outputs

        stage = stage_by_name[name]
        assert (stage_root / stage.script).is_file()
