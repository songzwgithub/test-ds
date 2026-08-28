from pathlib import Path
import pytest
from pypsds.config import (
    load_config,
    normalize_reference_contract,
    resolve_reference_date,
)


def test_public_global_reference_date():
    cfg = {"schema_version": 1, "reference_date": "20151212"}
    assert resolve_reference_date(cfg) == "20151212"


def test_global_reference_is_projected_to_legacy_internal_keys():
    cfg = {
        "schema_version": 1,
        "reference_date": "20151212",
        "geometry": {},
        "phase_correction": {},
    }
    normalize_reference_contract(cfg)
    assert cfg["geometry"]["reference_date"] == "20151212"
    assert cfg["phase_correction"]["geometric_reference_date"] == "20151212"


def test_legacy_reference_is_migrated_in_memory():
    cfg = {
        "schema_version": 1,
        "geometry": {"reference_date": "20151212"},
        "phase_correction": {},
    }
    normalize_reference_contract(cfg)
    assert cfg["reference_date"] == "20151212"
    assert cfg["phase_correction"]["geometric_reference_date"] == "20151212"


def test_conflicting_reference_dates_fail():
    cfg = {
        "schema_version": 1,
        "reference_date": "20151212",
        "geometry": {"reference_date": "20150101"},
    }
    with pytest.raises(ValueError, match="Conflicting GAMMA reference dates"):
        normalize_reference_contract(cfg)


def test_auto_reference_is_not_allowed():
    cfg = {"schema_version": 1, "reference_date": "auto"}
    with pytest.raises(ValueError, match="auto is not supported"):
        resolve_reference_date(cfg)


def test_reference_must_exist_in_stack():
    cfg = {"schema_version": 1, "reference_date": "20151212"}
    assert resolve_reference_date(
        cfg,
        available_dates=("20141006", "20151212", "20160410"),
    ) == "20151212"
    with pytest.raises(ValueError, match="not present"):
        resolve_reference_date(
            cfg,
            available_dates=("20141006", "20160410"),
        )


def test_load_config_projects_global_reference(tmp_path: Path):
    p = tmp_path / "pypsds.yaml"
    p.write_text(
        "schema_version: 1\nreference_date: 20151212\ngeometry: {}\nphase_correction: {}\n",
        encoding="utf-8",
    )
    cfg, _ = load_config(p)
    assert cfg["reference_date"] == "20151212"
    assert cfg["geometry"]["reference_date"] == "20151212"
    assert cfg["phase_correction"]["geometric_reference_date"] == "20151212"
