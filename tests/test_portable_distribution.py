from __future__ import annotations
from importlib import resources
from pathlib import Path
import pytest, yaml
from pypsds.cli import main as cli_main
from pypsds.phase_linking.temporal_plan import build_temporal_plan
from pypsds.pipeline import STAGES

def test_all_production_stages_are_packaged():
    import pypsds
    stage_root = Path(pypsds.__file__).resolve().parent / "stages"
    assert len(STAGES) == 38
    for stage in STAGES:
        assert (stage_root / stage.script).is_file()

def test_default_config_has_no_study_area_paths_or_coordinates():
    text = resources.files("pypsds.resources").joinpath("default_config.yaml").read_text(encoding="utf-8")
    assert "/home/" not in text and "/mnt/" not in text and "20000101" not in text
    cfg = yaml.safe_load(text)
    assert cfg["phase_correction"]["geometric_reference_date"] is None
    assert cfg["reference"]["radar_window"]["center_row"] is None
    assert cfg["reference"]["radar_window"]["center_col"] is None

@pytest.mark.parametrize("ndate", [12, 19, 20, 37, 38, 39, 57, 83])
def test_sequential_plan_covers_arbitrary_acquisition_counts(ndate):
    dates = tuple(f"D{i:04d}" for i in range(ndate))
    plan = build_temporal_plan(dates, strategy="sequential", ministack_size=19, max_num_compressed=5, reference_index=0)
    assert plan.execution_ready
    real = [idx for stage in plan.stages for idx in stage.real_indices]
    assert real == list(range(ndate))
    assert len(real) == len(set(real))
    assert plan.max_compressed_inputs <= 5

def test_cli_init_writes_portable_config(tmp_path):
    project = tmp_path / "project"
    cli_main(["init", str(project)])
    text = (project / "pypsds.yaml").read_text(encoding="utf-8")
    assert "/home/" not in text and "/mnt/" not in text
