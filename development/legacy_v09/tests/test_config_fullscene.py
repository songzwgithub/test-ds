from pathlib import Path
import yaml


def test_example_config_is_fullscene():
    p = Path(__file__).resolve().parents[1] / "config" / "pypsds_v09.yaml"
    cfg = yaml.safe_load(p.read_text())
    assert cfg["prototype"]["roi"]["rows"] is None
    assert cfg["prototype"]["roi"]["cols"] is None
    assert cfg["phase_correction"]["apply_sign"] == 1.0
