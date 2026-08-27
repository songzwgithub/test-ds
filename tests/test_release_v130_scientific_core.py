from pathlib import Path

import pypsds
from pypsds.modules import MODULES
from pypsds.pipeline import STAGES

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_release_identity_and_topology():
    assert pypsds.__version__ == "1.3.0"
    assert len(MODULES) == 9
    assert len(STAGES) == 38


def test_frozen_phase_linking_features_remain():
    assert "def compressed_coherence_all_pairs(" in read(
        "pypsds/phase_linking/coherence.py"
    )
    assert "def temporal_coherence_fused(" in read(
        "pypsds/phase_linking/emi.py"
    )
    assert "tc = temporal_coherence_fused(" in read(
        "pypsds/phase_linking/sequential_multistage.py"
    )
    assert "def _cache_compose_temporal_cell(" in read(
        "pypsds/phase_linking/phase_source.py"
    )
    assert "def configure_sequential_temporal_cache(" in read(
        "pypsds/phase_linking/phase_source.py"
    )


def test_no_rejected_or_profiler_path():
    text = read("pypsds/phase_linking/emi_threshold.py")
    assert "_PYPSDS_EMI_PROFILE_V1" not in text
    assert "PYPSDS_PROFILE_EMI" not in text
    assert not (
        ROOT / "pypsds/phase_linking/fullspan_quality_kernel.py"
    ).exists()
