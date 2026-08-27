from __future__ import annotations

from pathlib import Path

import pypsds

from pypsds.modules import MODULES
from pypsds.pipeline import STAGES


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def _read(
    rel,
):
    return (
        ROOT
        /
        rel
    ).read_text(
        encoding="utf-8"
    )


def test_v120_release_topology_is_frozen():

    assert pypsds.__version__ == "1.2.0"
    assert len(MODULES) == 9
    assert len(STAGES) == 38


def test_v120_production_optimizations_are_present():

    coherence = _read(
        "pypsds/phase_linking/coherence.py"
    )

    emi = _read(
        "pypsds/phase_linking/emi.py"
    )

    seq = _read(
        "pypsds/phase_linking/sequential_multistage.py"
    )

    phase_source = _read(
        "pypsds/phase_linking/phase_source.py"
    )

    production = _read(
        "pypsds/phase_linking/sequential_production.py"
    )

    assert (
        "def compressed_coherence_all_pairs("
        in coherence
    )

    assert (
        "def temporal_coherence_fused("
        in emi
    )

    assert (
        "tc = temporal_coherence_fused("
        in seq
    )

    assert (
        "def _cache_compose_temporal_cell("
        in phase_source
    )

    assert (
        "def configure_sequential_temporal_cache("
        in phase_source
    )

    assert (
        "clear_stage_cache=False"
        in production
    )


def test_v120_release_has_no_temporary_emi_profiler():

    text = _read(
        "pypsds/phase_linking/emi_threshold.py"
    )

    assert "_PYPSDS_EMI_PROFILE_V1" not in text
    assert "PYPSDS_PROFILE_EMI" not in text


def test_v120_release_has_no_unused_fullspan_fused_experiment():

    assert not (
        ROOT
        /
        "pypsds"
        /
        "phase_linking"
        /
        "fullspan_quality_kernel.py"
    ).exists()

    assert not (
        ROOT
        /
        "tests"
        /
        "test_fullspan_quality_fused_kernel.py"
    ).exists()

    fullspan = _read(
        "pypsds/phase_linking/fullspan_quality.py"
    )

    assert (
        "fullspan_quality_fused_kernel"
        not in fullspan
    )
