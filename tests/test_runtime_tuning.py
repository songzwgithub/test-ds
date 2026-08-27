from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from pypsds.runtime import RuntimePlan
from pypsds.runtime_tuning import (
    _apply_profile,
    _candidate_chunk_sizes,
    _candidate_worker_counts,
    runtime_signature,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return RuntimePlan(
        cpu_count=32,
        available_memory_bytes=64 * 1024**3,
        usable_memory_bytes=52 * 1024**3,
        io_workers=8,
        tile_rows=128,
        tile_cols=256,
        phase_link_tile_rows=384,
        phase_link_tile_cols=768,
        support_cache_tile_rows=768,
        support_cache_tile_cols=1536,
        support_cache_batch_size=49152,
        support_cache_support_block=1024,
        phase_link_workers=32,
        phase_link_chunk_size=2048,
        phase_link_batch_size=65536,
        phase_link_solver_size=24,
        blas_threads=1,
        numba_threads=32,
    )


def test_autotune_candidates_are_dynamic_and_bounded():
    assert _candidate_worker_counts(32) == (
        1, 2, 4, 8, 16, 32
    )
    chunks = _candidate_chunk_sizes(2048, 8192)
    assert 512 in chunks
    assert 2048 in chunks
    assert max(chunks) <= 2048


def test_runtime_profile_cannot_exceed_safe_plan():
    safe = _plan()
    profile = {
        "selected_schedule": {
            "workers": 64,
            "chunk_size": 4096,
            "batch_size": 131072,
        }
    }
    tuned = _apply_profile(safe, profile)
    assert tuned.phase_link_workers <= safe.phase_link_workers
    assert tuned.phase_link_chunk_size <= safe.phase_link_chunk_size
    assert tuned.phase_link_batch_size <= safe.phase_link_batch_size


def test_runtime_signature_tracks_solver_dimension():
    a = _plan()
    b = replace(a, phase_link_solver_size=19)
    assert runtime_signature(a)["sha256"] != runtime_signature(b)["sha256"]


def test_packaged_default_enables_runtime_autotune():
    cfg = yaml.safe_load(
        (
            ROOT
            / "pypsds"
            / "resources"
            / "default_config.yaml"
        ).read_text(encoding="utf-8")
    )
    a = cfg["runtime"]["autotune"]
    assert a["enabled"] is True
    assert int(a["sample_points"]) >= 256
    assert int(a["repeats"]) >= 1


def test_pipeline_has_no_stale_release_version_literals():
    text = (
        ROOT / "pypsds" / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert '"pyPSDS-GAMMA 1.1 "' not in text
    assert (
        '"version":\n'
        '                "1.0.0"'
        not in text
    )
    assert "__version__" in text


def test_runtime_profile_identity_is_process_stable():
    text = (
        ROOT
        / "pypsds"
        / "runtime_tuning.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "def _blas_identity(" not in text
    assert "_blas_identity()" not in text

    assert (
        '"numpy_build": _numpy_build_identity()'
        in text
    )


def test_runtime_autotune_enforces_single_thread_blas():
    text = (
        ROOT
        / "pypsds"
        / "runtime_tuning.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "threadpool_limits(" in text
    assert "limits=1" in text
