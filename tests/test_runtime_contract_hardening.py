from __future__ import annotations

import inspect
import time

import pytest
import yaml

import pypsds.gamma.phase_correction as phase_correction
import pypsds.phase_linking.phase_source as phase_source
from pypsds.gamma.phase_correction import PhaseCorrectionError
from pypsds.phase_linking.tile_prefetch import OneAheadTilePrefetcher
from pypsds.phase_linking.sequential_multistage import run_sequential_stage
from pypsds.phase_linking.sequential_plan_executor import run_sequential_plan


class _Stack:
    dates = ("20200101", "20200113", "20200125")


def _cfg(cpu="auto"):
    return {
        "runtime": {"cpu": cpu},
        "phase_correction": {
            "commands": {"phase_sim_orb_pt": "phase_sim_orb_pt"}
        },
    }


def test_runtime_cpu_caps_gamma(monkeypatch):
    monkeypatch.setattr(phase_source, "logical_cpu_count", lambda: 32)
    assert phase_source._effective_runtime_cpu_count(_cfg(16)) == 16
    assert phase_source._effective_runtime_cpu_count(_cfg("auto")) == 32


def test_autotune_cannot_change_canonical_grid(monkeypatch):
    monkeypatch.setattr(phase_source, "logical_cpu_count", lambda: 32)
    monkeypatch.setattr(phase_source, "_cpu_model_name", lambda: "test-cpu")
    identity = phase_source.canonical_autotune_runtime_identity(
        _cfg(32), _Stack(), phase_sim_path=None
    )
    tune = {
        "format": phase_source._CANONICAL_AUTOTUNE_FORMAT,
        "canonical_tile": [256, 512],
        "runtime_identity": identity,
        "winner": {"parity": True, "spatial_workers": 6, "pair_workers": 3},
    }
    with pytest.raises(ValueError, match="fixed at 128x256"):
        phase_source._validated_canonical_autotune(
            tune, _cfg(32), _Stack(), phase_sim_path=None
        )


def test_autotune_rejects_stale_runtime_identity(monkeypatch):
    monkeypatch.setattr(phase_source, "logical_cpu_count", lambda: 32)
    monkeypatch.setattr(phase_source, "_cpu_model_name", lambda: "test-cpu")
    identity = phase_source.canonical_autotune_runtime_identity(
        _cfg(32), _Stack(), phase_sim_path=None
    )
    identity["effective_cpu_count"] = 16
    tune = {
        "format": phase_source._CANONICAL_AUTOTUNE_FORMAT,
        "canonical_tile": [128, 256],
        "runtime_identity": identity,
        "winner": {"parity": True, "spatial_workers": 6, "pair_workers": 3},
    }
    with pytest.raises(ValueError, match="identity is stale"):
        phase_source._validated_canonical_autotune(
            tune, _cfg(32), _Stack(), phase_sim_path=None
        )


def test_gamma_retry_then_success(monkeypatch, tmp_path):
    calls = []
    def fake_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PhaseCorrectionError("transient")
        return 0.01
    monkeypatch.setattr(phase_correction, "_run_command_once", fake_once)
    monkeypatch.setenv("PYPSDS_GAMMA_COMMAND_RETRIES", "2")
    monkeypatch.setenv("PYPSDS_GAMMA_RETRY_BACKOFF_SECONDS", "0")
    phase_correction._run_command(
        ["fake"], log_file=tmp_path / "gamma.log", label="test"
    )
    assert len(calls) == 2


def test_gamma_retry_exhausted(monkeypatch, tmp_path):
    calls = []
    def fake_once(*args, **kwargs):
        calls.append(1)
        raise PhaseCorrectionError("persistent")
    monkeypatch.setattr(phase_correction, "_run_command_once", fake_once)
    monkeypatch.setenv("PYPSDS_GAMMA_COMMAND_RETRIES", "2")
    monkeypatch.setenv("PYPSDS_GAMMA_RETRY_BACKOFF_SECONDS", "0")
    with pytest.raises(PhaseCorrectionError, match="persistent"):
        phase_correction._run_command(
            ["fake"], log_file=tmp_path / "gamma.log", label="test"
        )
    assert len(calls) == 3


def test_prefetch_timeout_is_fail_fast(monkeypatch):
    monkeypatch.setenv("PYPSDS_PREFETCH_FUTURE_TIMEOUT_SECONDS", "0.02")
    def loader(_position):
        time.sleep(0.5)
        return 1
    p = OneAheadTilePrefetcher(positions=(1,), loader=loader, enabled=True)
    p.start()
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="timed out"):
        p.get(1)
    assert time.perf_counter() - t0 < 0.25
    assert p.abandoned == 1
    t1 = time.perf_counter()
    p.close()
    assert time.perf_counter() - t1 < 0.1


def test_runtime_default_is_conservative_and_scientific_chain_is_preserved():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "pypsds" / "resources" / "default_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    # Runtime default is conservative, while the established public
    # scientific correction chain remains unchanged.
    assert cfg["runtime"]["phase_link_prefetch_tiles"] == 0
    assert cfg["corrections"]["atmosphere"]["mode"] == "gacos"
    assert cfg["corrections"]["scla"]["mode"] == "stamps"
    assert cfg["corrections"]["scn"]["mode"] == "stamps"

def test_direct_sequential_api_prefetch_is_opt_in():
    assert (
        inspect.signature(run_sequential_stage)
        .parameters["prefetch_tiles"]
        .default
        == 0
    )
    assert (
        inspect.signature(run_sequential_plan)
        .parameters["prefetch_tiles"]
        .default
        == 0
    )


def test_manual_gamma_schedule_is_hard_capped(monkeypatch):
    monkeypatch.setattr(
        phase_source,
        "logical_cpu_count",
        lambda: 32,
    )

    spatial, pair, cpu = (
        phase_source._bounded_sync_gamma_schedule(
            _cfg(16),
            spatial_workers=6,
            pair_workers=3,
            n_pairs=74,
        )
    )

    assert cpu == 16
    assert pair == 3
    assert spatial == 5
    assert spatial * pair <= cpu


def test_autotune_over_runtime_cpu_is_rejected(monkeypatch):
    monkeypatch.setattr(
        phase_source,
        "logical_cpu_count",
        lambda: 32,
    )
    monkeypatch.setattr(
        phase_source,
        "_cpu_model_name",
        lambda: "test-cpu",
    )

    identity = phase_source.canonical_autotune_runtime_identity(
        _cfg(16),
        _Stack(),
        phase_sim_path=None,
    )

    tune = {
        "format": phase_source._CANONICAL_AUTOTUNE_FORMAT,
        "canonical_tile": [128, 256],
        "runtime_identity": identity,
        "parity_reference": {
            "spatial_workers": 1,
            "pair_workers": 1,
        },
        "winner": {
            "parity": True,
            "spatial_workers": 6,
            "pair_workers": 3,
        },
    }

    with pytest.raises(ValueError, match="exceeds runtime.cpu"):
        phase_source._validated_canonical_autotune(
            tune,
            _cfg(16),
            _Stack(),
            phase_sim_path=None,
        )


def test_benchmark_source_uses_serial_parity_reference_and_cpu_filter():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (
        root
        / "tools"
        / "benchmark_canonical_phase_parallel.py"
    ).read_text(encoding="utf-8")

    assert "def filter_candidates_by_cpu(" in text
    assert "Serial parity reference: 1x1" in text
    assert "serial_reference_by_mode[mode]" in text
    assert '"parity_reference": {' in text
    assert "--install-winner requires full-stack dates" in text



def test_autotune_requires_serial_parity_reference(monkeypatch):
    monkeypatch.setattr(
        phase_source,
        "logical_cpu_count",
        lambda: 32,
    )
    monkeypatch.setattr(
        phase_source,
        "_cpu_model_name",
        lambda: "test-cpu",
    )

    identity = phase_source.canonical_autotune_runtime_identity(
        _cfg(32),
        _Stack(),
        phase_sim_path=None,
    )

    tune = {
        "format": phase_source._CANONICAL_AUTOTUNE_FORMAT,
        "canonical_tile": [128, 256],
        "runtime_identity": identity,
        "winner": {
            "parity": True,
            "spatial_workers": 6,
            "pair_workers": 3,
        },
    }

    with pytest.raises(ValueError, match="1x1 serial"):
        phase_source._validated_canonical_autotune(
            tune,
            _cfg(32),
            _Stack(),
            phase_sim_path=None,
        )
