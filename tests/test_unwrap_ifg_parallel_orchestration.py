from pathlib import Path

from pypsds.stages import unwrap_all_ifgs


def test_ifg_parallel_orchestration_contract():
    source = Path(
        unwrap_all_ifgs.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "PYPSDS_UNWRAP_PAIR_PARALLEL_V1" in source
    assert "ThreadPoolExecutor" in source
    assert "as_completed" in source
    assert "ensure_pair_products" in source
    assert "resolve_ifg_workers" in source
    assert "PYPSDS_UNWRAP_IFG_WORKERS" in source
    assert "runtime.unwrap_ifg_workers" in source
    assert "parallel_precompute_done" in source

    assert "build_safe_fragment_integer_quality.py" in source
    assert "finalize_single_ifg_solution.py" in source

    assert "for idx, r in enumerate(" in source
    assert "results.append(" in source

    assert '"OPENBLAS_NUM_THREADS"' in source
    assert '"NUMBA_NUM_THREADS"' in source
