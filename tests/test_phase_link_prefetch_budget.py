from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile

import pytest

from pypsds.gamma.phase_correction import (
    PhaseCorrectionError,
    _run_command,
)

from pypsds.phase_linking.phase_source import (
    bounded_prefetch_gamma_parallelism,
)


def test_prefetch_budget_current_32_cpu_case():

    out = bounded_prefetch_gamma_parallelism(
        cpu_count=32,
        pl_workers=8,
        spatial_workers=8,
        pair_workers=4,
        reserve_cpus=4,
    )

    assert out["gamma_process_budget"] == 20

    assert (
        out["max_gamma_processes"]
        <=
        out["gamma_process_budget"]
    )

    assert out["spatial_workers"] == 5
    assert out["pair_workers"] == 4


@pytest.mark.parametrize(
    (
        "cpu",
        "pl",
        "spatial",
        "pair",
    ),
    [
        (1, 1, 8, 4),
        (4, 2, 8, 4),
        (8, 4, 8, 4),
        (16, 8, 8, 4),
        (64, 16, 16, 8),
    ],
)
def test_prefetch_budget_never_oversubscribes(
    cpu,
    pl,
    spatial,
    pair,
):

    out = bounded_prefetch_gamma_parallelism(
        cpu_count=cpu,
        pl_workers=pl,
        spatial_workers=spatial,
        pair_workers=pair,
        reserve_cpus=4,
    )

    assert (
        out["max_gamma_processes"]
        <=
        out["gamma_process_budget"]
    )

    assert (
        out["max_gamma_processes"]
        >=
        1
    )


def test_gamma_command_timeout_fail_fast():

    old = os.environ.get(
        "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS"
    )

    os.environ[
        "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS"
    ] = "0.05"

    try:
        with tempfile.TemporaryDirectory() as td:

            log_path = (
                Path(td)
                /
                "gamma_timeout.log"
            )

            with pytest.raises(
                PhaseCorrectionError,
                match="timeout",
            ):
                _run_command(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import time; "
                            "time.sleep(2)"
                        ),
                    ],
                    log_file=log_path,
                    label="timeout-test",
                )

            assert log_path.is_file()

            text = log_path.read_text(
                encoding="utf-8"
            )

            assert "TIMEOUT" in text

    finally:
        if old is None:
            os.environ.pop(
                "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS",
                None,
            )

        else:
            os.environ[
                "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS"
            ] = old
