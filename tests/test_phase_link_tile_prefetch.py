from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pypsds.phase_linking.tile_prefetch import (
    OneAheadTilePrefetcher,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def test_prefetch_preserves_execution_order_and_values():

    calls = []

    def loader(
        position,
    ):
        calls.append(
            int(position)
        )

        return np.asarray(
            [
                position,
                position + 1,
            ],
            dtype=np.int32,
        )

    p = OneAheadTilePrefetcher(
        positions=(
            2,
            5,
            9,
        ),
        loader=loader,
        enabled=True,
    )

    p.start()

    a = p.get(2)
    b = p.get(5)
    c = p.get(9)

    p.close()

    assert calls == [
        2,
        5,
        9,
    ]

    assert np.array_equal(
        a,
        np.asarray(
            [
                2,
                3,
            ],
            dtype=np.int32,
        ),
    )

    assert np.array_equal(
        b,
        np.asarray(
            [
                5,
                6,
            ],
            dtype=np.int32,
        ),
    )

    assert np.array_equal(
        c,
        np.asarray(
            [
                9,
                10,
            ],
            dtype=np.int32,
        ),
    )

    assert p.completed == 3

    assert p.read_seconds >= 0.0
    assert p.wait_seconds >= 0.0
    assert p.blocking_seconds >= 0.0
    assert p.scheduler_overhead_seconds >= 0.0

    # wait_seconds is the loader-I/O portion of raw blocking time.
    assert (
        p.wait_seconds
        <=
        p.read_seconds
        +
        1.0e-12
    )

    assert (
        p.wait_seconds
        <=
        p.blocking_seconds
        +
        1.0e-12
    )


def test_prefetch_rejects_out_of_order_consumption():

    p = OneAheadTilePrefetcher(
        positions=(
            1,
            3,
        ),
        loader=lambda x: x,
        enabled=True,
    )

    p.start()

    try:
        p.get(3)

    except RuntimeError as exc:
        assert (
            "execution-order mismatch"
            in
            str(exc)
        )

    else:
        raise AssertionError(
            "out-of-order prefetch was accepted"
        )

    finally:
        p.close()


def test_disabled_prefetch_is_synchronous():

    p = OneAheadTilePrefetcher(
        positions=(),
        loader=lambda x: (
            int(x)
            *
            7
        ),
        enabled=False,
    )

    assert p.get(4) == 28

    p.close()

    assert p.completed == 1


def test_packaged_perf_default_enables_bounded_prefetch():

    cfg = yaml.safe_load(
        (
            ROOT
            /
            "pypsds"
            /
            "resources"
            /
            "default_config.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert int(
        cfg[
            "runtime"
        ][
            "phase_link_prefetch_tiles"
        ]
    ) == 1


def test_sequential_stage_contains_bounded_prefetch_path():

    text = (
        ROOT
        /
        "pypsds"
        /
        "phase_linking"
        /
        "sequential_multistage.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "OneAheadTilePrefetcher"
        in
        text
    )

    assert (
        "prefetch_tiles must be 0 or 1"
        in
        text
    )

    assert (
        "prefetch hidden I/O"
        in
        text
    )
