from __future__ import annotations

import inspect
from pathlib import Path

from pypsds.phase_linking import (
    sequential_production,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def test_gamma_post_phase_fused_has_one_phase_read_site():

    src = inspect.getsource(
        sequential_production
        ._run_gamma_post_phase_fused
    )

    assert (
        src.count(
            "yxt.phase_source.read_tile("
        )
        ==
        1
    )


def test_gamma_post_phase_fused_contains_three_consumers():

    src = inspect.getsource(
        sequential_production
        ._run_gamma_post_phase_fused
    )

    assert (
        "evaluate_fullspan_quality_points("
        in
        src
    )

    assert (
        "run_full_scm_points("
        in
        src
    )

    assert (
        "stage_index=-2"
        in
        src
    )


def test_gamma_post_phase_fused_releases_each_band():

    src = inspect.getsource(
        sequential_production
        ._run_gamma_post_phase_fused
    )

    assert (
        "del phase_tile"
        in
        src
    )


def test_production_routes_proxy_to_fusion():

    text = (
        ROOT
        /
        "pypsds"
        /
        "phase_linking"
        /
        "sequential_production.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "gamma_post_fusion = bool("
        in
        text
    )

    assert (
        "post-PL orchestration  : fused GAMMA row-band"
        in
        text
    )


def test_non_gamma_legacy_path_is_retained():

    text = (
        ROOT
        /
        "pypsds"
        /
        "phase_linking"
        /
        "sequential_production.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "production ROW-BAND STREAMING FULL-SPAN quality"
        in
        text
    )
