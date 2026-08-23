from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import yaml

from pypsds.pipeline import STAGES


def test_adaptive_filter_remains_disabled_in_default_config():
    cfg = yaml.safe_load(
        Path("config/pypsds.yaml").read_text(encoding="utf-8")
    )

    af = cfg["adaptive_filter"]

    assert af["enabled"] is False
    assert af["status"] == (
        "experimental_rejected_as_default_after_P10B"
    )
    assert af["production_mode"] == (
        "raw_unfiltered_point_graph_unwrap"
    )
    assert af["preserve_unfiltered"] is True


def test_no_adaptive_filter_stage_is_registered_in_production_pipeline():
    scripts = [str(stage.script) for stage in STAGES]

    assert len(STAGES) == 32

    assert not any(
        "benchmark_adaptive_ifg_filter" in s
        for s in scripts
    )

    assert not any(
        "adaptive_ifg_filter" in s
        for s in scripts
    )


def test_packaged_policy_records_p10_rejection():
    text = (
        resources.files("pypsds.resources")
        .joinpath("ds_production_policy_v1.json")
        .read_text(encoding="utf-8")
    )

    policy = json.loads(text)
    af = policy["adaptive_interferogram_filter"]

    assert af["default_enabled"] is False
    assert af["status"] == (
        "benchmarked_rejected_as_default_after_P10B"
    )
    assert af["production_mode"] == (
        "raw_unfiltered_point_graph_unwrap"
    )
