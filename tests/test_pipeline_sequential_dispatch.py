from pathlib import Path

from pypsds.config import (
    cfg_get,
    load_config,
)
from pypsds.pipeline import (
    STAGES,
    _stage_args,
)
from pypsds.project import (
    ProjectPaths,
)
from pypsds.runtime import (
    build_runtime_plan,
)


CONFIG = Path(
    "pypsds/resources/default_config.yaml"
)


def _value(argv, flag):
    i = argv.index(flag)
    return argv[i + 1]


def _stage(name):
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(name)


def test_production_sequential_pipeline_dispatch(tmp_path):

    cfg, config_path = load_config(
        CONFIG
    )

    # Pure pipeline-dispatch unit test: no real RSLC dataset is required.
    # ProjectPaths is only used here to verify generated stage arguments and
    # output-product paths. Real path discovery is covered by project/doctor
    # integration checks using an actual project configuration.
    work_dir = tmp_path / "project"
    data_dir = work_dir / "data"
    rslc_dir = data_dir / "RSLC"
    output_dir = work_dir / "output"

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    rslc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rslc_tab = data_dir / "RSLC_tab"
    rslc_tab.write_text(
        "",
        encoding="utf-8",
    )

    paths = ProjectPaths(
        work_dir=work_dir,
        data_dir=data_dir,
        rslc_dir=rslc_dir,
        rslc_tab=rslc_tab,
        output_dir=output_dir,
    )

    runtime = build_runtime_plan(
        ndate=38,
        memory_fraction=float(
            cfg_get(
                cfg,
                "runtime.memory_fraction",
                0.85,
            )
        ),
    )

    names = [
        s.name
        for s in STAGES
    ]

    # Moraine KS is not a mandatory production stage.
    assert names[0] == "ds_statistics"

    i0 = names.index(
        "phase_linking"
    )

    i1 = names.index(
        "point_stack"
    )

    assert names[i0:i1 + 1] == [
        "phase_linking",
        "ds_selection",
        "ps_finalize",
        "point_stack",
    ]

    phase_args = _stage_args(
        _stage("phase_linking"),
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=False,
    )

    # Frozen sequential production semantics.
    assert float(
        _value(
            phase_args,
            "--beta",
        )
    ) == 0.0

    assert float(
        _value(
            phase_args,
            "--emi-mu",
        )
    ) == 0.99

    assert int(
        _value(
            phase_args,
            "--min-shp",
        )
    ) == 48

    assert (
        _value(
            phase_args,
            "--center-mode",
        )
        ==
        "all"
    )

    # Critical production contract:
    # sequential Phase linking must not inherit the
    # legacy full-SCM --resume flag.
    assert "--resume" not in phase_args

    ds_args = _stage_args(
        _stage("ds_selection"),
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=False,
    )

    assert float(
        _value(
            ds_args,
            "--tc-min",
        )
    ) == 0.80

    assert float(
        _value(
            ds_args,
            "--pair-min",
        )
    ) == 0.0

    assert (
        "--accept-evd"
        in ds_args
    )

    ps_args = _stage_args(
        _stage("ps_finalize"),
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=False,
    )

    assert (
        "--config"
        in ps_args
    )

    point_args = _stage_args(
        _stage("point_stack"),
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=False,
    )

    ds_mask = Path(
        _value(
            point_args,
            "--ds-mask",
        )
    )

    assert (
        ds_mask.name
        ==
        "final_ds_tc0.800_pc0.000_evd.npy"
    )

    assert (
        ds_mask.parent.name
        ==
        "processing"
    )
