from pathlib import Path

from pypsds.config import load_config
from pypsds.pipeline import STAGES, _stage_args
from pypsds.project import ProjectPaths
from pypsds.runtime import build_runtime_plan


CONFIG = Path("config/pypsds.yaml")


def _stage(name):
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(name)


def test_reference_point_ids_dispatch(tmp_path):
    cfg, config_path = load_config(CONFIG)

    cfg["reference"]["method"] = "point_ids"
    cfg["reference"]["point_ids_path"] = "reference_ids.npy"
    cfg["reference"]["min_points"] = 100

    work = tmp_path / "project"
    data = work / "data"
    output = work / "output"
    rslc = data / "RSLC"
    tab = data / "RSLC_tab"

    rslc.mkdir(parents=True)
    output.mkdir(parents=True)
    tab.write_text("", encoding="utf-8")

    paths = ProjectPaths(
        work_dir=work,
        data_dir=data,
        rslc_dir=rslc,
        rslc_tab=tab,
        output_dir=output,
    )

    runtime = build_runtime_plan(
        ndate=38,
        memory_fraction=0.85,
    )

    args = _stage_args(
        _stage("reference"),
        cfg=cfg,
        config_path=config_path,
        paths=paths,
        runtime=runtime,
        force=False,
    )

    assert "--point-ids-file" in args
    assert "--min-points" in args
    assert "--center-row" not in args
    assert "--center-col" not in args

    i = args.index("--point-ids-file")
    assert Path(args[i + 1]) == (
        work / "reference_ids.npy"
    ).resolve()
