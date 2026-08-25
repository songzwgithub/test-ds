from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import cfg_get


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    work_dir: Path
    data_dir: Path
    rslc_dir: Path
    rslc_tab: Path
    output_dir: Path


def _resolve(
    value,
    *,
    base: Path,
) -> Path | None:

    if value in (None, ""):
        return None

    p = Path(value).expanduser()

    if not p.is_absolute():
        p = base / p

    return p.resolve()


def resolve_project_paths(
    cfg: dict[str, Any],
    config_path: Path,
) -> ProjectPaths:

    config_path = Path(
        config_path
    ).expanduser().resolve()

    base = config_path.parent

    # Project working directory.
    # Explicit setting wins; otherwise use the directory
    # containing pypsds.yaml.
    work_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.work_dir",
                None,
            ),
            base=base,
        )
        or base
    )

    # Data directory.
    data_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.data_dir",
                None,
            ),
            base=work_dir,
        )
        or work_dir
    )

    # RSLC directory.
    rslc_dir = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_dir",
            None,
        ),
        base=data_dir,
    )

    if rslc_dir is None:
        for name in (
            "RSLC_cropped",
            "RSLC",
        ):
            candidate = (
                data_dir / name
            )

            if candidate.is_dir():
                rslc_dir = (
                    candidate.resolve()
                )
                break

    if (
        rslc_dir is None
        or
        not rslc_dir.is_dir()
    ):
        raise FileNotFoundError(
            "Unable to locate RSLC directory "
            f"below {data_dir}"
        )

    # RSLC_tab.
    rslc_tab = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_tab",
            None,
        ),
        base=data_dir,
    )

    if rslc_tab is None:
        direct = (
            data_dir / "RSLC_tab"
        )

        if direct.is_file():
            rslc_tab = direct.resolve()
        else:
            hits = sorted(
                data_dir.glob(
                    "*RSLC*tab*"
                )
            )

            if hits:
                rslc_tab = (
                    hits[0].resolve()
                )

    if (
        rslc_tab is None
        or
        not rslc_tab.is_file()
    ):
        raise FileNotFoundError(
            "Unable to locate RSLC_tab "
            f"below {data_dir}"
        )

    # Formal output default.
    output_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.output_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (work_dir / "output").resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return ProjectPaths(
        work_dir=work_dir,
        data_dir=data_dir,
        rslc_dir=rslc_dir,
        rslc_tab=rslc_tab,
        output_dir=output_dir,
    )


__all__ = [
    "ProjectPaths",
    "resolve_project_paths",
]
