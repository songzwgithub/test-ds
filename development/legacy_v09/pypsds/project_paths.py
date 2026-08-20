from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    work_dir: Path
    data_dir: Path
    rslc_dir: Path
    rslc_tab: Path
    output_dir: Path


def _resolve(raw: str | Path | None, base: Path) -> Path | None:
    if raw in (None, ""):
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def resolve_project_paths(cfg: dict[str, Any], config_path: Path | None = None) -> ProjectPaths:
    p = cfg.get("paths", {}) or {}
    base = config_path.parent if config_path is not None else Path.cwd()

    # Automatic discovery is anchored to the YAML location, never to shell cwd.
    default_work_dir = config_path.parent.resolve() if config_path is not None else Path.cwd().resolve()
    work_dir = _resolve(p.get("work_dir"), base) or default_work_dir
    data_dir = _resolve(p.get("data_dir"), work_dir) or work_dir.parent.resolve()

    rslc_dir = _resolve(p.get("rslc_dir"), data_dir)
    if rslc_dir is None:
        for name in ("RSLC_cropped", "RSLC"):
            candidate = data_dir / name
            if candidate.is_dir():
                rslc_dir = candidate.resolve()
                break
    if rslc_dir is None or not rslc_dir.is_dir():
        raise FileNotFoundError(f"Unable to discover RSLC directory below {data_dir}")

    rslc_tab = _resolve(p.get("rslc_tab"), data_dir)
    if rslc_tab is None:
        direct = data_dir / "RSLC_tab"
        if direct.is_file():
            rslc_tab = direct.resolve()
        else:
            hits = sorted(data_dir.glob("*RSLC*tab*"))
            if hits:
                rslc_tab = hits[0].resolve()
    if rslc_tab is None or not rslc_tab.is_file():
        raise FileNotFoundError(f"Unable to discover RSLC_tab below {data_dir}")

    output_dir = _resolve(p.get("output_dir"), work_dir) or (work_dir / "prototype_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        work_dir=work_dir,
        data_dir=data_dir,
        rslc_dir=rslc_dir,
        rslc_tab=rslc_tab,
        output_dir=output_dir,
    )
