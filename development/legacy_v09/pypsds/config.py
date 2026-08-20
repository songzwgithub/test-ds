from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_CANDIDATES = (
    "pypsds.yaml",
    "pypsds.yml",
    "prototype.yaml",
    "prototype.yml",
)


def find_config(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        return p

    for name in _CONFIG_CANDIDATES:
        p = (Path.cwd() / name).resolve()
        if p.is_file():
            return p

    packaged = Path(__file__).resolve().parents[1] / "config" / "prototype.yaml"
    if packaged.is_file():
        return packaged

    raise FileNotFoundError("No pypsds/prototype YAML config found")


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    p = find_config(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config top level must be a mapping")
    return raw, p


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
