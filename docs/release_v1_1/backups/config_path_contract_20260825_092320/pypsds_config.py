from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_NAMES = (
    "pypsds.yaml",
    "pypsds.yml",
)


def find_config(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path).expanduser().resolve()

        if not p.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {p}"
            )

        return p

    for name in CONFIG_NAMES:
        p = (Path.cwd() / name).resolve()

        if p.is_file():
            return p

    raise FileNotFoundError(
        "No pypsds.yaml configuration file was found."
    )


def load_config(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:

    p = find_config(path)

    raw = yaml.safe_load(
        p.read_text(encoding="utf-8")
    ) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            "Configuration top level must be a mapping."
        )

    schema = raw.get(
        "schema_version",
        None,
    )

    if schema != 1:
        raise ValueError(
            f"Unsupported schema_version={schema!r}; "
            "pyPSDS-GAMMA v1.0 requires schema_version: 1"
        )

    return raw, p


def cfg_get(
    cfg: dict[str, Any],
    dotted: str,
    default: Any = None,
) -> Any:

    value: Any = cfg

    for part in dotted.split("."):

        if (
            not isinstance(value, dict)
            or
            part not in value
        ):
            return default

        value = value[part]

    return value
