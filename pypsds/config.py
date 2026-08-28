from __future__ import annotations

from pathlib import Path
import re
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
            "pyPSDS-GAMMA requires schema_version: 1"
        )

    normalize_reference_contract(raw)

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


def _normalize_reference_value(
    value: Any,
    *,
    label: str,
) -> str | None:
    if value in (None, ""):
        return None

    text = str(value).strip()

    if text.lower() == "auto":
        raise ValueError(
            f"{label}=auto is not supported. "
            "Set the actual GAMMA co-registration reference "
            "acquisition explicitly as YYYYMMDD."
        )

    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(
            f"Invalid {label}={text!r}; expected YYYYMMDD."
        )

    return text


def resolve_reference_date(
    cfg: dict[str, Any],
    *,
    available_dates=None,
    required: bool = True,
) -> str | None:
    primary = _normalize_reference_value(
        cfg_get(cfg, "reference_date", None),
        label="reference_date",
    )

    geometry_legacy = _normalize_reference_value(
        cfg_get(cfg, "geometry.reference_date", None),
        label="geometry.reference_date",
    )

    phase_legacy = _normalize_reference_value(
        cfg_get(
            cfg,
            "phase_correction.geometric_reference_date",
            None,
        ),
        label="phase_correction.geometric_reference_date",
    )

    supplied = [
        x
        for x in (
            primary,
            geometry_legacy,
            phase_legacy,
        )
        if x is not None
    ]

    if not supplied:
        if not required:
            return None
        raise ValueError(
            "Required project setting is missing: reference_date. "
            "Set the actual GAMMA co-registration reference "
            "acquisition as YYYYMMDD."
        )

    unique = set(supplied)

    if len(unique) != 1:
        raise ValueError(
            "Conflicting GAMMA reference dates: "
            f"reference_date={primary!r}, "
            f"geometry.reference_date={geometry_legacy!r}, "
            "phase_correction.geometric_reference_date="
            f"{phase_legacy!r}. "
            "Use one project-level reference_date."
        )

    reference_date = supplied[0]

    if available_dates is not None:
        dates = tuple(str(x) for x in available_dates)
        if reference_date not in dates:
            raise ValueError(
                "GAMMA reference acquisition "
                f"{reference_date} is not present "
                "in the current RSLC stack. "
                f"Available dates: {dates}"
            )

    return reference_date


def normalize_reference_contract(
    cfg: dict[str, Any],
) -> str | None:
    reference_date = resolve_reference_date(
        cfg,
        required=False,
    )

    if reference_date is None:
        return None

    geometry = cfg.setdefault("geometry", {})
    phase_correction = cfg.setdefault("phase_correction", {})

    if not isinstance(geometry, dict):
        raise ValueError("geometry must be a mapping.")

    if not isinstance(phase_correction, dict):
        raise ValueError("phase_correction must be a mapping.")

    cfg["reference_date"] = reference_date
    geometry["reference_date"] = reference_date
    phase_correction["geometric_reference_date"] = reference_date

    return reference_date
