from __future__ import annotations

from pathlib import Path


class GammaParError(ValueError):
    """Invalid or incomplete GAMMA parameter file."""


def read_gamma_par(path: str | Path) -> dict[str, str]:
    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(p)

    values = {}

    for line in p.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        values[key.strip().lower()] = value.strip()

    return values


def gamma_par_scalar(
    values: dict[str, str],
    key: str,
) -> float:

    key = key.lower()

    if key not in values:
        raise GammaParError(
            f"Missing GAMMA parameter: {key}"
        )

    raw = values[key]

    try:
        return float(raw.split()[0])

    except (ValueError, IndexError) as exc:
        raise GammaParError(
            f"Invalid GAMMA parameter {key!r}: {raw!r}"
        ) from exc


def gamma_par_int(
    values: dict[str, str],
    key: str,
) -> int:

    value = gamma_par_scalar(
        values,
        key,
    )

    rounded = int(round(value))

    if abs(value - rounded) > 1.0e-9:
        raise GammaParError(
            f"GAMMA parameter {key!r} "
            f"is not integral: {value}"
        )

    return rounded
