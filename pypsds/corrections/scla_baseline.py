"""
Portable SCLA baseline helper core.

Project-path orchestration (`generate_one`) intentionally remains outside
this module and will be migrated with the production SCLA stage.
"""

from pathlib import Path
import hashlib
import re

import numpy as np


NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def sha256(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def parse_vector(
    text,
    labels,
):

    for line in text.splitlines():

        if not any(
            label in line
            for label in labels
        ):
            continue

        rhs = (
            line.split(
                ":",
                1,
            )[1]
            if ":" in line
            else line
        )

        x = NUM_RE.findall(
            rhs
        )

        if len(x) >= 3:

            return np.asarray(
                [
                    float(v)
                    for v in x[:3]
                ],
                dtype=np.float64,
            )

    return None


def parse_base(
    path,
):

    text = path.read_text(
        errors="ignore"
    )

    B = parse_vector(
        text,
        (
            "initial_baseline(TCN)",
            "initial_baseline",
        ),
    )

    Br = parse_vector(
        text,
        (
            "initial_baseline_rate",
            "baseline_rate(TCN)",
        ),
    )

    if (
        B is None
        or Br is None
        or not np.all(
            np.isfinite(B)
        )
        or not np.all(
            np.isfinite(Br)
        )
    ):

        raise RuntimeError(
            f"invalid baseline model: {path}"
        )

    return (
        B,
        Br,
    )


def source_path_from_entry(
    entry,
):

    if isinstance(
        entry,
        str,
    ):
        return Path(entry)

    if isinstance(
        entry,
        dict,
    ):

        for key in (
            "path",
            "base_file",
            "file",
        ):

            value = entry.get(
                key
            )

            if value:
                return Path(
                    value
                )

    return None


__all__ = [
    "sha256",
    "parse_vector",
    "parse_base",
    "source_path_from_entry",
]
