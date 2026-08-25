"""
Portable helpers for final LOS materialization.

Functions are copied from the frozen production implementation.
The full production orchestration is integrated separately after
numerical parity is frozen.
"""

import re

NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def read_par(path):

    result = {}

    for line in path.read_text(
        errors="ignore"
    ).splitlines():

        if ":" not in line:
            continue

        k, v = line.split(
            ":",
            1,
        )

        result[
            k.strip().lower()
        ] = v.strip()

    return result


def par_scalar(
    pars,
    keys,
):

    for key in keys:

        x = pars.get(
            key.lower()
        )

        if x is None:
            continue

        m = NUM_RE.search(x)

        if m:
            return float(
                m.group(0)
            )

    raise KeyError(
        " / ".join(keys)
    )


__all__ = [
    "read_par",
    "par_scalar",
]
