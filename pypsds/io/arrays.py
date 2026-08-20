from __future__ import annotations

from pathlib import Path

import numpy as np


def create_array(
    path,
    *,
    shape,
    dtype,
    fill=None,
):

    p = Path(path)

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arr = (
        np.lib.format.open_memmap(
            p,
            mode="w+",
            dtype=dtype,
            shape=tuple(shape),
        )
    )

    if fill is not None:
        arr[...] = fill
        arr.flush()

    return arr


def open_array(
    path,
    *,
    mode="r",
):

    return np.load(
        Path(path),
        mmap_mode=mode,
        allow_pickle=False,
    )


def require_array(
    path,
    *,
    shape=None,
    dtype=None,
    mode="r",
):

    arr = open_array(
        path,
        mode=mode,
    )

    if (
        shape is not None
        and
        tuple(arr.shape)
        != tuple(shape)
    ):
        raise RuntimeError(
            f"{path}: shape={arr.shape}, "
            f"expected={tuple(shape)}"
        )

    if (
        dtype is not None
        and
        arr.dtype
        != np.dtype(dtype)
    ):
        raise RuntimeError(
            f"{path}: dtype={arr.dtype}, "
            f"expected={np.dtype(dtype)}"
        )

    return arr
