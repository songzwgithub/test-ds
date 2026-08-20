from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .par import GammaInputError, first_int, parse_gamma_parameter_file

WIDTH_KEYS = ("interferogram_width", "range_samp_1", "range_samples", "width")
LENGTH_KEYS = ("interferogram_azimuth_lines", "az_samp_1", "azimuth_lines", "nlines")
FORMAT_KEYS = ("image_format", "data_type", "data_format")


@dataclass(frozen=True, slots=True)
class GammaRaster:
    path: Path
    par: Path
    width: int
    length: int
    dtype_name: str
    byte_order: str

    @property
    def bytes_per_pixel(self) -> int:
        return 8 if self.dtype_name == "fcomplex" else 4


def _normalize_dtype(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "fcomplex": "fcomplex",
        "complex": "fcomplex",
        "complex*8": "fcomplex",
        "scomplex": "scomplex",
        "short_complex": "scomplex",
    }
    if n not in aliases:
        raise GammaInputError(f"Unsupported RSLC dtype: {name}")
    return aliases[n]


def _format_from_par(params: dict[str, list[str]]) -> str | None:
    for key in FORMAT_KEYS:
        vals = params.get(key, [])
        if vals:
            try:
                return _normalize_dtype(vals[0])
            except GammaInputError:
                pass
    return None


def resolve_gamma_raster(
    path: str | Path,
    par: str | Path,
    *,
    dtype: str = "auto",
    byte_order: Literal["big", "little", "native"] = "big",
) -> GammaRaster:
    p = Path(path).expanduser().resolve()
    q = Path(par).expanduser().resolve()
    params = parse_gamma_parameter_file(q)
    width = first_int(params, WIDTH_KEYS)
    length = first_int(params, LENGTH_KEYS)
    if width is None or length is None:
        raise GammaInputError(f"Cannot determine width/length from {q}")

    if dtype == "auto":
        resolved = _format_from_par(params)
        if resolved is None:
            size = p.stat().st_size
            expected_f = width * length * 8
            expected_s = width * length * 4
            if size == expected_f:
                resolved = "fcomplex"
            elif size == expected_s:
                resolved = "scomplex"
            else:
                raise GammaInputError(
                    f"Cannot infer RSLC dtype from file size: {p} size={size}, "
                    f"expected fcomplex={expected_f}, scomplex={expected_s}"
                )
    else:
        resolved = _normalize_dtype(dtype)

    bpp = 8 if resolved == "fcomplex" else 4
    expected = width * length * bpp
    actual = p.stat().st_size
    if actual != expected:
        raise GammaInputError(
            f"RSLC size mismatch: {p.name}: {actual} bytes != {expected} "
            f"({width}x{length}, {resolved})"
        )
    return GammaRaster(p, q, width, length, resolved, byte_order)


def _endian_prefix(order: str) -> str:
    return {"big": ">", "little": "<", "native": "="}[order]


def read_complex_window(
    raster: GammaRaster,
    *,
    row0: int = 0,
    col0: int = 0,
    rows: int | None = None,
    cols: int | None = None,
) -> np.ndarray:
    if rows is None:
        rows = raster.length - row0
    if cols is None:
        cols = raster.width - col0
    if row0 < 0 or col0 < 0 or rows <= 0 or cols <= 0:
        raise ValueError("Invalid window")
    if row0 + rows > raster.length or col0 + cols > raster.width:
        raise ValueError(
            f"Window {row0}:{row0+rows}, {col0}:{col0+cols} exceeds "
            f"{raster.length}x{raster.width}"
        )

    endian = _endian_prefix(raster.byte_order)
    if raster.dtype_name == "fcomplex":
        mm = np.memmap(
            raster.path,
            dtype=np.dtype(endian + "c8"),
            mode="r",
            shape=(raster.length, raster.width),
            order="C",
        )
        out = np.asarray(mm[row0:row0+rows, col0:col0+cols]).astype(np.complex64, copy=True)
    else:
        mm = np.memmap(
            raster.path,
            dtype=np.dtype(endian + "i2"),
            mode="r",
            shape=(raster.length, raster.width, 2),
            order="C",
        )
        raw = mm[row0:row0+rows, col0:col0+cols, :]
        real = np.asarray(raw[..., 0], dtype=np.float32)
        imag = np.asarray(raw[..., 1], dtype=np.float32)
        out = (real + 1j * imag).astype(np.complex64, copy=False)
    del mm
    return out
