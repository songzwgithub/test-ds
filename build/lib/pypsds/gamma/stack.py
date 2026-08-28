from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .binary import GammaRaster, read_complex_window, resolve_gamma_raster
from .par import GammaInputError, RslcRecord, parse_rslc_tab


@dataclass(slots=True)
class GammaStack:
    records: list[RslcRecord]
    rasters: list[GammaRaster]
    io_workers: int = 4

    @classmethod
    def from_rslc_tab(
        cls,
        rslc_tab: str | Path,
        *,
        rslc_dir: str | Path | None = None,
        dtype: str = "auto",
        byte_order: str = "big",
        io_workers: int = 4,
    ) -> "GammaStack":
        records = parse_rslc_tab(rslc_tab, rslc_dir=rslc_dir)
        rasters = [
            resolve_gamma_raster(r.rslc, r.par, dtype=dtype, byte_order=byte_order)
            for r in records
        ]
        dims = {(r.length, r.width) for r in rasters}
        if len(dims) != 1:
            raise GammaInputError(f"RSLC dimensions differ: {sorted(dims)}")
        return cls(records, rasters, max(1, int(io_workers)))

    @property
    def dates(self) -> list[str]:
        return [r.date for r in self.records]

    @property
    def shape(self) -> tuple[int, int, int]:
        r = self.rasters[0]
        return len(self.rasters), r.length, r.width

    def read_window(
        self,
        *,
        row0: int,
        col0: int,
        rows: int,
        cols: int,
        date_indices: list[int] | None = None,
    ) -> np.ndarray:
        indices = date_indices if date_indices is not None else list(range(len(self.rasters)))

        def _read(i: int) -> np.ndarray:
            return read_complex_window(
                self.rasters[i], row0=row0, col0=col0, rows=rows, cols=cols
            )

        if self.io_workers == 1 or len(indices) == 1:
            blocks = [_read(i) for i in indices]
        else:
            with ThreadPoolExecutor(max_workers=min(self.io_workers, len(indices))) as ex:
                blocks = list(ex.map(_read, indices))
        return np.stack(blocks, axis=0).astype(np.complex64, copy=False)
