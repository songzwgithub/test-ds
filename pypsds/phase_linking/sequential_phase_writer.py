from __future__ import annotations

from pathlib import Path

import numpy as np


class SequentialPhaseWriter:

    def __init__(
        self,
        path,
        *,
        ndate: int,
        rows: int,
        cols: int,
        overwrite: bool = False,
    ):
        self.path = Path(path)

        self.ndate = int(ndate)
        self.rows = int(rows)
        self.cols = int(cols)

        if self.ndate < 1:
            raise ValueError(
                "ndate must be >= 1"
            )

        if (
            self.rows < 1
            or
            self.cols < 1
        ):
            raise ValueError(
                "rows/cols must be >= 1"
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shape = (
            self.ndate,
            self.rows,
            self.cols,
        )

        if (
            overwrite
            or
            not self.path.exists()
        ):

            # --------------------------------------------------
            # Sparse creation only.
            #
            # Do NOT bulk initialize:
            #
            #   self._arr[...] = NaN
            #
            # and do NOT scatter NaN into PS pixels.
            #
            # Filesystem holes read as 0+0j.
            # --------------------------------------------------

            self._arr = (
                np.lib.format.open_memmap(
                    self.path,
                    mode="w+",
                    dtype=np.complex64,
                    shape=shape,
                )
            )

        else:

            self._arr = np.load(
                self.path,
                mmap_mode="r+",
                allow_pickle=False,
            )

            if (
                self._arr.shape
                !=
                shape
            ):
                raise ValueError(
                    "existing phase cube shape "
                    f"{self._arr.shape} "
                    f"!= {shape}"
                )

            if (
                self._arr.dtype
                !=
                np.dtype(
                    np.complex64
                )
            ):
                raise ValueError(
                    "existing phase cube dtype "
                    f"{self._arr.dtype} "
                    "!= complex64"
                )

        self.written_counts = (
            np.zeros(
                self.ndate,
                dtype=np.int64,
            )
        )


    @property
    def shape(self):
        return self._arr.shape


    @property
    def dtype(self):
        return self._arr.dtype


    def __call__(
        self,
        *,
        stage_index: int,
        real_indices,
        rows,
        cols,
        phase,
    ):

        del stage_index

        real_indices = tuple(
            int(x)
            for x in real_indices
        )

        if not real_indices:
            raise ValueError(
                "empty real_indices"
            )

        rr = np.asarray(
            rows,
            dtype=np.int32,
        )

        cc = np.asarray(
            cols,
            dtype=np.int32,
        )

        ph = np.asarray(
            phase,
            dtype=np.complex64,
        )

        if (
            rr.ndim != 1
            or
            cc.ndim != 1
        ):
            raise ValueError(
                "rows/cols must be 1-D"
            )

        if rr.size != cc.size:
            raise ValueError(
                "rows/cols size mismatch"
            )

        expected_shape = (
            rr.size,
            len(real_indices),
        )

        if (
            ph.shape
            !=
            expected_shape
        ):
            raise ValueError(
                "phase shape mismatch: "
                f"{ph.shape} != "
                f"{expected_shape}"
            )

        idx = np.asarray(
            real_indices,
            dtype=np.int64,
        )

        if (
            np.any(idx < 0)
            or
            np.any(
                idx >= self.ndate
            )
        ):
            raise ValueError(
                "real acquisition index "
                "outside phase cube"
            )

        if (
            rr.size
            and
            (
                np.any(rr < 0)
                or
                np.any(
                    rr >= self.rows
                )
                or
                np.any(cc < 0)
                or
                np.any(
                    cc >= self.cols
                )
            )
        ):
            raise ValueError(
                "row/col outside "
                "phase cube"
            )

        # Phase-linking phase is a finite,
        # non-zero complex phasor.
        valid_phase = (
            np.isfinite(
                ph.real
            )
            &
            np.isfinite(
                ph.imag
            )
            &
            (
                ph
                !=
                np.complex64(0.0)
            )
        )

        if not np.all(
            valid_phase
        ):
            raise ValueError(
                "phase sink received "
                "invalid phase"
            )

        for j, date_index in enumerate(
            real_indices
        ):

            self._arr[
                date_index,
                rr,
                cc,
            ] = ph[
                :,
                j,
            ]

            self.written_counts[
                date_index
            ] += rr.size


    def flush(self):
        self._arr.flush()


    def finite_counts(self):
        return (
            self.written_counts.copy()
        )


    def close(self):

        self.flush()

        arr = self._arr

        self._arr = None

        del arr


    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        self.close()

        return False


__all__ = [
    "SequentialPhaseWriter",
]
