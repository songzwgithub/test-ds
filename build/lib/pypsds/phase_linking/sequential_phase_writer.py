from __future__ import annotations

from pathlib import Path

import numpy as np


class SequentialPhaseWriter:
    """
    Write sequential phase-linking results directly into the
    production phase cube:

        [date, row, col]

    The object is callable and therefore can be passed directly
    as run_sequential_stage(..., phase_sink=writer).

    Each sequential stage emits only its REAL acquisitions.
    Compressed-SLC columns are never written to this cube.
    """

    def __init__(
        self,
        path,
        *,
        ndate: int,
        rows: int,
        cols: int,
        overwrite: bool = False,
        strict_no_overwrite: bool = True,
    ):
        self.path = Path(path)

        self.ndate = int(ndate)
        self.rows = int(rows)
        self.cols = int(cols)

        self.strict_no_overwrite = bool(
            strict_no_overwrite
        )

        if self.ndate < 1:
            raise ValueError(
                "ndate must be >= 1"
            )

        if self.rows < 1 or self.cols < 1:
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

        if self.path.exists():

            if overwrite:
                self._arr = (
                    np.lib.format.open_memmap(
                        self.path,
                        mode="w+",
                        dtype=np.complex64,
                        shape=shape,
                    )
                )

                self._arr[...] = np.complex64(
                    np.nan
                    +
                    1j * np.nan
                )

                self._arr.flush()

            else:
                self._arr = np.load(
                    self.path,
                    mmap_mode="r+",
                )

                if self._arr.shape != shape:
                    raise ValueError(
                        "existing phase cube shape "
                        f"{self._arr.shape} != {shape}"
                    )

                if (
                    self._arr.dtype
                    !=
                    np.dtype(np.complex64)
                ):
                    raise ValueError(
                        "existing phase cube dtype "
                        f"{self._arr.dtype} "
                        "!= complex64"
                    )

        else:

            self._arr = (
                np.lib.format.open_memmap(
                    self.path,
                    mode="w+",
                    dtype=np.complex64,
                    shape=shape,
                )
            )

            self._arr[...] = np.complex64(
                np.nan
                +
                1j * np.nan
            )

            self._arr.flush()

        self.written_counts = np.zeros(
            self.ndate,
            dtype=np.int64,
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
        """
        Phase-sink callback used by run_sequential_stage().

        Parameters
        ----------
        stage_index
            Sequential ministack stage number.
        real_indices
            Original acquisition indices represented by phase
            columns.
        rows, cols
            Global center coordinates, shape [B].
        phase
            Referenced real-acquisition phase, shape
            [B, len(real_indices)].
        """

        del stage_index

        real_indices = tuple(
            int(x)
            for x in real_indices
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

        if rr.ndim != 1 or cc.ndim != 1:
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

        if ph.shape != expected_shape:
            raise ValueError(
                "phase shape mismatch: "
                f"{ph.shape} != "
                f"{expected_shape}"
            )

        if not real_indices:
            raise ValueError(
                "empty real_indices"
            )

        idx = np.asarray(
            real_indices,
            dtype=np.int64,
        )

        if (
            np.any(idx < 0)
            or
            np.any(idx >= self.ndate)
        ):
            raise ValueError(
                "real acquisition index "
                "outside phase cube"
            )

        if rr.size:

            if (
                np.any(rr < 0)
                or
                np.any(rr >= self.rows)
                or
                np.any(cc < 0)
                or
                np.any(cc >= self.cols)
            ):
                raise ValueError(
                    "row/col outside phase cube"
                )

        finite = (
            np.isfinite(ph.real)
            &
            np.isfinite(ph.imag)
        )

        if not np.all(finite):
            raise ValueError(
                "phase sink received "
                "non-finite phase"
            )

        # ----------------------------------------------------
        # Write one acquisition at a time.
        #
        # This preserves the existing [T,H,W] production
        # linked_phase.npy contract.
        # ----------------------------------------------------

        for j, date_index in enumerate(
            real_indices
        ):

            if self.strict_no_overwrite:

                old = self._arr[
                    date_index,
                    rr,
                    cc,
                ]

                old_finite = (
                    np.isfinite(old.real)
                    &
                    np.isfinite(old.imag)
                )

                if np.any(old_finite):
                    raise RuntimeError(
                        "sequential phase writer "
                        "would overwrite existing "
                        f"date={date_index} phase"
                    )

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
        """
        Count finite phase pixels for each acquisition.
        """

        out = np.zeros(
            self.ndate,
            dtype=np.int64,
        )

        for i in range(
            self.ndate
        ):
            x = self._arr[i]

            out[i] = np.count_nonzero(
                np.isfinite(x.real)
                &
                np.isfinite(x.imag)
            )

        return out


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
