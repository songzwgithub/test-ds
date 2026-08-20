from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.gamma.phase_correction import (
    GammaPointPhaseCorrectionProvider,
)

from pypsds.progress import log

from pypsds.runtime import (
    available_memory_bytes,
)


@dataclass(slots=True)
class PhaseTile:
    yxt: np.ndarray
    geometry_valid: np.ndarray

    read_seconds: float
    correction_seconds: float

    phase_min: float
    phase_max: float

    canonical_cells: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(slots=True)
class _CanonicalCell:
    row0: int
    row1: int
    col0: int
    col1: int

    yxt: np.ndarray
    geometry_valid: np.ndarray

    phase_min: float
    phase_max: float

    @property
    def nbytes(self) -> int:
        return (
            int(
                self.yxt.nbytes
            )
            +
            int(
                self.geometry_valid.nbytes
            )
        )


class CachedPhaseSource:
    """
    Existing full corrected Y-X-Time cache.

    This remains the regression/reference backend.
    """

    def __init__(
        self,
        *,
        processing_dir: Path,
        H: int,
        W: int,
        ndate: int,
    ):
        self.path = (
            processing_dir
            /
            "cache"
            /
            "phase_corrected_yxt.npy"
        )

        self.geom_path = (
            processing_dir
            /
            "cache"
            /
            "phase_geometry_valid.npy"
        )

        if not self.path.is_file():
            raise FileNotFoundError(
                self.path
            )

        if not self.geom_path.is_file():
            raise FileNotFoundError(
                self.geom_path
            )

        self.yxt = np.load(
            self.path,
            mmap_mode="r",
            allow_pickle=False,
        )

        self.geom = np.load(
            self.geom_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        expected = (
            H,
            W,
            ndate,
        )

        if self.yxt.shape != expected:

            raise RuntimeError(
                f"phase cache shape "
                f"{self.yxt.shape} != "
                f"{expected}"
            )

        if (
            self.yxt.dtype
            !=
            np.complex64
        ):

            raise RuntimeError(
                "phase cache must be complex64"
            )

        if self.geom.shape != (
            H,
            W,
        ):

            raise RuntimeError(
                "geometry-valid shape mismatch"
            )

    def read_tile(
        self,
        *,
        local_row0: int,
        local_row1: int,
        local_col0: int,
        local_col1: int,
    ) -> PhaseTile:

        t0 = perf_counter()

        yxt = np.ascontiguousarray(
            self.yxt[
                local_row0:local_row1,
                local_col0:local_col1,
                :
            ],
            dtype=np.complex64,
        )

        geom = np.ascontiguousarray(
            self.geom[
                local_row0:local_row1,
                local_col0:local_col1,
            ],
            dtype=np.bool_,
        )

        elapsed = (
            perf_counter()
            -
            t0
        )

        return PhaseTile(
            yxt=yxt,
            geometry_valid=geom,
            read_seconds=elapsed,
            correction_seconds=0.0,
            phase_min=float("nan"),
            phase_max=float("nan"),
        )


class GammaStreamingPhaseSource:
    """
    Canonical-grid GAMMA streaming phase source.

    Critical numerical rule
    -----------------------
    phase_sim_orb_pt is not numerically invariant to arbitrary
    point-list spatial grouping.

    The existing validated cache was generated using:

        canonical phase tile = 128 x 256

    Recomputing the same pixels using a larger 256x512 / DS-sized
    point list changes the simulated phase.

    Therefore the phase-correction grid is treated as a numerical
    grid, independent of the downstream DS tile grid.

    Processing:

        requested DS tile+halo
              |
              v
        identify all canonical 128x256 cells
              |
              v
        fetch cells from bounded LRU
              |
              +---- misses:
              |       read RSLC canonical cells
              |       8 spatial cells in parallel
              |       x 4 phase_sim pair workers
              |
              v
        mosaic exact canonical cells
              |
              v
        crop requested DS tile+halo

    No full-scene corrected YXT cube is required.
    """

    def __init__(
        self,
        *,
        cfg,
        paths,
        stack,
        base_row0: int,
        base_col0: int,
        io_workers: int,
    ):
        self.cfg = cfg
        self.paths = paths
        self.stack = stack

        self.base_row0 = int(
            base_row0
        )

        self.base_col0 = int(
            base_col0
        )

        (
            self.ndate,
            full_H,
            full_W,
        ) = stack.shape

        # ROI dimensions come from the caller's data products.
        # They are inferred lazily from requested local coordinates.
        self.H = None
        self.W = None

        self.stack.io_workers = max(
            1,
            int(
                io_workers
            ),
        )

        self.provider = (
            GammaPointPhaseCorrectionProvider(
                cfg,
                paths,
                stack,
            )
        )

        self.assets = (
            self.provider.prepare()
        )

        # ----------------------------------------------------
        # Machine-autotuned canonical phase layout.
        # ----------------------------------------------------

        self.canonical_rows = 128
        self.canonical_cols = 256

        self.spatial_workers = 1
        self.pair_workers = min(
            16,
            max(
                1,
                self.ndate - 1,
            ),
        )

        tune_path = (
            Path(
                paths.output_dir
            )
            /
            "processing"
            /
            "canonical_phase_parallel_autotune.json"
        )

        if tune_path.is_file():

            try:

                tune = json.loads(
                    tune_path.read_text(
                        encoding="utf-8"
                    )
                )

                winner = tune.get(
                    "winner",
                    {},
                )

                if (
                    winner.get(
                        "parity"
                    )
                    is True
                ):

                    canonical = tune.get(
                        "canonical_tile",
                        [
                            128,
                            256,
                        ],
                    )

                    self.canonical_rows = int(
                        canonical[0]
                    )

                    self.canonical_cols = int(
                        canonical[1]
                    )

                    self.spatial_workers = int(
                        winner[
                            "spatial_workers"
                        ]
                    )

                    self.pair_workers = int(
                        winner[
                            "pair_workers"
                        ]
                    )

            except Exception as exc:

                log(
                    "WARNING: invalid canonical phase "
                    f"autotune file ignored: {exc}"
                )

        self.spatial_workers = max(
            1,
            self.spatial_workers,
        )

        self.pair_workers = max(
            1,
            min(
                self.pair_workers,
                max(
                    1,
                    self.ndate - 1,
                ),
            ),
        )

        # The provider's pair-parallel engine is now explicitly
        # controlled by the canonical scheduler.
        self.provider._phase_sim_workers_override = (
            self.pair_workers
        )

        # ----------------------------------------------------
        # Bounded in-memory canonical-cell LRU.
        #
        # We want enough cells to retain DS halo overlap,
        # while RAM remains independent of scene H x W.
        #
        # Target:
        #     4 waves of spatial workers
        #
        # Memory safety:
        #     <= 10% of currently available RAM
        #
        # For the present 38-date stack this is only a few
        # hundred MiB.
        # ----------------------------------------------------

        nominal_cell_bytes = (
            self.canonical_rows
            *
            self.canonical_cols
            *
            self.ndate
            *
            np.dtype(
                np.complex64
            ).itemsize
            +
            self.canonical_rows
            *
            self.canonical_cols
        )

        desired_cells = max(
            4,
            self.spatial_workers
            *
            4,
        )

        avail = (
            available_memory_bytes()
        )

        if avail > 0:

            memory_cells = max(
                1,
                int(
                    (
                        avail
                        //
                        10
                    )
                    //
                    max(
                        1,
                        nominal_cell_bytes,
                    )
                ),
            )

            self.cache_max_cells = max(
                1,
                min(
                    desired_cells,
                    memory_cells,
                ),
            )

        else:

            self.cache_max_cells = (
                desired_cells
            )

        self._cache: OrderedDict[
            tuple[int, int],
            _CanonicalCell,
        ] = OrderedDict()

        self.cache_hits_total = 0
        self.cache_misses_total = 0

        log(
            "Canonical streaming phase source: "
            f"cell={self.canonical_rows}x"
            f"{self.canonical_cols}, "
            f"spatial_workers={self.spatial_workers}, "
            f"pair_workers={self.pair_workers}, "
            f"max_gamma_processes="
            f"{self.spatial_workers*self.pair_workers}, "
            f"LRU_cells={self.cache_max_cells}, "
            f"io_workers={self.stack.io_workers}"
        )

    # --------------------------------------------------------
    # Canonical geometry
    # --------------------------------------------------------

    def _canonical_keys(
        self,
        *,
        local_row0: int,
        local_row1: int,
        local_col0: int,
        local_col1: int,
    ) -> list[
        tuple[int, int]
    ]:

        cr = self.canonical_rows
        cc = self.canonical_cols

        first_r = (
            local_row0
            //
            cr
        ) * cr

        first_c = (
            local_col0
            //
            cc
        ) * cc

        keys = []

        for r0 in range(
            first_r,
            local_row1,
            cr,
        ):

            for c0 in range(
                first_c,
                local_col1,
                cc,
            ):

                keys.append(
                    (
                        r0,
                        c0,
                    )
                )

        return keys

    def _cell_shape(
        self,
        key,
    ):
        local_r0, local_c0 = key

        # Full raster coordinates determine edge clipping.
        #
        # The ROI itself begins at base_row0/base_col0.
        # Current processing ROI is known to end before or at
        # the underlying GAMMA raster boundary.
        full_H = (
            self.stack.rasters[0].length
        )

        full_W = (
            self.stack.rasters[0].width
        )

        global_r0 = (
            self.base_row0
            +
            local_r0
        )

        global_c0 = (
            self.base_col0
            +
            local_c0
        )

        rows = min(
            self.canonical_rows,
            full_H
            -
            global_r0,
        )

        cols = min(
            self.canonical_cols,
            full_W
            -
            global_c0,
        )

        if (
            rows <= 0
            or
            cols <= 0
        ):

            raise RuntimeError(
                "canonical phase cell outside "
                "GAMMA raster"
            )

        return (
            global_r0,
            global_c0,
            rows,
            cols,
        )

    # --------------------------------------------------------
    # LRU
    # --------------------------------------------------------

    def _cache_get(
        self,
        key,
    ):

        cell = self._cache.pop(
            key,
            None,
        )

        if cell is None:
            return None

        self._cache[
            key
        ] = cell

        self.cache_hits_total += 1

        return cell

    def _cache_put(
        self,
        key,
        cell,
    ):

        if key in self._cache:
            self._cache.pop(
                key
            )

        self._cache[
            key
        ] = cell

        while (
            len(
                self._cache
            )
            >
            self.cache_max_cells
        ):

            self._cache.popitem(
                last=False
            )

    # --------------------------------------------------------
    # Cell computation
    # --------------------------------------------------------

    def _correct_raw_cell(
        self,
        *,
        key,
        raw,
    ):

        (
            global_r0,
            global_c0,
            rows,
            cols,
        ) = self._cell_shape(
            key
        )

        (
            corrected,
            geometry_valid,
            stats,
        ) = self.provider.correct_block(
            raw,
            global_row0=global_r0,
            global_col0=global_c0,
            tile_label=(
                f"dscanonical_"
                f"r{global_r0}_{global_r0+rows}_"
                f"c{global_c0}_{global_c0+cols}"
            ),
        )

        yxt = np.ascontiguousarray(
            np.moveaxis(
                corrected,
                0,
                -1,
            ),
            dtype=np.complex64,
        )

        geometry_valid = (
            np.ascontiguousarray(
                geometry_valid,
                dtype=np.bool_,
            )
        )

        local_r0, local_c0 = key

        cell = _CanonicalCell(
            row0=local_r0,
            row1=(
                local_r0
                +
                rows
            ),
            col0=local_c0,
            col1=(
                local_c0
                +
                cols
            ),
            yxt=yxt,
            geometry_valid=(
                geometry_valid
            ),
            phase_min=float(
                stats.phase_min
            ),
            phase_max=float(
                stats.phase_max
            ),
        )

        return cell

    def _compute_missing(
        self,
        missing,
    ):
        """
        Compute canonical misses in waves.

        I/O:
            Cells are pre-read sequentially.
            Each GammaStack.read_window() may use date-parallel
            io_workers internally.

        GAMMA:
            A wave then runs spatial_workers cells concurrently,
            each cell using pair_workers phase_sim processes.

        This avoids nesting:
            spatial_workers x io_workers
        filesystem read thread pools.
        """

        computed = {}

        total_read = 0.0
        total_correction = 0.0

        phase_min = float("inf")
        phase_max = float("-inf")

        for start in range(
            0,
            len(missing),
            self.spatial_workers,
        ):

            wave = missing[
                start:
                start
                +
                self.spatial_workers
            ]

            # ------------------------------------------------
            # Stage A: pre-read canonical RSLC cells.
            # ------------------------------------------------

            raw_wave = {}

            for key in wave:

                (
                    global_r0,
                    global_c0,
                    rows,
                    cols,
                ) = self._cell_shape(
                    key
                )

                t0 = perf_counter()

                raw_wave[
                    key
                ] = (
                    self.stack.read_window(
                        row0=global_r0,
                        col0=global_c0,
                        rows=rows,
                        cols=cols,
                    )
                    .astype(
                        np.complex64,
                        copy=False,
                    )
                )

                total_read += (
                    perf_counter()
                    -
                    t0
                )

            # ------------------------------------------------
            # Stage B: canonical cells concurrently.
            # ------------------------------------------------

            t0 = perf_counter()

            with ThreadPoolExecutor(
                max_workers=len(
                    wave
                ),
                thread_name_prefix=(
                    "pypsds-phase-cell"
                ),
            ) as ex:

                futures = {
                    ex.submit(
                        self._correct_raw_cell,
                        key=key,
                        raw=raw_wave[
                            key
                        ],
                    ):
                    key
                    for key in wave
                }

                for fut in as_completed(
                    futures
                ):

                    key = futures[
                        fut
                    ]

                    cell = fut.result()

                    computed[
                        key
                    ] = cell

                    self._cache_put(
                        key,
                        cell,
                    )

                    if np.isfinite(
                        cell.phase_min
                    ):

                        phase_min = min(
                            phase_min,
                            cell.phase_min,
                        )

                    if np.isfinite(
                        cell.phase_max
                    ):

                        phase_max = max(
                            phase_max,
                            cell.phase_max,
                        )

            total_correction += (
                perf_counter()
                -
                t0
            )

            # Raw data are no longer needed.
            raw_wave.clear()

        if not np.isfinite(
            phase_min
        ):
            phase_min = float("nan")

        if not np.isfinite(
            phase_max
        ):
            phase_max = float("nan")

        return (
            computed,
            total_read,
            total_correction,
            phase_min,
            phase_max,
        )

    # --------------------------------------------------------
    # Public tile request
    # --------------------------------------------------------

    def read_tile(
        self,
        *,
        local_row0: int,
        local_row1: int,
        local_col0: int,
        local_col1: int,
    ) -> PhaseTile:

        if (
            local_row1
            <=
            local_row0
            or
            local_col1
            <=
            local_col0
        ):

            raise ValueError(
                "invalid requested phase tile"
            )

        rows = (
            local_row1
            -
            local_row0
        )

        cols = (
            local_col1
            -
            local_col0
        )

        keys = self._canonical_keys(
            local_row0=local_row0,
            local_row1=local_row1,
            local_col0=local_col0,
            local_col1=local_col1,
        )

        cells = {}

        missing = []

        hits = 0

        for key in keys:

            cell = self._cache_get(
                key
            )

            if cell is None:

                missing.append(
                    key
                )

            else:

                hits += 1

                cells[
                    key
                ] = cell

        misses = len(
            missing
        )

        self.cache_misses_total += (
            misses
        )

        read_seconds = 0.0
        correction_seconds = 0.0

        phase_min = float("nan")
        phase_max = float("nan")

        if missing:

            (
                created,
                read_seconds,
                correction_seconds,
                phase_min,
                phase_max,
            ) = self._compute_missing(
                missing
            )

            cells.update(
                created
            )

        # ----------------------------------------------------
        # Assemble only the requested DS tile+halo.
        # ----------------------------------------------------

        yxt = np.empty(
            (
                rows,
                cols,
                self.ndate,
            ),
            dtype=np.complex64,
        )

        geometry_valid = np.zeros(
            (
                rows,
                cols,
            ),
            dtype=np.bool_,
        )

        filled = np.zeros(
            (
                rows,
                cols,
            ),
            dtype=np.bool_,
        )

        for key in keys:

            cell = cells[
                key
            ]

            ir0 = max(
                local_row0,
                cell.row0,
            )

            ir1 = min(
                local_row1,
                cell.row1,
            )

            ic0 = max(
                local_col0,
                cell.col0,
            )

            ic1 = min(
                local_col1,
                cell.col1,
            )

            if (
                ir0 >= ir1
                or
                ic0 >= ic1
            ):
                continue

            dr0 = (
                ir0
                -
                local_row0
            )

            dr1 = (
                ir1
                -
                local_row0
            )

            dc0 = (
                ic0
                -
                local_col0
            )

            dc1 = (
                ic1
                -
                local_col0
            )

            sr0 = (
                ir0
                -
                cell.row0
            )

            sr1 = (
                ir1
                -
                cell.row0
            )

            sc0 = (
                ic0
                -
                cell.col0
            )

            sc1 = (
                ic1
                -
                cell.col0
            )

            yxt[
                dr0:dr1,
                dc0:dc1,
                :
            ] = cell.yxt[
                sr0:sr1,
                sc0:sc1,
                :
            ]

            geometry_valid[
                dr0:dr1,
                dc0:dc1,
            ] = (
                cell.geometry_valid[
                    sr0:sr1,
                    sc0:sc1,
                ]
            )

            filled[
                dr0:dr1,
                dc0:dc1,
            ] = True

        if not np.all(
            filled
        ):

            raise RuntimeError(
                "canonical phase mosaic did not "
                "fully cover requested tile"
            )

        log(
            "Canonical phase tile "
            f"r{local_row0}:{local_row1} "
            f"c{local_col0}:{local_col1}: "
            f"cells={len(keys)}, "
            f"hits={hits}, "
            f"misses={misses}, "
            f"read={read_seconds:.2f}s, "
            f"correction="
            f"{correction_seconds:.2f}s, "
            f"LRU={len(self._cache)}/"
            f"{self.cache_max_cells}"
        )

        return PhaseTile(
            yxt=yxt,
            geometry_valid=(
                geometry_valid
            ),
            read_seconds=(
                read_seconds
            ),
            correction_seconds=(
                correction_seconds
            ),
            phase_min=phase_min,
            phase_max=phase_max,
            canonical_cells=len(
                keys
            ),
            cache_hits=hits,
            cache_misses=misses,
        )


__all__ = [
    "PhaseTile",
    "CachedPhaseSource",
    "GammaStreamingPhaseSource",
]
