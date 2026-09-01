from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
import hashlib
import json
import platform
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.config import cfg_get

from pypsds.gamma.phase_correction import (
    GammaPointPhaseCorrectionProvider,
)

from pypsds.progress import log

from pypsds.runtime import (
    available_memory_bytes,
    logical_cpu_count,
)




def _plan_temporal_piece_cache_cells(
    *,
    H: int,
    W: int,
    canonical_rows: int,
    canonical_cols: int,
    ndate: int,
    temporal_parts: int,
    available_bytes: int,
    memory_fraction: float = 0.10,
):
    H = max(1, int(H))
    W = max(1, int(W))
    cr = max(1, int(canonical_rows))
    cc = max(1, int(canonical_cols))
    ndate = max(1, int(ndate))
    temporal_parts = max(1, int(temporal_parts))

    fraction = min(
        0.25,
        max(
            0.01,
            float(memory_fraction),
        ),
    )

    scene_rows = (H + cr - 1) // cr
    scene_cols = (W + cc - 1) // cc
    scene_cells = scene_rows * scene_cols

    nominal_piece_dates = max(
        1,
        (ndate + temporal_parts - 1)
        //
        temporal_parts,
    )

    nominal_piece_bytes = (
        cr
        *
        cc
        *
        nominal_piece_dates
        *
        np.dtype(np.complex64).itemsize
        +
        cr
        *
        cc
    )

    desired_entries = scene_cells * temporal_parts

    if int(available_bytes) > 0:
        budget_bytes = int(
            int(available_bytes)
            *
            fraction
        )

        memory_entries = max(
            1,
            budget_bytes
            //
            max(
                1,
                nominal_piece_bytes,
            ),
        )
    else:
        memory_entries = 32
        budget_bytes = memory_entries * nominal_piece_bytes

    target_entries = max(
        1,
        min(
            desired_entries,
            memory_entries,
        ),
    )

    return {
        "scene_rows": int(scene_rows),
        "scene_cols": int(scene_cols),
        "scene_cells": int(scene_cells),
        "temporal_parts": int(temporal_parts),
        "desired_entries": int(desired_entries),
        "nominal_piece_dates": int(nominal_piece_dates),
        "nominal_piece_bytes": int(nominal_piece_bytes),
        "memory_budget_bytes": int(budget_bytes),
        "memory_entries": int(memory_entries),
        "target_entries": int(target_entries),
    }


def _plan_fullspan_cache_cells(
    *,
    H: int,
    W: int,
    canonical_rows: int,
    canonical_cols: int,
    ndate: int,
    available_bytes: int,
    memory_fraction: float = 0.10,
):
    H = max(1, int(H))
    W = max(1, int(W))
    cr = max(1, int(canonical_rows))
    cc = max(1, int(canonical_cols))
    ndate = max(1, int(ndate))

    fraction = min(
        0.25,
        max(
            0.01,
            float(memory_fraction),
        ),
    )

    scene_rows = (H + cr - 1) // cr
    scene_cols = (W + cc - 1) // cc
    scene_cells = scene_rows * scene_cols

    nominal_cell_bytes = (
        cr
        *
        cc
        *
        ndate
        *
        np.dtype(np.complex64).itemsize
        +
        cr
        *
        cc
    )

    if int(available_bytes) > 0:
        budget_bytes = int(
            int(available_bytes)
            *
            fraction
        )

        memory_cells = max(
            1,
            budget_bytes
            //
            max(
                1,
                nominal_cell_bytes,
            ),
        )
    else:
        memory_cells = 32
        budget_bytes = (
            memory_cells
            *
            nominal_cell_bytes
        )

    target_cells = max(
        1,
        min(
            scene_cells,
            memory_cells,
        ),
    )

    return {
        "scene_rows": int(scene_rows),
        "scene_cols": int(scene_cols),
        "scene_cells": int(scene_cells),
        "nominal_cell_bytes": int(nominal_cell_bytes),
        "memory_budget_bytes": int(budget_bytes),
        "memory_cells": int(memory_cells),
        "target_cells": int(target_cells),
    }


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
    cache_composed_hits: int = 0


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


_CANONICAL_PHASE_ROWS = 128
_CANONICAL_PHASE_COLS = 256
_CANONICAL_AUTOTUNE_FORMAT = (
    "pyPSDS-GAMMA-canonical-phase-parallel-benchmark-v2"
)


def _effective_runtime_cpu_count(cfg) -> int:
    # CPU visible to pyPSDS and capped by runtime.cpu.
    cpu = max(1, int(logical_cpu_count()))
    raw = cfg_get(cfg, "runtime.cpu", None)
    if raw not in (None, "", "auto"):
        cpu = min(cpu, max(1, int(raw)))
    return int(cpu)


def _stack_dates_sha256(dates) -> str:
    raw = "\n".join(str(x) for x in dates).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cpu_model_name() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(
            encoding="utf-8",
            errors="replace",
        )
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _file_identity(path) -> dict | None:
    if path in (None, ""):
        return None
    p = Path(str(path)).expanduser()
    try:
        p = p.resolve()
    except Exception:
        return None
    if not p.is_file():
        return None
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def canonical_autotune_runtime_identity(
    cfg,
    stack,
    *,
    phase_sim_path=None,
) -> dict:
    # Identity required before a saved worker schedule can be reused.
    if phase_sim_path is None:
        raw_cmd = str(
            cfg_get(
                cfg,
                "phase_correction.commands.phase_sim_orb_pt",
                "phase_sim_orb_pt",
            )
        )
        phase_sim_path = shutil.which(raw_cmd)

    return {
        "effective_cpu_count": _effective_runtime_cpu_count(cfg),
        "cpu_model": _cpu_model_name(),
        "ndate": int(len(stack.dates)),
        "dates_sha256": _stack_dates_sha256(stack.dates),
        "phase_sim_orb_pt": _file_identity(phase_sim_path),
    }


def _validated_canonical_autotune(
    tune,
    cfg,
    stack,
    *,
    phase_sim_path=None,
) -> tuple[int, int]:
    # Validate schedule; canonical numerical grouping may never change.
    if tune.get("format") != _CANONICAL_AUTOTUNE_FORMAT:
        raise ValueError("autotune format is stale or unsupported")

    canonical = tune.get("canonical_tile")
    if canonical != [_CANONICAL_PHASE_ROWS, _CANONICAL_PHASE_COLS]:
        raise ValueError(
            "autotune canonical tile mismatch; "
            "production canonical grouping is fixed at 128x256"
        )

    expected_identity = canonical_autotune_runtime_identity(
        cfg,
        stack,
        phase_sim_path=phase_sim_path,
    )
    if tune.get("runtime_identity") != expected_identity:
        raise ValueError("autotune runtime/stack/GAMMA identity is stale")

    winner = tune.get("winner", {})
    if winner.get("parity") is not True:
        raise ValueError("autotune winner has no numerical-parity approval")

    spatial = int(winner["spatial_workers"])
    pair = int(winner["pair_workers"])
    if spatial < 1 or pair < 1:
        raise ValueError("autotune worker counts must be >= 1")

    return spatial, pair


def bounded_prefetch_gamma_parallelism(
    *,
    cpu_count: int,
    pl_workers: int,
    spatial_workers: int,
    pair_workers: int,
    reserve_cpus: int = 4,
):
    cpu = max(1, int(cpu_count))
    pl = max(1, int(pl_workers))

    reserve = max(
        1,
        min(
            int(reserve_cpus),
            max(1, cpu - 1),
        ),
    )

    gamma_budget = max(
        1,
        cpu
        -
        min(
            pl,
            max(1, cpu - 1),
        )
        -
        reserve,
    )

    pair = max(
        1,
        min(
            int(pair_workers),
            gamma_budget,
        ),
    )

    spatial = max(
        1,
        min(
            int(spatial_workers),
            max(
                1,
                gamma_budget // pair,
            ),
        ),
    )

    while (
        spatial * pair > gamma_budget
        and spatial > 1
    ):
        spatial -= 1

    while (
        spatial * pair > gamma_budget
        and pair > 1
    ):
        pair -= 1

    return {
        "cpu_count": cpu,
        "pl_workers": pl,
        "reserve_cpus": reserve,
        "gamma_process_budget": gamma_budget,
        "spatial_workers": spatial,
        "pair_workers": pair,
        "max_gamma_processes": spatial * pair,
    }


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

        self.canonical_rows = _CANONICAL_PHASE_ROWS
        self.canonical_cols = _CANONICAL_PHASE_COLS

        # FASTPATCH: hardware-aware fallback. The canonical 128x256
        # numerical grouping is unchanged; only execution scheduling changes.
        # On the validated 32-logical-CPU production host, 6 spatial x 3 pair
        # gave exact numerical parity and the best measured warm throughput.
        cpu_for_gamma = _effective_runtime_cpu_count(
            cfg
        )

        if cpu_for_gamma >= 28:
            default_spatial_workers = 6
            default_pair_workers = 3
        elif cpu_for_gamma >= 20:
            default_spatial_workers = 5
            default_pair_workers = 3
        elif cpu_for_gamma >= 14:
            default_spatial_workers = 4
            default_pair_workers = 3
        elif cpu_for_gamma >= 8:
            default_spatial_workers = 2
            default_pair_workers = 3
        else:
            default_spatial_workers = 1
            default_pair_workers = min(
                3,
                max(
                    1,
                    self.ndate - 1,
                ),
            )

        raw_spatial_workers = cfg_get(
            cfg,
            "phase_correction.parallel.spatial_workers",
            "auto",
        )
        raw_pair_workers = cfg_get(
            cfg,
            "phase_correction.parallel.pair_workers",
            "auto",
        )

        self.spatial_workers = (
            default_spatial_workers
            if raw_spatial_workers in (None, "", "auto")
            else int(raw_spatial_workers)
        )
        self.pair_workers = (
            default_pair_workers
            if raw_pair_workers in (None, "", "auto")
            else int(raw_pair_workers)
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

                phase_sim_path = getattr(
                    self.provider,
                    "_commands",
                    {},
                ).get("phase_sim_orb_pt")

                tuned_spatial, tuned_pair = (
                    _validated_canonical_autotune(
                        tune,
                        cfg,
                        stack,
                        phase_sim_path=phase_sim_path,
                    )
                )

                self.spatial_workers = int(tuned_spatial)
                self.pair_workers = int(tuned_pair)

                log(
                    "Loaded validated canonical GAMMA autotune: "
                    f"{self.spatial_workers}x{self.pair_workers}"
                )

            except Exception as exc:
                log(
                    "WARNING: canonical phase autotune ignored: "
                    f"{exc}"
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

    def configure_prefetch_concurrency(
        self,
        *,
        pl_workers: int,
    ):
        budget = bounded_prefetch_gamma_parallelism(
            cpu_count=_effective_runtime_cpu_count(self.cfg),
            pl_workers=pl_workers,
            spatial_workers=self.spatial_workers,
            pair_workers=self.pair_workers,
            reserve_cpus=4,
        )

        self.spatial_workers = int(
            budget["spatial_workers"]
        )

        self.pair_workers = int(
            budget["pair_workers"]
        )

        self.provider._phase_sim_workers_override = (
            self.pair_workers
        )

        log(
            "Prefetch GAMMA CPU budget: "
            f"cpu={budget['cpu_count']}, "
            f"PL={budget['pl_workers']}, "
            f"reserve={budget['reserve_cpus']}, "
            f"gamma_budget={budget['gamma_process_budget']}, "
            f"spatial_workers={budget['spatial_workers']}, "
            f"pair_workers={budget['pair_workers']}, "
            f"max_gamma_processes={budget['max_gamma_processes']}"
        )

        return budget




    def configure_sequential_temporal_cache(
        self,
        *,
        local_H: int,
        local_W: int,
        temporal_parts: int,
        memory_fraction: float = 0.10,
    ):
        plan = _plan_temporal_piece_cache_cells(
            H=local_H,
            W=local_W,
            canonical_rows=self.canonical_rows,
            canonical_cols=self.canonical_cols,
            ndate=self.ndate,
            temporal_parts=temporal_parts,
            available_bytes=available_memory_bytes(),
            memory_fraction=memory_fraction,
        )

        old_cells = int(self.cache_max_cells)

        self.cache_max_cells = max(
            old_cells,
            int(plan["target_entries"]),
        )

        log(
            "Sequential temporal phase cache: "
            f"ROI={int(local_H)}x{int(local_W)}, "
            f"scene_cells={plan['scene_cells']}, "
            f"temporal_parts={plan['temporal_parts']}, "
            f"desired={plan['desired_entries']}, "
            f"LRU={old_cells}->{self.cache_max_cells}, "
            f"budget={plan['memory_budget_bytes'] / (1024**3):.2f} GiB"
        )

        return {
            **plan,
            "old_cache_max_cells": old_cells,
            "cache_max_cells": int(self.cache_max_cells),
        }


    def configure_postphase_fullspan_cache(
        self,
        *,
        local_H: int,
        local_W: int,
        memory_fraction: float = 0.10,
        clear_stage_cache: bool = True,
    ):
        plan = _plan_fullspan_cache_cells(
            H=local_H,
            W=local_W,
            canonical_rows=self.canonical_rows,
            canonical_cols=self.canonical_cols,
            ndate=self.ndate,
            available_bytes=available_memory_bytes(),
            memory_fraction=memory_fraction,
        )

        old_cells = int(self.cache_max_cells)
        old_entries = len(self._cache)

        if clear_stage_cache:
            self._cache.clear()

        self.cache_max_cells = max(
            int(
                self.cache_max_cells
            ),
            int(
                plan[
                    "target_cells"
                ]
            ),
        )

        log(
            "Post-PL fullspan phase cache: "
            f"ROI={int(local_H)}x{int(local_W)}, "
            f"canonical={self.canonical_rows}x{self.canonical_cols}, "
            f"scene_cells={plan['scene_cells']}, "
            f"LRU={old_cells}->{self.cache_max_cells}, "
            f"cleared={old_entries if clear_stage_cache else 0}, "
            f"budget={plan['memory_budget_bytes'] / (1024**3):.2f} GiB"
        )

        return {
            **plan,
            "old_cache_max_cells": old_cells,
            "old_cache_entries": int(old_entries),
            "cache_max_cells": int(self.cache_max_cells),
            "cleared_entries": int(
                old_entries
                if
                clear_stage_cache
                else
                0
            ),
        }


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


    def _cache_compose_temporal_cell(
        self,
        *,
        spatial_key,
        date_indices,
    ):
        requested = tuple(
            int(x)
            for x in date_indices
        )

        if not requested:
            return None

        r0 = int(spatial_key[0])
        c0 = int(spatial_key[1])

        candidates = []

        for cache_key, cell in self._cache.items():
            if (
                len(cache_key) != 3
                or
                int(cache_key[0]) != r0
                or
                int(cache_key[1]) != c0
            ):
                continue

            dates = tuple(
                int(x)
                for x in cache_key[2]
            )

            if not dates:
                continue

            if dates == requested:
                continue

            candidates.append((dates, cell))

        if not candidates:
            return None

        pieces = []
        offset = 0

        while offset < len(requested):
            best = None
            remaining = requested[offset:]

            for dates, cell in candidates:
                n = len(dates)

                if (
                    n <= len(remaining)
                    and
                    dates == remaining[:n]
                ):
                    if (
                        best is None
                        or
                        n > len(best[0])
                    ):
                        best = (dates, cell)

            if best is None:
                return None

            pieces.append(best)
            offset += len(best[0])

        first = pieces[0][1]

        for _dates, cell in pieces[1:]:
            if (
                cell.row0 != first.row0
                or
                cell.row1 != first.row1
                or
                cell.col0 != first.col0
                or
                cell.col1 != first.col1
            ):
                return None

            if not np.array_equal(
                cell.geometry_valid,
                first.geometry_valid,
            ):
                return None

        yxt = np.ascontiguousarray(
            np.concatenate(
                [
                    cell.yxt
                    for _dates, cell
                    in pieces
                ],
                axis=2,
            ),
            dtype=np.complex64,
        )

        if yxt.shape[2] != len(requested):
            return None

        mins = [
            float(cell.phase_min)
            for _dates, cell
            in pieces
            if np.isfinite(cell.phase_min)
        ]

        maxs = [
            float(cell.phase_max)
            for _dates, cell
            in pieces
            if np.isfinite(cell.phase_max)
        ]

        return _CanonicalCell(
            row0=first.row0,
            row1=first.row1,
            col0=first.col0,
            col1=first.col1,
            yxt=yxt,
            geometry_valid=np.ascontiguousarray(
                first.geometry_valid,
                dtype=np.bool_,
            ),
            phase_min=(
                min(mins)
                if mins
                else float("nan")
            ),
            phase_max=(
                max(maxs)
                if maxs
                else float("nan")
            ),
        )


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
        date_indices,
    ):
        """
        Correct one canonical spatial cell for only the
        requested acquisition indices.
        """

        (
            global_r0,
            global_c0,
            rows,
            cols,
        ) = self._cell_shape(
            key
        )

        date_indices = tuple(
            int(x)
            for x in date_indices
        )

        date_tag = (
            f"d{date_indices[0]}_"
            f"{date_indices[-1]}_"
            f"n{len(date_indices)}"
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
                f"r{global_r0}_"
                f"{global_r0+rows}_"
                f"c{global_c0}_"
                f"{global_c0+cols}_"
                f"{date_tag}"
            ),
            date_indices=(
                date_indices
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

        return _CanonicalCell(
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


    def _compute_missing(
        self,
        missing,
        *,
        date_indices,
    ):
        """
        Compute canonical cache misses for one temporal subset.

        P11B-1:
          RSLC read_window() receives date_indices.
          GAMMA correction receives the same date_indices.

        Therefore sequential stage M reads/corrects only its
        real acquisitions, not the complete stack.
        """

        date_indices = tuple(
            int(x)
            for x in date_indices
        )

        computed = {}

        total_read = 0.0
        total_correction = 0.0

        phase_min = float(
            "inf"
        )

        phase_max = float(
            "-inf"
        )

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

            # ----------------------------------------------------
            # Stage A:
            # read ONLY selected acquisitions.
            # ----------------------------------------------------

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
                        date_indices=list(
                            date_indices
                        ),
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

            # ----------------------------------------------------
            # Stage B:
            # correct only selected acquisitions.
            # ----------------------------------------------------

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
                        date_indices=(
                            date_indices
                        ),
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

                    cache_key = (
                        int(key[0]),
                        int(key[1]),
                        date_indices,
                    )

                    self._cache_put(
                        cache_key,
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

            raw_wave.clear()

        if not np.isfinite(
            phase_min
        ):
            phase_min = float(
                "nan"
            )

        if not np.isfinite(
            phase_max
        ):
            phase_max = float(
                "nan"
            )

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
        date_indices=None,
    ) -> PhaseTile:
        """
        Read one corrected spatial tile.

        P11B-1:
            date_indices defines which global acquisitions are
            materialized.

        If date_indices=None the original all-date behavior is
        retained.
        """

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

        if date_indices is None:

            date_indices = tuple(
                range(
                    self.ndate
                )
            )

        else:

            date_indices = tuple(
                int(x)
                for x
                in date_indices
            )

        if not date_indices:

            raise ValueError(
                "date_indices must not be empty"
            )

        if len(
            set(date_indices)
        ) != len(
            date_indices
        ):

            raise ValueError(
                "date_indices contains duplicates"
            )

        for x in date_indices:

            if (
                x < 0
                or
                x >= self.ndate
            ):
                raise ValueError(
                    "date index outside stack: "
                    f"{x}"
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

        spatial_keys = (
            self._canonical_keys(
                local_row0=local_row0,
                local_row1=local_row1,
                local_col0=local_col0,
                local_col1=local_col1,
            )
        )

        cells = {}

        missing = []

        hits = 0
        composed_hits = 0

        for key in spatial_keys:

            cache_key = (
                int(key[0]),
                int(key[1]),
                date_indices,
            )

            cell = self._cache_get(
                cache_key
            )

            if cell is None:

                cell = (
                    self
                    ._cache_compose_temporal_cell(
                        spatial_key=key,
                        date_indices=date_indices,
                    )
                )

                if cell is None:

                    missing.append(
                        key
                    )

                else:

                    hits += 1
                    composed_hits += 1

                    cells[
                        key
                    ] = cell

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

        phase_min = float(
            "nan"
        )

        phase_max = float(
            "nan"
        )

        if missing:

            (
                created,
                read_seconds,
                correction_seconds,
                phase_min,
                phase_max,
            ) = self._compute_missing(
                missing,
                date_indices=(
                    date_indices
                ),
            )

            cells.update(
                created
            )

        # --------------------------------------------------------
        # Mosaic only requested dates.
        # --------------------------------------------------------

        yxt = np.empty(
            (
                rows,
                cols,
                len(
                    date_indices
                ),
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

        for key in spatial_keys:

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
                :,
            ] = (
                cell.yxt[
                    sr0:sr1,
                    sc0:sc1,
                    :,
                ]
            )

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
                "canonical phase mosaic "
                "did not fully cover "
                "requested tile"
            )

        log(
            "Canonical phase tile "
            f"r{local_row0}:"
            f"{local_row1} "
            f"c{local_col0}:"
            f"{local_col1}: "
            f"dates={len(date_indices)} "
            f"[{date_indices[0]}.."
            f"{date_indices[-1]}], "
            f"cells={len(spatial_keys)}, "
            f"hits={hits}, "
            f"composed={composed_hits}, "
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
                spatial_keys
            ),
            cache_hits=hits,
            cache_misses=misses,
            cache_composed_hits=(
                composed_hits
            ),
        )


__all__ = [
    "PhaseTile",
    "CachedPhaseSource",
    "GammaStreamingPhaseSource",
]
