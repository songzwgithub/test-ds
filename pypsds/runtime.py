from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os


def logical_cpu_count() -> int:

    try:
        return max(
            1,
            len(
                os.sched_getaffinity(0)
            ),
        )
    except Exception:
        return max(
            1,
            int(
                os.cpu_count()
                or 1
            ),
        )


def available_memory_bytes() -> int:

    try:
        with open(
            "/proc/meminfo",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.startswith(
                    "MemAvailable:"
                ):
                    return (
                        int(
                            line.split()[1]
                        )
                        * 1024
                    )

    except OSError:
        pass

    return 0


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    cpu_count: int
    available_memory_bytes: int
    usable_memory_bytes: int

    io_workers: int

    tile_rows: int
    tile_cols: int

    phase_link_workers: int
    phase_link_chunk_size: int
    phase_link_batch_size: int

    blas_threads: int
    numba_threads: int

    def as_dict(self):
        return asdict(self)


def build_runtime_plan(
    *,
    ndate: int,
    memory_fraction: float = 0.85,
    requested_cpu: int | None = None,
) -> RuntimePlan:

    if ndate < 2:
        raise ValueError(
            "ndate must be >= 2"
        )

    cpu = logical_cpu_count()

    if requested_cpu is not None:
        cpu = min(
            cpu,
            max(
                1,
                int(requested_cpu),
            ),
        )

    available = (
        available_memory_bytes()
    )

    frac = min(
        0.95,
        max(
            0.25,
            float(memory_fraction),
        ),
    )

    usable = int(
        available * frac
    )

    # -------------------------------------------------
    # PL workspace estimate.
    #
    # The production robust-EMI kernel transiently owns
    # several dense ndate x ndate complex/real matrices.
    # Use a deliberately conservative per-point estimate.
    # -------------------------------------------------

    npair = (
        ndate
        * (ndate - 1)
        // 2
    )

    bytes_per_point = int(
        npair * 8
        +
        ndate * ndate * (
            16 * 4
            +
            8 * 4
        )
        +
        ndate * 32
        +
        4096
    )

    # Reserve half of usable RAM for mmap cache,
    # operating system, input buffers and Python.
    pl_budget = max(
        256 * 1024**2,
        usable // 2,
    )

    worker_cap = max(
        1,
        min(
            cpu,
            32,
        ),
    )

    # Require at least a modest chunk per worker.
    min_chunk = 64

    max_workers_by_memory = max(
        1,
        pl_budget
        //
        max(
            1,
            bytes_per_point
            * min_chunk,
        ),
    )

    pl_workers = max(
        1,
        min(
            worker_cap,
            max_workers_by_memory,
        ),
    )

    per_worker_budget = max(
        bytes_per_point,
        pl_budget
        //
        pl_workers,
    )

    chunk = max(
        min_chunk,
        int(
            per_worker_budget
            //
            max(
                1,
                bytes_per_point,
            )
        ),
    )

    # Extremely large chunks reduce load balancing.
    chunk = min(
        chunk,
        2048,
    )

    batch = max(
        chunk,
        pl_workers * chunk,
    )

    # Avoid unnecessarily huge single-batch pair arrays.
    batch = min(
        batch,
        65536,
    )

    io_workers = min(
        max(
            1,
            cpu // 4,
        ),
        8,
    )

    # Current validated 128 x 256 tiles are a good
    # starting point. Later stages may override this
    # from their own exact workspace estimates.
    tile_rows = 128
    tile_cols = 256

    return RuntimePlan(
        cpu_count=cpu,
        available_memory_bytes=available,
        usable_memory_bytes=usable,
        io_workers=io_workers,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        phase_link_workers=pl_workers,
        phase_link_chunk_size=chunk,
        phase_link_batch_size=batch,
        blas_threads=1,
        numba_threads=cpu,
    )
