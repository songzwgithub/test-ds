from __future__ import annotations

from dataclasses import asdict, dataclass
import os


MiB = 1024 ** 2
GiB = 1024 ** 3


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
    """
    Hardware-aware execution plan.

    Scientific parameters are NOT defined here.  This object controls
    only execution geometry and parallelism.
    """

    cpu_count: int
    available_memory_bytes: int
    usable_memory_bytes: int

    io_workers: int

    # Legacy/general phase-cache tile.
    tile_rows: int
    tile_cols: int

    # Sequential Phase Linking execution tile.
    phase_link_tile_rows: int
    phase_link_tile_cols: int

    # Exact static SHP cache construction tile.
    support_cache_tile_rows: int
    support_cache_tile_cols: int
    support_cache_batch_size: int
    support_cache_support_block: int

    phase_link_workers: int
    phase_link_chunk_size: int
    phase_link_batch_size: int

    # Matrix dimension used by the runtime memory model.
    phase_link_solver_size: int

    blas_threads: int
    numba_threads: int

    def as_dict(self):
        return asdict(self)


def build_runtime_plan(
    *,
    ndate: int,
    memory_fraction: float = 0.85,
    requested_cpu: int | None = None,
    max_solver_size: int | None = None,
) -> RuntimePlan:
    """
    Build one conservative machine-aware execution plan.

    max_solver_size
        Maximum dense phase-linking matrix dimension actually solved
        in one sequential stage.  For the frozen M19 + max-compressed-5
        production path this is normally <= 24 even when the full stack
        contains many more acquisitions.

        Full-SCM callers should pass ndate.
    """

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

    if max_solver_size is None:
        solver_n = int(
            ndate
        )
    else:
        solver_n = min(
            int(ndate),
            max(
                2,
                int(max_solver_size),
            ),
        )

    # ---------------------------------------------------------
    # Phase-linking point workspace.
    #
    # This is intentionally conservative.  It accounts for
    # coherence vectors plus several temporary dense complex128 /
    # float64 matrices used by robust EMI / threshold-Cholesky.
    #
    # Crucially this uses solver_n rather than blindly using the
    # full acquisition count during sequential production.
    # ---------------------------------------------------------

    npair = (
        solver_n
        * (solver_n - 1)
        // 2
    )

    bytes_per_point = int(
        npair * 8
        +
        solver_n * solver_n * (
            16 * 4
            +
            8 * 4
        )
        +
        solver_n * 32
        +
        4096
    )

    # Leave substantial RAM for:
    #   GAMMA streaming/LRU,
    #   Linux page cache,
    #   tile stack,
    #   support arrays,
    #   output mmap,
    #   Python runtime.
    pl_budget = max(
        256 * MiB,
        usable // 2,
    )

    # Small LAPACK jobs scale through independent point chunks.
    # Beyond 64 Python workers overhead commonly starts to dominate;
    # a later benchmark/autotune stage may refine this ceiling.
    worker_cap = max(
        1,
        min(
            cpu,
            64,
        ),
    )

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

    # Preserve useful scheduling granularity.
    chunk = min(
        chunk,
        2048,
    )

    batch = max(
        chunk,
        pl_workers * chunk,
    )

    # Bound the largest transient coherence / solver batch.
    batch = min(
        batch,
        65536,
    )

    # ---------------------------------------------------------
    # Independent acquisition reads.
    # ---------------------------------------------------------

    io_workers = min(
        max(
            1,
            cpu // 4,
        ),
        8,
    )

    # ---------------------------------------------------------
    # General phase-cache tile.
    #
    # Keep the already validated canonical-friendly baseline.
    # ---------------------------------------------------------

    tile_rows = 128
    tile_cols = 256

    # ---------------------------------------------------------
    # Sequential PL spatial tile.
    #
    # Every candidate is an exact multiple of the canonical
    # 128 x 256 GAMMA phase cell:
    #
    #   128 x 256
    #   256 x 512
    #   384 x 768
    #   512 x 1024
    #
    # This avoids arbitrary phase-correction regrouping while
    # reducing orchestration overhead on larger machines.
    # ---------------------------------------------------------

    if (
        usable > 0
        and usable < 12 * GiB
    ):
        pl_tile_rows = 128
        pl_tile_cols = 256

    elif (
        usable >= 96 * GiB
        and cpu >= 32
    ):
        pl_tile_rows = 512
        pl_tile_cols = 1024

    elif (
        usable >= 32 * GiB
        and cpu >= 16
    ):
        pl_tile_rows = 384
        pl_tile_cols = 768

    else:
        pl_tile_rows = 256
        pl_tile_cols = 512

    # ---------------------------------------------------------
    # Exact SHP-support-cache construction.
    #
    # This stage does not carry the full complex stage cube, so
    # larger spatial tiles are safe.  Batch size controls the
    # transient [B, window] boolean GLRT workspace.
    # ---------------------------------------------------------

    if (
        usable > 0
        and usable < 12 * GiB
    ):
        support_tile_rows = 256
        support_tile_cols = 512
        support_batch = 16384

    elif (
        usable >= 96 * GiB
        and cpu >= 32
    ):
        support_tile_rows = 1024
        support_tile_cols = 2048
        support_batch = 65536

    elif (
        usable >= 32 * GiB
        and cpu >= 16
    ):
        support_tile_rows = 768
        support_tile_cols = 1536
        support_batch = 49152

    else:
        support_tile_rows = 512
        support_tile_cols = 1024
        support_batch = 32768

    # Keep the validated exact-GLRT internal support block unchanged
    # in P11A.  P11B can benchmark larger/vectorized packed blocks.
    support_block = 1024


    # ---------------------------------------------------------
    # P11D-4 calibrated sequential EMI schedule.
    #
    # Production benchmark on this CPU for the sequential
    # threshold-Cholesky EMI regime:
    #
    #   solver ~20
    #   B = 8k, 16k, 32k, 65k
    #
    # 4 workers / 512 centers was bit-exact at every tested B,
    # with:
    #
    #   geometric-mean EMI speedup = 1.358x
    #   worst tested speedup        = 1.109x
    #
    # This changes scheduling only. EMI mathematics, chunk
    # arithmetic and BLAS/LAPACK thread count are unchanged.
    #
    # Restrict the calibration to the validated small sequential
    # solver family. Larger solver sizes retain the generic P11A
    # memory-derived planner.
    # ---------------------------------------------------------

    _p11d_solver_size = int(
        solver_n
    )

    if (
        _p11d_solver_size <= 24
        and
        cpu >= 4
    ):

        pl_workers = 4

        chunk = 512

        batch = max(
            chunk,
            pl_workers
            *
            chunk,
        )

        batch = min(
            batch,
            65536,
        )

    return RuntimePlan(
        cpu_count=cpu,
        available_memory_bytes=available,
        usable_memory_bytes=usable,

        io_workers=io_workers,

        tile_rows=tile_rows,
        tile_cols=tile_cols,

        phase_link_tile_rows=(
            pl_tile_rows
        ),
        phase_link_tile_cols=(
            pl_tile_cols
        ),

        support_cache_tile_rows=(
            support_tile_rows
        ),
        support_cache_tile_cols=(
            support_tile_cols
        ),
        support_cache_batch_size=(
            support_batch
        ),
        support_cache_support_block=(
            support_block
        ),

        phase_link_workers=(
            pl_workers
        ),
        phase_link_chunk_size=(
            chunk
        ),
        phase_link_batch_size=(
            batch
        ),

        phase_link_solver_size=(
            solver_n
        ),

        blas_threads=1,
        numba_threads=cpu,
    )
