from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path


MiB = 1024 ** 2
GiB = 1024 ** 3

# These defaults work on ordinary Linux hosts, Docker/Podman
# containers, systemd scopes and Slurm cgroup jobs.
#
# Environment overrides are intentionally supported for unusual
# mount namespaces and deterministic tests.
_PROC_ROOT = Path(
    os.environ.get(
        "PYPSDS_PROC_ROOT",
        "/proc",
    )
)

_CGROUP_ROOT = Path(
    os.environ.get(
        "PYPSDS_CGROUP_ROOT",
        "/sys/fs/cgroup",
    )
)

# Linux cgroup-v1 often represents "unlimited" memory with a
# very large integer close to LONG_MAX.
_CGROUP_UNLIMITED_THRESHOLD = 1 << 60


def _read_text(
    path: Path,
) -> str | None:

    try:
        return path.read_text(
            encoding="utf-8",
        ).strip()

    except (
        OSError,
        UnicodeError,
    ):
        return None


def _read_int(
    path: Path,
) -> int | None:

    text = _read_text(path)

    if text is None:
        return None

    try:
        return int(text)

    except ValueError:
        return None


def _self_cgroup_membership():
    """
    Return the current process cgroup membership.

    Returns
    -------
    tuple
        (v2_relative_path, v1_controller_paths)
    """

    text = _read_text(
        _PROC_ROOT
        / "self"
        / "cgroup"
    )

    if not text:
        return None, {}

    v2_path = None
    v1_paths: dict[str, str] = {}

    for line in text.splitlines():

        parts = line.split(
            ":",
            2,
        )

        if len(parts) != 3:
            continue

        _hierarchy, controllers, path = parts

        path = (
            path.strip()
            or "/"
        )

        if controllers == "":
            v2_path = path
            continue

        for controller in controllers.split(","):

            controller = controller.strip()

            if controller:
                v1_paths[
                    controller
                ] = path

    return v2_path, v1_paths


def _join_cgroup_path(
    root: Path,
    relative: str,
) -> Path:

    relative = (
        relative
        .strip()
        .lstrip("/")
    )

    if not relative:
        return root

    return (
        root
        / relative
    )


def _v2_cgroup_dir() -> Path | None:

    v2_path, _ = (
        _self_cgroup_membership()
    )

    if v2_path is None:
        return None

    candidate = _join_cgroup_path(
        _CGROUP_ROOT,
        v2_path,
    )

    if candidate.is_dir():
        return candidate

    return None


def _v1_controller_dir(
    controller: str,
    probe: str,
) -> Path | None:

    _, membership = (
        _self_cgroup_membership()
    )

    relative = membership.get(
        controller
    )

    if relative is None:
        return None

    candidates = (
        _join_cgroup_path(
            _CGROUP_ROOT
            / controller,
            relative,
        ),
        _join_cgroup_path(
            _CGROUP_ROOT,
            relative,
        ),
    )

    for candidate in candidates:

        if (
            candidate
            / probe
        ).is_file():

            return candidate

    return None


def _parse_cpuset_count(
    text: str | None,
) -> int | None:

    if not text:
        return None

    total = 0

    try:

        for field in text.split(","):

            field = field.strip()

            if not field:
                continue

            if "-" in field:

                left, right = (
                    field.split(
                        "-",
                        1,
                    )
                )

                first = int(left)
                last = int(right)

                if last < first:
                    return None

                total += (
                    last
                    - first
                    + 1
                )

            else:
                int(field)
                total += 1

    except ValueError:
        return None

    if total < 1:
        return None

    return total


def _affinity_cpu_count() -> int:

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


def _cgroup_cpu_quota_count() -> int | None:
    """
    Return conservative CPU capacity imposed by cgroup quota.

    Supports:
      cgroup v2 : cpu.max
      cgroup v1 : cpu.cfs_quota_us / cpu.cfs_period_us
    """

    directory = (
        _v2_cgroup_dir()
    )

    if directory is not None:

        text = _read_text(
            directory
            / "cpu.max"
        )

        if text:

            parts = text.split()

            if len(parts) >= 2:

                quota_text = parts[0]

                if quota_text != "max":

                    try:
                        quota = int(
                            quota_text
                        )

                        period = int(
                            parts[1]
                        )

                    except ValueError:
                        pass

                    else:

                        if (
                            quota > 0
                            and period > 0
                        ):
                            return max(
                                1,
                                quota
                                // period,
                            )

    directory = (
        _v1_controller_dir(
            "cpu",
            "cpu.cfs_quota_us",
        )
    )

    if directory is None:

        directory = (
            _v1_controller_dir(
                "cpuacct",
                "cpu.cfs_quota_us",
            )
        )

    if directory is not None:

        quota = _read_int(
            directory
            / "cpu.cfs_quota_us"
        )

        period = _read_int(
            directory
            / "cpu.cfs_period_us"
        )

        if (
            quota is not None
            and period is not None
            and quota > 0
            and period > 0
        ):
            return max(
                1,
                quota
                // period,
            )

    return None


def _cgroup_cpuset_count() -> int | None:

    directory = (
        _v2_cgroup_dir()
    )

    if directory is not None:

        for name in (
            "cpuset.cpus.effective",
            "cpuset.cpus",
        ):

            count = (
                _parse_cpuset_count(
                    _read_text(
                        directory
                        / name
                    )
                )
            )

            if count is not None:
                return count

    directory = (
        _v1_controller_dir(
            "cpuset",
            "cpuset.cpus",
        )
    )

    if directory is not None:

        return (
            _parse_cpuset_count(
                _read_text(
                    directory
                    / "cpuset.cpus"
                )
            )
        )

    return None


def logical_cpu_count() -> int:
    """
    Effective CPU count available to this process.

    The result is bounded by all detected constraints:
      * Linux scheduler affinity
      * cgroup CPU quota
      * cgroup cpuset

    This prevents a container/job from using the host CPU count
    when only a smaller allocation was granted.
    """

    counts = [
        _affinity_cpu_count()
    ]

    quota = (
        _cgroup_cpu_quota_count()
    )

    if quota is not None:
        counts.append(quota)

    cpuset = (
        _cgroup_cpuset_count()
    )

    if cpuset is not None:
        counts.append(cpuset)

    return max(
        1,
        min(counts),
    )


def _host_available_memory_bytes() -> int:

    try:

        with (
            _PROC_ROOT
            / "meminfo"
        ).open(
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

    except (
        OSError,
        ValueError,
        IndexError,
    ):
        pass

    return 0


def _finite_cgroup_limit(
    value: int | None,
) -> int | None:

    if value is None:
        return None

    if value <= 0:
        return None

    if (
        value
        >=
        _CGROUP_UNLIMITED_THRESHOLD
    ):
        return None

    return value


def _cgroup_memory_remaining_bytes() -> int | None:
    """
    Return remaining memory under the active cgroup limit.

    Supports:
      cgroup v2 : memory.max / memory.current
      cgroup v1 : memory.limit_in_bytes / memory.usage_in_bytes

    None means no finite cgroup memory limit was detected.
    """

    directory = (
        _v2_cgroup_dir()
    )

    if directory is not None:

        limit_text = _read_text(
            directory
            / "memory.max"
        )

        if (
            limit_text
            and limit_text != "max"
        ):

            try:
                limit = int(
                    limit_text
                )

            except ValueError:
                limit = None

            limit = (
                _finite_cgroup_limit(
                    limit
                )
            )

            current = _read_int(
                directory
                / "memory.current"
            )

            if (
                limit is not None
                and current is not None
                and current >= 0
            ):
                return max(
                    0,
                    limit
                    - current,
                )

    directory = (
        _v1_controller_dir(
            "memory",
            "memory.limit_in_bytes",
        )
    )

    if directory is not None:

        limit = (
            _finite_cgroup_limit(
                _read_int(
                    directory
                    / "memory.limit_in_bytes"
                )
            )
        )

        current = _read_int(
            directory
            / "memory.usage_in_bytes"
        )

        if (
            limit is not None
            and current is not None
            and current >= 0
        ):
            return max(
                0,
                limit
                - current,
            )

    return None


def available_memory_bytes() -> int:
    """
    Effective currently available RAM.

    On an unrestricted host this is MemAvailable.

    Under a finite cgroup limit it is bounded by:
        min(host MemAvailable,
            cgroup limit - cgroup current usage)

    This prevents runtime planning against RAM that the process
    is not actually allowed to allocate.
    """

    host = (
        _host_available_memory_bytes()
    )

    cgroup = (
        _cgroup_memory_remaining_bytes()
    )

    if cgroup is None:
        return host

    if host > 0:
        return min(
            host,
            cgroup,
        )

    return cgroup


def runtime_resource_snapshot() -> dict:
    """
    Human-readable resource diagnostics for doctor/release checks.
    """

    host_memory = (
        _host_available_memory_bytes()
    )

    cgroup_memory = (
        _cgroup_memory_remaining_bytes()
    )

    affinity = (
        _affinity_cpu_count()
    )

    quota = (
        _cgroup_cpu_quota_count()
    )

    cpuset = (
        _cgroup_cpuset_count()
    )

    return {
        "affinity_cpu_count":
            affinity,

        "cgroup_cpu_quota_count":
            quota,

        "cgroup_cpuset_count":
            cpuset,

        "effective_cpu_count":
            logical_cpu_count(),

        "host_available_memory_bytes":
            host_memory,

        "cgroup_memory_remaining_bytes":
            cgroup_memory,

        "effective_available_memory_bytes":
            available_memory_bytes(),
    }

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
        in one sequential stage.  For the sequential ministack
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
    # in production.  production can benchmark larger/vectorized packed blocks.
    support_block = 1024


    # ---------------------------------------------------------
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
