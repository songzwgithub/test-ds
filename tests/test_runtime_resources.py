from pathlib import Path

import pypsds.runtime as rt


GiB = 1024 ** 3


def _roots(
    tmp_path,
    *,
    cgroup_text: str,
    available_gib: int = 64,
):

    proc = (
        tmp_path
        / "proc"
    )

    cg = (
        tmp_path
        / "cgroup"
    )

    (
        proc
        / "self"
    ).mkdir(
        parents=True
    )

    cg.mkdir()

    (
        proc
        / "meminfo"
    ).write_text(
        (
            f"MemTotal:       "
            f"{available_gib * 1024 * 1024} kB\n"
            f"MemAvailable:   "
            f"{available_gib * 1024 * 1024} kB\n"
        ),
        encoding="utf-8",
    )

    (
        proc
        / "self"
        / "cgroup"
    ).write_text(
        cgroup_text,
        encoding="utf-8",
    )

    return proc, cg


def test_unrestricted_v2_uses_host_memory(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text="0::/job\n",
    )

    job = (
        cg
        / "job"
    )

    job.mkdir()

    (
        job
        / "memory.max"
    ).write_text(
        "max\n",
        encoding="utf-8",
    )

    (
        job
        / "memory.current"
    ).write_text(
        str(2 * GiB),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    assert (
        rt.available_memory_bytes()
        ==
        64 * GiB
    )


def test_v2_memory_limit_bounds_host_memory(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text="0::/job\n",
    )

    job = (
        cg
        / "job"
    )

    job.mkdir()

    (
        job
        / "memory.max"
    ).write_text(
        str(8 * GiB),
        encoding="utf-8",
    )

    (
        job
        / "memory.current"
    ).write_text(
        str(2 * GiB),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    assert (
        rt._cgroup_memory_remaining_bytes()
        ==
        6 * GiB
    )

    assert (
        rt.available_memory_bytes()
        ==
        6 * GiB
    )


def test_v2_cpu_quota_and_cpuset_are_respected(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text="0::/job\n",
    )

    job = (
        cg
        / "job"
    )

    job.mkdir()

    (
        job
        / "cpu.max"
    ).write_text(
        "400000 100000\n",
        encoding="utf-8",
    )

    (
        job
        / "cpuset.cpus.effective"
    ).write_text(
        "0-7\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    monkeypatch.setattr(
        rt.os,
        "sched_getaffinity",
        lambda _pid: set(
            range(32)
        ),
    )

    assert (
        rt._cgroup_cpu_quota_count()
        ==
        4
    )

    assert (
        rt._cgroup_cpuset_count()
        ==
        8
    )

    assert (
        rt.logical_cpu_count()
        ==
        4
    )


def test_cpuset_parser_handles_ranges():

    assert (
        rt._parse_cpuset_count(
            "0-3,8,10-11"
        )
        ==
        7
    )

    assert (
        rt._parse_cpuset_count(
            "2"
        )
        ==
        1
    )

    assert (
        rt._parse_cpuset_count(
            ""
        )
        is None
    )


def test_v1_cpu_and_memory_limits(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text=(
            "5:memory:/docker/abc\n"
            "4:cpu,cpuacct:/docker/abc\n"
            "3:cpuset:/docker/abc\n"
        ),
    )

    memory = (
        cg
        / "memory"
        / "docker"
        / "abc"
    )

    cpu = (
        cg
        / "cpu"
        / "docker"
        / "abc"
    )

    cpuset = (
        cg
        / "cpuset"
        / "docker"
        / "abc"
    )

    memory.mkdir(
        parents=True
    )

    cpu.mkdir(
        parents=True
    )

    cpuset.mkdir(
        parents=True
    )

    (
        memory
        / "memory.limit_in_bytes"
    ).write_text(
        str(10 * GiB),
        encoding="utf-8",
    )

    (
        memory
        / "memory.usage_in_bytes"
    ).write_text(
        str(3 * GiB),
        encoding="utf-8",
    )

    (
        cpu
        / "cpu.cfs_quota_us"
    ).write_text(
        "600000\n",
        encoding="utf-8",
    )

    (
        cpu
        / "cpu.cfs_period_us"
    ).write_text(
        "100000\n",
        encoding="utf-8",
    )

    (
        cpuset
        / "cpuset.cpus"
    ).write_text(
        "0-7\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    monkeypatch.setattr(
        rt.os,
        "sched_getaffinity",
        lambda _pid: set(
            range(32)
        ),
    )

    assert (
        rt.available_memory_bytes()
        ==
        7 * GiB
    )

    assert (
        rt.logical_cpu_count()
        ==
        6
    )


def test_runtime_plan_uses_effective_resources(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text="0::/job\n",
    )

    job = (
        cg
        / "job"
    )

    job.mkdir()

    (
        job
        / "memory.max"
    ).write_text(
        str(8 * GiB),
        encoding="utf-8",
    )

    (
        job
        / "memory.current"
    ).write_text(
        str(2 * GiB),
        encoding="utf-8",
    )

    (
        job
        / "cpu.max"
    ).write_text(
        "400000 100000\n",
        encoding="utf-8",
    )

    (
        job
        / "cpuset.cpus.effective"
    ).write_text(
        "0-15\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    monkeypatch.setattr(
        rt.os,
        "sched_getaffinity",
        lambda _pid: set(
            range(32)
        ),
    )

    plan = (
        rt.build_runtime_plan(
            ndate=38,
            max_solver_size=24,
        )
    )

    assert plan.cpu_count == 4

    assert (
        plan.available_memory_bytes
        ==
        6 * GiB
    )

    assert (
        plan.phase_link_workers
        <=
        4
    )

    assert (
        plan.numba_threads
        ==
        4
    )

    # <12 GiB effective memory must select the
    # conservative low-memory spatial plan.
    assert (
        plan.phase_link_tile_rows,
        plan.phase_link_tile_cols,
    ) == (
        128,
        256,
    )

    assert (
        plan.support_cache_tile_rows,
        plan.support_cache_tile_cols,
    ) == (
        256,
        512,
    )


def test_requested_cpu_still_caps_effective_cpu(
    tmp_path,
    monkeypatch,
):

    proc, cg = _roots(
        tmp_path,
        cgroup_text="0::/job\n",
    )

    job = (
        cg
        / "job"
    )

    job.mkdir()

    (
        job
        / "cpu.max"
    ).write_text(
        "800000 100000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rt,
        "_PROC_ROOT",
        proc,
    )

    monkeypatch.setattr(
        rt,
        "_CGROUP_ROOT",
        cg,
    )

    monkeypatch.setattr(
        rt.os,
        "sched_getaffinity",
        lambda _pid: set(
            range(32)
        ),
    )

    plan = (
        rt.build_runtime_plan(
            ndate=38,
            requested_cpu=3,
            max_solver_size=24,
        )
    )

    assert plan.cpu_count == 3
    assert plan.numba_threads == 3
    assert plan.phase_link_workers <= 3
