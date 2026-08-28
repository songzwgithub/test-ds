from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import set_num_threads

from . import __version__
from .config import cfg_get
from .context import open_from_config
from .runtime import RuntimePlan, build_runtime_plan
from .phase_linking.coherence import compressed_coherence
from .phase_linking.emi import image_pairs
from .phase_linking.emi_threshold import robust_emi_threshold_threaded
from .phase_linking.phase_source import GammaStreamingPhaseSource
from .phase_linking.shp_policy import resolve_shp_policy
from .phase_linking.support_cache import load_exact_support_cache


PROFILE_FORMAT = "pyPSDS-GAMMA-runtime-profile-v1"
ALGORITHM = "sequential-threshold-cholesky-emi-v1"


def _requested_cpu(cfg) -> int | None:
    raw = cfg_get(cfg, "runtime.cpu", None)
    return None if raw in (None, "", "auto") else int(raw)


def _solver_size(cfg, ndate: int) -> int:
    strategy = str(
        cfg_get(cfg, "phase_linking.temporal.strategy", "full_scm")
    ).strip().lower()
    if strategy != "sequential":
        return int(ndate)
    return min(
        int(ndate),
        int(cfg_get(cfg, "phase_linking.temporal.ministack_size", 19))
        + int(cfg_get(cfg, "phase_linking.temporal.max_num_compressed", 5)),
    )


def _autotune_enabled(cfg) -> bool:
    return bool(cfg_get(cfg, "runtime.autotune.enabled", True))


def _manual_phase_link_schedule(cfg) -> bool:
    for key in (
        "runtime.phase_link_workers",
        "runtime.phase_link_chunk_size",
        "runtime.phase_link_batch_size",
    ):
        if cfg_get(cfg, key, None) not in (None, "", "auto"):
            return True
    return False


def build_safe_runtime_plan(cfg, *, ndate: int) -> RuntimePlan:
    return build_runtime_plan(
        ndate=int(ndate),
        memory_fraction=float(cfg_get(cfg, "runtime.memory_fraction", 0.85)),
        requested_cpu=_requested_cpu(cfg),
        max_solver_size=_solver_size(cfg, int(ndate)),
    )


def runtime_profile_path(paths) -> Path:
    return Path(paths.output_dir) / "runtime" / "phase_linking_profile.json"


def _cpu_model() -> str:
    p = Path("/proc/cpuinfo")
    if p.is_file():
        try:
            for line in p.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _numpy_build_identity() -> dict:
    import contextlib
    import io

    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            np.show_config()
    except Exception:
        build_text = ""
    else:
        build_text = buf.getvalue().strip()

    return {
        "numpy_version": np.__version__,
        "numpy_build_sha256": hashlib.sha256(
            build_text.encode("utf-8", errors="replace")
        ).hexdigest(),
    }

def runtime_signature(plan: RuntimePlan) -> dict:
    payload = {
        "software_version": __version__,
        "algorithm": ALGORITHM,
        "cpu_model": _cpu_model(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "effective_cpu_count": int(plan.cpu_count),
        "solver_size": int(plan.phase_link_solver_size),
        "numpy_build": _numpy_build_identity(),
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return {
        "payload": payload,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_profile_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("format") != PROFILE_FORMAT:
        return None
    return data


def _profile_valid(profile: dict, safe_plan: RuntimePlan) -> bool:
    return (
        profile.get("runtime_signature_sha256")
        == runtime_signature(safe_plan)["sha256"]
    )


def _apply_profile(safe_plan: RuntimePlan, profile: dict) -> RuntimePlan:
    schedule = profile["selected_schedule"]
    workers = max(
        1,
        min(
            int(schedule["workers"]),
            int(safe_plan.phase_link_workers),
            int(safe_plan.cpu_count),
        ),
    )
    chunk = max(
        64,
        min(
            int(schedule["chunk_size"]),
            int(safe_plan.phase_link_chunk_size),
        ),
    )
    batch = max(
        chunk,
        min(
            int(schedule["batch_size"]),
            int(safe_plan.phase_link_batch_size),
        ),
    )
    workers = min(workers, max(1, batch // chunk))
    return replace(
        safe_plan,
        phase_link_workers=workers,
        phase_link_chunk_size=chunk,
        phase_link_batch_size=batch,
    )


def resolve_runtime_plan(cfg, paths, *, ndate: int):
    safe = build_safe_runtime_plan(cfg, ndate=int(ndate))
    info = {
        "status": "safe_plan",
        "path": str(runtime_profile_path(paths)),
        "autotune_enabled": _autotune_enabled(cfg),
        "manual_override": _manual_phase_link_schedule(cfg),
    }

    if not info["autotune_enabled"]:
        info["status"] = "disabled"
        return safe, info

    if info["manual_override"]:
        info["status"] = "manual_override"
        return safe, info

    profile = _load_profile_file(runtime_profile_path(paths))
    if profile is None:
        info["status"] = "profile_missing"
        return safe, info

    if not _profile_valid(profile, safe):
        info["status"] = "profile_stale"
        return safe, info

    tuned = _apply_profile(safe, profile)
    info.update(
        {
            "status": "tuned",
            "selected_schedule": profile.get("selected_schedule"),
            "runtime_signature_sha256": profile.get(
                "runtime_signature_sha256"
            ),
        }
    )
    return tuned, info


def _candidate_worker_counts(max_workers: int) -> tuple[int, ...]:
    max_workers = max(1, int(max_workers))
    values = {1, max_workers}
    v = 2
    while v < max_workers:
        values.add(v)
        v *= 2
    return tuple(sorted(x for x in values if 1 <= x <= max_workers))


def _candidate_chunk_sizes(
    safe_chunk: int, sample_points: int
) -> tuple[int, ...]:
    safe_chunk = max(64, int(safe_chunk))
    sample_points = max(1, int(sample_points))
    values = {
        min(safe_chunk, x)
        for x in (128, 256, 512, 1024, 2048)
    }
    values.add(safe_chunk)
    return tuple(
        sorted(
            {
                max(64, min(x, sample_points))
                for x in values
            }
        )
    )


def _compare_solution(reference, candidate) -> float:
    ref_phase, ref_est, *_ = reference
    phase, est, *_ = candidate

    if not np.array_equal(ref_est, est):
        raise RuntimeError(
            "runtime tuning changed estimator classification"
        )

    rf = np.isfinite(ref_phase.real) & np.isfinite(ref_phase.imag)
    cf = np.isfinite(phase.real) & np.isfinite(phase.imag)

    if not np.array_equal(rf, cf):
        raise RuntimeError(
            "runtime tuning changed finite phase pattern"
        )

    if not np.any(rf):
        return 0.0

    maximum = float(np.max(np.abs(ref_phase[rf] - phase[rf])))
    if maximum > 5.0e-6:
        raise RuntimeError(
            "runtime tuning numerical parity failed: "
            f"max complex phase difference={maximum}"
        )
    return maximum


def _build_real_coherence_sample(
    *,
    cfg,
    paths,
    stack,
    row0: int,
    col0: int,
    H: int,
    W: int,
    safe_plan: RuntimePlan,
    sample_points: int,
):
    ndate = len(stack.dates)
    solver_n = int(safe_plan.phase_link_solver_size)
    policy = resolve_shp_policy(cfg, stack.dates)
    processing = Path(paths.output_dir) / "processing"

    count_path = (
        processing
        / "exact_support_cache"
        / "static_shp_count.npy"
    )
    ps_path = processing / "ds_statistics" / "ps_mask.npy"

    if not count_path.is_file():
        raise FileNotFoundError(
            "runtime autotuning requires exact support cache: "
            f"{count_path}"
        )
    if not ps_path.is_file():
        raise FileNotFoundError(
            "runtime autotuning requires PS statistics: "
            f"{ps_path}"
        )

    shp_count = np.load(
        count_path, mmap_mode="r", allow_pickle=False
    )
    ps = np.load(
        ps_path, mmap_mode="r", allow_pickle=False
    )

    support_cache = load_exact_support_cache(
        processing_dir=processing,
        H=H,
        W=W,
        ndate=ndate,
        half_row=policy.half_row,
        half_col=policy.half_col,
        alpha=float(cfg_get(cfg, "selection.shp.alpha", 0.005)),
        validate_input_hashes=True,
    )

    source = GammaStreamingPhaseSource(
        cfg=cfg,
        paths=paths,
        stack=stack,
        base_row0=row0,
        base_col0=col0,
        io_workers=safe_plan.io_workers,
    )

    pairs = image_pairs(solver_n)
    pair_i = np.asarray(pairs[:, 0], dtype=np.int32)
    pair_j = np.asarray(pairs[:, 1], dtype=np.int32)

    cr = max(1, int(source.canonical_rows))
    cc = max(1, int(source.canonical_cols))
    remaining = max(256, int(sample_points))
    blocks = []

    for r0 in range(0, H, cr):
        if remaining <= 0:
            break
        r1 = min(H, r0 + cr)

        for c0 in range(0, W, cc):
            if remaining <= 0:
                break
            c1 = min(W, c0 + cc)

            local = (
                np.asarray(shp_count[r0:r1, c0:c1])
                >= int(policy.formal_min_shp)
            )
            local &= ~np.asarray(
                ps[r0:r1, c0:c1], dtype=np.bool_
            )
            flat = np.flatnonzero(local)

            if flat.size == 0:
                continue

            take = min(remaining, flat.size)
            if take < flat.size:
                pos = np.linspace(
                    0, flat.size - 1, num=take, dtype=np.int64
                )
                flat = flat[pos]

            width = c1 - c0
            rr = (r0 + flat // width).astype(
                np.int32, copy=False
            )
            cols = (c0 + flat % width).astype(
                np.int32, copy=False
            )

            tr0 = max(0, r0 - policy.half_row)
            tr1 = min(H, r1 + policy.half_row)
            tc0 = max(0, c0 - policy.half_col)
            tc1 = min(W, c1 + policy.half_col)

            tile = source.read_tile(
                local_row0=tr0,
                local_row1=tr1,
                local_col0=tc0,
                local_col1=tc1,
            )
            support = support_cache.support(rr, cols)
            lr = (rr - tr0).astype(np.int32, copy=False)
            lc = (cols - tc0).astype(np.int32, copy=False)

            coh = compressed_coherence(
                tile.yxt,
                lr,
                lc,
                support,
                pair_i,
                pair_j,
            )

            finite = np.all(
                np.isfinite(coh.real) & np.isfinite(coh.imag),
                axis=1,
            )
            if np.any(finite):
                cur = np.ascontiguousarray(
                    coh[finite], dtype=np.complex64
                )
                blocks.append(cur)
                remaining -= int(cur.shape[0])

    if not blocks:
        raise RuntimeError(
            "no eligible real DS coherence sample found"
        )

    coherence = np.concatenate(blocks, axis=0)[: int(sample_points)]
    if coherence.shape[0] < 256:
        raise RuntimeError(
            "runtime autotuning sample too small: "
            f"{coherence.shape[0]}"
        )

    return coherence, pairs, policy


def tune_runtime_profile(
    config,
    *,
    sample_points: int | None = None,
    repeats: int | None = None,
    force: bool = False,
):
    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(config)

    strategy = str(
        cfg_get(
            cfg,
            "phase_linking.temporal.strategy",
            "full_scm",
        )
    ).strip().lower()

    if strategy != "sequential":
        raise RuntimeError(
            "runtime autotuning targets the production "
            "sequential Phase Linking path"
        )

    safe = build_safe_runtime_plan(
        cfg, ndate=len(stack.dates)
    )

    if _manual_phase_link_schedule(cfg):
        raise RuntimeError(
            "explicit Phase Linking schedule is configured; "
            "manual values take precedence over autotuning"
        )

    profile_path = runtime_profile_path(paths)
    existing = _load_profile_file(profile_path)

    if (
        not force
        and existing is not None
        and _profile_valid(existing, safe)
    ):
        return existing

    if sample_points is None:
        sample_points = int(
            cfg_get(
                cfg,
                "runtime.autotune.sample_points",
                16384,
            )
        )
    if repeats is None:
        repeats = int(
            cfg_get(cfg, "runtime.autotune.repeats", 2)
        )

    sample_points = max(256, int(sample_points))
    repeats = max(1, int(repeats))

    print("=" * 96)
    print("pyPSDS-GAMMA runtime autotuning")
    print("=" * 96)
    print("config          :", config_path)
    print("scene           :", f"{H} x {W}")
    print("acquisitions    :", len(stack.dates))
    print("solver size     :", safe.phase_link_solver_size)
    print("effective CPU   :", safe.cpu_count)
    print(
        "usable RAM GiB :",
        f"{safe.usable_memory_bytes / 1024**3:.3f}",
    )
    print("target sample   :", sample_points)
    print("repeats         :", repeats)

    set_num_threads(safe.numba_threads)

    # Match the production EMI BLAS/LAPACK regime.
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        _blas_limit_controller = None
    else:
        _blas_limit_controller = threadpool_limits(
            limits=1
        )

    t_sample = perf_counter()
    coherence, pairs, policy = _build_real_coherence_sample(
        cfg=cfg,
        paths=paths,
        stack=stack,
        row0=row0,
        col0=col0,
        H=H,
        W=W,
        safe_plan=safe,
        sample_points=sample_points,
    )
    sample_seconds = perf_counter() - t_sample
    B = int(coherence.shape[0])

    print("actual sample   :", B)
    print("sample build s  :", f"{sample_seconds:.3f}")
    print("SHP K threshold :", policy.formal_min_shp)

    workers_values = _candidate_worker_counts(
        safe.phase_link_workers
    )
    chunk_values = _candidate_chunk_sizes(
        safe.phase_link_chunk_size, B
    )

    candidates = []
    for chunk in chunk_values:
        nchunk = max(1, math.ceil(B / chunk))
        for workers in workers_values:
            if workers <= nchunk:
                candidates.append((workers, chunk))

    safe_pair = (
        min(
            safe.phase_link_workers,
            max(
                1,
                math.ceil(
                    B / safe.phase_link_chunk_size
                ),
            ),
        ),
        min(safe.phase_link_chunk_size, B),
    )
    candidates.append(safe_pair)
    candidates = sorted(set(candidates))

    print("candidate schedules:", candidates)

    beta = float(cfg_get(cfg, "phase_linking.beta", 0.0))
    jitter = float(
        cfg_get(cfg, "phase_linking.gamma_jitter", 1.0e-6)
    )
    mu = float(
        cfg_get(
            cfg,
            "phase_linking.target_eigenvalue",
            0.99,
        )
    )
    ref_idx = int(
        cfg_get(
            cfg,
            "phase_linking.temporal_reference_index",
            0,
        )
    )

    reference = robust_emi_threshold_threaded(
        coherence,
        n_images=safe.phase_link_solver_size,
        pairs=pairs,
        beta=beta,
        gamma_jitter=jitter,
        emi_mu=mu,
        reference_idx=ref_idx,
        workers=1,
        chunk_size=min(512, B),
    )

    rows = []

    print()
    print(
        f"{'workers':>8s} "
        f"{'chunk':>8s} "
        f"{'batch':>8s} "
        f"{'median_s':>12s} "
        f"{'pts/s':>12s} "
        f"{'maxdiff':>12s}"
    )

    for workers, chunk in candidates:
        batch = min(
            int(safe.phase_link_batch_size),
            max(int(chunk), int(workers) * int(chunk)),
        )
        timings = []
        maximum_difference = 0.0

        for _ in range(repeats):
            t0 = perf_counter()
            result = robust_emi_threshold_threaded(
                coherence,
                n_images=safe.phase_link_solver_size,
                pairs=pairs,
                beta=beta,
                gamma_jitter=jitter,
                emi_mu=mu,
                reference_idx=ref_idx,
                workers=workers,
                chunk_size=chunk,
            )
            elapsed = perf_counter() - t0
            timings.append(float(elapsed))
            maximum_difference = max(
                maximum_difference,
                _compare_solution(reference, result),
            )

        median_seconds = float(
            np.median(np.asarray(timings, dtype=np.float64))
        )
        throughput = float(B) / median_seconds

        row = {
            "workers": int(workers),
            "chunk_size": int(chunk),
            "batch_size": int(batch),
            "median_seconds": median_seconds,
            "throughput_point_fits_per_second": throughput,
            "max_complex_phase_difference": maximum_difference,
            "timings_seconds": timings,
        }
        rows.append(row)

        print(
            f"{workers:8d} "
            f"{chunk:8d} "
            f"{batch:8d} "
            f"{median_seconds:12.6f} "
            f"{throughput:12.1f} "
            f"{maximum_difference:12.3e}"
        )

    selected = min(
        rows,
        key=lambda x: (
            x["median_seconds"],
            x["workers"],
            x["chunk_size"],
        ),
    )

    signature = runtime_signature(safe)
    payload = {
        "format": PROFILE_FORMAT,
        "software_version": __version__,
        "algorithm": ALGORITHM,
        "config": str(config_path),
        "runtime_signature": signature["payload"],
        "runtime_signature_sha256": signature["sha256"],
        "sample_points": B,
        "sample_build_seconds": sample_seconds,
        "safe_plan": safe.as_dict(),
        "selected_schedule": {
            "workers": int(selected["workers"]),
            "chunk_size": int(selected["chunk_size"]),
            "batch_size": int(selected["batch_size"]),
        },
        "selected_median_seconds": float(
            selected["median_seconds"]
        ),
        "selected_throughput_point_fits_per_second": float(
            selected["throughput_point_fits_per_second"]
        ),
        "candidates": rows,
        "numerical_parity": "PASS",
    }

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = profile_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            payload, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(profile_path)

    print()
    print("selected schedule:", payload["selected_schedule"])
    print("profile          :", profile_path)
    print("numerical parity : PASS")

    return payload


def ensure_runtime_profile(
    cfg,
    config_path,
    paths,
    *,
    ndate: int,
):
    plan, info = resolve_runtime_plan(
        cfg, paths, ndate=int(ndate)
    )

    if info["status"] in {
        "tuned",
        "disabled",
        "manual_override",
    }:
        return plan, info

    if not _autotune_enabled(cfg):
        return plan, info

    print()
    print(
        "Runtime profile is not reusable; "
        "running bounded real-data autotuning once."
    )

    tune_runtime_profile(
        config_path,
        force=(info["status"] == "profile_stale"),
    )

    return resolve_runtime_plan(
        cfg, paths, ndate=int(ndate)
    )


__all__ = [
    "PROFILE_FORMAT",
    "build_safe_runtime_plan",
    "ensure_runtime_profile",
    "resolve_runtime_plan",
    "runtime_profile_path",
    "runtime_signature",
    "tune_runtime_profile",
]
