from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypsds.context import open_from_config
from pypsds.selection.shp import glrt_statistic, glrt_threshold
from pypsds.phase_linking.coherence import compressed_coherence
from pypsds.phase_linking.shp_policy import (
    resolve_shp_policy,
    write_shp_policy_json,
)
from pypsds.phase_linking.sequential_production import run_sequential_production
import pypsds.phase_linking.phase_source as phase_source_module
import pypsds.gamma.phase_correction as phase_correction_module
from pypsds.phase_linking.phase_source import (
    GammaStreamingPhaseSource,
)
from pypsds.runtime import build_runtime_plan
from pypsds.phase_linking.emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
    median_pair_coherence,
)


def make_support_batch(
    scale2,
    valid,
    ps,
    rows,
    cols,
    *,
    half_row,
    half_col,
    alpha,
    ndate,
):
    B = rows.size
    wh = 2 * half_row + 1
    ww = 2 * half_col + 1
    out = np.zeros((B, wh, ww), dtype=bool)
    center_scale = scale2[rows, cols]
    thr = glrt_threshold(alpha)
    H, W = valid.shape

    for ky, dy in enumerate(range(-half_row, half_row + 1)):
        for kx, dx in enumerate(range(-half_col, half_col + 1)):
            if dy == 0 and dx == 0:
                continue
            rr = rows + dy
            cc = cols + dx
            inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if not np.any(inside):
                continue
            ids = np.flatnonzero(inside)
            r2 = rr[ids]
            c2 = cc[ids]
            ngood = valid[r2, c2] & ~ps[r2, c2]
            if not np.any(ngood):
                continue
            ids2 = ids[ngood]
            r3 = rr[ids2]
            c3 = cc[ids2]
            stat = glrt_statistic(
                center_scale[ids2],
                scale2[r3, c3],
                nslc=ndate,
            )
            out[ids2, ky, kx] = np.isfinite(stat) & (stat < thr)
    return out


def open_or_create_npy(path: Path, *, shape, dtype, fill, resume: bool):
    if resume and path.is_file():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"resume cache mismatch: {path}: shape={arr.shape}, dtype={arr.dtype}, "
                f"expected={shape}/{np.dtype(dtype)}"
            )
        return arr

    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    if fill is not None:
        arr[...] = fill
        arr.flush()

    return arr


def save_map(arr, path, title, label, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(arr, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)




# ============================================================================
# production production PhaseSource boundary
# ============================================================================

def _phase_source_sha256_file(
    path,
):

    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                4 * 1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _phase_source_file_identity(
    path,
):

    if path is None:
        return None

    try:

        path = Path(
            path
        )

    except Exception:

        return None


    if not path.is_file():

        return None


    st = path.stat()


    return {
        "path":
            str(
                path.resolve()
            ),

        "size":
            int(
                st.st_size
            ),
    }


def _phase_source_raster_identity(
    raster,
):

    for attr in (
        "path",
        "filepath",
        "file_path",
        "filename",
        "data_path",
        "slc_path",
    ):

        value = getattr(
            raster,
            attr,
            None,
        )

        ident = (
            _phase_source_file_identity(
                value
            )
            if value is not None
            else None
        )

        if ident is not None:

            return ident


    return {
        "class":
            (
                raster.__class__.__module__
                +
                "."
                +
                raster.__class__.__qualname__
            ),

        "length":
            str(
                getattr(
                    raster,
                    "length",
                    "",
                )
            ),

        "width":
            str(
                getattr(
                    raster,
                    "width",
                    "",
                )
            ),
    }


def _write_phase_source_token(
    *,
    outdir,
    config_path,
    backend,
    H,
    W,
    ndate,
    stack,
    source=None,
    cache_path=None,
):

    module_path = Path(
        phase_source_module.__file__
    ).resolve()


    payload = {
        "format":
            "pyPSDS-GAMMA-phase-source-v1",

        "backend":
            str(
                backend
            ),

        "scene":
            [
                int(H),
                int(W),
                int(ndate),
            ],

        "dates":
            [
                str(x)
                for x
                in stack.dates
            ],

        "config_sha256":
            _phase_source_sha256_file(
                config_path
            ),

        "phase_source_sha256":
            _phase_source_sha256_file(
                module_path
            ),

        "phase_correction_sha256":
            _phase_source_sha256_file(
                Path(
                    phase_correction_module.__file__
                ).resolve()
            ),
    }


    if backend == "cache":

        payload[
            "corrected_yxt"
        ] = _phase_source_file_identity(
            cache_path
        )


    elif backend == "gamma":

        if source is None:

            raise RuntimeError(
                "gamma source token requires source"
            )


        payload.update(
            {
                "base_row0":
                    int(
                        source.base_row0
                    ),

                "base_col0":
                    int(
                        source.base_col0
                    ),

                # NUMERICAL grouping definition.
                "canonical_rows":
                    int(
                        source.canonical_rows
                    ),

                "canonical_cols":
                    int(
                        source.canonical_cols
                    ),

                "rasters":
                    [
                        _phase_source_raster_identity(
                            x
                        )
                        for x
                        in source.stack.rasters
                    ],
            }
        )


    else:

        raise ValueError(
            f"invalid phase-source backend: {backend}"
        )


    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    )


    path = (
        Path(
            outdir
        )
        /
        "cache"
        /
        "phase_source_checkpoint_token.json"
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Stable token:
    # do not change mtime if its scientific identity is unchanged.
    if path.is_file():

        try:

            if (
                path.read_text(
                    encoding="utf-8"
                )
                ==
                text
            ):

                return path

        except Exception:

            pass


    tmp = path.with_name(
        path.name
        +
        ".tmp"
    )


    tmp.write_text(
        text,
        encoding="utf-8",
    )


    tmp.replace(
        path
    )


    return path


class _PhaseSourceYXTProxy:
    """
    Lazy [Y,X,T] view over GammaStreamingPhaseSource.

    The numerical phase-source implementation remains unchanged.
    Only bounded spatial slices are materialized.
    """

    is_phase_source_proxy = True

    ndim = 3

    dtype = np.dtype(
        np.complex64
    )


    def __init__(
        self,
        source,
        *,
        H,
        W,
        ndate,
        token_path,
    ):

        self.phase_source = source

        self.shape = (
            int(H),
            int(W),
            int(ndate),
        )

        # production already fingerprints mmap-like sources through
        # the filename attribute. Point it at the stable source
        # identity token instead of a full corrected-YXT file.
        self.filename = str(
            Path(
                token_path
            ).resolve()
        )

        self.expected_geometry = None


    @staticmethod
    def _slice_bounds(
        key,
        size,
        label,
    ):

        if not isinstance(
            key,
            slice,
        ):

            raise TypeError(
                f"{label} must be a slice"
            )


        start, stop, step = key.indices(
            size
        )


        if step != 1:

            raise ValueError(
                f"{label} slice step must equal 1"
            )


        return (
            start,
            stop,
        )



    def __getitem__(
        self,
        key,
    ):
        """
        P11B-1 lazy date-aware access.

        Resolve the temporal slice BEFORE asking the phase source
        for data.  The old implementation read/corrected all dates
        and sliced afterwards.
        """

        if not (
            isinstance(
                key,
                tuple,
            )
            and
            len(key)
            ==
            3
        ):

            raise TypeError(
                "streaming YXT requires "
                "[row_slice, col_slice, date_slice]"
            )

        rk, ck, tk = key

        r0, r1 = self._slice_bounds(
            rk,
            self.shape[0],
            "row",
        )

        c0, c1 = self._slice_bounds(
            ck,
            self.shape[1],
            "column",
        )

        if not isinstance(
            tk,
            slice,
        ):

            raise TypeError(
                "streaming YXT date access "
                "must be a slice"
            )

        t0, t1 = self._slice_bounds(
            tk,
            self.shape[2],
            "date",
        )

        if t1 <= t0:

            return np.empty(
                (
                    r1 - r0,
                    c1 - c0,
                    0,
                ),
                dtype=np.complex64,
            )

        date_indices = tuple(
            range(
                t0,
                t1,
            )
        )

        tile = self.phase_source.read_tile(
            local_row0=r0,
            local_row1=r1,
            local_col0=c0,
            local_col1=c1,
            date_indices=(
                date_indices
            ),
        )

        if self.expected_geometry is not None:

            expected = np.asarray(
                self.expected_geometry[
                    r0:r1,
                    c0:c1,
                ],
                dtype=np.bool_,
            )

            if not np.array_equal(
                tile.geometry_valid,
                expected,
            ):

                raise RuntimeError(
                    "Gamma streaming geometry parity "
                    f"failure for "
                    f"rows={r0}:{r1}, "
                    f"cols={c0}:{c1}"
                )

        if tile.yxt.shape != (
            r1 - r0,
            c1 - c0,
            t1 - t0,
        ):

            raise RuntimeError(
                "date-aware phase-source shape "
                f"mismatch: {tile.yxt.shape}"
            )

        return np.ascontiguousarray(
            tile.yxt,
            dtype=np.complex64,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--center-mode", choices=["moraine", "all"], default="all")
    ap.add_argument("--moraine-npz", default=None)
    ap.add_argument("--yxt-cache", default=None)
    ap.add_argument("--geom-valid", default=None)
    ap.add_argument("--half-row", type=int, default=5)
    ap.add_argument("--half-col", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--min-shp", type=int, default=48)
    ap.add_argument("--adi-max", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma-jitter", type=float, default=1e-6)
    ap.add_argument("--emi-mu", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--pl-workers", type=int, default=0)
    ap.add_argument("--pl-chunk-size", type=int, default=0)
    ap.add_argument("--tile-rows", type=int, default=0)
    ap.add_argument("--tile-cols", type=int, default=0)
    ap.add_argument("--support-block", type=int, default=0)
    ap.add_argument(
        "--prefetch-tiles",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "One-ahead GAMMA tile prefetch (experimental, default off). "
            "0 disables prefetch; 1 overlaps the next tile read "
            "with current CPU processing."
        ),
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-centers", type=int, default=0)
    args = ap.parse_args()

    cfg, config_path, paths, stack, (row0, col0, H, W) = open_from_config(args.config)
    ndate = len(stack.dates)

    shp_policy = resolve_shp_policy(
        cfg,
        stack.dates,
        base_half_row=args.half_row,
        base_half_col=args.half_col,
        base_formal_min_shp=args.min_shp,
    )
    args.half_row = int(shp_policy.half_row)
    args.half_col = int(shp_policy.half_col)
    args.min_shp = int(shp_policy.formal_min_shp)

    # ---------------------------------------------------------
    # Hardware-aware execution geometry.
    #
    # This changes execution only.  Scientific SHP/EMI settings
    # remain owned by shp_policy/config.
    # ---------------------------------------------------------

    auto_plan = build_runtime_plan(
        ndate=ndate,
        max_solver_size=(
            shp_policy.max_solver_size
        ),
    )

    if args.batch_size <= 0:
        args.batch_size = (
            auto_plan.phase_link_batch_size
        )

    if args.pl_workers <= 0:
        args.pl_workers = (
            auto_plan.phase_link_workers
        )

    if args.pl_chunk_size <= 0:
        args.pl_chunk_size = (
            auto_plan.phase_link_chunk_size
        )

    if args.tile_rows <= 0:
        args.tile_rows = (
            auto_plan.phase_link_tile_rows
        )

    if args.tile_cols <= 0:
        args.tile_cols = (
            auto_plan.phase_link_tile_cols
        )

    if args.support_block <= 0:
        args.support_block = (
            auto_plan.support_cache_support_block
        )

    pairs = image_pairs(ndate)
    pi, pj = pairs[:, 0], pairs[:, 1]

    outdir = Path(paths.output_dir) / "processing"
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    write_shp_policy_json(
        outdir / "shp_policy.json",
        shp_policy,
    )

    print("=" * 80)
    print("pyPSDS-GAMMA - CPU production dispatcher")
    print("=" * 80)
    print(f"config          : {config_path}")
    print(f"scene           : {H} x {W}")
    print(f"dates           : {ndate}")
    print(f"center mode     : {args.center_mode}")
    print(f"GLRT            : {2*args.half_row+1} x {2*args.half_col+1}, alpha={args.alpha}")
    print(f"Kmin            : {args.min_shp}")
    print(f"SHP policy      : {shp_policy.mode}")
    print(f"solver max dim  : {shp_policy.max_solver_size}")
    print(f"state Kmin      : {shp_policy.state_min_shp}")
    print(f"full-SCM Kmin   : {shp_policy.full_scm_rank_min_shp}")
    print(f"window adapted  : {shp_policy.window_adapted}")
    print(f"rank guard      : {shp_policy.rank_guard}")
    print(f"batch size      : {args.batch_size}")
    print(f"PL workers      : {args.pl_workers}")
    print(f"PL chunk        : {args.pl_chunk_size}")
    print(
        f"PL tile         : "
        f"{args.tile_rows} x {args.tile_cols}"
    )
    print(f"support block   : {args.support_block}")
    print(f"prefetch tiles  : {args.prefetch_tiles}")
    print(f"Numba threads   : {os.environ.get('NUMBA_NUM_THREADS', 'runtime-default')}")
    print(f"resume          : {args.resume}")

    # ---------------------------------------------------------
    # Production DS amplitude statistics
    #
    # Prepared tile-wise by build_ds_statistics.py.
    # Do not reread the entire RSLC stack here.
    # ---------------------------------------------------------

    t_all = time.time()

    stats_dir = (
        outdir
        /
        "ds_statistics"
    )

    raw_valid_path = (
        stats_dir
        /
        "raw_valid.npy"
    )

    scale2_path = (
        stats_dir
        /
        "rayleigh_scale2.npy"
    )

    ps_stats_path = (
        stats_dir
        /
        "ps_mask.npy"
    )

    for required in (
        raw_valid_path,
        scale2_path,
        ps_stats_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                "Missing DS statistics product: "
                f"{required}\n"
                "Run scripts/build_ds_statistics.py first."
            )

    raw_valid = np.load(
        raw_valid_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    scale2 = np.load(
        scale2_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    ps_stats = np.load(
        ps_stats_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    for name, arr in (
        ("raw_valid", raw_valid),
        ("rayleigh_scale2", scale2),
        ("ps_mask", ps_stats),
    ):
        if arr.shape != (H, W):
            raise RuntimeError(
                f"{name} shape={arr.shape}, "
                f"expected={(H, W)}"
            )

    valid = np.asarray(
        raw_valid,
        dtype=np.bool_,
    ).copy()

    ps = np.asarray(
        ps_stats,
        dtype=np.bool_,
    ).copy()

    print(
        f"raw valid       : {valid.sum()} "
        f"({100*valid.mean():.3f}%)"
    )

    print(
        f"PS raw          : {ps.sum()} "
        f"({100*ps.mean():.3f}%)"
    )

    # ---------------------------------------------------------
    # Fast C-contiguous corrected stack
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # production production PhaseSource selection.
    #
    # auto:
    #   corrected-YXT present -> frozen cache backend
    #   corrected-YXT absent  -> canonical GAMMA streaming
    #
    # Explicit:
    #   PYPSDS_PHASE_SOURCE=cache
    #   PYPSDS_PHASE_SOURCE=gamma
    # ---------------------------------------------------------

    _pre_temporal_strategy = str(
        cfg.get(
            "phase_linking",
            {},
        )
        .get(
            "temporal",
            {},
        )
        .get(
            "strategy",
            "full_scm",
        )
    ).lower()


    _source_mode = os.environ.get(
        "PYPSDS_PHASE_SOURCE",
        "auto",
    ).strip().lower()


    if _source_mode not in {
        "auto",
        "cache",
        "gamma",
    }:

        raise ValueError(
            "PYPSDS_PHASE_SOURCE must be "
            "auto/cache/gamma"
        )


    _candidate_yxt_path = (
        Path(
            args.yxt_cache
        )
        if args.yxt_cache
        else
        outdir
        /
        "cache"
        /
        "phase_corrected_yxt.npy"
    )


    if (
        _pre_temporal_strategy
        ==
        "sequential"
        and
        _source_mode
        ==
        "auto"
    ):

        _source_mode = (
            "cache"
            if _candidate_yxt_path.is_file()
            else
            "gamma"
        )


    if (
        _pre_temporal_strategy
        ==
        "sequential"
        and
        _source_mode
        ==
        "gamma"
    ):

        _runtime_plan = auto_plan


        _phase_source = (
            GammaStreamingPhaseSource(
                cfg=cfg,
                paths=paths,
                stack=stack,

                base_row0=row0,
                base_col0=col0,

                io_workers=(
                    _runtime_plan.io_workers
                ),
            )
        )


        _phase_source_token = (
            _write_phase_source_token(
                outdir=outdir,
                config_path=config_path,

                backend="gamma",

                H=H,
                W=W,
                ndate=ndate,

                stack=stack,

                source=_phase_source,
            )
        )


        rslc_yxt = (
            _PhaseSourceYXTProxy(
                _phase_source,

                H=H,
                W=W,
                ndate=ndate,

                token_path=(
                    _phase_source_token
                ),
            )
        )


        print(
            "phase source     : gamma"
        )

        print(
            "full YXT cache   : not required"
        )

    else:

        if args.yxt_cache:
            yxt_path = Path(args.yxt_cache)
        else:
            yxt_path = outdir / "cache" / "phase_corrected_yxt.npy"
        if not yxt_path.is_file():
            raise FileNotFoundError(
                f"Missing v1.0 YXT cache: {yxt_path}\n"
                "Run the phase_cache stage (scripts/build_phase_cache.py) first."
            )
        rslc_yxt = np.load(yxt_path, mmap_mode="r")
        if rslc_yxt.shape != (H, W, ndate):
            raise RuntimeError(f"YXT cache shape={rslc_yxt.shape}, expected={(H,W,ndate)}")


        if (
            _pre_temporal_strategy
            ==
            "sequential"
        ):

            _write_phase_source_token(
                outdir=outdir,
                config_path=config_path,

                backend="cache",

                H=H,
                W=W,
                ndate=ndate,

                stack=stack,

                cache_path=yxt_path,
            )


            print(
                "phase source     : cache"
            )


    if args.geom_valid:
        geom_path = Path(args.geom_valid)
    else:
        geom_path = Path(paths.output_dir) / "processing" / "cache" / "phase_geometry_valid.npy"
    geom_valid = np.load(geom_path, mmap_mode="r").astype(bool, copy=False)


    if getattr(
        rslc_yxt,
        "is_phase_source_proxy",
        False,
    ):

        rslc_yxt.expected_geometry = (
            geom_valid
        )

    # DS processing is restricted to the geometry-valid domain.
    #
    # IMPORTANT:
    # Keep `ps` as the RAW ADI PS mask here.
    # Step06b owns the formal PS geometry/phase finalization and
    # must remain able to diagnose geometry-invalid PS pixels.
    valid &= geom_valid

    print(
        f"geometry-valid DS domain : {valid.sum()} "
        f"({100*valid.mean():.3f}%)"
    )

    print(
        f"raw ADI PS               : {ps.sum()} "
        f"({100*ps.mean():.3f}%)"
    )

    # ---------------------------------------------------------
    # Center domain
    # ---------------------------------------------------------
    if args.center_mode == "moraine":
        prior_path = Path(args.moraine_npz) if args.moraine_npz else Path(paths.output_dir) / "processing" / "moraine_center_prior.npz"
        z = np.load(prior_path)
        center_prior = z["candidate_mask"].astype(bool)
    else:
        # Production scientific domain:
        # every geometry-valid non-PS pixel.
        center_prior = valid.copy()

    center_prior &= valid
    center_prior &= ~ps
    rr, cc = np.where(center_prior)
    rr = rr.astype(np.int32)
    cc = cc.astype(np.int32)
    if args.max_centers > 0 and len(rr) > args.max_centers:
        ids = np.linspace(0, len(rr) - 1, args.max_centers, dtype=np.int64)
        rr, cc = rr[ids], cc[ids]
    total = len(rr)
    print(f"DS centers      : {total}")

    # Save exact center order for reliable resume.
    rows_path = outdir / "center_rows.npy"
    cols_path = outdir / "center_cols.npy"
    if args.resume and rows_path.is_file() and cols_path.is_file():
        old_r = np.load(rows_path)
        old_c = np.load(cols_path)
        if not (np.array_equal(old_r, rr) and np.array_equal(old_c, cc)):
            raise RuntimeError("Resume center list does not match current configuration")
    else:
        np.save(rows_path, rr)
        np.save(cols_path, cc)

    # ---------------------------------------------------------
    # Temporal production dispatcher
    # ---------------------------------------------------------

    temporal_strategy = str(
        cfg.get(
            "phase_linking",
            {},
        )
        .get(
            "temporal",
            {},
        )
        .get(
            "strategy",
            "full_scm",
        )
    ).lower()

    print(
        f"temporal strategy: {temporal_strategy}"
    )

    if temporal_strategy == "sequential":

        if args.resume:
            print(
                "NOTE: --resume is ignored for sequential "
                "production; Phase linking is rebuilt deterministically."
            )

        run_sequential_production(
            cfg=cfg,
            config_path=config_path,

            paths=paths,
            stack=stack,

            H=H,
            W=W,

            outdir=outdir,

            yxt=rslc_yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom_valid,
            ps=ps,

            center_prior=center_prior,

            args=args,
        )

        return

    if temporal_strategy != "full_scm":

        raise RuntimeError(
            "unsupported temporal strategy: "
            f"{temporal_strategy!r}"
        )

    # ---------------------------------------------------------
    # Output mmap arrays
    # ---------------------------------------------------------
    shp_count = open_or_create_npy(outdir / "shp_count.npy", shape=(H, W), dtype=np.int16, fill=-1, resume=args.resume)
    tc_map = open_or_create_npy(outdir / "temporal_coherence.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    pair_map = open_or_create_npy(outdir / "median_pair_coherence.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    estimator = open_or_create_npy(outdir / "estimator_code.npy", shape=(H, W), dtype=np.uint8, fill=255, resume=args.resume)
    emi_eig_map = open_or_create_npy(outdir / "emi_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    evd_eig_map = open_or_create_npy(outdir / "evd_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    gamma_min_map = open_or_create_npy(outdir / "gamma_min_eigenvalue.npy", shape=(H, W), dtype=np.float32, fill=np.nan, resume=args.resume)
    pl_valid = open_or_create_npy(outdir / "pl_valid.npy", shape=(H, W), dtype=np.bool_, fill=False, resume=args.resume)
    processed = open_or_create_npy(outdir / "processed_centers.npy", shape=(total,), dtype=np.bool_, fill=False, resume=args.resume)
    linked_phase = open_or_create_npy(
        outdir / "linked_phase.npy",
        shape=(ndate, H, W),
        dtype=np.complex64,
        fill=None,
        resume=args.resume,
    )

    # Persist static masks for downstream use.
    np.save(outdir / "ps_mask.npy", ps)
    np.save(outdir / "center_prior.npy", center_prior)

    # Warm-up covariance JIT with tiny legal shapes, excluded from timings.
    if total:
        warm_support = make_support_batch(
            scale2, valid, ps, rr[:1], cc[:1],
            half_row=args.half_row, half_col=args.half_col,
            alpha=args.alpha, ndate=ndate,
        )
        _ = compressed_coherence(rslc_yxt, rr[:1], cc[:1], warm_support, pi, pj)

    # ---------------------------------------------------------
    # Production batches
    # ---------------------------------------------------------
    linked_n = int(np.count_nonzero(pl_valid))
    t0 = time.time()
    support_s = covariance_s = pl_s = write_s = 0.0

    for b0 in range(0, total, args.batch_size):
        b1 = min(total, b0 + args.batch_size)
        if np.all(processed[b0:b1]):
            print(f"centers {b1:8d}/{total:8d}: RESUME skip")
            continue

        br = rr[b0:b1]
        bc = cc[b0:b1]

        ts = time.time()
        support = make_support_batch(
            scale2, valid, ps, br, bc,
            half_row=args.half_row,
            half_col=args.half_col,
            alpha=args.alpha,
            ndate=ndate,
        )
        K = np.sum(support, axis=(1, 2)).astype(np.int16)
        shp_count[br, bc] = K
        support_s += time.time() - ts

        good = K >= args.min_shp
        nlinked_batch = 0
        if np.any(good):
            gr = br[good]
            gc = bc[good]
            gs = support[good]

            ts = time.time()
            coh = compressed_coherence(rslc_yxt, gr, gc, gs, pi, pj)
            covariance_s += time.time() - ts

            ts = time.time()
            ph, est, emi_eig, evd_eig, gamma_min = robust_emi_threaded(
                coh,
                n_images=ndate,
                pairs=pairs,
                beta=args.beta,
                gamma_jitter=args.gamma_jitter,
                emi_mu=args.emi_mu,
                reference_idx=0,
                workers=args.pl_workers,
                chunk_size=args.pl_chunk_size,
            )
            tc = temporal_coherence(coh, ph, pairs)
            pc = median_pair_coherence(coh)
            pl_s += time.time() - ts

            ok = (est != 255) & np.isfinite(tc)
            gr2 = gr[ok]
            gc2 = gc[ok]
            if gr2.size:
                ts = time.time()
                tc_map[gr2, gc2] = tc[ok]
                pair_map[gr2, gc2] = pc[ok]
                estimator[gr2, gc2] = est[ok]
                emi_eig_map[gr2, gc2] = emi_eig[ok]
                evd_eig_map[gr2, gc2] = evd_eig[ok]
                gamma_min_map[gr2, gc2] = gamma_min[ok]
                pl_valid[gr2, gc2] = True
                linked_phase[:, gr2, gc2] = ph[ok].T
                write_s += time.time() - ts
                nlinked_batch = int(gr2.size)
                linked_n += nlinked_batch

        processed[b0:b1] = True
        for arr in (
            shp_count, tc_map, pair_map, estimator, emi_eig_map,
            evd_eig_map, gamma_min_map, pl_valid, processed, linked_phase,
        ):
            arr.flush()

        elapsed = time.time() - t0
        done = b1
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"centers {done:8d}/{total:8d} ({100*done/total:6.2f}%) "
            f"linked={linked_n:8d} batch_link={nlinked_batch:6d} "
            f"wall={elapsed/60:6.2f}m ETA={eta/60:6.2f}m | "
            f"support={support_s:6.1f}s cov={covariance_s:6.1f}s "
            f"PL={pl_s:6.1f}s write={write_s:6.1f}s"
        )

    # ---------------------------------------------------------
    # Fill PS phase, but DO NOT set DS TC=1 for PS QA.
    # ---------------------------------------------------------
    pr, pcx = np.where(ps & geom_valid)
    if pr.size:
        pph = rslc_yxt[pr, pcx, :]
        pph = np.exp(1j * np.angle(pph)).astype(np.complex64)
        pph *= np.exp(-1j * np.angle(pph[:, 0]))[:, None]
        linked_phase[:, pr, pcx] = pph.T
        linked_phase.flush()

    # ---------------------------------------------------------
    # QA only on actually calculated DS pixels.
    # ---------------------------------------------------------
    ds_mask = np.asarray(pl_valid) & ~ps
    tc_view = np.where(ds_mask, np.asarray(tc_map), np.nan)
    pc_view = np.where(ds_mask, np.asarray(pair_map), np.nan)
    shp_view = np.where(np.asarray(shp_count) >= 0, np.asarray(shp_count).astype(np.float32), np.nan)
    est_view = np.where(ds_mask, np.asarray(estimator).astype(np.float32), np.nan)

    save_map(shp_view, figdir / "processing_glrt_shp_count.png", "v1.0 - GLRT covariance support count", "SHP count")
    save_map(tc_view, figdir / "processing_temporal_coherence.png", "v1.0 - DS temporal coherence", "TC", 0, 1)
    save_map(pc_view, figdir / "processing_median_pair_coherence.png", "v1.0 - DS median pair coherence", "Median |Cij|", 0, 1)
    save_map(est_view, figdir / "processing_estimator.png", "v1.0 - Phase linking estimator (0=EVD, 1=EMI)", "Estimator", 0, 1)

    ds_tc = np.asarray(tc_map)[ds_mask]
    print("=" * 80)
    print("v1.0 complete")
    print("=" * 80)
    print(f"PS                 : {ps.sum()} ({100*ps.mean():.3f}%)")
    print(f"phase-linked DS    : {ds_mask.sum()} ({100*ds_mask.mean():.3f}%)")
    if ds_tc.size:
        print(f"DS TC median       : {np.nanmedian(ds_tc):.4f}")
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            m = ds_mask & (np.asarray(tc_map) >= t)
            print(f"TC >= {t:.1f}          : {m.sum()} ({100*m.mean():.3f}%)")
    print(f"EMI used           : {np.sum(ds_mask & (np.asarray(estimator)==ESTIMATOR_EMI))}")
    print(f"EVD fallback       : {np.sum(ds_mask & (np.asarray(estimator)==ESTIMATOR_EVD))}")
    print(f"total wall         : {(time.time()-t_all)/60:.2f} min")
    print(f"stage seconds      : support={support_s:.1f}, covariance={covariance_s:.1f}, PL={pl_s:.1f}, write={write_s:.1f}")
    print(f"outputs            : {outdir}")


if __name__ == "__main__":
    main()
