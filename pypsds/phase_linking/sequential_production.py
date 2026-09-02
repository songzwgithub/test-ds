from __future__ import annotations

import hashlib
import json

import os

from pathlib import Path
from time import perf_counter

import numpy as np

from .emi import (
    ESTIMATOR_EVD,
    ESTIMATOR_EMI,
    ESTIMATOR_INVALID,
)
from .full_scm_points import (
    run_full_scm_points,
)
from .fullspan_quality import (
    aggregate_stage_estimators,
    evaluate_fullspan_quality_points,
)
from .sequential_phase_writer import (
    SequentialPhaseWriter,
)
from .sequential_plan_executor import (
    run_sequential_plan,
)
from .sequential_routing import (
    build_sequential_routing,
)
from .shp_vectorized_exact import (
    prepare_glrt_window_context,
)
from .shp_policy import (
    resolve_shp_policy,
    split_fallback_by_rank,
    write_shp_policy_json,
)
from .support_cache import (
    load_exact_support_cache,
)

from .state_domain import (
    build_fixed_point_state_core,
    compute_original_K,
    effective_counts,
)
from .temporal_plan import (
    build_temporal_plan,
)



# ============================================================================
# production completed-Phase linking fast resume
# ============================================================================

_PHASE_LINKING_COMPLETE_FORMAT = (
    "pyPSDS-GAMMA-phase_linking-complete-v1"
)


def _sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with Path(
        path
    ).open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _sha256_array(
    arr,
) -> str:
    """
    Content hash for the comparatively small spatial
    science-control arrays.

    The full corrected YXT cube is deliberately represented by
    a file identity token instead of being re-hashed on every
    fast-resume attempt.
    """

    x = np.ascontiguousarray(
        np.asarray(
            arr
        )
    )

    h = hashlib.sha256()

    h.update(
        str(
            x.dtype
        ).encode(
            "utf-8"
        )
    )

    h.update(
        json.dumps(
            [
                int(v)
                for v in x.shape
            ]
        ).encode(
            "utf-8"
        )
    )

    h.update(
        memoryview(
            x
        ).cast(
            "B"
        )
    )

    return h.hexdigest()


def _file_identity(
    path,
):
    """
    Lightweight identity for a pipeline-owned large mmap product.

    Volatile mtime is deliberately excluded here. Scientific source
    provenance is supplied separately by the phase-source checkpoint
    token produced upstream.
    """

    if path is None:
        return None

    path = Path(
        path
    )

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


def _source_file_hashes():
    """
    Hash only modules that materially define Phase linking numerical
    behaviour or its production routing.

    Any future change to these files invalidates fast resume.
    """

    root = Path(
        __file__
    ).resolve().parent

    names = (
        "sequential_production.py",
        "sequential_multistage.py",
        "sequential_plan_executor.py",
        "fullspan_quality.py",
        "full_scm_points.py",
        "state_domain.py",
        "support_cache.py",
        "shp_policy.py",
        "coherence.py",
        "compression.py",
        "emi.py",
        "emi_threshold.py",
        "temporal_plan.py",
        "sequential_routing.py",
    )


    out = {}


    for name in names:

        path = (
            root
            /
            name
        )

        if path.is_file():

            out[
                name
            ] = _sha256_file(
                path
            )


    return out


def _phase_linking_completion_fingerprint(
    *,
    cfg,
    config_path,

    outdir,

    yxt,
    scale2,

    valid,
    geom_valid,
    ps,

    H,
    W,
    ndate,

    args,
):

    outdir = Path(
        outdir
    )


    yxt_filename = getattr(
        yxt,
        "filename",
        None,
    )


    if yxt_filename is None:

        # Standard production currently uses the corrected mmap.
        # If a future backend supplies another array type, fast
        # reuse is disabled conservatively rather than guessing.
        yxt_identity = None

    else:

        yxt_identity = (
            _file_identity(
                yxt_filename
            )
        )


    phase_source_token = (
        outdir
        /
        "cache"
        /
        "phase_source_checkpoint_token.json"
    )


    phase_source_token_sha = (
        _sha256_file(
            phase_source_token
        )
        if phase_source_token.is_file()
        else None
    )


    support_manifest = (
        outdir
        /
        "exact_support_cache"
        /
        "manifest.json"
    )


    support_manifest_sha = (
        _sha256_file(
            support_manifest
        )
        if support_manifest.is_file()
        else None
    )


    checkpoint_manifests = {}


    checkpoint_root = (
        outdir
        /
        "sequential"
        /
        "checkpoints"
    )


    if checkpoint_root.is_dir():

        for path in sorted(
            checkpoint_root.glob(
                "sequential_stage*/manifest.json"
            )
        ):

            checkpoint_manifests[
                path.parent.name
            ] = _sha256_file(
                path
            )


    science_cfg = {
        "phase_linking":
            cfg.get(
                "phase_linking",
                {},
            ),

        "selection":
            cfg.get(
                "selection",
                {},
            ),
    }


    return {
        "format":
            _PHASE_LINKING_COMPLETE_FORMAT,

        "scene":
            [
                int(H),
                int(W),
            ],

        "ndate":
            int(
                ndate
            ),

        "yxt":
            yxt_identity,

        "scale2_sha256":
            _sha256_array(
                scale2
            ),

        "valid_sha256":
            _sha256_array(
                valid
            ),

        "geometry_sha256":
            _sha256_array(
                geom_valid
            ),

        "ps_sha256":
            _sha256_array(
                ps
            ),

        "config_sha256":
            (
                _sha256_file(
                    config_path
                )
                if
                Path(
                    config_path
                ).is_file()
                else None
            ),

        "science_cfg":
            science_cfg,

        "runtime_args":
            {
                "half_row":
                    int(
                        args.half_row
                    ),

                "half_col":
                    int(
                        args.half_col
                    ),

                "alpha":
                    float(
                        args.alpha
                    ),

                "min_shp":
                    int(
                        args.min_shp
                    ),

                "beta":
                    float(
                        args.beta
                    ),

                "gamma_jitter":
                    float(
                        args.gamma_jitter
                    ),

                "emi_mu":
                    float(
                        args.emi_mu
                    ),

                "batch_size":
                    int(
                        args.batch_size
                    ),

                "pl_workers":
                    int(
                        args.pl_workers
                    ),

                "pl_chunk_size":
                    int(
                        args.pl_chunk_size
                    ),

                "tile_rows":
                    int(
                        args.tile_rows
                    ),

                "tile_cols":
                    int(
                        args.tile_cols
                    ),

                "support_block":
                    int(
                        args.support_block
                    ),
            },

        "phase_source_checkpoint_token_sha256":
            phase_source_token_sha,

        "exact_support_manifest_sha256":
            support_manifest_sha,

        "sequential_checkpoint_manifests":
            checkpoint_manifests,

        "source_sha256":
            _source_file_hashes(),
    }


def _fingerprint_hash(
    fingerprint,
):

    raw = json.dumps(
        fingerprint,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


def _completion_required_outputs(
    *,
    outdir,
    H,
    W,
    ndate,
):

    outdir = Path(
        outdir
    )


    return {
        "linked_phase.npy":
            (
                outdir
                /
                "linked_phase.npy",

                (
                    ndate,
                    H,
                    W,
                ),

                np.dtype(
                    np.complex64
                ),
            ),

        "temporal_coherence.npy":
            (
                outdir
                /
                "temporal_coherence.npy",

                (
                    H,
                    W,
                ),

                np.dtype(
                    np.float32
                ),
            ),

        "median_pair_coherence.npy":
            (
                outdir
                /
                "median_pair_coherence.npy",

                (
                    H,
                    W,
                ),

                np.dtype(
                    np.float32
                ),
            ),

        "estimator_code.npy":
            (
                outdir
                /
                "estimator_code.npy",

                (
                    H,
                    W,
                ),

                np.dtype(
                    np.uint8
                ),
            ),

        "pl_valid.npy":
            (
                outdir
                /
                "pl_valid.npy",

                (
                    H,
                    W,
                ),

                np.dtype(
                    np.bool_
                ),
            ),

        "shp_count.npy":
            (
                outdir
                /
                "shp_count.npy",

                (
                    H,
                    W,
                ),

                np.dtype(
                    np.int16
                ),
            ),
    }


def _validate_completion_outputs(
    *,
    outdir,
    H,
    W,
    ndate,
):

    required = (
        _completion_required_outputs(
            outdir=outdir,

            H=H,
            W=W,
            ndate=ndate,
        )
    )


    for name, (
        path,
        expected_shape,
        expected_dtype,
    ) in required.items():

        if not path.is_file():

            return (
                False,
                f"missing {name}",
            )


        try:

            x = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

        except Exception as exc:

            return (
                False,
                f"cannot open {name}: {exc}",
            )


        if x.shape != expected_shape:

            return (
                False,
                f"{name} shape mismatch",
            )


        if x.dtype != expected_dtype:

            return (
                False,
                f"{name} dtype mismatch",
            )


    return (
        True,
        "outputs valid",
    )


def _atomic_write_json(
    path,
    payload,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        +
        ".tmp"
    )


    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


    os.replace(
        tmp,
        path,
    )


def _try_reuse_completed_phase_linking(
    *,
    cfg,
    config_path,

    outdir,

    yxt,
    scale2,

    valid,
    geom_valid,
    ps,

    H,
    W,
    ndate,

    args,
):

    outdir = Path(
        outdir
    )


    manifest_path = (
        outdir
        /
        "sequential"
        /
        "phase_linking_complete.json"
    )


    force_fresh = (
        os.environ.get(
            "PYPSDS_FORCE_FRESH_PHASE_LINKING",
            "0",
        )
        ==
        "1"

        or

        os.environ.get(
            "PYPSDS_FORCE_FRESH_TILES",
            "0",
        )
        ==
        "1"
    )


    if force_fresh:

        if manifest_path.exists():

            manifest_path.unlink()

        return False


    if not manifest_path.is_file():

        return False


    fingerprint = (
        _phase_linking_completion_fingerprint(
            cfg=cfg,
            config_path=config_path,

            outdir=outdir,

            yxt=yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom_valid,
            ps=ps,

            H=H,
            W=W,
            ndate=ndate,

            args=args,
        )
    )


    current_hash = (
        _fingerprint_hash(
            fingerprint
        )
    )


    try:

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        print(
            "Phase linking completion manifest invalid:",
            exc,
        )

        return False


    if (
        manifest.get(
            "format"
        )
        !=
        _PHASE_LINKING_COMPLETE_FORMAT
    ):

        return False


    if not bool(
        manifest.get(
            "complete",
            False,
        )
    ):

        return False


    if (
        manifest.get(
            "fingerprint_sha256"
        )
        !=
        current_hash
    ):

        print(
            "Phase linking completion cache : INVALIDATED "
            "(fingerprint changed)"
        )

        return False


    ok, reason = (
        _validate_completion_outputs(
            outdir=outdir,

            H=H,
            W=W,
            ndate=ndate,
        )
    )


    if not ok:

        print(
            "Phase linking completion cache : INVALIDATED "
            f"({reason})"
        )

        return False


    print()
    print(
        "=" * 88
    )

    print(
        "Phase linking completion cache"
    )

    print(
        "=" * 88
    )

    print(
        "status             : VALID"
    )

    print(
        "action             : reuse complete production outputs"
    )

    print(
        "linked_phase       :",
        outdir
        /
        "linked_phase.npy",
    )

    print(
        "PHASE LINKING FAST RESUME : PASS"
    )

    return True


def _write_completed_phase_linking_manifest(
    *,
    cfg,
    config_path,

    outdir,

    yxt,
    scale2,

    valid,
    geom_valid,
    ps,

    H,
    W,
    ndate,

    args,

    summary,
):

    outdir = Path(
        outdir
    )


    ok, reason = (
        _validate_completion_outputs(
            outdir=outdir,

            H=H,
            W=W,
            ndate=ndate,
        )
    )


    if not ok:

        raise RuntimeError(
            "cannot commit Phase linking completion manifest: "
            f"{reason}"
        )


    fingerprint = (
        _phase_linking_completion_fingerprint(
            cfg=cfg,
            config_path=config_path,

            outdir=outdir,

            yxt=yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom_valid,
            ps=ps,

            H=H,
            W=W,
            ndate=ndate,

            args=args,
        )
    )


    payload = {
        "format":
            _PHASE_LINKING_COMPLETE_FORMAT,

        "complete":
            True,

        "fingerprint_sha256":
            _fingerprint_hash(
                fingerprint
            ),

        "fingerprint":
            fingerprint,

        "summary":
            summary,
    }


    path = (
        outdir
        /
        "sequential"
        /
        "phase_linking_complete.json"
    )


    _atomic_write_json(
        path,
        payload,
    )


    return path


def _new_map(
    path,
    *,
    shape,
    dtype,
    fill,
):
    x = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    x[...] = fill
    x.flush()

    return x


def _run_gamma_post_phase_fused(
    *,
    yxt,
    linked_path,
    writer,

    routing,
    fallback_exec_mask,

    original_k,
    state_core,
    effective_k,

    scale2,
    valid,
    geom_valid,
    ps,

    stage_estimator_maps,

    seqdir,

    tc_map,
    pair_map,
    estimator,
    emi_eig_map,
    evd_eig_map,
    gamma_min_map,
    pl_valid,

    shp_policy,

    H,
    W,
    ndate,

    args,
    fullspan_batch_size,
    static_support_cache,
):
    # Fused GAMMA post-Phase-Linking row-band executor.
    #
    # One full-date PhaseTile is produced per row band and consumed by:
    #   1. sequential full-span quality;
    #   2. original-support full-SCM fallback;
    #   3. PS phase fill.
    #
    # Memory remains scene-size independent because only one row-band
    # PhaseTile is resident at a time.

    t_fused = perf_counter()

    phase_cube = np.load(
        linked_path,
        mmap_mode="r",
    )

    sequential_total = int(
        routing.sequential_count
    )

    fullspan_target_pixels = max(
        1,
        int(
            os.environ.get(
                "PYPSDS_FULLSPAN_TARGET_PIXELS",
                "500000",
            )
        ),
    )

    fullspan_row_block = max(
        1,
        min(
            H,
            fullspan_target_pixels
            //
            max(
                1,
                W,
            ),
        ),
    )

    fullspan_band_count = (
        H
        +
        fullspan_row_block
        -
        1
    ) // fullspan_row_block


    postphase_cache_plan = None

    phase_source = getattr(
        yxt,
        "phase_source",
        None,
    )

    if (
        phase_source is not None
        and
        hasattr(
            phase_source,
            "configure_postphase_fullspan_cache",
        )
    ):
        postphase_cache_plan = (
            phase_source
            .configure_postphase_fullspan_cache(
                local_H=H,
                local_W=W,
                memory_fraction=0.10,
                clear_stage_cache=False,
            )
        )

    print(
        "post-PL orchestration  : fused GAMMA row-band"
    )

    print(
        "post-PL target pixels  :",
        f"{fullspan_target_pixels:,}",
    )

    print(
        "post-PL row block      :",
        fullspan_row_block,
    )

    print(
        "post-PL row bands      :",
        fullspan_band_count,
    )

    seq_rows_out = np.lib.format.open_memmap(
        seqdir
        /
        "production_sequential_rows.npy",

        mode="w+",
        dtype=np.int32,
        shape=(
            sequential_total,
        ),
    )

    seq_cols_out = np.lib.format.open_memmap(
        seqdir
        /
        "production_sequential_cols.npy",

        mode="w+",
        dtype=np.int32,
        shape=(
            sequential_total,
        ),
    )

    seq_tc_out = np.lib.format.open_memmap(
        seqdir
        /
        "production_sequential_fullspan_tc.npy",

        mode="w+",
        dtype=np.float32,
        shape=(
            sequential_total,
        ),
    )

    seq_pair_out = np.lib.format.open_memmap(
        seqdir
        /
        "production_sequential_pair_coherence.npy",

        mode="w+",
        dtype=np.float32,
        shape=(
            sequential_total,
        ),
    )

    seq_estimator_out = np.lib.format.open_memmap(
        seqdir
        /
        "production_sequential_estimator_code.npy",

        mode="w+",
        dtype=np.uint8,
        shape=(
            sequential_total,
        ),
    )

    sequential_offset = 0
    seq_ok_count = 0

    fallback_valid_count = 0
    fallback_estimator_parts = []

    phase_read_wall = 0.0
    source_read_seconds = 0.0
    source_correction_seconds = 0.0

    quality_seconds = 0.0
    fallback_seconds = 0.0
    ps_fill_seconds = 0.0

    phase_reads = 0
    phase_cache_hits = 0
    phase_cache_misses = 0
    phase_cache_composed_hits = 0

    for band_index, r0 in enumerate(
        range(
            0,
            H,
            fullspan_row_block,
        ),
        start=1,
    ):

        r1 = min(
            H,
            r0
            +
            fullspan_row_block,
        )

        seq_lr, seq_lc = np.where(
            routing.sequential[
                r0:r1,
                :
            ]
        )

        flr, flc = np.where(
            fallback_exec_mask[
                r0:r1,
                :
            ]
        )

        local_ps_mask = (
            np.asarray(
                ps[
                    r0:r1,
                    :
                ],
                dtype=np.bool_,
            )
            &
            np.asarray(
                geom_valid[
                    r0:r1,
                    :
                ],
                dtype=np.bool_,
            )
        )

        ps_lr, ps_lc = np.where(
            local_ps_mask
        )

        nseq = int(
            seq_lr.size
        )

        nfallback = int(
            flr.size
        )

        nps = int(
            ps_lr.size
        )

        if (
            nseq == 0
            and
            nfallback == 0
            and
            nps == 0
        ):
            print(
                f"[post-PL-fused] "
                f"band={band_index}/"
                f"{fullspan_band_count} "
                f"rows={r0}:{r1} "
                "seq=0 fallback=0 ps=0 "
                "phase_read=SKIP"
            )
            continue

        need_halo = (
            nseq > 0
            or
            nfallback > 0
        )

        if need_halo:
            ir0 = max(
                0,
                r0
                -
                args.half_row,
            )

            ir1 = min(
                H,
                r1
                +
                args.half_row,
            )

        else:
            ir0 = r0
            ir1 = r1

        t0 = perf_counter()

        phase_tile = (
            yxt.phase_source.read_tile(
                local_row0=ir0,
                local_row1=ir1,

                local_col0=0,
                local_col1=W,
            )
        )

        phase_read_wall += (
            perf_counter()
            -
            t0
        )

        phase_reads += 1

        phase_cache_hits += int(
            getattr(
                phase_tile,
                "cache_hits",
                0,
            )
        )

        phase_cache_misses += int(
            getattr(
                phase_tile,
                "cache_misses",
                0,
            )
        )


        phase_cache_composed_hits += int(
            getattr(
                phase_tile,
                "cache_composed_hits",
                0,
            )
        )

        source_read_seconds += float(
            getattr(
                phase_tile,
                "read_seconds",
                0.0,
            )
        )

        source_correction_seconds += float(
            getattr(
                phase_tile,
                "correction_seconds",
                0.0,
            )
        )

        if not np.array_equal(
            phase_tile.geometry_valid,

            np.asarray(
                geom_valid[
                    ir0:ir1,
                    :
                ],
                dtype=np.bool_,
            ),
        ):
            raise RuntimeError(
                "Gamma fused post-PL geometry parity failure "
                f"for rows={ir0}:{ir1}"
            )

        if nseq:

            tq = perf_counter()

            sr = (
                seq_lr
                +
                r0
            ).astype(
                np.int32,
                copy=False,
            )

            sc = seq_lc.astype(
                np.int32,
                copy=False,
            )

            local_sr = (
                sr
                -
                ir0
            ).astype(
                np.int32,
                copy=False,
            )

            local_sc = sc

            local_phase_points = (
                np.ascontiguousarray(
                    phase_cube[
                        :,
                        sr,
                        sc,
                    ].T,

                    dtype=np.complex64,
                )
            )

            q = evaluate_fullspan_quality_points(
                yxt=phase_tile.yxt,

                phase_points=(
                    local_phase_points
                ),

                rows=local_sr,
                cols=local_sc,

                scale2=scale2[
                    ir0:ir1,
                    :
                ],

                valid=valid[
                    ir0:ir1,
                    :
                ],

                ps=ps[
                    ir0:ir1,
                    :
                ],

                state_core=state_core[
                    ir0:ir1,
                    :
                ],

                expected_effective_k=(
                    effective_k[
                        ir0:ir1,
                        :
                    ]
                ),

                half_row=args.half_row,
                half_col=args.half_col,
                alpha=args.alpha,

                batch=(
                    fullspan_batch_size
                ),

                support_block=1024,

                static_support_cache=None,

                support_rows=sr,
                support_cols=sc,
            )

            seq_estimator_band = (
                aggregate_stage_estimators(
                    stage_estimator_maps,

                    rows=sr,
                    cols=sc,
                )
            )

            seq_ok_band = (
                q.phase_complete
                &
                np.isfinite(
                    q.temporal_coherence
                )
                &
                (
                    seq_estimator_band
                    !=
                    ESTIMATOR_INVALID
                )
            )

            tc_map[
                sr,
                sc,
            ] = (
                q.temporal_coherence
            )

            pair_map[
                sr,
                sc,
            ] = (
                q.median_pair_coherence
            )

            estimator[
                sr,
                sc,
            ] = (
                seq_estimator_band
            )

            pl_valid[
                sr,
                sc,
            ] = (
                seq_ok_band
            )

            o0 = sequential_offset
            o1 = (
                o0
                +
                nseq
            )

            seq_rows_out[
                o0:o1
            ] = sr

            seq_cols_out[
                o0:o1
            ] = sc

            seq_tc_out[
                o0:o1
            ] = (
                q.temporal_coherence
            )

            seq_pair_out[
                o0:o1
            ] = (
                q.median_pair_coherence
            )

            seq_estimator_out[
                o0:o1
            ] = (
                seq_estimator_band
            )

            sequential_offset = o1

            seq_ok_count += int(
                np.count_nonzero(
                    seq_ok_band
                )
            )

            quality_seconds += (
                perf_counter()
                -
                tq
            )

        if nfallback:

            tf = perf_counter()

            fr = (
                flr
                +
                r0
            ).astype(
                np.int32,
                copy=False,
            )

            fc = flc.astype(
                np.int32,
                copy=False,
            )

            lfr = (
                fr
                -
                ir0
            ).astype(
                np.int32,
                copy=False,
            )

            lfc = fc

            result = run_full_scm_points(
                yxt=phase_tile.yxt,

                rows=lfr,
                cols=lfc,

                scale2=scale2[
                    ir0:ir1,
                    :
                ],

                valid=valid[
                    ir0:ir1,
                    :
                ],

                ps=ps[
                    ir0:ir1,
                    :
                ],

                expected_original_k=(
                    original_k[
                        ir0:ir1,
                        :
                    ]
                ),

                phase_sink=None,

                half_row=args.half_row,
                half_col=args.half_col,
                alpha=args.alpha,

                min_shp=(
                    shp_policy
                    .full_scm_rank_min_shp
                ),

                beta=args.beta,

                gamma_jitter=(
                    args.gamma_jitter
                ),

                emi_mu=args.emi_mu,

                batch=2048,
                support_block=1024,

                pl_workers=(
                    args.pl_workers
                ),

                pl_chunk_size=(
                    args.pl_chunk_size
                ),
            )

            ok = np.asarray(
                result.pl_valid,
                dtype=np.bool_,
            )

            if np.any(
                ok
            ):
                writer(
                    stage_index=-1,

                    real_indices=tuple(
                        range(
                            ndate
                        )
                    ),

                    rows=fr[
                        ok
                    ],

                    cols=fc[
                        ok
                    ],

                    phase=result.phase[
                        ok
                    ],
                )

            tc_map[
                fr,
                fc,
            ] = (
                result
                .temporal_coherence
            )

            pair_map[
                fr,
                fc,
            ] = (
                result
                .median_pair_coherence
            )

            estimator[
                fr,
                fc,
            ] = (
                result.estimator
            )

            emi_eig_map[
                fr,
                fc,
            ] = (
                result
                .emi_eigenvalue
            )

            evd_eig_map[
                fr,
                fc,
            ] = (
                result
                .evd_eigenvalue
            )

            gamma_min_map[
                fr,
                fc,
            ] = (
                result
                .gamma_min_eigenvalue
            )

            pl_valid[
                fr,
                fc,
            ] = (
                result.pl_valid
            )

            fallback_valid_count += int(
                result.valid_count
            )

            fallback_estimator_parts.append(
                np.asarray(
                    result.estimator,
                    dtype=np.uint8,
                )
            )

            fallback_seconds += (
                perf_counter()
                -
                tf
            )

        if nps:

            tp = perf_counter()

            gr = (
                ps_lr
                +
                r0
            ).astype(
                np.int32,
                copy=False,
            )

            gc = ps_lc.astype(
                np.int32,
                copy=False,
            )

            tile_ps_r = (
                gr
                -
                ir0
            ).astype(
                np.int32,
                copy=False,
            )

            pph = (
                phase_tile.yxt[
                    tile_ps_r,
                    gc,
                    :
                ]
            )

            pph = np.exp(
                1j
                *
                np.angle(
                    pph
                )
            ).astype(
                np.complex64
            )

            pph *= np.exp(
                -1j
                *
                np.angle(
                    pph[
                        :,
                        0,
                    ]
                )
            )[
                :,
                None
            ]

            writer(
                stage_index=-2,

                real_indices=tuple(
                    range(
                        ndate
                    )
                ),

                rows=gr,
                cols=gc,

                phase=pph,
            )

            ps_fill_seconds += (
                perf_counter()
                -
                tp
            )

        print(
            f"[post-PL-fused] "
            f"band={band_index}/"
            f"{fullspan_band_count} "
            f"rows={r0}:{r1} "
            f"seq={nseq:,} "
            f"fallback={nfallback:,} "
            f"ps={nps:,} "
            f"seq_done="
            f"{sequential_offset:,}/"
            f"{sequential_total:,}"
        )

        del phase_tile

    if (
        sequential_offset
        !=
        sequential_total
    ):
        raise RuntimeError(
            "fused post-PL sequential point-count mismatch: "
            f"{sequential_offset} != "
            f"{sequential_total}"
        )

    for sparse_arr in (
        seq_rows_out,
        seq_cols_out,
        seq_tc_out,
        seq_pair_out,
        seq_estimator_out,
    ):
        sparse_arr.flush()

    seq_estimator = (
        seq_estimator_out
    )

    fallback_result = None

    if fallback_estimator_parts:

        class _ProductionFallback:
            pass

        fallback_result = (
            _ProductionFallback()
        )

        fallback_result.valid_count = (
            fallback_valid_count
        )

        fallback_result.estimator = (
            np.concatenate(
                fallback_estimator_parts
            )
        )

    writer.flush()
    writer.close()

    linked_phase = np.load(
        linked_path,
        mmap_mode="r+",
    )

    fused_seconds = (
        perf_counter()
        -
        t_fused
    )

    print()
    print(
        "post-PL fused phase reads :",
        phase_reads,
    )

    print(
        "post-PL phase wall seconds:",
        f"{phase_read_wall:.3f}",
    )


    phase_cache_total = (
        phase_cache_hits
        +
        phase_cache_misses
    )

    phase_cache_hit_rate = (
        100.0
        *
        phase_cache_hits
        /
        phase_cache_total
        if
        phase_cache_total > 0
        else
        0.0
    )

    print(
        "post-PL phase cache hits  :",
        phase_cache_hits,
    )

    print(
        "post-PL phase cache misses:",
        phase_cache_misses,
    )


    print(
        "post-PL phase composed hits:",
        phase_cache_composed_hits,
    )

    print(
        "post-PL phase cache hit % :",
        f"{phase_cache_hit_rate:.1f}%",
    )

    if postphase_cache_plan is not None:
        print(
            "post-PL phase cache cells :",
            f"{postphase_cache_plan['cache_max_cells']}/"
            f"{postphase_cache_plan['scene_cells']}",
        )

    print(
        "post-PL raw read seconds  :",
        f"{source_read_seconds:.3f}",
    )

    print(
        "post-PL correction seconds:",
        f"{source_correction_seconds:.3f}",
    )

    print(
        "post-PL quality seconds   :",
        f"{quality_seconds:.3f}",
    )

    print(
        "post-PL fallback seconds  :",
        f"{fallback_seconds:.3f}",
    )

    print(
        "post-PL PS fill seconds   :",
        f"{ps_fill_seconds:.3f}",
    )

    print(
        "post-PL fused total       :",
        f"{fused_seconds:.3f}",
    )

    return (
        seq_estimator,
        seq_ok_count,
        fallback_result,
        linked_phase,
    )


def _load_or_build_state_domain(
    *,
    seqdir,
    scale2,
    valid,
    ps,

    ndate,
    state_min_shp,

    half_row,
    half_col,
    alpha,

    batch,
    support_block,
    support_cache=None,
):
    """
    Load the validated state-domain cache when available.

    On a fresh project, construct the same state domain through
    the promoted production state_domain implementation.
    """

    H, W = valid.shape

    original_path = (
        seqdir
        /
        "compression_all_valid_nonps_shp_count.npy"
    )

    core_path = (
        seqdir
        /
        f"compression_state_core_K"
        f"{state_min_shp:02d}.npy"
    )

    effective_path = (
        seqdir
        /
        f"compression_state_core_K"
        f"{state_min_shp:02d}"
        "_effective_shp_count.npy"
    )

    cache_complete = all(
        p.is_file()
        for p in (
            original_path,
            core_path,
            effective_path,
        )
    )

    if cache_complete:

        original_k = np.load(
            original_path,
            mmap_mode="r",
        )

        state_core = np.load(
            core_path,
            mmap_mode="r",
        )

        effective_k = np.load(
            effective_path,
            mmap_mode="r",
        )

        for name, arr in (
            ("original_K", original_k),
            ("state_core", state_core),
            ("effective_K", effective_k),
        ):
            if arr.shape != (H, W):
                raise RuntimeError(
                    f"{name} cache shape mismatch: "
                    f"{arr.shape} != {(H, W)}"
                )

        print(
            "state domain       : reuse validated cache"
        )

        return (
            original_k,
            state_core,
            effective_k,
        )

    print(
        "state domain       : building production K-state core"
    )

    valid_nonps = (
        np.asarray(
            valid,
            dtype=np.bool_,
        )
        &
        ~np.asarray(
            ps,
            dtype=np.bool_,
        )
    )

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps,
        half_row=half_row,
        half_col=half_col,
    )

    original_k = compute_original_K(
        ctx=ctx,
        mask=valid_nonps,
        alpha=alpha,
        ndate=ndate,
        batch=batch,
        support_block=support_block,
    
        support_cache=support_cache,)

    state_core, _ = (
        build_fixed_point_state_core(
            ctx=ctx,
            valid_nonps=valid_nonps,
            original_K=original_k,

            threshold=state_min_shp,

            alpha=alpha,
            ndate=ndate,

            batch=batch,
            support_block=support_block,

            half_row=half_row,
            half_col=half_col,
        
            support_cache=support_cache,)
    )

    sr, sc, sk = effective_counts(
        ctx=ctx,

        center_mask=state_core,
        state_mask=state_core,

        alpha=alpha,
        ndate=ndate,

        batch=batch,
        support_block=support_block,

        half_row=half_row,
        half_col=half_col,
    
        support_cache=support_cache,)

    effective_k = np.full(
        (H, W),
        -1,
        dtype=np.int16,
    )

    effective_k[
        sr,
        sc,
    ] = sk

    np.save(
        original_path,
        original_k,
    )

    np.save(
        core_path,
        state_core,
    )

    np.save(
        effective_path,
        effective_k,
    )

    return (
        original_k,
        state_core,
        effective_k,
    )


def run_sequential_production(
    *,
    cfg,
    config_path,

    paths,
    stack,

    H,
    W,

    outdir,

    yxt,
    scale2,

    valid,
    geom_valid,
    ps,

    center_prior,

    args,
):
    """
    Production sequential phase-linking branch for Phase linking.

    Output contract intentionally remains compatible with the
    existing downstream Step05/06b/06 implementation.
    """

    t_all = perf_counter()

    ndate = int(
        yxt.shape[2]
    )

    if yxt.shape != (
        H,
        W,
        ndate,
    ):
        raise RuntimeError(
            f"YXT shape mismatch: {yxt.shape}"
        )

    phase_cfg = cfg.get(
        "phase_linking",
        {},
    )

    temporal_cfg = phase_cfg.get(
        "temporal",
        {},
    )

    strategy = str(
        temporal_cfg.get(
            "strategy",
            "full_scm",
        )
    ).lower()

    if strategy != "sequential":
        raise RuntimeError(
            "run_sequential_production called "
            f"for strategy={strategy!r}"
        )

    M = int(
        temporal_cfg.get(
            "ministack_size",
            19,
        )
    )

    max_compressed = int(
        temporal_cfg.get(
            "max_num_compressed",
            5,
        )
    )

    full_scm_fallback = bool(
        temporal_cfg.get(
            "full_scm_fallback",
            True,
        )
    )

    shp_policy = resolve_shp_policy(
        cfg,
        stack.dates,
        base_half_row=args.half_row,
        base_half_col=args.half_col,
        base_formal_min_shp=args.min_shp,
    )

    if (
        int(args.half_row) != int(shp_policy.half_row)
        or int(args.half_col) != int(shp_policy.half_col)
        or int(args.min_shp) != int(shp_policy.formal_min_shp)
    ):
        raise RuntimeError(
            "run_phase_linking / sequential SHP-policy mismatch"
        )

    state_min_shp = int(
        shp_policy.state_min_shp
    )


    fullspan_batch_size = int(
        temporal_cfg.get(
            "fullspan_batch_size",
            16384,
        )
    )

    if fullspan_batch_size < 1:
        raise ValueError(
            "fullspan_batch_size must be >= 1"
        )

    emi_backend = str(
        temporal_cfg.get(
            "emi_backend",
            "current_eigh",
        )
    ).strip().lower()

    if emi_backend not in {
        "current_eigh",
        "threshold_cholesky",
    }:
        raise ValueError(
            f"unsupported production EMI backend: {emi_backend}"
        )

    reference_index = int(
        phase_cfg.get(
            "temporal_reference_index",
            0,
        )
    )

    if reference_index != 0:
        raise RuntimeError(
            "validated sequential production "
            "currently requires reference_index=0"
        )

    seqdir = (
        Path(outdir)
        /
        "sequential"
    )

    seqdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    if _try_reuse_completed_phase_linking(
        cfg=cfg,
        config_path=config_path,

        outdir=outdir,

        yxt=yxt,
        scale2=scale2,

        valid=valid,
        geom_valid=geom_valid,
        ps=ps,

        H=H,
        W=W,
        ndate=ndate,

        args=args,
    ):

        return


    use_exact_support_cache = bool(
        temporal_cfg.get(
            "use_exact_support_cache",
            True,
        )
    )

    if use_exact_support_cache:

        static_support_cache = (
            load_exact_support_cache(
                processing_dir=Path(
                    outdir
                ),

                H=H,
                W=W,

                ndate=ndate,

                half_row=args.half_row,
                half_col=args.half_col,

                alpha=args.alpha,

                validate_input_hashes=True,
            )
        )

    else:

        static_support_cache = None

    print()
    print("=" * 88)
    print(
        "pyPSDS-GAMMA - sequential production"
    )
    print("=" * 88)

    print("config             :", config_path)
    print("scene              :", f"{H} x {W}")
    print("dates              :", ndate)
    print("strategy           :", strategy)
    print("ministack          :", M)
    print("max compressed     :", max_compressed)
    print("state min SHP      :", state_min_shp)
    print("formal min SHP     :", args.min_shp)
    print("fullspan batch     :", fullspan_batch_size)
    print(
        "PL tile            :",
        f"{args.tile_rows} x {args.tile_cols}",
    )
    print(
        "PL workers         :",
        args.pl_workers,
    )
    print(
        "PL chunk           :",
        args.pl_chunk_size,
    )
    print(
        "support block      :",
        args.support_block,
    )

    print(
        "tile prefetch      :",
        args.prefetch_tiles,
    )
    print("beta               :", args.beta)
    print("EMI mu             :", args.emi_mu)
    print("EMI backend        :", emi_backend)
    print(
        "exact SHP cache     :",
        (
            "enabled"
            if static_support_cache is not None
            else "disabled"
        ),
    )
    print()

    # --------------------------------------------------------
    # State domain
    # --------------------------------------------------------

    (
        original_k,
        state_core,
        effective_k,
    ) = _load_or_build_state_domain(
        seqdir=seqdir,

        scale2=scale2,
        valid=valid,
        ps=ps,

        ndate=ndate,
        state_min_shp=state_min_shp,

        half_row=args.half_row,
        half_col=args.half_col,
        alpha=args.alpha,

        batch=args.batch_size,
        support_block=args.support_block,
    
        support_cache=static_support_cache,)

    print(
        "state pixels       :",
        f"{np.count_nonzero(state_core):,}",
    )

    # --------------------------------------------------------
    # Formal DS + routing
    # --------------------------------------------------------

    routing = build_sequential_routing(
        center_prior=center_prior,

        valid=valid,
        ps=ps,

        original_shp_count=original_k,
        effective_shp_count=effective_k,

        formal_min_shp=args.min_shp,
        state_min_shp=state_min_shp,
    )

    print(
        "formal DS          :",
        f"{routing.formal_count:,}",
    )

    print(
        "sequential route   :",
        f"{routing.sequential_count:,}",
    )

    print(
        "full-SCM fallback  :",
        f"{routing.fallback_count:,}",
    )

    (
        fallback_exec_mask,
        fallback_under_supported_mask,
    ) = split_fallback_by_rank(
        routing.fallback,
        original_k,
        full_scm_min_shp=(
            shp_policy.full_scm_rank_min_shp
        ),
        rank_guard=(
            shp_policy.rank_guard
        ),
    )

    fallback_exec_count = int(
        np.count_nonzero(
            fallback_exec_mask
        )
    )
    fallback_under_count = int(
        np.count_nonzero(
            fallback_under_supported_mask
        )
    )

    print(
        "fallback rank-supported:",
        f"{fallback_exec_count:,}",
    )
    print(
        "fallback under-supported:",
        f"{fallback_under_count:,}",
    )

    write_shp_policy_json(
        seqdir
        /
        "shp_policy.json",
        shp_policy,
    )

    if (
        routing.fallback_count > 0
        and
        not full_scm_fallback
    ):
        raise RuntimeError(
            "formal DS fallback centers exist, "
            "but full_scm_fallback=false"
        )

    np.save(
        seqdir
        /
        "production_formal_ds_mask.npy",
        routing.formal_ds,
    )

    np.save(
        seqdir
        /
        "production_sequential_mask.npy",
        routing.sequential,
    )

    np.save(
        seqdir
        /
        "production_full_scm_fallback_mask.npy",
        routing.fallback,
    )

    np.save(
        seqdir
        /
        "production_full_scm_rank_supported_mask.npy",
        fallback_exec_mask,
    )

    np.save(
        seqdir
        /
        "production_full_scm_under_supported_mask.npy",
        fallback_under_supported_mask,
    )

    np.save(
        seqdir
        /
        "production_effective_shp_count.npy",
        np.asarray(
            effective_k,
            dtype=np.int16,
        ),
    )

    # --------------------------------------------------------
    # Existing Phase linking output contract
    # --------------------------------------------------------

    shp_count = _new_map(
        outdir
        /
        "shp_count.npy",

        shape=(H, W),
        dtype=np.int16,
        fill=-1,
    )

    tc_map = _new_map(
        outdir
        /
        "temporal_coherence.npy",

        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    pair_map = _new_map(
        outdir
        /
        "median_pair_coherence.npy",

        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    estimator = _new_map(
        outdir
        /
        "estimator_code.npy",

        shape=(H, W),
        dtype=np.uint8,
        fill=ESTIMATOR_INVALID,
    )

    emi_eig_map = _new_map(
        outdir
        /
        "emi_eigenvalue.npy",

        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    evd_eig_map = _new_map(
        outdir
        /
        "evd_eigenvalue.npy",

        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    gamma_min_map = _new_map(
        outdir
        /
        "gamma_min_eigenvalue.npy",

        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    pl_valid = _new_map(
        outdir
        /
        "pl_valid.npy",

        shape=(H, W),
        dtype=np.bool_,
        fill=False,
    )

    # Formal shp_count remains ORIGINAL GLRT K.
    cp = np.asarray(
        center_prior,
        dtype=np.bool_,
    )

    shp_count[
        cp
    ] = np.asarray(
        original_k[
            cp
        ],
        dtype=np.int16,
    )

    # Preserve normal Phase linking center-list artifacts.
    cr, cc0 = np.where(
        center_prior
    )

    cr = cr.astype(
        np.int32,
        copy=False,
    )

    cc0 = cc0.astype(
        np.int32,
        copy=False,
    )

    np.save(
        outdir
        /
        "center_rows.npy",
        cr,
    )

    np.save(
        outdir
        /
        "center_cols.npy",
        cc0,
    )

    np.save(
        outdir
        /
        "processed_centers.npy",
        np.ones(
            cr.size,
            dtype=np.bool_,
        ),
    )

    np.save(
        outdir
        /
        "ps_mask.npy",
        ps,
    )

    np.save(
        outdir
        /
        "center_prior.npy",
        center_prior,
    )

    # --------------------------------------------------------
    # Temporal plan
    # --------------------------------------------------------

    dates = tuple(
        str(x)
        for x
        in stack.dates
    )

    plan = build_temporal_plan(
        dates,

        strategy="sequential",

        ministack_size=M,

        max_num_compressed=(
            max_compressed
        ),

        reference_index=(
            reference_index
        ),
    )

    if not plan.execution_ready:
        raise RuntimeError(
            "sequential temporal plan "
            "is not execution-ready"
        )


    sequential_cache_plan = None

    phase_source = getattr(
        yxt,
        "phase_source",
        None,
    )

    if (
        phase_source is not None
        and
        hasattr(
            phase_source,
            "configure_sequential_temporal_cache",
        )
    ):
        sequential_cache_plan = (
            phase_source
            .configure_sequential_temporal_cache(
                local_H=H,
                local_W=W,
                temporal_parts=len(
                    plan.stages
                ),
                memory_fraction=0.10,
            )
        )

    print(
        "stage solver sizes :",
        [
            s.solver_size
            for s
            in plan.stages
        ],
    )

    # --------------------------------------------------------
    # Final linked_phase writer.
    #
    # Fresh creation is sparse.
    # Unwritten phase remains 0+0j.
    # --------------------------------------------------------

    linked_path = (
        outdir
        /
        "linked_phase.npy"
    )

    # --------------------------------------------------------
    # production linked-phase resume contract.
    #
    # Stage0's checkpoint manifest is the authoritative signal
    # that linked_phase.npy belongs to an interrupted/resumable
    # sequential execution.
    # --------------------------------------------------------

    checkpoint_force_fresh = (
        os.environ.get(
            "PYPSDS_FORCE_FRESH_TILES",
            "0",
        )
        ==
        "1"
    )


    stage0_checkpoint_manifest = (
        seqdir
        /
        "checkpoints"
        /
        "sequential_stage0000"
        /
        "manifest.json"
    )


    checkpoint_phase_resume = (
        stage0_checkpoint_manifest.is_file()
        and
        not checkpoint_force_fresh
    )


    if (
        checkpoint_phase_resume
        and
        not linked_path.is_file()
    ):

        raise RuntimeError(
            "stage checkpoint exists but linked_phase.npy "
            "is missing; use PYPSDS_FORCE_FRESH_TILES=1 "
            "for a deliberate fresh recomputation"
        )


    print(
        "phase checkpoint mode :",
        (
            "resume existing linked_phase"
            if checkpoint_phase_resume
            else "fresh linked_phase"
        ),
    )


    writer = SequentialPhaseWriter(
        linked_path,

        ndate=ndate,
        rows=H,
        cols=W,

        overwrite=(
            not checkpoint_phase_resume
        ),

    )

    # --------------------------------------------------------
    # Sequential ministacks
    # --------------------------------------------------------

    plan_result = run_sequential_plan(
        plan=plan,

        yxt=yxt,

        scale2=scale2,
        valid=valid,
        ps=ps,

        state_core=state_core,

        expected_effective_k=(
            effective_k
        ),

        output_dir=seqdir,

        phase_sink=writer,

        full_glrt_nslc=ndate,

        state_min_shp=(
            state_min_shp
        ),

        half_row=args.half_row,
        half_col=args.half_col,
        alpha=args.alpha,

        beta=args.beta,

        gamma_jitter=(
            args.gamma_jitter
        ),

        emi_mu=args.emi_mu,

        emi_backend=emi_backend,

        tile_rows=(
            args.tile_rows
        ),
        tile_cols=(
            args.tile_cols
        ),

        center_batch=(
            args.batch_size
        ),

        support_block=(
            args.support_block
        ),

        pl_workers=(
            args.pl_workers
        ),

        pl_chunk_size=(
            args.pl_chunk_size
        ),

        prefetch_tiles=(
            args.prefetch_tiles
        ),

        formula_audit_points=1000,
    
        static_support_cache=static_support_cache,)

    writer.flush()

    # --------------------------------------------------------
    # Post-Phase-Linking processing.
    #
    # GAMMA streaming:
    #   one full-date row-band PhaseTile is shared by full-span
    #   quality, full-SCM fallback and PS fill.
    #
    # Cached/mmap backend:
    #   preserve the validated legacy orchestration below.
    # --------------------------------------------------------

    gamma_post_fusion = bool(
        getattr(
            yxt,
            "is_phase_source_proxy",
            False,
        )
    )

    if gamma_post_fusion:

        stage_estimator_maps = tuple(
            np.load(
                x.estimator_path,
                mmap_mode="r",
            )
            for x
            in plan_result.stage_results
        )

        (
            seq_estimator,
            seq_ok_count,
            fallback_result,
            linked_phase,
        ) = _run_gamma_post_phase_fused(
            yxt=yxt,
            linked_path=linked_path,
            writer=writer,

            routing=routing,
            fallback_exec_mask=(
                fallback_exec_mask
            ),

            original_k=original_k,
            state_core=state_core,
            effective_k=effective_k,

            scale2=scale2,
            valid=valid,
            geom_valid=geom_valid,
            ps=ps,

            stage_estimator_maps=(
                stage_estimator_maps
            ),

            seqdir=seqdir,

            tc_map=tc_map,
            pair_map=pair_map,
            estimator=estimator,
            emi_eig_map=emi_eig_map,
            evd_eig_map=evd_eig_map,
            gamma_min_map=gamma_min_map,
            pl_valid=pl_valid,

            shp_policy=shp_policy,

            H=H,
            W=W,
            ndate=ndate,

            args=args,

            fullspan_batch_size=(
                fullspan_batch_size
            ),
            static_support_cache=(
                static_support_cache
            ),

        )

    else:
        # --------------------------------------------------------
        # --------------------------------------------------------
        # production ROW-BAND STREAMING FULL-SPAN quality.
        #
        # Scientific operation is unchanged:
        #
        #   exact static GLRT support
        #       intersect K24 state core
        #   -> full-span coherence
        #   -> TC(final sequential phase)
        #   -> median pair coherence
        #
        # Only the point orchestration changes.
        #
        # We DO NOT construct:
        #
        #   sr, sc = np.where(routing.sequential)
        #
        # for the entire scene.
        #
        # Instead, full-width ROW bands are processed in increasing
        # row order.  Full-width bands are deliberate: concatenating
        # np.where() results from these bands is exactly equivalent to
        # the original full-scene C-order np.where() point order.
        # --------------------------------------------------------

        phase_cube = np.load(
            linked_path,
            mmap_mode="r",
        )


        stage_estimator_maps = tuple(
            np.load(
                x.estimator_path,
                mmap_mode="r",
            )
            for x
            in plan_result.stage_results
        )


        sequential_total = int(
            routing.sequential_count
        )


        # --------------------------------------------------------
        # Adaptive row-band size.
        #
        # Keep at most approximately this many spatial pixels in one
        # orchestration band, independent of total scene area.
        #
        # Current 600x2000 scene:
        #     target=500000 -> 250 rows -> 3 bands.
        #
        # Environment override is engineering-only and does not alter
        # the mathematical solution:
        #
        #   PYPSDS_FULLSPAN_TARGET_PIXELS
        # --------------------------------------------------------

        fullspan_target_pixels = max(
            1,
            int(
                os.environ.get(
                    "PYPSDS_FULLSPAN_TARGET_PIXELS",
                    "500000",
                )
            ),
        )


        fullspan_row_block = max(
            1,
            min(
                H,
                fullspan_target_pixels
                //
                max(
                    1,
                    W,
                ),
            ),
        )


        fullspan_band_count = (
            H
            +
            fullspan_row_block
            -
            1
        ) // fullspan_row_block


        print(
            "fullspan orchestration : row-band streaming"
        )

        print(
            "fullspan target pixels :",
            f"{fullspan_target_pixels:,}",
        )

        print(
            "fullspan row block     :",
            fullspan_row_block,
        )

        print(
            "fullspan row bands     :",
            fullspan_band_count,
        )


        # --------------------------------------------------------
        # Sparse production artifacts.
        #
        # These retain the EXACT existing filenames, dtype, shape,
        # and row-major point order, but are filled incrementally.
        # --------------------------------------------------------

        seq_rows_out = np.lib.format.open_memmap(
            seqdir
            /
            "production_sequential_rows.npy",

            mode="w+",
            dtype=np.int32,
            shape=(
                sequential_total,
            ),
        )


        seq_cols_out = np.lib.format.open_memmap(
            seqdir
            /
            "production_sequential_cols.npy",

            mode="w+",
            dtype=np.int32,
            shape=(
                sequential_total,
            ),
        )


        seq_tc_out = np.lib.format.open_memmap(
            seqdir
            /
            "production_sequential_fullspan_tc.npy",

            mode="w+",
            dtype=np.float32,
            shape=(
                sequential_total,
            ),
        )


        seq_pair_out = np.lib.format.open_memmap(
            seqdir
            /
            "production_sequential_pair_coherence.npy",

            mode="w+",
            dtype=np.float32,
            shape=(
                sequential_total,
            ),
        )


        seq_estimator_out = np.lib.format.open_memmap(
            seqdir
            /
            "production_sequential_estimator_code.npy",

            mode="w+",
            dtype=np.uint8,
            shape=(
                sequential_total,
            ),
        )


        sequential_offset = 0
        seq_ok_count = 0


        # --------------------------------------------------------
        # Process complete-width row bands.
        # --------------------------------------------------------

        for band_index, r0 in enumerate(
            range(
                0,
                H,
                fullspan_row_block,
            ),
            start=1,
        ):

            r1 = min(
                H,
                r0
                +
                fullspan_row_block,
            )


            local_r, local_c = np.where(
                routing.sequential[
                    r0:r1,
                    :
                ]
            )


            nband = int(
                local_r.size
            )


            if nband == 0:

                print(
                    f"fullspan band "
                    f"{band_index}/"
                    f"{fullspan_band_count} "
                    f"rows={r0}:{r1} "
                    "points=0"
                )

                continue


            # Convert to global int32 coordinates immediately so the
            # temporary int64 np.where arrays remain band-local.
            sr = (
                local_r
                +
                r0
            ).astype(
                np.int32,
                copy=False,
            )


            sc = local_c.astype(
                np.int32,
                copy=False,
            )


            if getattr(
                yxt,
                "is_phase_source_proxy",
                False,
            ):

                phase_ir0 = max(
                    0,
                    r0
                    -
                    args.half_row,
                )

                phase_ir1 = min(
                    H,
                    r1
                    +
                    args.half_row,
                )


                phase_tile = (
                    yxt.phase_source.read_tile(
                        local_row0=phase_ir0,
                        local_row1=phase_ir1,

                        local_col0=0,
                        local_col1=W,
                    )
                )


                if not np.array_equal(
                    phase_tile.geometry_valid,

                    np.asarray(
                        geom_valid[
                            phase_ir0:
                            phase_ir1,
                            :
                        ],
                        dtype=np.bool_,
                    ),
                ):

                    raise RuntimeError(
                        "Gamma fullspan geometry "
                        "parity failure"
                    )


                local_sr = (
                    sr
                    -
                    phase_ir0
                ).astype(
                    np.int32,
                    copy=False,
                )


                local_sc = sc


                # Bounded current-band phase vector only.
                # This does NOT restore the former global
                # N_DS x N_date materialization.
                local_phase_points = (
                    np.ascontiguousarray(
                        phase_cube[
                            :,
                            sr,
                            sc,
                        ].T,

                        dtype=np.complex64,
                    )
                )


                q = evaluate_fullspan_quality_points(
                    yxt=phase_tile.yxt,

                    phase_points=(
                        local_phase_points
                    ),

                    rows=local_sr,
                    cols=local_sc,

                    scale2=scale2[
                        phase_ir0:
                        phase_ir1,
                        :
                    ],

                    valid=valid[
                        phase_ir0:
                        phase_ir1,
                        :
                    ],

                    ps=ps[
                        phase_ir0:
                        phase_ir1,
                        :
                    ],

                    state_core=state_core[
                        phase_ir0:
                        phase_ir1,
                        :
                    ],

                    expected_effective_k=(
                        effective_k[
                            phase_ir0:
                            phase_ir1,
                            :
                        ]
                    ),

                    half_row=args.half_row,
                    half_col=args.half_col,
                    alpha=args.alpha,

                    batch=(
                        fullspan_batch_size
                    ),

                    support_block=1024,

                    # Same exact GLRT definition.
                    # Local coordinates cannot index
                    # the global packed cache directly.
                    static_support_cache=None,
                )

            else:

                q = evaluate_fullspan_quality_points(
                    yxt=yxt,

                    phase_cube=phase_cube,

                    rows=sr,
                    cols=sc,

                    scale2=scale2,
                    valid=valid,
                    ps=ps,

                    state_core=state_core,

                    expected_effective_k=(
                        effective_k
                    ),

                    half_row=args.half_row,
                    half_col=args.half_col,
                    alpha=args.alpha,

                    batch=fullspan_batch_size,
                    support_block=1024,

                    static_support_cache=(
                        static_support_cache
                    ),
                )


            seq_estimator_band = (
                aggregate_stage_estimators(
                    stage_estimator_maps,

                    rows=sr,
                    cols=sc,
                )
            )


            seq_ok_band = (
                q.phase_complete
                &
                np.isfinite(
                    q.temporal_coherence
                )
                &
                (
                    seq_estimator_band
                    !=
                    ESTIMATOR_INVALID
                )
            )


            # ----------------------------------------------------
            # Dense production maps: direct scatter write.
            # ----------------------------------------------------

            tc_map[
                sr,
                sc,
            ] = q.temporal_coherence


            pair_map[
                sr,
                sc,
            ] = q.median_pair_coherence


            estimator[
                sr,
                sc,
            ] = seq_estimator_band


            pl_valid[
                sr,
                sc,
            ] = seq_ok_band


            # ----------------------------------------------------
            # Existing sparse artifacts: sequential append.
            #
            # Full-width row-band ordering guarantees this is exactly
            # the previous global np.where ordering.
            # ----------------------------------------------------

            o0 = sequential_offset
            o1 = (
                o0
                +
                nband
            )


            seq_rows_out[
                o0:o1
            ] = sr


            seq_cols_out[
                o0:o1
            ] = sc


            seq_tc_out[
                o0:o1
            ] = q.temporal_coherence


            seq_pair_out[
                o0:o1
            ] = q.median_pair_coherence


            seq_estimator_out[
                o0:o1
            ] = seq_estimator_band


            sequential_offset = o1


            seq_ok_count += int(
                np.count_nonzero(
                    seq_ok_band
                )
            )


            print(
                f"fullspan band "
                f"{band_index}/"
                f"{fullspan_band_count} "
                f"rows={r0}:{r1} "
                f"points={nband:,} "
                f"done="
                f"{sequential_offset:,}/"
                f"{sequential_total:,}"
            )


        # --------------------------------------------------------
        # Strong row-major completeness invariant.
        # --------------------------------------------------------

        if (
            sequential_offset
            !=
            sequential_total
        ):

            raise RuntimeError(
                "production fullspan row-band point-count mismatch: "
                f"{sequential_offset} != "
                f"{sequential_total}"
            )


        for sparse_arr in (
            seq_rows_out,
            seq_cols_out,
            seq_tc_out,
            seq_pair_out,
            seq_estimator_out,
        ):

            sparse_arr.flush()


        # Preserve downstream variable contract without materializing
        # another full in-memory estimator vector.
        seq_estimator = (
            seq_estimator_out
        )


        # Sparse original-support full-SCM fallback
        # --------------------------------------------------------

        if getattr(
            yxt,
            "is_phase_source_proxy",
            False,
        ):

            fallback_result = None

            fallback_valid_count = 0

            fallback_estimator_parts = []


            for fallback_r0 in range(
                0,
                H,
                fullspan_row_block,
            ):

                fallback_r1 = min(
                    H,
                    fallback_r0
                    +
                    fullspan_row_block,
                )


                flr, flc = np.where(
                    fallback_exec_mask[
                        fallback_r0:
                        fallback_r1,
                        :
                    ]
                )


                if flr.size == 0:
                    continue


                fr = (
                    flr
                    +
                    fallback_r0
                ).astype(
                    np.int32,
                    copy=False,
                )

                fc = flc.astype(
                    np.int32,
                    copy=False,
                )


                fir0 = max(
                    0,
                    fallback_r0
                    -
                    args.half_row,
                )

                fir1 = min(
                    H,
                    fallback_r1
                    +
                    args.half_row,
                )


                tile = (
                    yxt.phase_source.read_tile(
                        local_row0=fir0,
                        local_row1=fir1,

                        local_col0=0,
                        local_col1=W,
                    )
                )


                if not np.array_equal(
                    tile.geometry_valid,

                    np.asarray(
                        geom_valid[
                            fir0:fir1,
                            :
                        ],
                        dtype=np.bool_,
                    ),
                ):

                    raise RuntimeError(
                        "Gamma fallback geometry "
                        "parity failure"
                    )


                lfr = (
                    fr
                    -
                    fir0
                ).astype(
                    np.int32,
                    copy=False,
                )


                lfc = fc


                result = run_full_scm_points(
                    yxt=tile.yxt,

                    rows=lfr,
                    cols=lfc,

                    scale2=scale2[
                        fir0:fir1,
                        :
                    ],

                    valid=valid[
                        fir0:fir1,
                        :
                    ],

                    ps=ps[
                        fir0:fir1,
                        :
                    ],

                    expected_original_k=(
                        original_k[
                            fir0:fir1,
                            :
                        ]
                    ),

                    phase_sink=None,

                    half_row=args.half_row,
                    half_col=args.half_col,
                    alpha=args.alpha,

                    min_shp=shp_policy.full_scm_rank_min_shp,

                    beta=args.beta,

                    gamma_jitter=(
                        args.gamma_jitter
                    ),

                    emi_mu=args.emi_mu,

                    batch=2048,
                    support_block=1024,

                    pl_workers=(
                        args.pl_workers
                    ),

                    pl_chunk_size=(
                        args.pl_chunk_size
                    ),
                )


                ok = np.asarray(
                    result.pl_valid,
                    dtype=np.bool_,
                )


                if np.any(
                    ok
                ):

                    writer(
                        stage_index=-1,

                        real_indices=tuple(
                            range(
                                ndate
                            )
                        ),

                        rows=fr[
                            ok
                        ],

                        cols=fc[
                            ok
                        ],

                        phase=result.phase[
                            ok
                        ],
                    )


                tc_map[
                    fr,
                    fc,
                ] = (
                    result
                    .temporal_coherence
                )


                pair_map[
                    fr,
                    fc,
                ] = (
                    result
                    .median_pair_coherence
                )


                estimator[
                    fr,
                    fc,
                ] = result.estimator


                emi_eig_map[
                    fr,
                    fc,
                ] = (
                    result
                    .emi_eigenvalue
                )


                evd_eig_map[
                    fr,
                    fc,
                ] = (
                    result
                    .evd_eigenvalue
                )


                gamma_min_map[
                    fr,
                    fc,
                ] = (
                    result
                    .gamma_min_eigenvalue
                )


                pl_valid[
                    fr,
                    fc,
                ] = result.pl_valid


                fallback_valid_count += int(
                    result.valid_count
                )


                fallback_estimator_parts.append(
                    np.asarray(
                        result.estimator,
                        dtype=np.uint8,
                    )
                )


                print(
                    "[full-SCM-fallback-stream] "
                    f"rows={fallback_r0}:"
                    f"{fallback_r1} "
                    f"points={fr.size:,} "
                    f"valid={result.valid_count:,}"
                )


            if fallback_estimator_parts:

                class _production:
                    pass


                fallback_result = (
                    _production()
                )


                fallback_result.valid_count = (
                    fallback_valid_count
                )


                fallback_result.estimator = (
                    np.concatenate(
                        fallback_estimator_parts
                    )
                )

        else:

            fr, fc = np.where(
                fallback_exec_mask
            )

            fr = fr.astype(
                np.int32,
                copy=False,
            )

            fc = fc.astype(
                np.int32,
                copy=False,
            )

            fallback_result = None

            if fr.size:

                fallback_result = (
                    run_full_scm_points(
                        yxt=yxt,

                        rows=fr,
                        cols=fc,

                        scale2=scale2,
                        valid=valid,
                        ps=ps,

                        expected_original_k=(
                            original_k
                        ),

                        phase_sink=writer,

                        half_row=args.half_row,
                        half_col=args.half_col,
                        alpha=args.alpha,

                        min_shp=shp_policy.full_scm_rank_min_shp,

                        beta=args.beta,

                        gamma_jitter=(
                            args.gamma_jitter
                        ),

                        emi_mu=args.emi_mu,

                        batch=2048,
                        support_block=1024,

                        pl_workers=(
                            args.pl_workers
                        ),

                        pl_chunk_size=(
                            args.pl_chunk_size
                        ),
                    )
                )

                tc_map[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .temporal_coherence
                )

                pair_map[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .median_pair_coherence
                )

                estimator[
                    fr,
                    fc,
                ] = fallback_result.estimator

                emi_eig_map[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .emi_eigenvalue
                )

                evd_eig_map[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .evd_eigenvalue
                )

                gamma_min_map[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .gamma_min_eigenvalue
                )

                pl_valid[
                    fr,
                    fc,
                ] = (
                    fallback_result
                    .pl_valid
                )

        writer.flush()
        writer.close()

        # --------------------------------------------------------
        # PS phase fill -- preserve existing Phase linking semantics.
        # --------------------------------------------------------

        linked_phase = np.load(
            linked_path,
            mmap_mode="r+",
        )

        pr, pcx = np.where(
            np.asarray(
                ps,
                dtype=np.bool_,
            )
            &
            np.asarray(
                geom_valid,
                dtype=np.bool_,
            )
        )

        if getattr(
            yxt,
            "is_phase_source_proxy",
            False,
        ):

            for ps_r0 in range(
                0,
                H,
                fullspan_row_block,
            ):

                ps_r1 = min(
                    H,
                    ps_r0
                    +
                    fullspan_row_block,
                )


                local_ps = (
                    np.asarray(
                        ps[
                            ps_r0:ps_r1,
                            :
                        ],
                        dtype=np.bool_,
                    )
                    &
                    np.asarray(
                        geom_valid[
                            ps_r0:ps_r1,
                            :
                        ],
                        dtype=np.bool_,
                    )
                )


                lr, lc = np.where(
                    local_ps
                )


                if lr.size == 0:
                    continue


                tile = (
                    yxt.phase_source.read_tile(
                        local_row0=ps_r0,
                        local_row1=ps_r1,

                        local_col0=0,
                        local_col1=W,
                    )
                )


                if not np.array_equal(
                    tile.geometry_valid,

                    np.asarray(
                        geom_valid[
                            ps_r0:ps_r1,
                            :
                        ],
                        dtype=np.bool_,
                    ),
                ):

                    raise RuntimeError(
                        "Gamma PS geometry "
                        "parity failure"
                    )


                gr = (
                    lr
                    +
                    ps_r0
                ).astype(
                    np.int32,
                    copy=False,
                )


                gc = lc.astype(
                    np.int32,
                    copy=False,
                )


                pph = tile.yxt[
                    lr,
                    lc,
                    :
                ]


                pph = np.exp(
                    1j
                    *
                    np.angle(
                        pph
                    )
                ).astype(
                    np.complex64
                )


                pph *= np.exp(
                    -1j
                    *
                    np.angle(
                        pph[
                            :,
                            0,
                        ]
                    )
                )[
                    :,
                    None
                ]


                linked_phase[
                    :,
                    gr,
                    gc,
                ] = pph.T

        else:

            if pr.size:

                pph = yxt[
                    pr,
                    pcx,
                    :,
                ]

                pph = np.exp(
                    1j
                    *
                    np.angle(
                        pph
                    )
                ).astype(
                    np.complex64
                )

                pph *= np.exp(
                    -1j
                    *
                    np.angle(
                        pph[
                            :,
                            0,
                        ]
                    )
                )[
                    :,
                    None
                ]

                linked_phase[
                    :,
                    pr,
                    pcx,
                ] = pph.T

    # --------------------------------------------------------
    # Flush output contract.
    # --------------------------------------------------------

    for arr in (
        shp_count,
        tc_map,
        pair_map,
        estimator,
        emi_eig_map,
        evd_eig_map,
        gamma_min_map,
        pl_valid,
        linked_phase,
    ):
        arr.flush()

    ds_mask = (
        np.asarray(
            pl_valid,
            dtype=np.bool_,
        )
        &
        ~np.asarray(
            ps,
            dtype=np.bool_,
        )
    )

    tc_threshold = float(
        cfg.get(
            "selection",
            {},
        )
        .get(
            "ds",
            {},
        )
        .get(
            "temporal_coherence_min",
            0.80,
        )
    )

    tc_pass = (
        ds_mask
        &
        (
            np.asarray(
                tc_map
            )
            >=
            tc_threshold
        )
    )

    seq_emi = int(
        np.count_nonzero(
            seq_estimator
            ==
            ESTIMATOR_EMI
        )
    )

    seq_evd = int(
        np.count_nonzero(
            seq_estimator
            ==
            ESTIMATOR_EVD
        )
    )

    seq_invalid = int(
        np.count_nonzero(
            seq_estimator
            ==
            ESTIMATOR_INVALID
        )
    )

    print()
    print("=" * 88)
    print(
        "Sequential Phase linking production complete"
    )
    print("=" * 88)

    print(
        "formal DS          :",
        f"{routing.formal_count:,}",
    )

    print(
        "sequential route   :",
        f"{routing.sequential_count:,}",
    )

    print(
        "  PL valid         :",
        f"{seq_ok_count:,}",
    )

    print(
        "  EMI all stages   :",
        f"{seq_emi:,}",
    )

    print(
        "  >=1 EVD stage    :",
        f"{seq_evd:,}",
    )

    print(
        "  invalid          :",
        f"{seq_invalid:,}",
    )

    print(
        "full-SCM fallback  :",
        f"{routing.fallback_count:,}",
    )

    if fallback_result is not None:

        print(
            "  PL valid         :",
            f"{fallback_result.valid_count:,}",
        )

        print(
            "  EMI              :",
            f"{np.count_nonzero(fallback_result.estimator == ESTIMATOR_EMI):,}",
        )

        print(
            "  EVD              :",
            f"{np.count_nonzero(fallback_result.estimator == ESTIMATOR_EVD):,}",
        )

    print(
        "combined PL valid  :",
        f"{np.count_nonzero(ds_mask):,}",
    )

    print(
        f"combined TC>={tc_threshold:.2f} :",
        f"{np.count_nonzero(tc_pass):,}",
    )

    print(
        "stage seconds      :",
        f"{plan_result.total_stage_seconds:.3f}",
    )

    print(
        "total wall         :",
        f"{perf_counter() - t_all:.3f}s",
    )

    print(
        "linked_phase       :",
        linked_path,
    )

    complete_manifest = (
        _write_completed_phase_linking_manifest(
            cfg=cfg,
            config_path=config_path,

            outdir=outdir,

            yxt=yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom_valid,
            ps=ps,

            H=H,
            W=W,
            ndate=ndate,

            args=args,

            summary={
                "formal_ds":
                    int(
                        routing.formal_count
                    ),

                "sequential_route":
                    int(
                        routing.sequential_count
                    ),

                "fallback":
                    int(
                        routing.fallback_count
                    ),

                "combined_pl_valid":
                    int(
                        np.count_nonzero(
                            ds_mask
                        )
                    ),

                "tc_pass":
                    int(
                        np.count_nonzero(
                            tc_pass
                        )
                    ),
            },
        )
    )


    print(
        "completion manifest :",
        complete_manifest,
    )


    print(
        "production SEQUENTIAL PRODUCTION: PASS"
    )
