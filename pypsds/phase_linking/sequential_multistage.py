from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..progress import ProgressReporter

from .coherence import compressed_coherence
from .compression import compress_stage_slcs
from .emi import (
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
    temporal_coherence_fused,
)
from .emi_threshold import (
    robust_emi_threshold_threaded,
)
from .tile_prefetch import (
    OneAheadTilePrefetcher,
)

from .shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


STATE_OUTSIDE = np.uint8(0)
STATE_VALID = np.uint8(1)
STATE_LOW_K = np.uint8(2)
STATE_PL_INVALID = np.uint8(3)
STATE_COMPRESSION_INVALID = np.uint8(4)
STATE_CENTER_INPUT_INVALID = np.uint8(5)


@dataclass(slots=True)
class StageResult:
    stage_index: int

    real_indices: tuple[int, ...]
    compressed_input_ids: tuple[str, ...]

    solver_size: int
    first_real_idx: int
    reference_idx: int

    state_pixels: int
    state_valid: int

    low_k: int
    pl_invalid: int
    compression_invalid: int
    center_input_invalid: int

    static_k_excess: int
    static_k_mismatch: int

    compression_formula_max_abs_diff: float

    support_seconds: float
    covariance_seconds: float
    phase_linking_seconds: float
    compression_seconds: float
    elapsed_seconds: float

    compressed_path: Path
    valid_path: Path
    state_code_path: Path
    shp_count_path: Path
    temporal_coherence_path: Path
    estimator_path: Path


def _new_memmap(
    path: Path,
    *,
    shape,
    dtype,
    fill,
    resume_existing: bool = False,
):
    """
    Create a fresh stage map, or reopen a checkpointed map.

    resume_existing=True is allowed only when the file already
    exists with exactly the expected shape and dtype.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_shape = tuple(
        int(x)
        for x in shape
    )

    expected_dtype = np.dtype(
        dtype
    )


    if resume_existing:

        if not path.is_file():
            raise RuntimeError(
                "checkpoint exists but stage output is missing: "
                f"{path}"
            )

        arr = np.load(
            path,
            mmap_mode="r+",
            allow_pickle=False,
        )

        if arr.shape != expected_shape:
            raise RuntimeError(
                "checkpoint stage-map shape mismatch: "
                f"{path}: {arr.shape} != {expected_shape}"
            )

        if arr.dtype != expected_dtype:
            raise RuntimeError(
                "checkpoint stage-map dtype mismatch: "
                f"{path}: {arr.dtype} != {expected_dtype}"
            )

        return arr


    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=expected_dtype,
        shape=expected_shape,
    )

    arr[...] = fill

    return arr



# ============================================================================
# production stage-major tile checkpointing
# ============================================================================

_CHECKPOINT_FORMAT = (
    "pyPSDS-GAMMA-sequential-tile-checkpoint-v1"
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


def _path_token(
    path,
):
    """
    Conservative source token.

    Small files are content-hashed.
    Large files use size+mtime to avoid repeatedly hashing
    the complete full-scene phase cube.
    """

    if path is None:
        return None

    path = Path(
        path
    )

    if not path.is_file():
        return None

    st = path.stat()

    out = {
        "path":
            str(
                path.resolve()
            ),

        "size":
            int(
                st.st_size
            ),
    }

    if st.st_size <= (
        64
        *
        1024
        *
        1024
    ):

        out[
            "sha256"
        ] = _sha256_file(
            path
        )

    else:

        out[
            "mtime_ns"
        ] = int(
            st.st_mtime_ns
        )

    return out


def _array_source_token(
    arr,
):

    out = {
        "shape":
            [
                int(x)
                for x
                in np.shape(
                    arr
                )
            ],

        "dtype":
            str(
                np.asarray(
                    arr
                ).dtype
            ),
    }

    filename = getattr(
        arr,
        "filename",
        None,
    )

    if filename is not None:

        out[
            "file"
        ] = _path_token(
            filename
        )

    return out


def _atomic_json(
    path: Path,
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
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def _stage_checkpoint_fingerprint(
    *,
    output_dir,
    stage_index,
    compressed_input_ids,
    compressed_inputs,
    real_indices,

    H,
    W,
    ndate,

    full_glrt_nslc,
    state_min_shp,

    half_row,
    half_col,
    alpha,

    beta,
    gamma_jitter,
    emi_mu,
    emi_backend,

    tile_rows,
    tile_cols,

    center_batch,
    support_block,

    pl_workers,
    pl_chunk_size,

    static_support_cache,
):

    output_dir = Path(
        output_dir
    )

    processing = (
        output_dir.parent
    )

    phase_cache = (
        processing
        /
        "cache"
        /
        "phase_source_checkpoint_token.json"
    )

    state_core_path = (
        output_dir
        /
        "compression_state_core_K24.npy"
    )

    effective_path = (
        output_dir
        /
        "compression_state_core_K24_effective_shp_count.npy"
    )


    if static_support_cache is not None:

        support_manifest = (
            static_support_cache.manifest
        )

    else:

        support_manifest = None


    return {
        "format":
            _CHECKPOINT_FORMAT,

        "code_sha256":
            _sha256_file(
                Path(
                    __file__
                )
            ),

        "stage_index":
            int(
                stage_index
            ),

        "scene":
            [
                int(H),
                int(W),
            ],

        "ndate":
            int(
                ndate
            ),

        "real_indices":
            [
                int(x)
                for x
                in real_indices
            ],

        "compressed_input_ids":
            [
                str(x)
                for x
                in compressed_input_ids
            ],

        "compressed_inputs":
            [
                _array_source_token(
                    x
                )
                for x
                in compressed_inputs
            ],

        "phase_cache":
            _path_token(
                phase_cache
            ),

        "state_core":
            _path_token(
                state_core_path
            ),

        "effective_k":
            _path_token(
                effective_path
            ),

        "support_manifest":
            support_manifest,

        "full_glrt_nslc":
            int(
                full_glrt_nslc
            ),

        "state_min_shp":
            int(
                state_min_shp
            ),

        "half_row":
            int(
                half_row
            ),

        "half_col":
            int(
                half_col
            ),

        "alpha":
            float(
                alpha
            ),

        "beta":
            float(
                beta
            ),

        "gamma_jitter":
            float(
                gamma_jitter
            ),

        "emi_mu":
            float(
                emi_mu
            ),

        "emi_backend":
            str(
                emi_backend
            ),

        "tile_rows":
            int(
                tile_rows
            ),

        "tile_cols":
            int(
                tile_cols
            ),

        "center_batch":
            int(
                center_batch
            ),

        "support_block":
            int(
                support_block
            ),

        "pl_workers":
            int(
                pl_workers
            ),

        "pl_chunk_size":
            int(
                pl_chunk_size
            ),
    }


def _fingerprint_sha256(
    fingerprint,
):

    raw = json.dumps(
        fingerprint,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


def _prepare_stage_checkpoint(
    *,
    output_dir,
    prefix,
    fingerprint,
    tile_count,
):

    root = (
        Path(
            output_dir
        )
        /
        "checkpoints"
        /
        prefix
    )

    force_fresh = (
        os.environ.get(
            "PYPSDS_FORCE_FRESH_TILES",
            "0",
        )
        ==
        "1"
    )


    if (
        force_fresh
        and
        root.exists()
    ):

        shutil.rmtree(
            root
        )


    root.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest_path = (
        root
        /
        "manifest.json"
    )


    fp_hash = (
        _fingerprint_sha256(
            fingerprint
        )
    )


    expected_manifest = {
        "format":
            _CHECKPOINT_FORMAT,

        "fingerprint_sha256":
            fp_hash,

        "fingerprint":
            fingerprint,
    }


    if manifest_path.is_file():

        old = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            old.get(
                "format"
            )
            !=
            _CHECKPOINT_FORMAT
        ):

            raise RuntimeError(
                "unsupported stage checkpoint format"
            )


        if (
            old.get(
                "fingerprint_sha256"
            )
            !=
            fp_hash
        ):

            raise RuntimeError(
                "sequential checkpoint fingerprint changed. "
                "For a deliberate fresh recomputation use:\n"
                "  PYPSDS_FORCE_FRESH_TILES=1"
            )

    else:

        _atomic_json(
            manifest_path,
            expected_manifest,
        )


    # --------------------------------------------------------
    # Only a contiguous completed prefix is resumable.
    #
    # This matches stage-major execution and removes ambiguity
    # about dependency state.
    # --------------------------------------------------------

    prefix_count = 0
    last_record = None
    gap_seen = False


    for tile_index in range(
        1,
        int(tile_count) + 1,
    ):

        marker = (
            root
            /
            f"tile_{tile_index:04d}.json"
        )


        if marker.is_file():

            if gap_seen:

                raise RuntimeError(
                    "non-contiguous sequential checkpoint "
                    f"at tile {tile_index}"
                )


            record = json.loads(
                marker.read_text(
                    encoding="utf-8"
                )
            )


            if int(
                record.get(
                    "tile_index",
                    -1,
                )
            ) != tile_index:

                raise RuntimeError(
                    "checkpoint tile index mismatch"
                )


            if (
                record.get(
                    "fingerprint_sha256"
                )
                !=
                fp_hash
            ):

                raise RuntimeError(
                    "checkpoint marker fingerprint mismatch"
                )


            prefix_count = (
                tile_index
            )

            last_record = (
                record
            )

        else:

            gap_seen = True


    resume_done = (
        int(
            last_record.get(
                "total_done",
                0,
            )
        )
        if
        last_record is not None
        else
        0
    )


    return (
        root,
        prefix_count,
        resume_done,
        fp_hash,
    )


def _commit_stage_checkpoint(
    *,
    root,
    fingerprint_sha256,
    tile_index,
    bounds,
    total_done,
):

    payload = {
        "format":
            _CHECKPOINT_FORMAT,

        "fingerprint_sha256":
            fingerprint_sha256,

        "tile_index":
            int(
                tile_index
            ),

        "bounds":
            [
                int(x)
                for x
                in bounds
            ],

        "total_done":
            int(
                total_done
            ),

        "time_unix":
            float(
                time.time()
            ),
    }


    _atomic_json(
        Path(
            root
        )
        /
        f"tile_{tile_index:04d}.json",

        payload,
    )


def _bool_windows(
    arr: np.ndarray,
    *,
    half_row: int,
    half_col: int,
):
    x = np.asarray(
        arr,
        dtype=np.bool_,
    )

    padded = np.pad(
        x,
        (
            (half_row, half_row),
            (half_col, half_col),
        ),
        mode="constant",
        constant_values=False,
    )

    return np.lib.stride_tricks.sliding_window_view(
        padded,
        (
            2 * half_row + 1,
            2 * half_col + 1,
        ),
    )


def _iter_tiles(
    H: int,
    W: int,
    tile_rows: int,
    tile_cols: int,
):
    for r0 in range(
        0,
        H,
        tile_rows,
    ):
        r1 = min(
            H,
            r0 + tile_rows,
        )

        for c0 in range(
            0,
            W,
            tile_cols,
        ):
            c1 = min(
                W,
                c0 + tile_cols,
            )

            yield (
                r0,
                r1,
                c0,
                c1,
            )


def _manual_stage_compression(
    stage_z: np.ndarray,
    stage_phase: np.ndarray,
    *,
    first_real_idx: int,
    reference_idx: int,
):
    """
    Independent audit implementation.

    Intentionally does NOT call compress_stage_slcs().
    """

    z = np.asarray(
        stage_z,
        dtype=np.complex64,
    )

    ph = np.asarray(
        stage_phase,
        dtype=np.complex64,
    )

    ref = ph[
        :,
        reference_idx
    ][
        :,
        None
    ]

    ph_ref = (
        ph
        *
        np.conj(
            ref
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    real_z = z[
        :,
        first_real_idx:
    ]

    real_ph = ph_ref[
        :,
        first_real_idx:
    ]

    with np.errstate(
        invalid="ignore",
    ):
        projected = np.nansum(
            real_z
            *
            np.conj(
                real_ph
            ),
            axis=1,
        )

        amp = np.nanmean(
            np.abs(
                real_z
            ),
            axis=1,
            dtype=np.float64,
        ).astype(
            np.float32
        )

    out = (
        amp
        *
        np.exp(
            1j
            *
            np.angle(
                projected
            )
        )
    ).astype(
        np.complex64
    )

    bad = (
        ~np.isfinite(
            projected.real
        )
        |
        ~np.isfinite(
            projected.imag
        )
        |
        ~np.isfinite(
            amp
        )
        |
        (
            np.abs(
                projected
            )
            ==
            0
        )
    )

    out[
        bad
    ] = np.complex64(
        np.nan
        +
        1j
        *
        np.nan
    )

    return out



def reference_real_phase(
    stage_phase: np.ndarray,
    *,
    first_real_idx: int,
    reference_idx: int,
) -> np.ndarray:
    """
    Re-reference one solved sequential stage and return only
    its real-acquisition phase histories.

    The operation follows the same ALWAYS_FIRST stage-reference
    semantics used before compressed-SLC generation:

        phase_ref = phase * conj(phase[:, reference_idx])

    Compressed input columns are then removed.

    Parameters
    ----------
    stage_phase
        Complex unit phase, shape [n_point, n_stage].
    first_real_idx
        First real-acquisition column in the stage solver.
    reference_idx
        Stage reference column.

    Returns
    -------
    np.ndarray
        complex64 array, shape [n_point, n_real].
    """

    ph = np.asarray(
        stage_phase,
        dtype=np.complex64,
    )

    if ph.ndim != 2:
        raise ValueError(
            "stage_phase must be 2-D"
        )

    nstage = ph.shape[1]

    if not (
        0 <= reference_idx < nstage
    ):
        raise ValueError(
            "reference_idx outside stage"
        )

    if not (
        0 <= first_real_idx <= nstage
    ):
        raise ValueError(
            "first_real_idx outside stage"
        )

    ref = ph[
        :,
        reference_idx,
    ][
        :,
        None
    ]

    referenced = (
        ph
        *
        np.conj(
            ref
        )
    ).astype(
        np.complex64,
        copy=False,
    )

    return np.asarray(
        referenced[
            :,
            first_real_idx:
        ],
        dtype=np.complex64,
    )


def run_sequential_stage(
    *,
    stage_index: int,
    compressed_input_ids: tuple[str, ...],
    compressed_inputs: tuple[np.ndarray, ...],
    yxt: np.ndarray,
    real_indices: tuple[int, ...],

    scale2: np.ndarray,
    valid: np.ndarray,
    ps: np.ndarray,

    state_core: np.ndarray,
    expected_effective_k: np.ndarray,

    output_dir: Path,

    full_glrt_nslc: int,
    state_min_shp: int,

    inputs_complete: bool,

    half_row: int = 5,
    half_col: int = 11,
    alpha: float = 0.005,

    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,

    emi_backend: str = "current_eigh",

    tile_rows: int = 256,
    tile_cols: int = 512,

    center_batch: int = 16000,
    support_block: int = 1024,

    static_support_cache=None,

    pl_workers: int = 16,
    pl_chunk_size: int = 512,

    prefetch_tiles: int = 1,

    formula_audit_points: int = 5000,

    phase_sink=None,
) -> StageResult:

    if not getattr(
        yxt,
        "is_phase_source_proxy",
        False,
    ):

        yxt = np.asarray(
            yxt
        )

    H, W, ndate = yxt.shape

    prefetch_tiles = int(
        prefetch_tiles
    )

    if prefetch_tiles not in (
        0,
        1,
    ):
        raise ValueError(
            "prefetch_tiles must be 0 or 1"
        )

    real_indices = tuple(
        int(x)
        for x
        in real_indices
    )

    compressed_input_ids = tuple(
        str(x)
        for x
        in compressed_input_ids
    )

    compressed_inputs = tuple(
        compressed_inputs
    )

    ncomp = len(
        compressed_inputs
    )

    if len(
        compressed_input_ids
    ) != ncomp:
        raise ValueError(
            "compressed input id/data count mismatch"
        )

    if not real_indices:
        raise ValueError(
            "empty real_indices"
        )

    # Temporal planner currently creates contiguous real blocks.
    start_real = real_indices[0]
    stop_real = real_indices[-1] + 1

    if real_indices != tuple(
        range(
            start_real,
            stop_real,
        )
    ):
        raise ValueError(
            "Sequential execution requires contiguous "
            "real acquisition blocks"
        )

    if (
        start_real < 0
        or
        stop_real > ndate
    ):
        raise ValueError(
            "real acquisition index outside YXT"
        )

    for i, arr in enumerate(
        compressed_inputs
    ):
        if arr.shape != (
            H,
            W,
        ):
            raise ValueError(
                f"compressed input {i} shape mismatch: "
                f"{arr.shape}"
            )

    if state_core.shape != (
        H,
        W,
    ):
        raise ValueError(
            "state_core shape mismatch"
        )

    if expected_effective_k.shape != (
        H,
        W,
    ):
        raise ValueError(
            "effective-K reference shape mismatch"
        )

    if full_glrt_nslc != ndate:
        raise ValueError(
            "frozen GLRT nslc must equal full stack size"
        )

    # --------------------------------------------------------
    # ALWAYS_FIRST semantics.
    #
    # stage 0:
    #   no compressed input
    #   reference = first real = index 0
    #
    # later:
    #   compressed SLCs are prepended
    #   latest compressed = ncomp - 1
    # --------------------------------------------------------

    first_real_idx = ncomp

    reference_idx = (
        0
        if
        ncomp == 0
        else
        ncomp - 1
    )

    stage_n = (
        ncomp
        +
        len(
            real_indices
        )
    )

    if not (
        0
        <= reference_idx
        < stage_n
    ):
        raise RuntimeError(
            "invalid stage reference index"
        )


    emi_backend = str(
        emi_backend
    ).strip().lower()

    if emi_backend not in {
        "current_eigh",
        "threshold_cholesky",
    }:
        raise ValueError(
            f"unsupported sequential EMI backend: {emi_backend}"
        )

    if emi_backend == "current_eigh":

        emi_solver = (
            robust_emi_threaded
        )

    else:

        emi_solver = (
            robust_emi_threshold_threaded
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"sequential_stage"
        f"{stage_index:04d}"
    )


    # ------------------------------------------------------------------
    # production checkpoint setup.
    #
    # Numerical algorithms are unchanged.  This only determines whether
    # persistent stage maps may be reopened and which completed tile
    # prefix can be skipped.
    # ------------------------------------------------------------------

    checkpoint_tiles = list(
        _iter_tiles(
            H,
            W,
            tile_rows,
            tile_cols,
        )
    )


    checkpoint_fingerprint = (
        _stage_checkpoint_fingerprint(
            output_dir=output_dir,

            stage_index=stage_index,

            compressed_input_ids=(
                compressed_input_ids
            ),

            compressed_inputs=(
                compressed_inputs
            ),

            real_indices=(
                real_indices
            ),

            H=H,
            W=W,
            ndate=ndate,

            full_glrt_nslc=(
                full_glrt_nslc
            ),

            state_min_shp=(
                state_min_shp
            ),

            half_row=half_row,
            half_col=half_col,
            alpha=alpha,

            beta=beta,

            gamma_jitter=(
                gamma_jitter
            ),

            emi_mu=emi_mu,
            emi_backend=emi_backend,

            tile_rows=tile_rows,
            tile_cols=tile_cols,

            center_batch=(
                center_batch
            ),

            support_block=(
                support_block
            ),

            pl_workers=(
                pl_workers
            ),

            pl_chunk_size=(
                pl_chunk_size
            ),

            static_support_cache=(
                static_support_cache
            ),
        )
    )


    (
        checkpoint_root,
        checkpoint_prefix,
        checkpoint_resume_done,
        checkpoint_fp_hash,
    ) = _prepare_stage_checkpoint(
        output_dir=output_dir,
        prefix=prefix,

        fingerprint=(
            checkpoint_fingerprint
        ),

        tile_count=len(
            checkpoint_tiles
        ),
    )


    resume_existing = (
        checkpoint_prefix
        >
        0
    )


    def new_stage_map(
        path,
        *,
        shape,
        dtype,
        fill,
    ):

        return _new_memmap(
            path,

            shape=shape,
            dtype=dtype,
            fill=fill,

            resume_existing=(
                resume_existing
            ),
        )


    print(
        "tile checkpoint         :",
        (
            f"resume "
            f"{checkpoint_prefix}/"
            f"{len(checkpoint_tiles)}"
            if checkpoint_prefix
            else
            f"fresh 0/"
            f"{len(checkpoint_tiles)}"
        ),
    )


    # ------------------------------------------------------------------
    # production grouped durable checkpoint.
    #
    # Flushing every individual tile is prohibitively expensive because
    # numpy.memmap.flush() synchronizes large mapped files.
    #
    # We therefore make durability periodic:
    #
    #   compute N tiles
    #       -> flush persistent maps once
    #       -> flush linked phase once
    #       -> atomically publish all N tile markers
    #
    # Crash semantics remain conservative:
    # any tile without a marker is recomputed after restart.
    #
    # Default N=12 means at most eleven completed-but-uncommitted tiles
    # are lost after a hard crash.
    # ------------------------------------------------------------------

    checkpoint_every_tiles = (
        max(
            1,
            int(
                os.environ.get(
                    "PYPSDS_CHECKPOINT_EVERY_TILES",
                    "12",
                )
            ),
        )
    )


    checkpoint_pending = []


    print(
        "checkpoint cadence      :",
        (
            f"every "
            f"{checkpoint_every_tiles} "
            "tile(s)"
        ),
    )


    compressed_path = (
        output_dir
        /
        f"{prefix}_compressed.npy"
    )

    valid_path = (
        output_dir
        /
        f"{prefix}_state_valid.npy"
    )

    state_code_path = (
        output_dir
        /
        f"{prefix}_state_code.npy"
    )

    shp_count_path = (
        output_dir
        /
        f"{prefix}_effective_shp_count.npy"
    )

    tc_path = (
        output_dir
        /
        f"{prefix}_temporal_coherence.npy"
    )

    estimator_path = (
        output_dir
        /
        f"{prefix}_estimator.npy"
    )

    compressed_out = new_stage_map(
        compressed_path,
        shape=(H, W),
        dtype=np.complex64,
        fill=(
            np.nan
            +
            1j
            *
            np.nan
        ),
    )

    state_valid_out = new_stage_map(
        valid_path,
        shape=(H, W),
        dtype=np.bool_,
        fill=False,
    )

    state_code_out = new_stage_map(
        state_code_path,
        shape=(H, W),
        dtype=np.uint8,
        fill=STATE_OUTSIDE,
    )

    k_out = new_stage_map(
        shp_count_path,
        shape=(H, W),
        dtype=np.int16,
        fill=-1,
    )

    tc_out = new_stage_map(
        tc_path,
        shape=(H, W),
        dtype=np.float32,
        fill=np.nan,
    )

    est_out = new_stage_map(
        estimator_path,
        shape=(H, W),
        dtype=np.uint8,
        fill=ESTIMATOR_INVALID,
    )

    state_core = np.asarray(
        state_core,
        dtype=np.bool_,
    )

    valid = np.asarray(
        valid,
        dtype=np.bool_,
    )

    ps = np.asarray(
        ps,
        dtype=np.bool_,
    )

    expected_effective_k = np.asarray(
        expected_effective_k,
        dtype=np.int16,
    )

    pairs = image_pairs(
        stage_n
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    tiles = checkpoint_tiles

    total_state = int(
        np.count_nonzero(
            state_core
        )
    )

    total_done = checkpoint_resume_done
    total_valid = 0

    low_k_n = 0
    pl_invalid_n = 0
    comp_invalid_n = 0
    center_invalid_n = 0

    static_k_excess = 0
    static_k_mismatch = 0

    formula_checked = 0
    formula_max_diff = 0.0

    support_seconds = 0.0
    covariance_seconds = 0.0
    phase_seconds = 0.0
    compression_seconds = 0.0

    t_all = perf_counter()

    stage_progress = ProgressReporter(
        label=f"sequential-stage-{stage_index}",
        total=total_state,
        unit="center",
        min_interval=10.0,
        log_path=(
            output_dir
            /
            f"{prefix}_progress.jsonl"
        ),
    )

    print()
    print(
        "=" * 108
    )
    print(
        f"Sequential PL stage {stage_index}"
    )
    print(
        "=" * 108
    )

    print(
        "compressed inputs      :",
        list(
            compressed_input_ids
        ),
    )

    print(
        "real indices           :",
        list(
            real_indices
        ),
    )

    print(
        "solver size            :",
        stage_n,
    )

    print(
        "first real idx         :",
        first_real_idx,
    )

    print(
        "reference idx          :",
        reference_idx,
    )

    print(
        "input state complete   :",
        inputs_complete,
    )


    print(
        "EMI backend            :",
        emi_backend,
    )


    print(
        "SHP support source      :",
        (
            "exact_static_cache"
            if static_support_cache is not None
            else "exact_GLRT_runtime"
        ),
    )

    print()

    # Bounded one-ahead GAMMA streaming.
    #
    # The current tile remains the only compute tile. At most one
    # additional real-acquisition tile is read/corrected in a background
    # thread while the current tile performs SHP/coherence/EMI/compression.
    #
    # NPY/memmap phase caches keep their existing synchronous path.
    prefetch_enabled = (
        prefetch_tiles == 1
        and
        bool(
            getattr(
                yxt,
                "is_phase_source_proxy",
                False,
            )
        )
    )

    prefetch_budget = None

    if prefetch_enabled:
        phase_source = getattr(
            yxt,
            "phase_source",
            None,
        )

        if (
            phase_source is not None
            and hasattr(
                phase_source,
                "configure_prefetch_concurrency",
            )
        ):
            prefetch_budget = (
                phase_source.configure_prefetch_concurrency(
                    pl_workers=pl_workers,
                )
            )

            print(
                "prefetch CPU budget    :",
                prefetch_budget[
                    "gamma_process_budget"
                ],
            )

            print(
                "prefetch GAMMA layout  :",
                (
                    f"{prefetch_budget['spatial_workers']} spatial x "
                    f"{prefetch_budget['pair_workers']} pair = "
                    f"{prefetch_budget['max_gamma_processes']} processes"
                ),
            )

    active_prefetch_positions = tuple(
        position
        for position, (
            _r0,
            _r1,
            _c0,
            _c1,
        )
        in enumerate(
            tiles
        )
        if (
            position + 1
            >
            checkpoint_prefix
            and
            bool(
                np.any(
                    state_core[
                        _r0:_r1,
                        _c0:_c1,
                    ]
                )
            )
        )
    )

    def load_real_tile(
        position: int,
    ):
        (
            _r0,
            _r1,
            _c0,
            _c1,
        ) = tiles[
            int(
                position
            )
        ]

        _ir0 = max(
            0,
            _r0 - half_row,
        )

        _ir1 = min(
            H,
            _r1 + half_row,
        )

        _ic0 = max(
            0,
            _c0 - half_col,
        )

        _ic1 = min(
            W,
            _c1 + half_col,
        )

        return np.ascontiguousarray(
            yxt[
                _ir0:_ir1,
                _ic0:_ic1,
                start_real:stop_real,
            ],
            dtype=np.complex64,
        )

    tile_prefetcher = (
        OneAheadTilePrefetcher(
            positions=(
                active_prefetch_positions
            ),
            loader=load_real_tile,
            enabled=prefetch_enabled,
        )
    )

    tile_prefetcher.start()

    print(
        "tile prefetch          :",
        (
            "gamma one-ahead"
            if prefetch_enabled
            else "disabled"
        ),
    )

    for tile_index, (
        r0,
        r1,
        c0,
        c1,
    ) in enumerate(
        tiles,
        start=1,
    ):

        if tile_index <= checkpoint_prefix:

            print(
                f"stage {stage_index} "
                f"tile {tile_index:2d}/"
                f"{len(tiles):2d} "
                "CHECKPOINT RESUME",
                flush=True,
            )

            continue

        stage_progress.update(
            total_done,
            force=(tile_index == 1),
            detail=(
                f"tile={tile_index}/{len(tiles)} "
                f"support={support_seconds:.1f}s "
                f"cov={covariance_seconds:.1f}s "
                f"PL={phase_seconds:.1f}s "
                f"compress={compression_seconds:.1f}s"
            ),
        )

        core_sub = state_core[
            r0:r1,
            c0:c1,
        ]

        sr, sc = np.where(
            core_sub
        )

        if sr.size == 0:

            # FASTPATCH: an empty tile performs no numerical work, but it is
            # still a completed tile and MUST participate in the contiguous
            # checkpoint prefix.
            checkpoint_pending.append(
                (
                    tile_index,
                    (
                        r0,
                        r1,
                        c0,
                        c1,
                    ),
                    total_done,
                )
            )

            checkpoint_due = (
                len(
                    checkpoint_pending
                )
                >=
                checkpoint_every_tiles
                or
                tile_index
                ==
                len(
                    tiles
                )
            )

            if checkpoint_due:
                checkpoint_flush_t0 = (
                    perf_counter()
                )

                for checkpoint_arr in (
                    compressed_out,
                    state_valid_out,
                    state_code_out,
                    k_out,
                    tc_out,
                    est_out,
                ):
                    checkpoint_arr.flush()

                if (
                    phase_sink is not None
                    and
                    hasattr(
                        phase_sink,
                        "flush",
                    )
                ):
                    phase_sink.flush()

                for (
                    pending_tile,
                    pending_bounds,
                    pending_done,
                ) in checkpoint_pending:
                    _commit_stage_checkpoint(
                        root=checkpoint_root,
                        fingerprint_sha256=(
                            checkpoint_fp_hash
                        ),
                        tile_index=pending_tile,
                        bounds=pending_bounds,
                        total_done=pending_done,
                    )

                checkpoint_flush_seconds = (
                    perf_counter()
                    -
                    checkpoint_flush_t0
                )

                print(
                    f"stage {stage_index} "
                    "checkpoint flush "
                    f"tiles "
                    f"{checkpoint_pending[0][0]}-"
                    f"{checkpoint_pending[-1][0]} "
                    f"{checkpoint_flush_seconds:.2f}s",
                    flush=True,
                )

                checkpoint_pending.clear()

            print(
                f"stage {stage_index} "
                f"tile {tile_index:2d}/"
                f"{len(tiles):2d} "
                "EMPTY CHECKPOINT",
                flush=True,
            )

            continue

        gr = (
            sr
            +
            r0
        ).astype(
            np.int32,
            copy=False,
        )

        gc = (
            sc
            +
            c0
        ).astype(
            np.int32,
            copy=False,
        )

        # ----------------------------------------------------
        # Tile + exact spatial halo.
        # ----------------------------------------------------

        ir0 = max(
            0,
            r0 - half_row,
        )

        ir1 = min(
            H,
            r1 + half_row,
        )

        ic0 = max(
            0,
            c0 - half_col,
        )

        ic1 = min(
            W,
            c1 + half_col,
        )

        th = ir1 - ir0
        tw = ic1 - ic0

        # ----------------------------------------------------
        # Assemble stage stack in RAM:
        #
        # [old compressed states | current real SLCs]
        # ----------------------------------------------------

        stage_tile = np.empty(
            (
                th,
                tw,
                stage_n,
            ),
            dtype=np.complex64,
        )

        for j, comp in enumerate(
            compressed_inputs
        ):
            stage_tile[
                :,
                :,
                j,
            ] = comp[
                ir0:ir1,
                ic0:ic1,
            ]

        if prefetch_enabled:

            real_tile = (
                tile_prefetcher.get(
                    tile_index - 1
                )
            )

        else:

            real_tile = (
                np.ascontiguousarray(
                    yxt[
                        ir0:ir1,
                        ic0:ic1,
                        start_real:stop_real,
                    ],
                    dtype=np.complex64,
                )
            )

        expected_real_shape = (
            th,
            tw,
            len(
                real_indices
            ),
        )

        if (
            real_tile.shape
            !=
            expected_real_shape
        ):
            raise RuntimeError(
                "prefetched real-tile shape mismatch: "
                f"{real_tile.shape} != "
                f"{expected_real_shape}"
            )

        stage_tile[
            :,
            :,
            first_real_idx:
        ] = real_tile

        # A sample may participate in this stage covariance
        # only if every stage layer is finite at that sample.
        stage_sample_valid = np.all(
            np.isfinite(
                stage_tile.real
            )
            &
            np.isfinite(
                stage_tile.imag
            ),
            axis=2,
        )

        scale_tile = np.ascontiguousarray(
            scale2[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.float32,
        )

        valid_tile = np.ascontiguousarray(
            valid[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        ps_tile = np.ascontiguousarray(
            ps[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        state_tile = np.ascontiguousarray(
            state_core[
                ir0:ir1,
                ic0:ic1,
            ],
            dtype=np.bool_,
        )

        ctx = prepare_glrt_window_context(
            scale_tile,
            valid_tile,
            ps_tile,
            half_row=half_row,
            half_col=half_col,
        )

        state_windows = _bool_windows(
            state_tile,
            half_row=half_row,
            half_col=half_col,
        )

        stage_valid_windows = _bool_windows(
            stage_sample_valid,
            half_row=half_row,
            half_col=half_col,
        )

        lr = (
            gr - ir0
        ).astype(
            np.int32,
            copy=False,
        )

        lc = (
            gc - ic0
        ).astype(
            np.int32,
            copy=False,
        )

        for b0 in range(
            0,
            gr.size,
            center_batch,
        ):

            b1 = min(
                gr.size,
                b0 + center_batch,
            )

            br = lr[b0:b1]
            bc = lc[b0:b1]

            bgr = gr[b0:b1]
            bgc = gc[b0:b1]

            # ------------------------------------------------
            # Frozen 38-date GLRT.
            # ------------------------------------------------

            ts = perf_counter()

            if static_support_cache is None:

                support, _ = (
                    glrt_support_vectorized_exact(
                        ctx,
                        br,
                        bc,
                        alpha=alpha,
                        nslc=full_glrt_nslc,
                        block_size=support_block,
                    )
                )

            else:

                # ------------------------------------------------
                # Exact full-stack GLRT support was generated once
                # by glrt_support_vectorized_exact() and losslessly
                # packed to uint64.
                #
                # bgr/bgc are GLOBAL center coordinates.
                # ------------------------------------------------

                support = (
                    static_support_cache.support(
                        bgr,
                        bgc,
                    )
                )

            # Static K24 core.
            support &= np.asarray(
                state_windows[
                    br,
                    bc,
                ],
                dtype=np.bool_,
            )

            # Dynamic sequential-state availability.
            support &= np.asarray(
                stage_valid_windows[
                    br,
                    bc,
                ],
                dtype=np.bool_,
            )

            K = np.sum(
                support,
                axis=(1, 2),
                dtype=np.int32,
            ).astype(
                np.int16
            )

            support_seconds += (
                perf_counter()
                -
                ts
            )

            k_out[
                bgr,
                bgc,
            ] = K

            expected = expected_effective_k[
                bgr,
                bgc,
            ]

            # Dynamic filtering is only allowed to REDUCE K.
            excess = (
                K > expected
            )

            if np.any(
                excess
            ):
                static_k_excess += int(
                    np.count_nonzero(
                        excess
                    )
                )

                bad = int(
                    np.flatnonzero(
                        excess
                    )[0]
                )

                raise RuntimeError(
                    "Dynamic K exceeds frozen K24 K at "
                    f"({int(bgr[bad])},"
                    f"{int(bgc[bad])}): "
                    f"dynamic={int(K[bad])}, "
                    f"static={int(expected[bad])}"
                )

            # If all input compressed states are complete,
            # the dynamic support MUST equal frozen K24.
            if inputs_complete:

                mismatch = (
                    K != expected
                )

                if np.any(
                    mismatch
                ):
                    static_k_mismatch += int(
                        np.count_nonzero(
                            mismatch
                        )
                    )

                    bad = int(
                        np.flatnonzero(
                            mismatch
                        )[0]
                    )

                    raise RuntimeError(
                        "Dense-input K parity failure at "
                        f"({int(bgr[bad])},"
                        f"{int(bgc[bad])}): "
                        f"dynamic={int(K[bad])}, "
                        f"static={int(expected[bad])}"
                    )

            center_input_ok = stage_sample_valid[
                br,
                bc,
            ]

            center_bad = (
                ~center_input_ok
            )

            if np.any(
                center_bad
            ):
                state_code_out[
                    bgr[center_bad],
                    bgc[center_bad],
                ] = (
                    STATE_CENTER_INPUT_INVALID
                )

                center_invalid_n += int(
                    np.count_nonzero(
                        center_bad
                    )
                )

            good_k = (
                center_input_ok
                &
                (
                    K >= state_min_shp
                )
            )

            low_k = (
                center_input_ok
                &
                ~good_k
            )

            if np.any(
                low_k
            ):
                state_code_out[
                    bgr[low_k],
                    bgc[low_k],
                ] = (
                    STATE_LOW_K
                )

                low_k_n += int(
                    np.count_nonzero(
                        low_k
                    )
                )

            if not np.any(
                good_k
            ):
                total_done += int(
                    bgr.size
                )
                continue

            lr2 = br[
                good_k
            ]

            lc2 = bc[
                good_k
            ]

            gr2 = bgr[
                good_k
            ]

            gc2 = bgc[
                good_k
            ]

            support2 = support[
                good_k
            ]

            # ------------------------------------------------
            # SCM/coherence.
            # ------------------------------------------------

            ts = perf_counter()

            coh = compressed_coherence(
                stage_tile,
                lr2,
                lc2,
                support2,
                pi,
                pj,
            )

            covariance_seconds += (
                perf_counter()
                -
                ts
            )

            # ------------------------------------------------
            # EMI.
            #
            # ALWAYS_FIRST:
            # latest compressed input is the stage reference.
            # ------------------------------------------------

            ts = perf_counter()

            (
                ph,
                est,
                _,
                _,
                _,
            ) = emi_solver(
                coh,
                n_images=stage_n,
                pairs=pairs,
                beta=beta,
                gamma_jitter=gamma_jitter,
                emi_mu=emi_mu,
                reference_idx=reference_idx,
                workers=pl_workers,
                chunk_size=pl_chunk_size,
            )

            tc = temporal_coherence_fused(
                coh,
                ph,
                pairs,
            )

            phase_seconds += (
                perf_counter()
                -
                ts
            )

            est_out[
                gr2,
                gc2,
            ] = est

            tc_out[
                gr2,
                gc2,
            ] = tc

            phase_finite = np.all(
                np.isfinite(
                    ph.real
                )
                &
                np.isfinite(
                    ph.imag
                ),
                axis=1,
            )

            pl_ok = (
                (est != ESTIMATOR_INVALID)
                &
                np.isfinite(
                    tc
                )
                &
                phase_finite
            )

            if np.any(
                ~pl_ok
            ):

                bad_r = gr2[
                    ~pl_ok
                ]

                bad_c = gc2[
                    ~pl_ok
                ]

                state_code_out[
                    bad_r,
                    bad_c,
                ] = (
                    STATE_PL_INVALID
                )

                pl_invalid_n += int(
                    bad_r.size
                )

            if not np.any(
                pl_ok
            ):
                total_done += int(
                    bgr.size
                )
                continue

            lr3 = lr2[
                pl_ok
            ]

            lc3 = lc2[
                pl_ok
            ]

            gr3 = gr2[
                pl_ok
            ]

            gc3 = gc2[
                pl_ok
            ]

            ph3 = ph[
                pl_ok
            ]

            z3 = stage_tile[
                lr3,
                lc3,
                :,
            ]

            # ------------------------------------------------
            # Emit real-acquisition phase from this SAME PL
            # solution before immediate compression.
            #
            # No second phase-linking pass is performed.
            # ------------------------------------------------

            if phase_sink is not None:

                real_phase = reference_real_phase(
                    ph3,
                    first_real_idx=first_real_idx,
                    reference_idx=reference_idx,
                )

                if real_phase.shape != (
                    gr3.size,
                    len(real_indices),
                ):
                    raise RuntimeError(
                        "phase sink shape invariant failed: "
                        f"{real_phase.shape} != "
                        f"{(gr3.size, len(real_indices))}"
                    )

                phase_sink(
                    stage_index=stage_index,
                    real_indices=real_indices,
                    rows=gr3,
                    cols=gc3,
                    phase=real_phase,
                )

            # ------------------------------------------------
            # Immediate compression.
            # ------------------------------------------------

            ts = perf_counter()

            comp = compress_stage_slcs(
                z3,
                ph3,
                first_real_idx=first_real_idx,
                compressed_reference_idx=reference_idx,
                mean_amplitude=None,
            )

            compression_seconds += (
                perf_counter()
                -
                ts
            )

            comp_ok = (
                np.isfinite(
                    comp.real
                )
                &
                np.isfinite(
                    comp.imag
                )
            )

            if np.any(
                ~comp_ok
            ):

                bad_r = gr3[
                    ~comp_ok
                ]

                bad_c = gc3[
                    ~comp_ok
                ]

                state_code_out[
                    bad_r,
                    bad_c,
                ] = (
                    STATE_COMPRESSION_INVALID
                )

                comp_invalid_n += int(
                    bad_r.size
                )

            if np.any(
                comp_ok
            ):

                wr = gr3[
                    comp_ok
                ]

                wc = gc3[
                    comp_ok
                ]

                compressed_out[
                    wr,
                    wc,
                ] = comp[
                    comp_ok
                ]

                state_valid_out[
                    wr,
                    wc,
                ] = True

                state_code_out[
                    wr,
                    wc,
                ] = (
                    STATE_VALID
                )

                total_valid += int(
                    wr.size
                )

            # ------------------------------------------------
            # Independent compression parity audit.
            # ------------------------------------------------

            remaining = (
                formula_audit_points
                -
                formula_checked
            )

            if (
                remaining > 0
                and
                np.any(
                    comp_ok
                )
            ):

                ids = np.flatnonzero(
                    comp_ok
                )[
                    :remaining
                ]

                manual = _manual_stage_compression(
                    z3[
                        ids
                    ],
                    ph3[
                        ids
                    ],
                    first_real_idx=first_real_idx,
                    reference_idx=reference_idx,
                )

                delta = np.abs(
                    manual
                    -
                    comp[
                        ids
                    ]
                )

                delta = delta[
                    np.isfinite(
                        delta
                    )
                ]

                if delta.size:

                    formula_max_diff = max(
                        formula_max_diff,
                        float(
                            np.max(
                                delta
                            )
                        ),
                    )

                formula_checked += int(
                    ids.size
                )

            total_done += int(
                bgr.size
            )

        elapsed = (
            perf_counter()
            -
            t_all
        )

        # FASTPATCH: historical checkpoint work must not be divided by
        # only the elapsed time of this restarted process.
        run_done = max(
            0,
            int(
                total_done
                -
                checkpoint_resume_done
            ),
        )

        rate = (
            run_done
            /
            elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"stage {stage_index} "
            f"tile {tile_index:2d}/"
            f"{len(tiles):2d} "
            f"state="
            f"{total_done:,}/"
            f"{total_state:,} "
            f"valid="
            f"{total_valid:,} "
            f"rate="
            f"{rate:,.0f} center/s"
        )

        # --------------------------------------------------------------
        # production GROUPED CHECKPOINT COMMIT
        #
        # Add this completed tile to the in-memory pending group.
        # No durability marker is published before the backing maps
        # themselves have been flushed.
        # --------------------------------------------------------------

        checkpoint_pending.append(
            (
                tile_index,

                (
                    r0,
                    r1,
                    c0,
                    c1,
                ),

                total_done,
            )
        )


        checkpoint_due = (
            len(
                checkpoint_pending
            )
            >=
            checkpoint_every_tiles

            or

            tile_index
            ==
            len(
                tiles
            )
        )


        if checkpoint_due:

            checkpoint_flush_t0 = (
                perf_counter()
            )


            # ----------------------------------------------------------
            # 1. Make stage products durable.
            # ----------------------------------------------------------

            for checkpoint_arr in (
                compressed_out,
                state_valid_out,
                state_code_out,
                k_out,
                tc_out,
                est_out,
            ):

                checkpoint_arr.flush()


            # ----------------------------------------------------------
            # 2. Make linked phase durable.
            # ----------------------------------------------------------

            if (
                phase_sink is not None
                and
                hasattr(
                    phase_sink,
                    "flush",
                )
            ):

                phase_sink.flush()


            # ----------------------------------------------------------
            # 3. Only NOW publish atomic completion markers.
            #
            # If the process crashes while markers are being written,
            # the next run sees only the contiguous marker prefix and
            # conservatively recomputes the remainder.
            # ----------------------------------------------------------

            for (
                pending_tile,
                pending_bounds,
                pending_done,
            ) in checkpoint_pending:

                _commit_stage_checkpoint(
                    root=checkpoint_root,

                    fingerprint_sha256=(
                        checkpoint_fp_hash
                    ),

                    tile_index=(
                        pending_tile
                    ),

                    bounds=(
                        pending_bounds
                    ),

                    total_done=(
                        pending_done
                    ),
                )


            checkpoint_flush_seconds = (
                perf_counter()
                -
                checkpoint_flush_t0
            )


            print(
                f"stage {stage_index} "
                "checkpoint flush "
                f"tiles "
                f"{checkpoint_pending[0][0]}-"
                f"{checkpoint_pending[-1][0]} "
                f"{checkpoint_flush_seconds:.2f}s",
                flush=True,
            )


            checkpoint_pending.clear()


    tile_prefetcher.close()

    if prefetch_enabled:

        print(
            "prefetch reads         :",
            tile_prefetcher.completed,
        )

        print(
            "prefetch read seconds  :",
            f"{tile_prefetcher.read_seconds:.3f}",
        )

        print(
            "prefetch wait seconds  :",
            f"{tile_prefetcher.wait_seconds:.3f}",
        )

        print(
            "prefetch block seconds :",
            f"{tile_prefetcher.blocking_seconds:.3f}",
        )

        print(
            "prefetch sched overhead:",
            f"{tile_prefetcher.scheduler_overhead_seconds:.3f}",
        )

        print(
            "prefetch overlap sec   :",
            f"{tile_prefetcher.overlap_seconds:.3f}",
        )

        if (
            tile_prefetcher.read_seconds
            >
            0.0
        ):
            hidden_fraction = (
                tile_prefetcher.overlap_seconds
                /
                tile_prefetcher.read_seconds
            )

            print(
                "prefetch hidden I/O   :",
                f"{100.0 * hidden_fraction:.1f}%",
            )

    # ------------------------------------------------------------------
    # Rebuild status counts from persistent stage maps.
    #
    # This makes StageResult diagnostics correct for both:
    #   fresh execution
    #   resumed execution
    # ------------------------------------------------------------------

    state_codes_final = np.asarray(
        state_code_out[
            state_core
        ],
        dtype=np.uint8,
    )


    total_done = total_state


    total_valid = int(
        np.count_nonzero(
            state_codes_final
            ==
            STATE_VALID
        )
    )


    low_k_n = int(
        np.count_nonzero(
            state_codes_final
            ==
            STATE_LOW_K
        )
    )


    pl_invalid_n = int(
        np.count_nonzero(
            state_codes_final
            ==
            STATE_PL_INVALID
        )
    )


    comp_invalid_n = int(
        np.count_nonzero(
            state_codes_final
            ==
            STATE_COMPRESSION_INVALID
        )
    )


    center_invalid_n = int(
        np.count_nonzero(
            state_codes_final
            ==
            STATE_CENTER_INPUT_INVALID
        )
    )


    # Flush once per completed stage.
    for arr in (
        compressed_out,
        state_valid_out,
        state_code_out,
        k_out,
        tc_out,
        est_out,
    ):
        arr.flush()

    elapsed = (
        perf_counter()
        -
        t_all
    )

    stage_progress.finish(
        total_done,
        detail=(
            f"valid={total_valid:,} "
            f"support={support_seconds:.1f}s "
            f"cov={covariance_seconds:.1f}s "
            f"PL={phase_seconds:.1f}s "
            f"compress={compression_seconds:.1f}s"
        ),
    )

    return StageResult(
        stage_index=stage_index,

        real_indices=real_indices,
        compressed_input_ids=(
            compressed_input_ids
        ),

        solver_size=stage_n,
        first_real_idx=first_real_idx,
        reference_idx=reference_idx,

        state_pixels=total_state,
        state_valid=total_valid,

        low_k=low_k_n,
        pl_invalid=pl_invalid_n,
        compression_invalid=comp_invalid_n,
        center_input_invalid=center_invalid_n,

        static_k_excess=static_k_excess,
        static_k_mismatch=static_k_mismatch,

        compression_formula_max_abs_diff=(
            formula_max_diff
        ),

        support_seconds=support_seconds,
        covariance_seconds=(
            covariance_seconds
        ),
        phase_linking_seconds=(
            phase_seconds
        ),
        compression_seconds=(
            compression_seconds
        ),
        elapsed_seconds=elapsed,

        compressed_path=compressed_path,
        valid_path=valid_path,
        state_code_path=(
            state_code_path
        ),
        shp_count_path=(
            shp_count_path
        ),
        temporal_coherence_path=(
            tc_path
        ),
        estimator_path=(
            estimator_path
        ),
    )


__all__ = [
    "StageResult",
    "run_sequential_stage",
    "reference_real_phase",
]
