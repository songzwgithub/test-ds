from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def support_geometry(
    *,
    half_row: int,
    half_col: int,
):
    wh = (
        2 * int(half_row)
        +
        1
    )

    ww = (
        2 * int(half_col)
        +
        1
    )

    nwin = (
        wh
        *
        ww
    )

    nbytes = (
        nwin
        +
        7
    ) // 8

    nwords = (
        nwin
        +
        63
    ) // 64

    return (
        wh,
        ww,
        nwin,
        nbytes,
        nwords,
    )


def pack_support_bool(
    support: np.ndarray,
) -> np.ndarray:
    """
    Losslessly pack [B,wh,ww] bool support into uint64.

    Bit convention matches shp_coherence_bitset.py:

        flat = ky * ww + kx
        word = flat >> 6
        bit  = flat & 63

    No GLRT is evaluated here.
    """

    x = np.asarray(
        support,
        dtype=np.bool_,
    )

    if x.ndim != 3:
        raise ValueError(
            "support must have shape [B,wh,ww]"
        )

    B = x.shape[0]

    flat = np.ascontiguousarray(
        x.reshape(
            B,
            -1,
        ),
        dtype=np.uint8,
    )

    packed = np.packbits(
        flat,
        axis=1,
        bitorder="little",
    )

    nbytes = packed.shape[1]

    nwords = (
        nbytes
        +
        7
    ) // 8

    out = np.zeros(
        (
            B,
            nwords,
        ),
        dtype=np.uint64,
    )

    # Explicit byte -> uint64 assembly avoids relying on
    # platform endianness.
    for byte_idx in range(
        nbytes
    ):
        word = (
            byte_idx
            //
            8
        )

        shift = (
            byte_idx
            %
            8
        ) * 8

        out[
            :,
            word,
        ] |= (
            packed[
                :,
                byte_idx,
            ].astype(
                np.uint64
            )
            <<
            np.uint64(
                shift
            )
        )

    return out


def unpack_support_bits(
    bits: np.ndarray,
    *,
    half_row: int,
    half_col: int,
) -> np.ndarray:
    """
    Exact inverse of pack_support_bool().
    """

    bits = np.asarray(
        bits,
        dtype=np.uint64,
    )

    if bits.ndim != 2:
        raise ValueError(
            "bits must have shape [B,nwords]"
        )

    (
        wh,
        ww,
        nwin,
        nbytes,
        nwords,
    ) = support_geometry(
        half_row=half_row,
        half_col=half_col,
    )

    if bits.shape[1] != nwords:
        raise ValueError(
            f"nwords={bits.shape[1]} "
            f"expected={nwords}"
        )

    B = bits.shape[0]

    packed = np.empty(
        (
            B,
            nbytes,
        ),
        dtype=np.uint8,
    )

    for byte_idx in range(
        nbytes
    ):
        word = (
            byte_idx
            //
            8
        )

        shift = (
            byte_idx
            %
            8
        ) * 8

        packed[
            :,
            byte_idx,
        ] = (
            (
                bits[
                    :,
                    word,
                ]
                >>
                np.uint64(
                    shift
                )
            )
            &
            np.uint64(
                0xFF
            )
        ).astype(
            np.uint8
        )

    flat = np.unpackbits(
        packed,
        axis=1,
        bitorder="little",
    )[
        :,
        :nwin
    ]

    return (
        flat.reshape(
            B,
            wh,
            ww,
        )
        .astype(
            np.bool_,
            copy=False,
        )
    )


def popcount_support_bits(
    bits: np.ndarray,
) -> np.ndarray:
    """
    Return number of set SHP bits per center.
    """

    bits = np.asarray(
        bits,
        dtype=np.uint64,
    )

    # Vectorized SWAR popcount for uint64.
    x = bits.copy()

    m1 = np.uint64(
        0x5555555555555555
    )

    m2 = np.uint64(
        0x3333333333333333
    )

    m4 = np.uint64(
        0x0F0F0F0F0F0F0F0F
    )

    h01 = np.uint64(
        0x0101010101010101
    )

    x = (
        x
        -
        (
            (
                x
                >>
                np.uint64(1)
            )
            &
            m1
        )
    )

    x = (
        (
            x
            &
            m2
        )
        +
        (
            (
                x
                >>
                np.uint64(2)
            )
            &
            m2
        )
    )

    x = (
        x
        +
        (
            x
            >>
            np.uint64(4)
        )
    ) & m4

    x = (
        x
        *
        h01
    ) >> np.uint64(56)

    return (
        np.sum(
            x,
            axis=1,
            dtype=np.uint64,
        )
        .astype(
            np.int16
        )
    )


def bool_windows(
    mask: np.ndarray,
    *,
    half_row: int,
    half_col: int,
):
    """
    Read-only spatial window view matching production semantics.
    """

    x = np.asarray(
        mask,
        dtype=np.bool_,
    )

    padded = np.pad(
        x,
        (
            (
                half_row,
                half_row,
            ),
            (
                half_col,
                half_col,
            ),
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



# ============================================================================
# Production runtime cache
# ============================================================================

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
                8
                *
                1024
                *
                1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


@dataclass(slots=True)
class ExactSupportCache:
    """
    Read-only exact static SHP support.

    The scientific GLRT is evaluated only during cache construction
    by glrt_support_vectorized_exact().

    Runtime use is lossless unpacking only.
    """

    bits: np.ndarray
    static_k: np.ndarray

    half_row: int
    half_col: int

    ndate: int
    alpha: float

    manifest: dict
    directory: Path

    def support(
        self,
        rows,
        cols,
    ) -> np.ndarray:

        rows = np.asarray(
            rows,
            dtype=np.int32,
        )

        cols = np.asarray(
            cols,
            dtype=np.int32,
        )

        if rows.shape != cols.shape:
            raise ValueError(
                "support-cache row/col shape mismatch"
            )

        packed = np.asarray(
            self.bits[
                rows,
                cols,
                :,
            ],
            dtype=np.uint64,
        )

        return unpack_support_bits(
            packed,
            half_row=self.half_row,
            half_col=self.half_col,
        )


def load_exact_support_cache(
    *,
    processing_dir,
    H: int,
    W: int,
    ndate: int,
    half_row: int,
    half_col: int,
    alpha: float,
    validate_input_hashes: bool = True,
) -> ExactSupportCache:
    """
    Strict production loader.

    Cache reuse is rejected if:
      - scientific GLRT parameters differ;
      - scene or date count differs;
      - cache is incomplete;
      - original-K audit failed;
      - support inputs changed.
    """

    processing = Path(
        processing_dir
    )

    directory = (
        processing
        /
        "exact_support_cache"
    )

    manifest_path = (
        directory
        /
        "manifest.json"
    )

    bits_path = (
        directory
        /
        "static_support_bits.npy"
    )

    k_path = (
        directory
        /
        "static_shp_count.npy"
    )

    for p in (
        manifest_path,
        bits_path,
        k_path,
    ):

        if not p.is_file():
            raise FileNotFoundError(
                "exact support cache missing: "
                f"{p}"
            )


    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )


    if manifest.get(
        "format"
    ) != (
        "pyPSDS-GAMMA-exact-static-support-cache-v1"
    ):
        raise RuntimeError(
            "unsupported exact-support cache format"
        )


    expected_scene = [
        int(H),
        int(W),
    ]


    if manifest.get(
        "scene"
    ) != expected_scene:

        raise RuntimeError(
            "exact-support scene mismatch: "
            f"{manifest.get('scene')} "
            f"!= {expected_scene}"
        )


    if int(
        manifest.get(
            "ndate",
            -1,
        )
    ) != int(
        ndate
    ):
        raise RuntimeError(
            "exact-support ndate mismatch"
        )


    if int(
        manifest.get(
            "half_row",
            -1,
        )
    ) != int(
        half_row
    ):
        raise RuntimeError(
            "exact-support half_row mismatch"
        )


    if int(
        manifest.get(
            "half_col",
            -1,
        )
    ) != int(
        half_col
    ):
        raise RuntimeError(
            "exact-support half_col mismatch"
        )


    if not np.isclose(
        float(
            manifest.get(
                "alpha",
                np.nan,
            )
        ),
        float(
            alpha
        ),
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            "exact-support alpha mismatch"
        )


    if not bool(
        manifest.get(
            "complete",
            False,
        )
    ):
        raise RuntimeError(
            "exact-support cache is incomplete"
        )


    parity_bad = manifest.get(
        "original_k_parity_bad"
    )

    if (
        parity_bad is not None
        and
        int(parity_bad) != 0
    ):
        raise RuntimeError(
            "exact-support original-K audit failed"
        )


    (
        _,
        _,
        _,
        _,
        nwords,
    ) = support_geometry(
        half_row=half_row,
        half_col=half_col,
    )


    bits = np.load(
        bits_path,
        mmap_mode="r",
    )

    static_k = np.load(
        k_path,
        mmap_mode="r",
    )


    if bits.shape != (
        H,
        W,
        nwords,
    ):

        raise RuntimeError(
            "exact-support bits shape mismatch: "
            f"{bits.shape}"
        )


    if bits.dtype != np.uint64:

        raise RuntimeError(
            "exact-support bits dtype mismatch"
        )


    if static_k.shape != (
        H,
        W,
    ):

        raise RuntimeError(
            "exact-support K shape mismatch"
        )


    if static_k.dtype != np.int16:

        raise RuntimeError(
            "exact-support K dtype mismatch"
        )


    # ------------------------------------------------------------------
    # Strong standalone-run protection.
    #
    # This is done once per phase-linking execution, not per batch.
    # ------------------------------------------------------------------

    if validate_input_hashes:

        expected_inputs = {
            "scale2_sha256":
                processing
                /
                "ds_statistics"
                /
                "rayleigh_scale2.npy",

            "raw_valid_sha256":
                processing
                /
                "ds_statistics"
                /
                "raw_valid.npy",

            "ps_sha256":
                processing
                /
                "ds_statistics"
                /
                "ps_mask.npy",

            "geometry_sha256":
                processing
                /
                "cache"
                /
                "phase_geometry_valid.npy",
        }


        for key, path in (
            expected_inputs.items()
        ):

            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            actual = _sha256_file(
                path
            )

            expected = manifest.get(
                key
            )

            if actual != expected:

                raise RuntimeError(
                    "exact-support cache input changed: "
                    f"{key}"
                )


    return ExactSupportCache(
        bits=bits,
        static_k=static_k,

        half_row=int(
            half_row
        ),

        half_col=int(
            half_col
        ),

        ndate=int(
            ndate
        ),

        alpha=float(
            alpha
        ),

        manifest=manifest,
        directory=directory,
    )


__all__ = [
    "support_geometry",
    "pack_support_bool",
    "unpack_support_bits",
    "popcount_support_bits",
    "bool_windows",
    "ExactSupportCache",
    "load_exact_support_cache",
]
