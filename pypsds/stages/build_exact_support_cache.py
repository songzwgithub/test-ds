#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import time

import numpy as np

from pypsds.context import (
    open_from_config,
)

from pypsds.progress import (
    ProgressReporter,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)

from pypsds.phase_linking.support_cache import (
    pack_support_bool,
    support_geometry,
)


FORMAT = (
    "pyPSDS-GAMMA-exact-static-support-cache-v1"
)


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            x = f.read(
                8 * 1024 * 1024
            )

            if not x:
                break

            h.update(
                x
            )

    return h.hexdigest()


def make_tiles(
    H,
    W,
    tile_rows,
    tile_cols,
):

    out = []

    for r0 in range(
        0,
        H,
        tile_rows,
    ):

        r1 = min(
            H,
            r0
            +
            tile_rows,
        )

        for c0 in range(
            0,
            W,
            tile_cols,
        ):

            c1 = min(
                W,
                c0
                +
                tile_cols,
            )

            out.append(
                (
                    r0,
                    r1,
                    c0,
                    c1,
                )
            )

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--half-row",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--half-col",
        type=int,
        default=11,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.005,
    )

    ap.add_argument(
        "--tile-rows",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--tile-cols",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=32000,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--resume",
        action="store_true",
    )

    args = ap.parse_args()


    (
        cfg,
        config_path,
        paths,
        stack,
        (_, _, H, W),
    ) = open_from_config(
        args.config
    )

    ndate = len(
        stack.dates
    )

    processing = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    stats = (
        processing
        /
        "ds_statistics"
    )

    outdir = (
        processing
        /
        "exact_support_cache"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    scale_path = (
        stats
        /
        "rayleigh_scale2.npy"
    )

    raw_valid_path = (
        stats
        /
        "raw_valid.npy"
    )

    ps_path = (
        stats
        /
        "ps_mask.npy"
    )

    geom_path = (
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy"
    )


    scale2 = np.load(
        scale_path,
        mmap_mode="r",
    )

    raw_valid = np.load(
        raw_valid_path,
        mmap_mode="r",
    )

    ps_raw = np.load(
        ps_path,
        mmap_mode="r",
    )

    geom = np.load(
        geom_path,
        mmap_mode="r",
    )


    valid = (
        np.asarray(
            raw_valid
        )
        &
        np.asarray(
            geom
        )
    )

    ps = (
        np.asarray(
            ps_raw
        )
        &
        valid
    )

    center = (
        valid
        &
        ~ps
    )


    (
        wh,
        ww,
        nwin,
        nbytes,
        nwords,
    ) = support_geometry(
        half_row=args.half_row,
        half_col=args.half_col,
    )


    tiles = make_tiles(
        H,
        W,
        args.tile_rows,
        args.tile_cols,
    )


    ntr = (
        H
        +
        args.tile_rows
        -
        1
    ) // args.tile_rows

    ntc = (
        W
        +
        args.tile_cols
        -
        1
    ) // args.tile_cols


    fingerprint = {
        "format":
            FORMAT,

        "scene":
            [
                int(H),
                int(W),
            ],

        "ndate":
            int(ndate),

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

        "window":
            [
                int(wh),
                int(ww),
            ],

        "nwin":
            int(nwin),

        "nwords":
            int(nwords),

        "scale2_sha256":
            sha256_file(
                scale_path
            ),

        "raw_valid_sha256":
            sha256_file(
                raw_valid_path
            ),

        "ps_sha256":
            sha256_file(
                ps_path
            ),

        "geometry_sha256":
            sha256_file(
                geom_path
            ),
    }


    manifest_path = (
        outdir
        /
        "manifest.json"
    )

    bits_path = (
        outdir
        /
        "static_support_bits.npy"
    )

    k_path = (
        outdir
        /
        "static_shp_count.npy"
    )

    done_path = (
        outdir
        /
        "tile_done.npy"
    )


    # ------------------------------------------------------------------
    # Resume contract.
    # ------------------------------------------------------------------

    existing_ok = False

    if (
        args.resume
        and
        manifest_path.is_file()
        and
        bits_path.is_file()
        and
        k_path.is_file()
        and
        done_path.is_file()
    ):

        old = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        for key, value in (
            fingerprint.items()
        ):

            if old.get(
                key
            ) != value:

                raise RuntimeError(
                    "support-cache fingerprint mismatch: "
                    f"{key}: "
                    f"old={old.get(key)!r}, "
                    f"new={value!r}"
                )

        existing_ok = True


    if existing_ok:

        bits_map = np.load(
            bits_path,
            mmap_mode="r+",
        )

        k_map = np.load(
            k_path,
            mmap_mode="r+",
        )

        tile_done = np.load(
            done_path,
            mmap_mode="r+",
        )

        if bits_map.shape != (
            H,
            W,
            nwords,
        ):
            raise RuntimeError(
                "support bits shape mismatch"
            )

        if k_map.shape != (
            H,
            W,
        ):
            raise RuntimeError(
                "support K shape mismatch"
            )

        if tile_done.shape != (
            ntr,
            ntc,
        ):
            raise RuntimeError(
                "tile_done shape mismatch"
            )

    else:

        bits_map = (
            np.lib.format.open_memmap(
                bits_path,
                mode="w+",
                dtype=np.uint64,
                shape=(
                    H,
                    W,
                    nwords,
                ),
            )
        )

        bits_map[...] = 0

        k_map = (
            np.lib.format.open_memmap(
                k_path,
                mode="w+",
                dtype=np.int16,
                shape=(
                    H,
                    W,
                ),
            )
        )

        k_map[...] = -1

        tile_done = (
            np.lib.format.open_memmap(
                done_path,
                mode="w+",
                dtype=np.bool_,
                shape=(
                    ntr,
                    ntc,
                ),
            )
        )

        tile_done[...] = False

        manifest_path.write_text(
            json.dumps(
                fingerprint,
                indent=2,
            )
            +
            "\n",
            encoding="utf-8",
        )


    total_centers = int(
        np.count_nonzero(
            center
        )
    )


    # Count already-completed centers without constructing
    # a global center coordinate list.
    already_done = 0

    tile_id = 0

    for tr in range(
        ntr
    ):

        r0 = (
            tr
            *
            args.tile_rows
        )

        r1 = min(
            H,
            r0
            +
            args.tile_rows,
        )

        for tc in range(
            ntc
        ):

            c0 = (
                tc
                *
                args.tile_cols
            )

            c1 = min(
                W,
                c0
                +
                args.tile_cols,
            )

            if tile_done[
                tr,
                tc,
            ]:

                already_done += int(
                    np.count_nonzero(
                        center[
                            r0:r1,
                            c0:c1,
                        ]
                    )
                )


    run_stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    progress = ProgressReporter(
        label="exact-support-cache",
        total=total_centers,
        unit="center",
        min_interval=5.0,
        log_path=(
            outdir
            /
            f"build_progress_{run_stamp}.jsonl"
        ),
    )


    done_centers = (
        already_done
    )

    if done_centers:

        progress.update(
            done_centers,
            force=True,
            detail="resume",
        )


    print()
    print("=" * 92)
    print(
        "pyPSDS-GAMMA exact static support cache"
    )
    print("=" * 92)

    print(
        "scene          :",
        f"{H} x {W}",
    )

    print(
        "dates          :",
        ndate,
    )

    print(
        "GLRT           :",
        f"{wh} x {ww}, "
        f"alpha={args.alpha}",
    )

    print(
        "centers        :",
        f"{total_centers:,}",
    )

    print(
        "cache words    :",
        nwords,
    )

    print(
        "bytes/center   :",
        nwords * 8,
    )

    print(
        "tile           :",
        f"{args.tile_rows} x "
        f"{args.tile_cols}",
    )

    print(
        "tiles          :",
        f"{ntr} x {ntc} "
        f"= {ntr*ntc}",
    )

    print(
        "center batch   :",
        args.batch,
    )

    print(
        "resume         :",
        args.resume,
    )

    print()


    t_all = perf_counter()

    tile_number = 0

    for tr in range(
        ntr
    ):

        r0 = (
            tr
            *
            args.tile_rows
        )

        r1 = min(
            H,
            r0
            +
            args.tile_rows,
        )

        for tc in range(
            ntc
        ):

            tile_number += 1

            c0 = (
                tc
                *
                args.tile_cols
            )

            c1 = min(
                W,
                c0
                +
                args.tile_cols,
            )

            if tile_done[
                tr,
                tc,
            ]:

                print(
                    f"tile "
                    f"{tile_number:4d}/"
                    f"{ntr*ntc:4d} "
                    f"r={r0}:{r1} "
                    f"c={c0}:{c1} "
                    f"RESUME",
                    flush=True,
                )

                continue


            sub = center[
                r0:r1,
                c0:c1,
            ]

            sr, sc = np.where(
                sub
            )


            if sr.size == 0:

                tile_done[
                    tr,
                    tc,
                ] = True

                tile_done.flush()

                continue


            # ----------------------------------------------------------
            # Exact spatial halo.  RAM remains proportional to tile,
            # not full-scene size.
            # ----------------------------------------------------------

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

            ic0 = max(
                0,
                c0
                -
                args.half_col,
            )

            ic1 = min(
                W,
                c1
                +
                args.half_col,
            )


            scale_tile = (
                np.ascontiguousarray(
                    scale2[
                        ir0:ir1,
                        ic0:ic1,
                    ],
                    dtype=np.float32,
                )
            )

            valid_tile = (
                np.ascontiguousarray(
                    valid[
                        ir0:ir1,
                        ic0:ic1,
                    ],
                    dtype=np.bool_,
                )
            )

            ps_tile = (
                np.ascontiguousarray(
                    ps[
                        ir0:ir1,
                        ic0:ic1,
                    ],
                    dtype=np.bool_,
                )
            )


            ctx = (
                prepare_glrt_window_context(
                    scale_tile,
                    valid_tile,
                    ps_tile,
                    half_row=(
                        args.half_row
                    ),
                    half_col=(
                        args.half_col
                    ),
                )
            )


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


            lr = (
                gr
                -
                ir0
            ).astype(
                np.int32,
                copy=False,
            )

            lc = (
                gc
                -
                ic0
            ).astype(
                np.int32,
                copy=False,
            )


            for b0 in range(
                0,
                gr.size,
                args.batch,
            ):

                b1 = min(
                    gr.size,
                    b0
                    +
                    args.batch,
                )

                support, K = (
                    glrt_support_vectorized_exact(
                        ctx,
                        lr[
                            b0:b1
                        ],
                        lc[
                            b0:b1
                        ],
                        alpha=args.alpha,
                        nslc=ndate,
                        block_size=(
                            args.support_block
                        ),
                    )
                )


                bits = pack_support_bool(
                    support
                )


                bits_map[
                    gr[b0:b1],
                    gc[b0:b1],
                    :,
                ] = bits


                k_map[
                    gr[b0:b1],
                    gc[b0:b1],
                ] = K


                done_centers += (
                    b1 - b0
                )


                progress.update(
                    done_centers,
                    detail=(
                        f"tile="
                        f"{tile_number}/"
                        f"{ntr*ntc}"
                    ),
                )


            bits_map.flush()
            k_map.flush()

            tile_done[
                tr,
                tc,
            ] = True

            tile_done.flush()


            print(
                f"tile "
                f"{tile_number:4d}/"
                f"{ntr*ntc:4d} "
                f"centers="
                f"{gr.size:,} "
                f"DONE",
                flush=True,
            )


    elapsed = (
        perf_counter()
        -
        t_all
    )


    progress.finish(
        done_centers,
        detail="complete",
    )


    if not np.all(
        tile_done
    ):
        raise RuntimeError(
            "support cache incomplete"
        )


    # ------------------------------------------------------------------
    # Exact original-K parity against validated K cache.
    # ------------------------------------------------------------------

    original_k_path = (
        processing
        /
        "sequential"
        /
        "compression_all_valid_nonps_shp_count.npy"
    )

    parity_bad = None

    if original_k_path.is_file():

        original_k = np.load(
            original_k_path,
            mmap_mode="r",
        )

        parity_bad = 0

        for r0 in range(
            0,
            H,
            args.tile_rows,
        ):

            r1 = min(
                H,
                r0
                +
                args.tile_rows,
            )

            m = center[
                r0:r1,
                :
            ]

            bad = (
                np.asarray(
                    k_map[
                        r0:r1,
                        :
                    ]
                )[m]
                !=
                np.asarray(
                    original_k[
                        r0:r1,
                        :
                    ]
                )[m]
            )

            parity_bad += int(
                np.count_nonzero(
                    bad
                )
            )

        if parity_bad != 0:
            raise RuntimeError(
                "exact support cache original-K "
                f"parity failed: {parity_bad}"
            )


    final_manifest = {
        **fingerprint,

        "complete":
            True,

        "center_count":
            total_centers,

        "tile_rows":
            args.tile_rows,

        "tile_cols":
            args.tile_cols,

        "batch":
            args.batch,

        "build_seconds_this_run":
            elapsed,

        "original_k_parity_bad":
            parity_bad,

        "support_path":
            str(
                bits_path
            ),

        "k_path":
            str(
                k_path
            ),
    }


    manifest_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 92)
    print("EXACT SUPPORT CACHE RESULT")
    print("=" * 92)

    print(
        "center count       :",
        f"{total_centers:,}",
    )

    print(
        "cache shape        :",
        (
            H,
            W,
            nwords,
        ),
    )

    print(
        "cache disk         :",
        f"{bits_path.stat().st_size/1024**2:.2f} MiB",
    )

    print(
        "K disk             :",
        f"{k_path.stat().st_size/1024**2:.2f} MiB",
    )

    print(
        "elapsed this run   :",
        f"{elapsed:.3f} s",
    )

    print(
        "original-K mismatch:",
        parity_bad,
    )

    print()
    print(
        "EXACT STATIC SUPPORT CACHE: PASS"
    )


if __name__ == "__main__":
    main()
