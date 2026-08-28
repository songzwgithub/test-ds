from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config
from pypsds.phase_linking.coherence import (
    compressed_coherence,
)
from pypsds.phase_linking.emi import (
    image_pairs,
)
from pypsds.phase_linking.phase_source import (
    GammaStreamingPhaseSource,
)
from pypsds.phase_linking.reliability_qa import (
    connected_support_count,
    crlb_median_std_from_compressed,
    deterministic_sample_positions,
    dolphin_style_num_looks,
    finite_quantiles,
    nearest_triplet_closure_metrics,
    sampled_phase_similarity,
    write_json,
)
from pypsds.phase_linking.shp_policy import (
    resolve_shp_policy,
)
from pypsds.phase_linking.support_cache import (
    load_exact_support_cache,
)
from pypsds.runtime import (
    build_runtime_plan,
)


def _new_map(
    path,
    *,
    shape,
    dtype,
    fill,
):
    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    arr[...] = fill
    arr.flush()
    return arr


def _sample_formal_points(
    formal,
    pl_valid,
    ps,
    *,
    max_points,
):
    candidate = (
        np.asarray(
            formal,
            dtype=np.bool_,
        )
        &
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

    flat = np.flatnonzero(
        candidate
    )

    pos = deterministic_sample_positions(
        flat.size,
        max_points,
    )

    selected = flat[
        pos
    ]

    W = candidate.shape[1]

    rows = (
        selected
        //
        W
    ).astype(
        np.int32,
    )

    cols = (
        selected
        %
        W
    ).astype(
        np.int32,
    )

    return candidate, rows, cols


def _coherence_qa_sample(
    *,
    source,
    support_cache,
    rows,
    cols,
    H,
    W,
    ndate,
    half_row,
    half_col,
    pairs,
    batch_cells=True,
):
    """
    Compute sampled coherence QA grouped by canonical phase-source cell.

    Each group is read with an SHP halo, so exact cached support can be
    evaluated without changing the production GLRT definition.
    """
    n = rows.size

    closure_rms = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    closure_med = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    closure_max = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    crlb = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    if n == 0:
        return (
            closure_rms,
            closure_med,
            closure_max,
            crlb,
        )

    cr = int(
        source.canonical_rows
    )

    cc = int(
        source.canonical_cols
    )

    key_r = (
        rows
        //
        cr
    )

    key_c = (
        cols
        //
        cc
    )

    keys = np.stack(
        [
            key_r,
            key_c,
        ],
        axis=1,
    )

    unique_keys = np.unique(
        keys,
        axis=0,
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    num_looks = dolphin_style_num_looks(
        half_row,
        half_col,
    )

    for group_index, (kr, kc) in enumerate(
        unique_keys,
        start=1,
    ):
        ids = np.flatnonzero(
            (key_r == kr)
            &
            (key_c == kc)
        )

        gr = rows[
            ids
        ]

        gc = cols[
            ids
        ]

        cell_r0 = int(
            kr
            *
            cr
        )

        cell_c0 = int(
            kc
            *
            cc
        )

        r0 = max(
            0,
            cell_r0
            -
            half_row,
        )

        r1 = min(
            H,
            cell_r0
            +
            cr
            +
            half_row,
        )

        c0 = max(
            0,
            cell_c0
            -
            half_col,
        )

        c1 = min(
            W,
            cell_c0
            +
            cc
            +
            half_col,
        )

        tile = source.read_tile(
            local_row0=r0,
            local_row1=r1,
            local_col0=c0,
            local_col1=c1,
        )

        support = support_cache.support(
            gr,
            gc,
        )

        lr = (
            gr
            -
            r0
        ).astype(
            np.int32,
            copy=False,
        )

        lc = (
            gc
            -
            c0
        ).astype(
            np.int32,
            copy=False,
        )

        coh = compressed_coherence(
            tile.yxt,
            lr,
            lc,
            support,
            pi,
            pj,
        )

        (
            rms_b,
            med_b,
            max_b,
        ) = nearest_triplet_closure_metrics(
            coh,
            pairs,
            ndate,
        )

        crlb_b = crlb_median_std_from_compressed(
            coh,
            pairs,
            ndate,
            num_looks=num_looks,
            reference_idx=0,
            gamma_jitter=1.0e-6,
            fim_jitter=1.0e-6,
        )

        closure_rms[
            ids
        ] = rms_b

        closure_med[
            ids
        ] = med_b

        closure_max[
            ids
        ] = max_b

        crlb[
            ids
        ] = crlb_b

        print(
            f"[coherence-QA] group "
            f"{group_index}/{unique_keys.shape[0]} "
            f"cell=({int(kr)},{int(kc)}) "
            f"points={ids.size}"
        )

    return (
        closure_rms,
        closure_med,
        closure_max,
        crlb,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--coherence-sample",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--similarity-sample",
        type=int,
        default=8192,
    )

    ap.add_argument(
        "--similarity-radius",
        type=int,
        default=7,
    )

    ap.add_argument(
        "--similarity-nearest-n",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--connected-row-block",
        type=int,
        default=32,
    )

    args = ap.parse_args()

    t0 = perf_counter()

    (
        cfg,
        config_path,
        paths,
        stack,
        (
            row0,
            col0,
            H,
            W,
        ),
    ) = open_from_config(
        args.config
    )

    ndate = len(
        stack.dates
    )

    policy = resolve_shp_policy(
        cfg,
        stack.dates,
    )

    alpha = float(
        cfg.get(
            "selection",
            {},
        )
        .get(
            "shp",
            {},
        )
        .get(
            "alpha",
            0.005,
        )
    )

    outdir = (
        Path(
            paths.output_dir
        )
        /
        "processing"
    )

    qadir = (
        outdir
        /
        "reliability_qa"
    )

    qadir.mkdir(
        parents=True,
        exist_ok=True,
    )

    formal = np.load(
        outdir
        /
        "sequential"
        /
        "production_formal_ds_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    pl_valid = np.load(
        outdir
        /
        "pl_valid.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    ps = np.load(
        outdir
        /
        "ps_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    shp_count = np.load(
        outdir
        /
        "shp_count.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    phase = np.load(
        outdir
        /
        "linked_phase.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    support_cache = load_exact_support_cache(
        processing_dir=outdir,
        H=H,
        W=W,
        ndate=ndate,
        half_row=(
            policy.half_row
        ),
        half_col=(
            policy.half_col
        ),
        alpha=alpha,
        validate_input_hashes=True,
    )

    print("=" * 88)
    print("P9B/C reliability QA")
    print("=" * 88)
    print("config             :", config_path)
    print("scene              :", f"{H} x {W}")
    print("dates              :", ndate)
    print(
        "SHP window         :",
        f"{2*policy.half_row+1} x "
        f"{2*policy.half_col+1}",
    )
    print(
        "formal DS          :",
        f"{np.count_nonzero(formal):,}",
    )

    # -----------------------------------------------------------------
    # Full-scene connected SHP count. This is diagnostic only.
    # -----------------------------------------------------------------
    connected = _new_map(
        qadir
        /
        "connected_raw_shp_count.npy",
        shape=(H, W),
        dtype=np.int16,
        fill=-1,
    )

    for r0 in range(
        0,
        H,
        int(
            args.connected_row_block
        ),
    ):
        r1 = min(
            H,
            r0
            +
            int(
                args.connected_row_block
            ),
        )

        lr, cc = np.where(
            np.asarray(
                formal[
                    r0:r1,
                    :
                ],
                dtype=np.bool_,
            )
        )

        if lr.size == 0:
            continue

        rr = (
            lr
            +
            r0
        ).astype(
            np.int32,
            copy=False,
        )

        cc = cc.astype(
            np.int32,
            copy=False,
        )

        support = support_cache.support(
            rr,
            cc,
        )

        connected[
            rr,
            cc,
        ] = connected_support_count(
            support
        )

        print(
            f"[connected-SHP] rows={r0}:{r1} "
            f"points={rr.size:,}"
        )

    connected.flush()

    # -----------------------------------------------------------------
    # Deterministic formal-DS sample for coherence-domain QA.
    # -----------------------------------------------------------------
    (
        candidate,
        cr,
        cc,
    ) = _sample_formal_points(
        formal,
        pl_valid,
        ps,
        max_points=(
            args.coherence_sample
        ),
    )

    runtime = build_runtime_plan(
        ndate=ndate,
        memory_fraction=float(
            cfg.get(
                "runtime",
                {},
            ).get(
                "memory_fraction",
                0.85,
            )
        ),
    )

    source = GammaStreamingPhaseSource(
        cfg=cfg,
        paths=paths,
        stack=stack,
        base_row0=row0,
        base_col0=col0,
        io_workers=(
            runtime.io_workers
        ),
    )

    pairs = image_pairs(
        ndate
    )

    (
        closure_rms,
        closure_med,
        closure_max,
        crlb,
    ) = _coherence_qa_sample(
        source=source,
        support_cache=support_cache,
        rows=cr,
        cols=cc,
        H=H,
        W=W,
        ndate=ndate,
        half_row=(
            policy.half_row
        ),
        half_col=(
            policy.half_col
        ),
        pairs=pairs,
    )

    for name, values in (
        (
            "pl_closure_rms_rad_sampled.npy",
            closure_rms,
        ),
        (
            "pl_closure_median_abs_rad_sampled.npy",
            closure_med,
        ),
        (
            "pl_closure_max_abs_rad_sampled.npy",
            closure_max,
        ),
        (
            "pl_crlb_median_std_rad_sampled.npy",
            crlb,
        ),
    ):
        arr = _new_map(
            qadir
            /
            name,
            shape=(H, W),
            dtype=np.float32,
            fill=np.nan,
        )

        arr[
            cr,
            cc,
        ] = values

        arr.flush()

    np.save(
        qadir
        /
        "coherence_sample_rows.npy",
        cr,
    )

    np.save(
        qadir
        /
        "coherence_sample_cols.npy",
        cc,
    )

    # -----------------------------------------------------------------
    # Dolphin-style phase similarity on deterministic formal-DS sample.
    # -----------------------------------------------------------------
    (
        sr,
        sc,
        sim_med,
        sim_max,
    ) = sampled_phase_similarity(
        phase,
        candidate,
        max_points=(
            args.similarity_sample
        ),
        search_radius=(
            args.similarity_radius
        ),
        nearest_n=(
            args.similarity_nearest_n
        ),
    )

    for name, values in (
        (
            "phase_similarity_median_sampled.npy",
            sim_med,
        ),
        (
            "phase_similarity_max_sampled.npy",
            sim_max,
        ),
    ):
        arr = _new_map(
            qadir
            /
            name,
            shape=(H, W),
            dtype=np.float32,
            fill=np.nan,
        )

        arr[
            sr,
            sc,
        ] = values

        arr.flush()

    np.save(
        qadir
        /
        "similarity_sample_rows.npy",
        sr,
    )

    np.save(
        qadir
        /
        "similarity_sample_cols.npy",
        sc,
    )

    # -----------------------------------------------------------------
    # Summary. No new gate is applied here.
    # -----------------------------------------------------------------
    m = np.asarray(
        formal,
        dtype=np.bool_,
    )

    raw_k = np.asarray(
        shp_count[
            m
        ],
        dtype=np.float64,
    )

    conn_k = np.asarray(
        connected[
            m
        ],
        dtype=np.float64,
    )

    ratio = np.full(
        raw_k.shape,
        np.nan,
        dtype=np.float32,
    )

    good = (
        raw_k
        >
        0
    )

    ratio[
        good
    ] = (
        conn_k[
            good
        ]
        /
        raw_k[
            good
        ]
    ).astype(
        np.float32
    )

    connected_survive = int(
        np.count_nonzero(
            conn_k
            >=
            int(
                policy.formal_min_shp
            )
        )
    )

    formal_count = int(
        np.count_nonzero(
            m
        )
    )

    tc = np.load(
        outdir
        /
        "temporal_coherence.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    payload = {
        "format":
            "pyPSDS-GAMMA-reliability-QA-v1",

        "production_gate_changed":
            False,

        "scene":
            [
                int(H),
                int(W),
            ],

        "ndate":
            int(
                ndate
            ),

        "formal_min_shp":
            int(
                policy.formal_min_shp
            ),

        "formal_ds_count":
            formal_count,

        "connected_formal_survival_count":
            connected_survive,

        "connected_formal_survival_fraction":
            float(
                connected_survive
                /
                formal_count
            ),

        "connected_raw_fraction":
            finite_quantiles(
                ratio
            ),

        "connected_raw_shp_count":
            finite_quantiles(
                connected[
                    m
                ]
            ),

        "coherence_sample_points":
            int(
                cr.size
            ),

        "closure_rms_rad_sampled":
            finite_quantiles(
                closure_rms
            ),

        "closure_median_abs_rad_sampled":
            finite_quantiles(
                closure_med
            ),

        "closure_max_abs_rad_sampled":
            finite_quantiles(
                closure_max
            ),

        "crlb_median_std_rad_sampled":
            finite_quantiles(
                crlb
            ),

        "similarity_sample_points":
            int(
                sr.size
            ),

        "phase_similarity_median_sampled":
            finite_quantiles(
                sim_med
            ),

        "phase_similarity_max_sampled":
            finite_quantiles(
                sim_max
            ),

        "temporal_coherence_formal":
            finite_quantiles(
                tc[
                    m
                ]
            ),

        "elapsed_seconds":
            float(
                perf_counter()
                -
                t0
            ),
    }

    write_json(
        qadir
        /
        "reliability_summary.json",
        payload,
    )

    print()
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "QA output:",
        qadir,
    )


if __name__ == "__main__":
    main()
