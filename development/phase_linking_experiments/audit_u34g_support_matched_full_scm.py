#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pypsds.context import open_from_config

from pypsds.phase_linking.coherence import compressed_coherence

from pypsds.phase_linking.emi import (
    ESTIMATOR_EMI,
    ESTIMATOR_EVD,
    ESTIMATOR_INVALID,
    image_pairs,
    robust_emi_threaded,
    temporal_coherence,
)

from pypsds.phase_linking.shp_vectorized_exact import (
    glrt_support_vectorized_exact,
    prepare_glrt_window_context,
)


def bool_windows(x, hr, hc):
    x = np.asarray(x, dtype=np.bool_)

    p = np.pad(
        x,
        ((hr, hr), (hc, hc)),
        mode="constant",
        constant_values=False,
    )

    return np.lib.stride_tricks.sliding_window_view(
        p,
        (2 * hr + 1, 2 * hc + 1),
    )


def unit_ref(ph):
    ph = np.asarray(ph)

    out = np.exp(
        1j * np.angle(ph)
    ).astype(np.complex64)

    out *= np.conj(
        out[:, 0]
    )[:, None]

    return out


def phase_difference(a, b):
    a = unit_ref(a)
    b = unit_ref(b)

    delta = (
        a[:, 1:]
        *
        np.conj(
            b[:, 1:]
        )
    )

    similarity = np.abs(
        np.mean(
            delta,
            axis=1,
        )
    ).astype(np.float32)

    deg = (
        np.abs(
            np.angle(delta)
        )
        *
        180.0
        /
        np.pi
    ).astype(np.float32)

    return (
        similarity,
        np.median(
            deg,
            axis=1,
        ).astype(np.float32),
        np.percentile(
            deg,
            95,
            axis=1,
        ).astype(np.float32),
    )


def qs(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    p = np.percentile(
        x,
        [5, 25, 50, 75, 95, 99],
    )

    return {
        "p05": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
    }


def summarize(
    mask,
    *,
    K_original,
    K_core,
    tc_original,
    tc_original_on_core,
    tc_core38,
    tc_seq,
    sim_core_seq,
    med_core_seq,
    p95_core_seq,
    sim_core_orig,
    med_core_orig,
    p95_core_orig,
    original_estimator,
    core_estimator,
):
    mask = np.asarray(mask, dtype=np.bool_)

    if not np.any(mask):
        return {"n": 0}

    loss = (
        K_original[mask].astype(np.int32)
        -
        K_core[mask].astype(np.int32)
    )

    scs = sim_core_seq[mask]
    sco = sim_core_orig[mask]

    return {
        "n":
            int(mask.sum()),

        "K_original":
            qs(
                K_original[mask]
            ),

        "K_core":
            qs(
                K_core[mask]
            ),

        "support_loss":
            qs(loss),

        "original_TC_original_support":
            qs(
                tc_original[mask]
            ),

        "original_phase_TC_on_core":
            qs(
                tc_original_on_core[mask]
            ),

        "core38_TC_on_core":
            qs(
                tc_core38[mask]
            ),

        "sequential_TC_on_core":
            qs(
                tc_seq[mask]
            ),

        "core38_vs_sequential": {
            "similarity":
                qs(scs),

            "median_error_deg":
                qs(
                    med_core_seq[mask]
                ),

            "p95_error_deg":
                qs(
                    p95_core_seq[mask]
                ),

            "similarity_ge_0p99_fraction":
                float(
                    np.mean(
                        scs >= 0.99
                    )
                ),
        },

        "core38_vs_original": {
            "similarity":
                qs(sco),

            "median_error_deg":
                qs(
                    med_core_orig[mask]
                ),

            "p95_error_deg":
                qs(
                    p95_core_orig[mask]
                ),

            "similarity_ge_0p99_fraction":
                float(
                    np.mean(
                        sco >= 0.99
                    )
                ),
        },

        "core38_closer_to_sequential_fraction":
            float(
                np.mean(
                    scs > sco
                )
            ),

        "original_estimator": {
            "EMI":
                int(
                    np.count_nonzero(
                        original_estimator[mask]
                        ==
                        ESTIMATOR_EMI
                    )
                ),

            "EVD":
                int(
                    np.count_nonzero(
                        original_estimator[mask]
                        ==
                        ESTIMATOR_EVD
                    )
                ),

            "invalid":
                int(
                    np.count_nonzero(
                        original_estimator[mask]
                        ==
                        ESTIMATOR_INVALID
                    )
                ),
        },

        "core38_estimator": {
            "EMI":
                int(
                    np.count_nonzero(
                        core_estimator[mask]
                        ==
                        ESTIMATOR_EMI
                    )
                ),

            "EVD":
                int(
                    np.count_nonzero(
                        core_estimator[mask]
                        ==
                        ESTIMATOR_EVD
                    )
                ),

            "invalid":
                int(
                    np.count_nonzero(
                        core_estimator[mask]
                        ==
                        ESTIMATOR_INVALID
                    )
                ),
        },
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=2048,
    )

    ap.add_argument(
        "--support-block",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--pl-workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--pl-chunk",
        type=int,
        default=256,
    )

    args = ap.parse_args()

    (
        cfg,
        config_path,
        paths,
        stack,
        (row0, col0, H, W),
    ) = open_from_config(
        args.config
    )

    processing = (
        Path(paths.output_dir)
        /
        "processing"
    )

    seqdir = (
        processing
        /
        "sequential"
    )

    # --------------------------------------------------------
    # Verify current sequential phase is M16.
    # --------------------------------------------------------

    phase_report = json.loads(
        (
            seqdir
            /
            "u34a_phase_parity_report.json"
        ).read_text()
    )

    if phase_report["ministack_size"] != 16:
        raise RuntimeError(
            "Current U3.4a phase is not M16"
        )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    yxt = np.load(
        processing
        /
        "cache"
        /
        "phase_corrected_yxt.npy",
        mmap_mode="r",
    )

    scale2 = np.load(
        processing
        /
        "ds_statistics"
        /
        "rayleigh_scale2.npy",
        mmap_mode="r",
    )

    raw_valid = np.load(
        processing
        /
        "ds_statistics"
        /
        "raw_valid.npy",
        mmap_mode="r",
    )

    geom = np.load(
        processing
        /
        "cache"
        /
        "phase_geometry_valid.npy",
        mmap_mode="r",
    )

    ps = np.load(
        processing
        /
        "ps_mask.npy",
        mmap_mode="r",
    )

    core = np.load(
        seqdir
        /
        "compression_state_core_K24.npy",
        mmap_mode="r",
    )

    original_phase = np.load(
        processing
        /
        "linked_phase.npy",
        mmap_mode="r",
    )

    original_tc_map = np.load(
        processing
        /
        "temporal_coherence.npy",
        mmap_mode="r",
    )

    original_K_map = np.load(
        processing
        /
        "shp_count.npy",
        mmap_mode="r",
    )

    original_estimator_map = np.load(
        processing
        /
        "estimator_code.npy",
        mmap_mode="r",
    )

    seq_phase = np.load(
        seqdir
        /
        "u34a_sequential_phase_points.npy",
        mmap_mode="r",
    )

    rr = np.load(
        seqdir
        /
        "u34a_rows.npy"
    )

    cc = np.load(
        seqdir
        /
        "u34a_cols.npy"
    )

    K_core_all = np.load(
        seqdir
        /
        "u34a_effective_K.npy"
    )

    ze = np.load(
        seqdir
        /
        "u34e_fullspan_sequential_tc.npz"
    )

    metrics = np.load(
        seqdir
        /
        "u34c_M16_phase_metrics.npz"
    )

    if not (
        np.array_equal(
            rr,
            ze["rows"]
        )
        and
        np.array_equal(
            cc,
            ze["cols"]
        )
        and
        np.array_equal(
            rr,
            metrics["rows"]
        )
        and
        np.array_equal(
            cc,
            metrics["cols"]
        )
    ):
        raise RuntimeError(
            "Point ordering mismatch"
        )

    full_accept = ze[
        "full_accept"
    ].astype(bool)

    seq_accept = ze[
        "sequential_accept"
    ].astype(bool)

    p95_old = metrics[
        "p95_abs_error_deg"
    ]

    false_accept = (
        seq_accept
        &
        ~full_accept
    )

    catastrophic = (
        false_accept
        &
        (
            p95_old > 90.0
        )
    )

    stable_pool = (
        seq_accept
        &
        full_accept
        &
        (
            p95_old <= 30.0
        )
    )

    bad_ids = np.flatnonzero(
        false_accept
    )

    stable_ids_all = np.flatnonzero(
        stable_pool
    )

    if stable_ids_all.size < bad_ids.size:
        raise RuntimeError(
            "Not enough stable controls"
        )

    # Deterministic evenly distributed control sample.
    sample_pos = np.linspace(
        0,
        stable_ids_all.size - 1,
        bad_ids.size,
        dtype=np.int64,
    )

    control_ids = stable_ids_all[
        sample_pos
    ]

    audit_ids = np.concatenate(
        [
            bad_ids,
            control_ids,
        ]
    )

    is_false_accept = np.zeros(
        audit_ids.size,
        dtype=np.bool_,
    )

    is_false_accept[
        :bad_ids.size
    ] = True

    is_control = ~is_false_accept

    catastrophic_selected = (
        is_false_accept
        &
        (
            p95_old[
                audit_ids
            ]
            >
            90.0
        )
    )

    ar = rr[
        audit_ids
    ]

    ac = cc[
        audit_ids
    ]

    ndate = yxt.shape[2]

    print("=" * 125)
    print(
        "U3.4g SUPPORT-MATCHED 38-DATE FULL-SCM CONTROL"
    )
    print("=" * 125)

    print(
        "false accepts          :",
        f"{bad_ids.size:,}",
    )

    print(
        "  catastrophic >90 deg :",
        f"{np.count_nonzero(catastrophic_selected):,}",
    )

    print(
        "stable controls        :",
        f"{control_ids.size:,}",
    )

    print(
        "audit total            :",
        f"{audit_ids.size:,}",
    )

    print()

    # --------------------------------------------------------
    # Exact core-filtered covariance.
    # --------------------------------------------------------

    valid = (
        np.asarray(
            raw_valid,
            dtype=np.bool_,
        )
        &
        np.asarray(
            geom,
            dtype=np.bool_,
        )
    )

    ps_bool = np.asarray(
        ps,
        dtype=np.bool_,
    )

    core_bool = np.asarray(
        core,
        dtype=np.bool_,
    )

    ctx = prepare_glrt_window_context(
        scale2,
        valid,
        ps_bool,
        half_row=5,
        half_col=11,
    )

    core_windows = bool_windows(
        core_bool,
        5,
        11,
    )

    pairs = image_pairs(
        ndate
    )

    pi = np.asarray(
        pairs[:, 0],
        dtype=np.int32,
    )

    pj = np.asarray(
        pairs[:, 1],
        dtype=np.int32,
    )

    na = audit_ids.size

    core_phase = np.full(
        (na, ndate),
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )

    core_tc = np.full(
        na,
        np.nan,
        dtype=np.float32,
    )

    seq_tc_core = np.full(
        na,
        np.nan,
        dtype=np.float32,
    )

    orig_tc_core = np.full(
        na,
        np.nan,
        dtype=np.float32,
    )

    core_estimator = np.full(
        na,
        ESTIMATOR_INVALID,
        dtype=np.uint8,
    )

    K_core = np.full(
        na,
        -1,
        dtype=np.int16,
    )

    t0 = perf_counter()

    for b0 in range(
        0,
        na,
        args.batch,
    ):

        b1 = min(
            na,
            b0 + args.batch,
        )

        br = ar[
            b0:b1
        ]

        bc = ac[
            b0:b1
        ]

        support, _ = (
            glrt_support_vectorized_exact(
                ctx,
                br,
                bc,
                alpha=0.005,
                nslc=ndate,
                block_size=args.support_block,
            )
        )

        support &= np.asarray(
            core_windows[
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

        expected = K_core_all[
            audit_ids[
                b0:b1
            ]
        ]

        if not np.array_equal(
            K,
            expected,
        ):
            raise RuntimeError(
                "K-core parity mismatch"
            )

        K_core[
            b0:b1
        ] = K

        coh = compressed_coherence(
            yxt,
            br,
            bc,
            support,
            pi,
            pj,
        )

        (
            ph,
            est,
            _,
            _,
            _,
        ) = robust_emi_threaded(
            coh,
            n_images=ndate,
            pairs=pairs,
            beta=0.05,
            gamma_jitter=1.0e-6,
            emi_mu=0.99,
            reference_idx=0,
            workers=args.pl_workers,
            chunk_size=args.pl_chunk,
        )

        if np.any(
            est
            ==
            ESTIMATOR_INVALID
        ):
            raise RuntimeError(
                "core-matched full-SCM estimator invalid"
            )

        core_phase[
            b0:b1
        ] = ph

        core_estimator[
            b0:b1
        ] = est

        core_tc[
            b0:b1
        ] = temporal_coherence(
            coh,
            ph,
            pairs,
        )

        seq_ph = np.asarray(
            seq_phase[
                audit_ids[
                    b0:b1
                ]
            ]
        )

        seq_tc_core[
            b0:b1
        ] = temporal_coherence(
            coh,
            seq_ph,
            pairs,
        )

        orig_ph = (
            original_phase[
                :,
                br,
                bc,
            ]
            .T
        )

        orig_tc_core[
            b0:b1
        ] = temporal_coherence(
            coh,
            orig_ph,
            pairs,
        )

        elapsed = (
            perf_counter()
            -
            t0
        )

        print(
            f"{b1:7,d}/{na:7,d} "
            f"({100*b1/na:6.2f}%) "
            f"rate={b1/elapsed:,.0f} center/s"
        )

    # --------------------------------------------------------
    # Computational parity with U3.4e TC.
    # --------------------------------------------------------

    seq_tc_saved = ze[
        "sequential_fullspan_tc"
    ][
        audit_ids
    ]

    max_tc_diff = float(
        np.max(
            np.abs(
                seq_tc_core
                -
                seq_tc_saved
            )
        )
    )

    if max_tc_diff > 1e-5:
        raise RuntimeError(
            f"Sequential TC parity failure: {max_tc_diff}"
        )

    # --------------------------------------------------------
    # Candidate phase comparisons.
    # --------------------------------------------------------

    orig_ph_all = (
        original_phase[
            :,
            ar,
            ac,
        ]
        .T
    )

    seq_ph_all = np.asarray(
        seq_phase[
            audit_ids
        ]
    )

    (
        sim_core_seq,
        med_core_seq,
        p95_core_seq,
    ) = phase_difference(
        core_phase,
        seq_ph_all,
    )

    (
        sim_core_orig,
        med_core_orig,
        p95_core_orig,
    ) = phase_difference(
        core_phase,
        orig_ph_all,
    )

    K_original = np.asarray(
        original_K_map[
            ar,
            ac,
        ],
        dtype=np.int16,
    )

    original_tc = np.asarray(
        original_tc_map[
            ar,
            ac,
        ],
        dtype=np.float32,
    )

    original_estimator = np.asarray(
        original_estimator_map[
            ar,
            ac,
        ],
        dtype=np.uint8,
    )

    groups = {
        "false_accept_all":
            is_false_accept,

        "false_accept_catastrophic_gt90":
            catastrophic_selected,

        "stable_control":
            is_control,
    }

    summaries = {
        name:
            summarize(
                mask,

                K_original=K_original,
                K_core=K_core,

                tc_original=original_tc,
                tc_original_on_core=orig_tc_core,
                tc_core38=core_tc,
                tc_seq=seq_tc_core,

                sim_core_seq=sim_core_seq,
                med_core_seq=med_core_seq,
                p95_core_seq=p95_core_seq,

                sim_core_orig=sim_core_orig,
                med_core_orig=med_core_orig,
                p95_core_orig=p95_core_orig,

                original_estimator=original_estimator,
                core_estimator=core_estimator,
            )

        for name, mask
        in groups.items()
    }

    print()
    print("=" * 125)
    print(
        "SUPPORT-MATCHED RESULT"
    )
    print("=" * 125)

    print(
        "group                         n      "
        "Kloss50 coreTC50 seqTC50 origOnCore50 "
        "core-v-seq sim50 med50 p9550 "
        "core-v-orig sim50 med50 p9550 "
        "closerSeq%"
    )

    print("-" * 125)

    for name in groups:

        x = summaries[
            name
        ]

        if not x.get(
            "n",
            0
        ):
            continue

        print(
            f"{name:<29s} "
            f"{x['n']:7,d} "
            f"{x['support_loss']['median']:8.1f} "
            f"{x['core38_TC_on_core']['median']:8.4f} "
            f"{x['sequential_TC_on_core']['median']:7.4f} "
            f"{x['original_phase_TC_on_core']['median']:12.4f} "
            f"{x['core38_vs_sequential']['similarity']['median']:9.4f} "
            f"{x['core38_vs_sequential']['median_error_deg']['median']:6.2f} "
            f"{x['core38_vs_sequential']['p95_error_deg']['median']:6.2f} "
            f"{x['core38_vs_original']['similarity']['median']:10.4f} "
            f"{x['core38_vs_original']['median_error_deg']['median']:6.2f} "
            f"{x['core38_vs_original']['p95_error_deg']['median']:6.2f} "
            f"{100*x['core38_closer_to_sequential_fraction']:9.3f}%"
        )

    print()

    print(
        "sequential TC recomputation max diff:",
        max_tc_diff,
    )

    # --------------------------------------------------------
    # Automatic interpretation: diagnostic only.
    # --------------------------------------------------------

    cat = summaries[
        "false_accept_catastrophic_gt90"
    ]

    seq_sim = (
        cat[
            "core38_vs_sequential"
        ][
            "similarity"
        ][
            "median"
        ]
    )

    orig_sim = (
        cat[
            "core38_vs_original"
        ][
            "similarity"
        ][
            "median"
        ]
    )

    if (
        seq_sim >= 0.99
        and
        seq_sim > orig_sim + 0.05
    ):

        decision = (
            "support_change_dominates_"
            "original_full_scm_disagreement"
        )

    elif (
        orig_sim >= 0.99
        and
        orig_sim > seq_sim + 0.05
    ):

        decision = (
            "sequential_temporal_propagation_"
            "dominates_disagreement"
        )

    else:

        decision = (
            "mixed_or_solver_branch_ambiguity"
        )

    report = {
        "format":
            "pyPSDS-GAMMA-U3.4g-support-matched-full-SCM-v1",

        "ministack_size":
            16,

        "false_accept_count":
            int(
                bad_ids.size
            ),

        "stable_control_count":
            int(
                control_ids.size
            ),

        "seq_tc_recompute_max_abs_diff":
            max_tc_diff,

        "groups":
            summaries,

        "decision":
            decision,
    }

    out_json = (
        seqdir
        /
        "u34g_support_matched_full_scm.json"
    )

    out_npz = (
        seqdir
        /
        "u34g_support_matched_full_scm.npz"
    )

    np.savez_compressed(
        out_npz,

        audit_ids=audit_ids,

        rows=ar,
        cols=ac,

        is_false_accept=is_false_accept,

        is_catastrophic=catastrophic_selected,

        K_original=K_original,
        K_core=K_core,

        original_tc=original_tc,

        original_tc_on_core=orig_tc_core,

        core38_tc=core_tc,

        sequential_tc_on_core=seq_tc_core,

        core38_vs_seq_similarity=sim_core_seq,

        core38_vs_orig_similarity=sim_core_orig,

        core38_vs_seq_median_error_deg=med_core_seq,

        core38_vs_orig_median_error_deg=med_core_orig,
    )

    out_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "decision:",
        decision,
    )

    print(
        "json    :",
        out_json,
    )

    print(
        "npz     :",
        out_npz,
    )

    print()

    print(
        "U3.4g SUPPORT-MATCHED CONTROL: PASS"
    )


if __name__ == "__main__":
    main()
