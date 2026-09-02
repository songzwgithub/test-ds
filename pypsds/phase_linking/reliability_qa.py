from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def connected_support_count(
    support: np.ndarray,
) -> np.ndarray:
    """
    Count SHPs in the 8-connected component attached to the center.

    The center is excluded from the SHP support itself, so the flood fill is
    seeded by accepted SHPs in the 8-neighborhood immediately around center.
    """
    B, wh, ww = support.shape
    cr = wh // 2
    cc = ww // 2

    out = np.zeros(
        B,
        dtype=np.int16,
    )

    for p in prange(B):
        visited = np.zeros(
            (wh, ww),
            dtype=np.uint8,
        )

        qr = np.empty(
            wh * ww,
            dtype=np.int16,
        )

        qc = np.empty(
            wh * ww,
            dtype=np.int16,
        )

        head = 0
        tail = 0

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue

                r = cr + dr
                c = cc + dc

                if (
                    r >= 0
                    and r < wh
                    and c >= 0
                    and c < ww
                    and support[p, r, c]
                    and visited[r, c] == 0
                ):
                    visited[r, c] = 1
                    qr[tail] = r
                    qc[tail] = c
                    tail += 1

        count = 0

        while head < tail:
            r0 = qr[head]
            c0 = qc[head]
            head += 1
            count += 1

            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue

                    r = r0 + dr
                    c = c0 + dc

                    if (
                        r < 0
                        or r >= wh
                        or c < 0
                        or c >= ww
                    ):
                        continue

                    if (
                        support[p, r, c]
                        and visited[r, c] == 0
                    ):
                        visited[r, c] = 1
                        qr[tail] = r
                        qc[tail] = c
                        tail += 1

        out[p] = count

    return out


def deterministic_sample_positions(
    n: int,
    max_points: int,
) -> np.ndarray:
    n = int(n)
    max_points = int(max_points)

    if n <= 0 or max_points <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    m = min(
        n,
        max_points,
    )

    if m == n:
        return np.arange(
            n,
            dtype=np.int64,
        )

    return np.unique(
        np.linspace(
            0,
            n - 1,
            num=m,
            dtype=np.int64,
        )
    )


def _pair_index(
    pairs: np.ndarray,
    n_images: int,
) -> np.ndarray:
    idx = np.full(
        (n_images, n_images),
        -1,
        dtype=np.int32,
    )

    for q, (i, j) in enumerate(
        np.asarray(
            pairs,
            dtype=np.int32,
        )
    ):
        idx[i, j] = q

    return idx


def nearest_triplet_closure_metrics(
    coherence: np.ndarray,
    pairs: np.ndarray,
    n_images: int,
):
    """
    Nearest-consecutive coherence closure before phase linking.

    closure(i,i+1,i+2) =
        angle(gamma_i,i+1 * gamma_i+1,i+2 * conj(gamma_i,i+2))
    """
    coh = np.asarray(
        coherence,
        dtype=np.complex64,
    )

    B = coh.shape[0]

    rms = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )

    med = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )

    mx = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )

    if B == 0 or n_images < 3:
        return rms, med, mx

    idx = _pair_index(
        pairs,
        n_images,
    )

    q01 = []
    q12 = []
    q02 = []

    for i in range(n_images - 2):
        a = int(idx[i, i + 1])
        b = int(idx[i + 1, i + 2])
        c = int(idx[i, i + 2])

        if min(a, b, c) >= 0:
            q01.append(a)
            q12.append(b)
            q02.append(c)

    if not q01:
        return rms, med, mx

    z = (
        coh[:, q01]
        *
        coh[:, q12]
        *
        np.conj(
            coh[:, q02]
        )
    )

    ang = np.angle(
        z
    ).astype(
        np.float32,
        copy=False,
    )

    for p in range(B):
        vals = ang[
            p,
            np.isfinite(
                ang[p]
            ),
        ]

        if vals.size == 0:
            continue

        av = np.abs(
            vals
        )

        rms[p] = np.float32(
            np.sqrt(
                np.mean(
                    vals.astype(
                        np.float64
                    )
                    ** 2
                )
            )
        )

        med[p] = np.float32(
            np.median(
                av
            )
        )

        mx[p] = np.float32(
            np.max(
                av
            )
        )

    return rms, med, mx


def dolphin_style_num_looks(
    half_row: int,
    half_col: int,
) -> float:
    """
    Conservative effective-look convention used in Dolphin CRLB processing.
    """
    return float(
        math.sqrt(
            max(
                1,
                int(half_row)
                *
                int(half_col),
            )
        )
    )


def crlb_median_std_from_compressed(
    coherence: np.ndarray,
    pairs: np.ndarray,
    n_images: int,
    *,
    num_looks: float,
    reference_idx: int = 0,
    gamma_jitter: float = 1.0e-6,
    fim_jitter: float = 1.0e-6,
) -> np.ndarray:
    """
    Median per-epoch CRLB standard deviation [rad].

    The Fisher-information construction mirrors the Tebaldini-style
    phase-linking CRLB implemented by Dolphin. This is a QA diagnostic.
    """
    coh = np.asarray(
        coherence,
        dtype=np.complex64,
    )

    pairs = np.asarray(
        pairs,
        dtype=np.int32,
    )

    B = int(
        coh.shape[0]
    )

    out = np.full(
        B,
        np.nan,
        dtype=np.float32,
    )

    eye = np.eye(
        n_images,
        dtype=np.float64,
    )

    keep = np.asarray(
        [
            i
            for i in range(n_images)
            if i != int(reference_idx)
        ],
        dtype=np.int32,
    )

    for p in range(B):
        G = np.eye(
            n_images,
            dtype=np.float64,
        )

        mag = np.abs(
            coh[p]
        ).astype(
            np.float64,
            copy=False,
        )

        for q, (i, j) in enumerate(
            pairs
        ):
            v = mag[q]

            if not np.isfinite(v):
                G[i, j] = np.nan
                G[j, i] = np.nan
            else:
                G[i, j] = v
                G[j, i] = v

        if not np.all(
            np.isfinite(
                G
            )
        ):
            continue

        try:
            Ginv = np.linalg.inv(
                G
                +
                float(gamma_jitter)
                *
                eye
            )

            X = (
                2.0
                *
                float(num_looks)
                *
                (
                    G
                    *
                    Ginv
                    -
                    eye
                )
            )

            F = X[
                np.ix_(
                    keep,
                    keep,
                )
            ]

            F = (
                F
                +
                float(fim_jitter)
                *
                np.eye(
                    n_images - 1,
                    dtype=np.float64,
                )
            )

            Sigma = np.linalg.inv(
                F
            )

            d = np.diag(
                Sigma
            )

            if (
                np.any(
                    ~np.isfinite(
                        d
                    )
                )
                or
                np.any(
                    d < 0
                )
            ):
                continue

            out[p] = np.float32(
                np.median(
                    np.sqrt(
                        d
                    )
                )
            )

        except np.linalg.LinAlgError:
            continue

    return out


def nearest_n_edges(
    n_images: int,
    nearest_n: int,
) -> np.ndarray:
    edges = []

    for i in range(
        int(n_images)
    ):
        for j in range(
            i + 1,
            min(
                int(n_images),
                i + int(nearest_n) + 1,
            ),
        ):
            edges.append(
                (
                    i,
                    j,
                )
            )

    return np.asarray(
        edges,
        dtype=np.int32,
    )


def disk_offsets(
    search_radius: int,
) -> np.ndarray:
    r = int(
        search_radius
    )

    if r < 2:
        raise ValueError(
            "search_radius must be >= 2"
        )

    offsets = []

    for dy in range(
        -(r - 1),
        r,
    ):
        for dx in range(
            -(r - 1),
            r,
        ):
            if dy == 0 and dx == 0:
                continue

            if (
                dy * dy
                +
                dx * dx
                <
                r * r
            ):
                offsets.append(
                    (
                        dy,
                        dx,
                    )
                )

    return np.asarray(
        sorted(
            offsets
        ),
        dtype=np.int16,
    )


def sampled_phase_similarity(
    phase_cube: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    max_points: int = 8192,
    search_radius: int = 7,
    nearest_n: int = 3,
    batch_size: int = 128,
):
    """
    Spatial phase similarity on a deterministic point sample.

    The similarity metric is the temporal mean of
        Re(z_center * conj(z_neighbor))
    after converting linked phases to nearest-N interferometric differences.
    """
    cube = np.asarray(
        phase_cube
    )

    mask = np.asarray(
        candidate_mask,
        dtype=np.bool_,
    )

    if cube.ndim != 3:
        raise ValueError(
            "phase_cube must be [date,row,col]"
        )

    n_images, H, W = cube.shape

    if mask.shape != (
        H,
        W,
    ):
        raise ValueError(
            "candidate mask shape mismatch"
        )

    flat = np.flatnonzero(
        mask
    )

    pos = deterministic_sample_positions(
        flat.size,
        max_points,
    )

    selected = flat[
        pos
    ]

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

    edges = nearest_n_edges(
        n_images,
        nearest_n,
    )

    offsets = disk_offsets(
        search_radius
    )

    median_out = np.full(
        rows.size,
        np.nan,
        dtype=np.float32,
    )

    max_out = np.full(
        rows.size,
        np.nan,
        dtype=np.float32,
    )

    for b0 in range(
        0,
        rows.size,
        int(batch_size),
    ):
        b1 = min(
            rows.size,
            b0 + int(batch_size),
        )

        br = rows[
            b0:b1
        ]

        bc = cols[
            b0:b1
        ]

        center = np.asarray(
            cube[
                :,
                br,
                bc,
            ].T,
            dtype=np.complex64,
        )

        complete_center = np.all(
            np.isfinite(
                center.real
            )
            &
            np.isfinite(
                center.imag
            )
            &
            (
                center
                !=
                np.complex64(0.0)
            ),
            axis=1,
        )

        ce = (
            center[
                :,
                edges[:, 1],
            ]
            *
            np.conj(
                center[
                    :,
                    edges[:, 0],
                ]
            )
        )

        sim = np.full(
            (
                br.size,
                offsets.shape[0],
            ),
            np.nan,
            dtype=np.float32,
        )

        for q, (dy, dx) in enumerate(
            offsets
        ):
            nr = (
                br
                +
                int(dy)
            )

            nc = (
                bc
                +
                int(dx)
            )

            inside = (
                (nr >= 0)
                &
                (nr < H)
                &
                (nc >= 0)
                &
                (nc < W)
                &
                complete_center
            )

            if not np.any(
                inside
            ):
                continue

            ids = np.flatnonzero(
                inside
            )

            neigh = np.asarray(
                cube[
                    :,
                    nr[ids],
                    nc[ids],
                ].T,
                dtype=np.complex64,
            )

            complete = np.all(
                np.isfinite(
                    neigh.real
                )
                &
                np.isfinite(
                    neigh.imag
                )
                &
                (
                    neigh
                    !=
                    np.complex64(0.0)
                ),
                axis=1,
            )

            if not np.any(
                complete
            ):
                continue

            ids2 = ids[
                complete
            ]

            neigh = neigh[
                complete
            ]

            ne = (
                neigh[
                    :,
                    edges[:, 1],
                ]
                *
                np.conj(
                    neigh[
                        :,
                        edges[:, 0],
                    ]
                )
            )

            vals = np.mean(
                np.real(
                    ce[
                        ids2
                    ]
                    *
                    np.conj(
                        ne
                    )
                ),
                axis=1,
            )

            sim[
                ids2,
                q,
            ] = vals.astype(
                np.float32,
                copy=False,
            )

        with np.errstate(
            all="ignore"
        ):
            median_out[
                b0:b1
            ] = np.nanmedian(
                sim,
                axis=1,
            ).astype(
                np.float32
            )

            max_out[
                b0:b1
            ] = np.nanmax(
                sim,
                axis=1,
            ).astype(
                np.float32
            )

    return (
        rows,
        cols,
        median_out,
        max_out,
    )


def finite_quantiles(
    x: np.ndarray,
):
    x = np.asarray(
        x
    )

    v = x[
        np.isfinite(
            x
        )
    ]

    if v.size == 0:
        return {
            "count": 0,
        }

    q = np.quantile(
        v,
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ],
    )

    return {
        "count": int(
            v.size
        ),
        "min": float(
            np.min(v)
        ),
        "q01": float(
            q[0]
        ),
        "q05": float(
            q[1]
        ),
        "q25": float(
            q[2]
        ),
        "median": float(
            q[3]
        ),
        "q75": float(
            q[4]
        ),
        "q95": float(
            q[5]
        ),
        "q99": float(
            q[6]
        ),
        "max": float(
            np.max(v)
        ),
    }


def write_json(
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

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )


__all__ = [
    "connected_support_count",
    "crlb_median_std_from_compressed",
    "deterministic_sample_positions",
    "disk_offsets",
    "dolphin_style_num_looks",
    "finite_quantiles",
    "nearest_n_edges",
    "nearest_triplet_closure_metrics",
    "sampled_phase_similarity",
    "write_json",
]
