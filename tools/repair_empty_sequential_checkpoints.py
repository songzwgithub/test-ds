#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np


def atomic_json(
    path: Path,
    payload: dict,
) -> None:
    tmp = path.with_suffix(
        path.suffix
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processing-dir",
        required=True,
    )
    ap.add_argument(
        "--stage",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--apply",
        action="store_true",
    )
    args = ap.parse_args()

    processing = Path(
        args.processing_dir
    ).resolve()

    root = (
        processing
        /
        "sequential"
        /
        "checkpoints"
        /
        f"sequential_stage{args.stage:04d}"
    )

    manifest_path = (
        root
        /
        "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    fp = manifest[
        "fingerprint"
    ]
    fp_hash = manifest[
        "fingerprint_sha256"
    ]

    state_token = fp.get(
        "state_core"
    )

    if (
        not isinstance(
            state_token,
            dict,
        )
        or
        not state_token.get(
            "path"
        )
    ):
        raise RuntimeError(
            "manifest does not expose state_core path"
        )

    state_path = Path(
        state_token[
            "path"
        ]
    )

    state = np.load(
        state_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    tile_rows = int(
        fp[
            "tile_rows"
        ]
    )
    tile_cols = int(
        fp[
            "tile_cols"
        ]
    )

    H, W = state.shape

    tiles = []

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

            tiles.append(
                (
                    r0,
                    r1,
                    c0,
                    c1,
                )
            )

    existing = {}

    for p in root.glob(
        "tile_*.json"
    ):
        m = re.fullmatch(
            r"tile_(\d+)\.json",
            p.name,
        )

        if m:
            existing[
                int(
                    m.group(1)
                )
            ] = p

    if not existing:
        print(
            "no existing tile markers"
        )
        return

    max_existing = max(
        existing
    )

    cumulative = 0
    repair = []
    errors = []

    for idx, bounds in enumerate(
        tiles,
        start=1,
    ):
        if idx > max_existing:
            break

        r0, r1, c0, c1 = bounds

        nstate = int(
            np.count_nonzero(
                state[
                    r0:r1,
                    c0:c1,
                ]
            )
        )

        cumulative += nstate

        p = (
            root
            /
            f"tile_{idx:04d}.json"
        )

        if p.is_file():
            rec = json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )

            if int(
                rec.get(
                    "tile_index",
                    -1,
                )
            ) != idx:
                errors.append(
                    (
                        idx,
                        "tile_index",
                    )
                )

            if list(
                rec.get(
                    "bounds",
                    [],
                )
            ) != list(
                bounds
            ):
                errors.append(
                    (
                        idx,
                        "bounds",
                    )
                )

            if int(
                rec.get(
                    "total_done",
                    -1,
                )
            ) != cumulative:
                errors.append(
                    (
                        idx,
                        "total_done",
                        rec.get(
                            "total_done"
                        ),
                        cumulative,
                    )
                )

            if (
                rec.get(
                    "fingerprint_sha256"
                )
                !=
                fp_hash
            ):
                errors.append(
                    (
                        idx,
                        "fingerprint",
                    )
                )

            continue

        if nstate != 0:
            errors.append(
                (
                    idx,
                    "missing_nonempty",
                    nstate,
                    bounds,
                )
            )
        else:
            repair.append(
                (
                    idx,
                    bounds,
                    cumulative,
                )
            )

    print(
        "scene             :",
        state.shape,
    )
    print(
        "total tiles       :",
        len(
            tiles
        ),
    )
    print(
        "existing markers  :",
        len(
            existing
        ),
    )
    print(
        "max marker        :",
        max_existing,
    )
    print(
        "empty gaps        :",
        len(
            repair
        ),
    )
    print(
        "errors            :",
        len(
            errors
        ),
    )

    if errors:
        for x in errors[:20]:
            print(
                "ERROR",
                x,
            )

        raise SystemExit(
            "REFUSE: checkpoint audit failed"
        )

    if not args.apply:
        print(
            "AUDIT PASS; rerun with --apply "
            "to create empty-tile markers"
        )
        return

    for (
        idx,
        bounds,
        cumulative,
    ) in repair:
        atomic_json(
            root
            /
            f"tile_{idx:04d}.json",
            {
                "format":
                    manifest[
                        "format"
                    ],

                "fingerprint_sha256":
                    fp_hash,

                "tile_index":
                    idx,

                "bounds":
                    [
                        int(x)
                        for x
                        in bounds
                    ],

                "total_done":
                    int(
                        cumulative
                    ),

                "time_unix":
                    float(
                        time.time()
                    ),
            },
        )

    print(
        "created markers   :",
        len(
            repair
        ),
    )
    print(
        "resume prefix     :",
        max_existing,
    )


if __name__ == "__main__":
    main()
