from __future__ import annotations

import hashlib
import os

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pypsds.stages.run_phase_linking import (
    _phase_source_file_identity,
)

from pypsds.phase_linking.sequential_production import (
    _phase_linking_completion_fingerprint,
)


def _sha256(path):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def test_external_phase_source_identity_tracks_mtime(
    tmp_path,
):

    p = (
        tmp_path
        /
        "20200101.rslc"
    )

    p.write_bytes(
        b"12345678"
    )

    a = (
        _phase_source_file_identity(
            p
        )
    )

    assert a["path"] == str(
        p.resolve()
    )

    assert a["size"] == 8

    assert (
        "mtime_ns"
        in a
    )

    old = int(
        a["mtime_ns"]
    )

    new = (
        old
        +
        2_000_000_000
    )

    os.utime(
        p,
        ns=(
            new,
            new,
        ),
    )

    b = (
        _phase_source_file_identity(
            p
        )
    )

    assert (
        b["mtime_ns"]
        !=
        a["mtime_ns"]
    )


def test_completion_fingerprint_binds_phase_source_token(
    tmp_path,
):

    outdir = (
        tmp_path
        /
        "processing"
    )

    cache = (
        outdir
        /
        "cache"
    )

    cache.mkdir(
        parents=True,
    )

    token = (
        cache
        /
        "phase_source_checkpoint_token.json"
    )

    token.write_text(
        '{"source":"A"}\n',
        encoding="utf-8",
    )

    config = (
        tmp_path
        /
        "pypsds.yaml"
    )

    config.write_text(
        "reference_date: 20200101\n",
        encoding="utf-8",
    )

    yxt_path = (
        tmp_path
        /
        "corrected_yxt.npy"
    )

    yxt = (
        np.lib.format.open_memmap(
            yxt_path,
            mode="w+",
            dtype=np.complex64,
            shape=(
                2,
                2,
                2,
            ),
        )
    )

    scale2 = np.ones(
        (2, 2),
        dtype=np.float32,
    )

    valid = np.ones(
        (2, 2),
        dtype=np.bool_,
    )

    geom = np.ones(
        (2, 2),
        dtype=np.bool_,
    )

    ps = np.zeros(
        (2, 2),
        dtype=np.bool_,
    )

    args = SimpleNamespace(
        half_row=1,
        half_col=1,
        alpha=0.005,
        min_shp=48,
        beta=0.0,
        gamma_jitter=1.0e-6,
        emi_mu=0.99,
        batch_size=16,
        pl_workers=1,
        pl_chunk_size=16,
        tile_rows=4,
        tile_cols=4,
        support_block=16,
    )

    fp = (
        _phase_linking_completion_fingerprint(
            cfg={
                "phase_linking": {},
                "selection": {},
            },

            config_path=config,

            outdir=outdir,

            yxt=yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom,
            ps=ps,

            H=2,
            W=2,
            ndate=2,

            args=args,
        )
    )

    assert (
        fp[
            "phase_source_checkpoint_token_sha256"
        ]
        ==
        _sha256(
            token
        )
    )

    first = (
        fp[
            "phase_source_checkpoint_token_sha256"
        ]
    )

    token.write_text(
        '{"source":"B"}\n',
        encoding="utf-8",
    )

    fp2 = (
        _phase_linking_completion_fingerprint(
            cfg={
                "phase_linking": {},
                "selection": {},
            },

            config_path=config,

            outdir=outdir,

            yxt=yxt,
            scale2=scale2,

            valid=valid,
            geom_valid=geom,
            ps=ps,

            H=2,
            W=2,
            ndate=2,

            args=args,
        )
    )

    assert (
        fp2[
            "phase_source_checkpoint_token_sha256"
        ]
        !=
        first
    )
