#!/usr/bin/env python3
from __future__ import annotations

import math

from pypsds.runtime import (
    available_memory_bytes,
    logical_cpu_count,
)


def gib(x):
    return x / 1024**3


def mib(x):
    return x / 1024**2


def model(
    ndate: int,
    *,
    target_batch: int = 16000,
    memory_fraction: float = 0.50,
):
    npair = (
        ndate
        *
        (ndate - 1)
        //
        2
    )

    # --------------------------------------------------------
    # Persistent compressed coherence per point
    #
    # complex64
    # --------------------------------------------------------

    coh_per_point = (
        npair
        *
        8
    )

    # --------------------------------------------------------
    # Fast-Cholesky EMI approximate workspace / point.
    #
    # This is deliberately conservative.
    #
    # C             complex128       N² * 16
    # Gamma         float64          N² * 8
    # Gamma_inv     float64          N² * 8
    # A             complex128       N² * 16
    # eigenvectors  complex128       N² * 16
    # misc / eigvals / temporaries
    # --------------------------------------------------------

    emi_matrix_per_point = (
        ndate
        *
        ndate
        *
        (
            16
            +
            8
            +
            8
            +
            16
            +
            16
            +
            16
        )
    )

    misc_per_point = (
        ndate
        *
        64
        +
        4096
    )

    workspace_per_point = (
        coh_per_point
        +
        emi_matrix_per_point
        +
        misc_per_point
    )

    avail = (
        available_memory_bytes()
    )

    budget = int(
        avail
        *
        memory_fraction
    )

    max_batch_by_ram = max(
        1,
        budget
        //
        max(
            1,
            workspace_per_point,
        )
    )

    practical_batch = max(
        1,
        min(
            target_batch,
            max_batch_by_ram,
        )
    )

    # --------------------------------------------------------
    # Relative algebra cost.
    #
    # Cholesky ~ N³/3
    # Hermitian eigendecomposition ~ O(N³)
    #
    # We use a dimensionless N³ indicator here.
    # --------------------------------------------------------

    cubic = (
        ndate
        **
        3
    )

    relative_to_38 = (
        ndate
        /
        38.0
    ) ** 3

    return {
        "ndate":
            ndate,

        "npair":
            npair,

        "coh_per_point":
            coh_per_point,

        "workspace_per_point":
            workspace_per_point,

        "max_batch_by_ram":
            max_batch_by_ram,

        "practical_batch":
            practical_batch,

        "batch_workspace":
            practical_batch
            *
            workspace_per_point,

        "relative_cubic_to_38":
            relative_to_38,
    }


def main():

    cpu = (
        logical_cpu_count()
    )

    avail = (
        available_memory_bytes()
    )

    print(
        "=" * 108
    )

    print(
        "pyPSDS-GAMMA temporal full-SCM scaling model"
    )

    print(
        "=" * 108
    )

    print(
        "CPU              :",
        cpu,
    )

    print(
        "available RAM    :",
        f"{gib(avail):.2f} GiB",
    )

    print(
        "workspace budget :",
        "50% available RAM",
    )

    print()

    print(
        f"{'N':>6s}"
        f"{'pairs':>12s}"
        f"{'coh/pt':>12s}"
        f"{'work/pt':>12s}"
        f"{'batch':>10s}"
        f"{'batch RAM':>12s}"
        f"{'N^3 / 38':>12s}"
    )

    print(
        "-" * 92
    )

    for ndate in (
        38,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
    ):

        x = model(
            ndate
        )

        print(
            f"{ndate:6d}"
            f"{x['npair']:12,d}"
            f"{mib(x['coh_per_point']):12.3f}"
            f"{mib(x['workspace_per_point']):12.3f}"
            f"{x['practical_batch']:10,d}"
            f"{gib(x['batch_workspace']):12.2f}"
            f"{x['relative_cubic_to_38']:12.1f}"
        )

    print()

    print(
        "Notes:"
    )

    print(
        "  coh/pt   = compressed complex64 upper triangle"
    )

    print(
        "  work/pt  = conservative full-SCM fast-Cholesky workspace"
    )

    print(
        "  N^3/38   = approximate algebra scaling relative to 38 acquisitions"
    )


if __name__ == "__main__":
    main()
