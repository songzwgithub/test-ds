from pathlib import Path
import inspect

import numpy as np

from pypsds.corrections import scla_baseline
from pypsds.corrections import scla_k
from pypsds.corrections import scla_c


GEOMETRY_ARGS = {
    "azimuth_looks",
    "earth_radius",
    "mean_azimuth",
    "near_range",
    "prf",
    "range_looks",
    "range_spacing",
    "sar_to_earth",
}


def test_scla_baseline_parse_base(tmp_path):

    p = tmp_path / "x.base"

    # Exercise only the authoritative parser contract.
    p.write_text(
        "initial_baseline(TCN): 1.0 2.0 3.0\n"
        "initial_baseline_rate: 0.1 0.2 0.3\n",
        encoding="utf-8",
    )

    B, Br = scla_baseline.parse_base(
        p
    )

    assert B.shape == (3,)
    assert Br.shape == (3,)
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(Br))


def test_scla_c_gls_projector():

    A = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    covariance = np.eye(
        3,
        dtype=np.float64,
    )

    projector = scla_c.gls_projector(
        A,
        covariance,
    )

    assert projector.shape == (2, 3)

    recovered = (
        projector
        @
        A
    )

    np.testing.assert_allclose(
        recovered,
        np.eye(2),
        rtol=0.0,
        atol=1e-12,
    )


def test_scla_geometry_contract_is_explicit():

    for func in (
        scla_k.geometry_factors,
        scla_c.geometry_factors,
    ):

        sig = inspect.signature(
            func
        )

        names = set(
            sig.parameters
        )

        assert GEOMETRY_ARGS <= names

        for name in GEOMETRY_ARGS:
            assert (
                sig.parameters[name].default
                is
                inspect.Parameter.empty
            )


def test_scla_c_projection_contract_is_explicit():

    sig = inspect.signature(
        scla_c.baseline_c_projection
    )

    names = set(
        sig.parameters
    )

    assert GEOMETRY_ARGS <= names

    assert {
        "CW_C",
        "CW_CR",
        "CW_N",
        "CW_NR",
    } <= names
