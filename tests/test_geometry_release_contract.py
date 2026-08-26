from pathlib import Path

import yaml

from pypsds.pipeline import STAGES


PUBLIC_CONFIGS = (
    Path("config/pypsds.yaml"),
    Path("config/pypsds_template.yaml"),
    Path("pypsds/resources/default_config.yaml"),
)


def _contract(path):
    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    g = cfg["geometry"]

    return {
        "dem_dir":
            cfg["paths"]["dem_dir"],

        "reference_date":
            g["reference_date"],

        "geometry_par":
            g["geometry_par"],

        "longitude_raster":
            g["longitude_raster"],

        "latitude_raster":
            g["latitude_raster"],

        "height_raster":
            g["height_raster"],
    }


def test_public_geometry_contract_is_consistent():

    values = [
        _contract(path)
        for path in PUBLIC_CONFIGS
    ]

    assert (
        values[0]
        ==
        values[1]
        ==
        values[2]
    )


def test_public_geometry_contract_is_portable():

    expected = {
        "dem_dir":
            "DEM_prep",

        "reference_date":
            None,

        "geometry_par":
            None,

        "longitude_raster":
            None,

        "latitude_raster":
            None,

        "height_raster":
            None,
    }

    for path in PUBLIC_CONFIGS:

        assert (
            _contract(path)
            ==
            expected
        )


def test_release_contract_has_point_geometry():

    names = [
        stage.name
        for stage in STAGES
    ]

    assert len(names) == 38

    assert names[-8:] == [
        "timeseries_inversion",
        "point_geometry",
        "reference",
        "atmosphere_correction",
        "scla",
        "scn",
        "final_los",
        "point_products",
    ]

    text = Path(
        "tools/release_gate.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "build_point_geometry.py"
        in text
    )
