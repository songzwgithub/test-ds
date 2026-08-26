from pathlib import Path

import yaml

from pypsds.pipeline import STAGES


PUBLIC_CONFIGS = (
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

    assert PUBLIC_CONFIGS == (
        Path("pypsds/resources/default_config.yaml"),
    )

    path = PUBLIC_CONFIGS[0]

    assert path.is_file()

    assert _contract(path) == {
        "dem_dir":
            None,

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


def test_public_geometry_contract_is_portable():

    expected = {
        "dem_dir":
            None,

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

    point_geometry = next(
        stage
        for stage in STAGES
        if stage.name == "point_geometry"
    )

    stage_file = (
        Path("pypsds/stages")
        / point_geometry.script
    )

    assert stage_file.is_file()
