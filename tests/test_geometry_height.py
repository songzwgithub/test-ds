from pathlib import Path
from types import SimpleNamespace

import pytest

from pypsds.geometry import (
    GeometryInputs,
    HeightGeometryError,
    resolve_height_raster,
)


def geometry(tmp_path):
    return GeometryInputs(
        reference_date="20200102",
        reference_rslc_par=
            tmp_path / "x.rslc.par",
        geometry_par=
            tmp_path / "x.mli.par",
        longitude_raster=
            tmp_path / "x.rdc.lon",
        latitude_raster=
            tmp_path / "x.rdc.lat",
        radar_width=10,
        radar_length=20,
        float32_expected_bytes=800,
    )


def test_default_height_resolution(tmp_path):
    dem = tmp_path / "DEM_prep"
    dem.mkdir()

    h = dem / "20200102.hgt"
    h.write_bytes(b"\0" * 800)

    paths = SimpleNamespace(
        dem_dir=dem,
        data_dir=tmp_path,
    )

    assert resolve_height_raster(
        {},
        paths,
        geometry(tmp_path),
    ) == h.resolve()


def test_height_size_validation(tmp_path):
    dem = tmp_path / "DEM_prep"
    dem.mkdir()

    h = dem / "20200102.hgt"
    h.write_bytes(b"\0" * 16)

    paths = SimpleNamespace(
        dem_dir=dem,
        data_dir=tmp_path,
    )

    with pytest.raises(
        HeightGeometryError,
        match="byte size",
    ):
        resolve_height_raster(
            {},
            paths,
            geometry(tmp_path),
        )
