from types import SimpleNamespace

from pypsds.gamma.phase_correction import (
    _discover_height,
)


def _write_geometry_par(path):
    path.write_text(
        "range_samples: 2\n"
        "azimuth_lines: 2\n",
        encoding="utf-8",
    )


def test_canonical_reference_height_beats_pixel_area_height(
    tmp_path,
):
    dem = tmp_path / "DEM_prep"
    dem.mkdir()

    work = tmp_path / "project"
    work.mkdir()

    canonical = dem / "20151212.hgt"
    pixel_area = dem / "20151212.pixel_area.hgt"

    # 2 x 2 FLOAT raster = 16 bytes.
    canonical.write_bytes(b"\x00" * 16)
    pixel_area.write_bytes(b"\x00" * 16)

    par = dem / "20151212_4_1.vv.mli.par"
    _write_geometry_par(par)

    cfg = {
        "reference_date": "20151212",
        "geometry": {
            "height_raster": None,
        },
        "phase_correction": {
            "radar_height": {
                "path": None,
                "geometry_par": str(par),
            },
        },
    }

    paths = SimpleNamespace(
        work_dir=work,
        data_dir=tmp_path,
        dem_dir=dem,
    )

    stack = SimpleNamespace(
        dates=(
            "20141006",
            "20151212",
            "20160410",
        )
    )

    height, geometry_par = _discover_height(
        cfg,
        paths,
        stack,
    )

    assert height == canonical.resolve()
    assert geometry_par == par.resolve()
