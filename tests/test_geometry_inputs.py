from pathlib import Path

import pytest

from pypsds.geometry import (
    GeometryInputError,
    gamma_par_int,
    read_gamma_par,
    resolve_geometry_inputs,
)

from pypsds.project import ProjectPaths


DATE = "20200102"


def write_par(path, width, length):
    path.write_text(
        f"range_samples: {width}\n"
        f"azimuth_lines: {length}\n",
        encoding="utf-8",
    )


def write_float_raster(path, width, length):
    path.write_bytes(
        b"\x00" * (width * length * 4)
    )


def make_project(tmp_path):
    work = tmp_path / "project"
    data = work / "data"
    rslc = data / "RSLC"
    dem = data / "DEM_prep"
    output = work / "output"

    rslc.mkdir(parents=True)
    dem.mkdir(parents=True)
    output.mkdir(parents=True)

    tab = data / "RSLC_tab"
    tab.write_text("", encoding="utf-8")

    paths = ProjectPaths(
        work_dir=work,
        data_dir=data,
        rslc_dir=rslc,
        rslc_tab=tab,
        output_dir=output,
        dem_dir=dem,
    )

    return paths, rslc, dem


def test_gamma_par_parser(tmp_path):
    par = tmp_path / "test.par"

    par.write_text(
        "range_samples: 500\n"
        "azimuth_lines: 600\n",
        encoding="utf-8",
    )

    values = read_gamma_par(par)

    assert gamma_par_int(
        values,
        "range_samples",
    ) == 500

    assert gamma_par_int(
        values,
        "azimuth_lines",
    ) == 600


def test_geometry_semantic_discovery(tmp_path):
    paths, rslc, dem = make_project(
        tmp_path
    )

    (rslc / f"{DATE}.rslc.par").write_text(
        "dummy: 1\n",
        encoding="utf-8",
    )

    width = 500
    length = 600

    write_float_raster(
        dem / f"{DATE}_8_2.rdc.lon",
        width,
        length,
    )

    write_float_raster(
        dem / f"{DATE}_8_2.rdc.lat",
        width,
        length,
    )

    write_par(
        dem / f"{DATE}_wrong.mli.par",
        100,
        100,
    )

    correct = (
        dem
        / f"{DATE}_custom.hh.mli.par"
    )

    write_par(
        correct,
        width,
        length,
    )

    cfg = {
        "phase_correction": {
            "geometric_reference_date":
                DATE,
        }
    }

    result = resolve_geometry_inputs(
        cfg,
        paths,
    )

    assert result.reference_date == DATE
    assert result.geometry_par == correct.resolve()
    assert result.radar_width == width
    assert result.radar_length == length


def test_geometry_fails_on_ambiguity(tmp_path):
    paths, rslc, dem = make_project(
        tmp_path
    )

    (rslc / f"{DATE}.rslc.par").write_text(
        "dummy: 1\n",
        encoding="utf-8",
    )

    width = 10
    length = 20

    write_float_raster(
        dem / f"{DATE}_a.rdc.lon",
        width,
        length,
    )

    write_float_raster(
        dem / f"{DATE}_b.rdc.lon",
        width,
        length,
    )

    write_float_raster(
        dem / f"{DATE}.rdc.lat",
        width,
        length,
    )

    write_par(
        dem / f"{DATE}.mli.par",
        width,
        length,
    )

    cfg = {
        "phase_correction": {
            "geometric_reference_date":
                DATE,
        }
    }

    with pytest.raises(
        GeometryInputError,
        match="Ambiguous longitude radar raster",
    ):
        resolve_geometry_inputs(
            cfg,
            paths,
        )
