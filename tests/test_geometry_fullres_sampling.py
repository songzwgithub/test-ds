import numpy as np

from pypsds.geometry import (
    build_ipta_point_list,
    sample_radar_raster_at_points,
)


def _write_par(path, *, width, length, rg_spacing, az_spacing, rg_looks=None, az_looks=None):
    lines = [
        f"range_samples: {width}",
        f"azimuth_lines: {length}",
        f"range_pixel_spacing: {rg_spacing} m",
        f"azimuth_pixel_spacing: {az_spacing} m",
    ]
    if rg_looks is not None:
        lines.append(f"range_looks: {rg_looks}")
    if az_looks is not None:
        lines.append(f"azimuth_looks: {az_looks}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_multilook_bilinear_sampling_preserves_singlelook_coordinates(tmp_path):
    raw_par = tmp_path / "raw.rslc.par"
    mli_par = tmp_path / "geom.mli.par"

    _write_par(
        raw_par,
        width=31,
        length=9,
        rg_spacing=2.0,
        az_spacing=10.0,
    )
    _write_par(
        mli_par,
        width=4,
        length=5,
        rg_spacing=20.0,
        az_spacing=20.0,
        rg_looks=10,
        az_looks=2,
    )

    # A perfectly linear field makes the expected continuous interpolation exact:
    # z(v,u) = 100*v + 10*u
    yy, xx = np.mgrid[0:5, 0:4]
    grid = (
        100.0 * yy
        + 10.0 * xx
    ).astype(">f4")

    raster = tmp_path / "geom.bin"
    grid.tofile(raster)

    cols = np.array(
        [0, 1, 5, 9, 10, 11, 19, 20, 29, 30],
        dtype=np.int32,
    )
    rows = np.array(
        [0, 1, 1, 2, 2, 3, 4, 5, 7, 8],
        dtype=np.int32,
    )

    plist = build_ipta_point_list(
        cols,
        rows,
        tmp_path / "points.plist",
    )

    got = sample_radar_raster_at_points(
        source_raster=raster,
        geometry_par=mli_par,
        point_list=plist,
        reference_rslc_par=raw_par,
        output_path=tmp_path / "values.pt",
        expected_count=cols.size,
    )

    expected = (
        100.0 * (rows.astype(np.float64) / 2.0)
        +
        10.0 * (cols.astype(np.float64) / 10.0)
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )

    # Adjacent raw pixels inside one 10x2 cell must not collapse.
    assert got[1] != got[2]
    assert got[2] != got[3]


def test_multilook_integer_centers_equal_source_grid(tmp_path):
    raw_par = tmp_path / "raw.rslc.par"
    mli_par = tmp_path / "geom.mli.par"

    _write_par(
        raw_par,
        width=31,
        length=9,
        rg_spacing=2.0,
        az_spacing=10.0,
    )
    _write_par(
        mli_par,
        width=4,
        length=5,
        rg_spacing=20.0,
        az_spacing=20.0,
        rg_looks=10,
        az_looks=2,
    )

    grid = np.arange(
        20,
        dtype=np.float32,
    ).reshape(5, 4).astype(">f4")

    raster = tmp_path / "geom.bin"
    grid.tofile(raster)

    rows = np.array(
        [0, 2, 4, 6, 8],
        dtype=np.int32,
    )
    cols = np.array(
        [0, 10, 20, 30, 30],
        dtype=np.int32,
    )

    plist = build_ipta_point_list(
        cols,
        rows,
        tmp_path / "points.plist",
    )

    got = sample_radar_raster_at_points(
        source_raster=raster,
        geometry_par=mli_par,
        point_list=plist,
        reference_rslc_par=raw_par,
        output_path=tmp_path / "values.pt",
        expected_count=cols.size,
    )

    expected = np.array(
        [
            grid[0, 0],
            grid[1, 1],
            grid[2, 2],
            grid[3, 3],
            grid[4, 3],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        got,
        expected,
    )


def test_final_partial_look_interval_uses_linear_edge_extrapolation(tmp_path):
    raw_par = tmp_path / "raw.rslc.par"
    mli_par = tmp_path / "geom.mli.par"

    _write_par(
        raw_par,
        width=42,
        length=11,
        rg_spacing=2.0,
        az_spacing=10.0,
    )
    _write_par(
        mli_par,
        width=4,
        length=5,
        rg_spacing=20.0,
        az_spacing=20.0,
        rg_looks=10,
        az_looks=2,
    )

    yy, xx = np.mgrid[0:5, 0:4]
    grid = (
        100.0 * yy
        + 10.0 * xx
    ).astype(">f4")
    raster = tmp_path / "geom.bin"
    grid.tofile(raster)

    rows = np.array(
        [10],
        dtype=np.int32,
    )
    cols = np.array(
        [41],
        dtype=np.int32,
    )

    plist = build_ipta_point_list(
        cols,
        rows,
        tmp_path / "points.plist",
    )

    got = sample_radar_raster_at_points(
        source_raster=raster,
        geometry_par=mli_par,
        point_list=plist,
        reference_rslc_par=raw_par,
        output_path=tmp_path / "values.pt",
        expected_count=1,
    )

    expected = np.array(
        [100.0 * 5.0 + 10.0 * 4.1],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        got,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_joint_geometry_masks_invalid_zero_zero_corner(tmp_path):
    from types import SimpleNamespace
    from pypsds.geometry import sample_full_resolution_point_geometry

    raw_par = tmp_path / "raw.rslc.par"
    mli_par = tmp_path / "geom.mli.par"

    _write_par(
        raw_par,
        width=21,
        length=5,
        rg_spacing=2.0,
        az_spacing=10.0,
    )
    _write_par(
        mli_par,
        width=3,
        length=3,
        rg_spacing=20.0,
        az_spacing=20.0,
        rg_looks=10,
        az_looks=2,
    )

    yy, xx = np.mgrid[0:3, 0:3]
    lon = (121.0 + 0.01 * xx + 0.001 * yy).astype(">f4")
    lat = (30.0 + 0.01 * yy + 0.001 * xx).astype(">f4")
    hgt = (5.0 + xx + yy).astype(">f4")

    lon[0, 0] = 0.0
    lat[0, 0] = 0.0

    lon_path = tmp_path / "lon.bin"
    lat_path = tmp_path / "lat.bin"
    hgt_path = tmp_path / "hgt.bin"
    lon.tofile(lon_path)
    lat.tofile(lat_path)
    hgt.tofile(hgt_path)

    geometry = SimpleNamespace(
        geometry_par=mli_par,
        reference_rslc_par=raw_par,
        longitude_raster=lon_path,
        latitude_raster=lat_path,
    )

    out = sample_full_resolution_point_geometry(
        rows=np.array([1], dtype=np.int32),
        cols=np.array([5], dtype=np.int32),
        geometry=geometry,
        height_raster=hgt_path,
        work_dir=tmp_path / "work",
    )

    assert out.valid_mask.tolist() == [True]
    assert out.partial_support_count == 1
    assert out.zero_support_count == 0
    assert 120.0 < out.longitude_deg[0] < 122.0
    assert 29.0 < out.latitude_deg[0] < 31.0
