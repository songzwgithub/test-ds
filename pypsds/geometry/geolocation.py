from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from .gamma_par import (
    GammaParError,
    gamma_par_int,
    gamma_par_scalar,
    read_gamma_par,
)
from .inputs import GeometryInputs


class GeolocationError(RuntimeError):
    """Radar-point geolocation failed."""


@dataclass(
    frozen=True,
    slots=True,
)
class PointGeolocation:
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    valid_mask: np.ndarray

    point_list: Path
    longitude_gamma_pt: Path
    latitude_gamma_pt: Path


def resolve_data2pt(
    executable: str | Path | None = None,
) -> Path:
    """
    Resolve the GAMMA data2pt executable.

    Kept for public API/backward compatibility. Full-resolution point geometry
    no longer uses nearest-cell data2pt sampling when the geometry raster is
    multilooked.
    """

    if executable is not None:
        path = Path(executable).expanduser()

        if not path.is_absolute():
            found = shutil.which(str(path))

            if found is None:
                raise GeolocationError(
                    f"Cannot resolve data2pt executable: {executable}"
                )

            path = Path(found)

        path = path.resolve()

        if not path.is_file():
            raise GeolocationError(
                f"data2pt executable does not exist: {path}"
            )

        return path

    found = shutil.which(
        "data2pt"
    )

    if found is None:
        raise GeolocationError(
            "GAMMA data2pt is not available on PATH."
        )

    return Path(found).resolve()


def build_ipta_point_list(
    cols,
    rows,
    output_path: str | Path,
) -> Path:
    """
    Write the GAMMA/IPTA point list.

    Column 0 = range pixel  = col
    Column 1 = azimuth line = row

    Coordinates remain 0-based.

    Binary representation:
        big-endian signed int32 (>i4)
    """

    cols = np.asarray(cols)
    rows = np.asarray(rows)

    if cols.ndim != 1 or rows.ndim != 1:
        raise GeolocationError(
            "cols and rows must be one-dimensional arrays."
        )

    if cols.shape != rows.shape:
        raise GeolocationError(
            "cols and rows must have identical shape."
        )

    if not (
        np.issubdtype(cols.dtype, np.integer)
        and
        np.issubdtype(rows.dtype, np.integer)
    ):
        raise GeolocationError(
            "cols and rows must contain integer radar coordinates."
        )

    i32 = np.iinfo(np.int32)

    if cols.size:
        if (
            cols.min() < i32.min
            or cols.max() > i32.max
            or rows.min() < i32.min
            or rows.max() > i32.max
        ):
            raise GeolocationError(
                "Radar coordinates exceed int32 range."
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.column_stack((cols, rows)).astype(
        ">i4",
        copy=False,
    ).tofile(output)

    expected_bytes = cols.size * 8
    actual_bytes = output.stat().st_size

    if actual_bytes != expected_bytes:
        raise GeolocationError(
            "Invalid IPTA point-list byte size: "
            f"{actual_bytes} != {expected_bytes}"
        )

    return output


def read_gamma_point_values(
    path: str | Path,
    *,
    expected_count: int,
) -> np.ndarray:
    """
    Read a GAMMA-compatible point-value file.

    On-disk diagnostic representation:
        big-endian float32 -> returned float64
    """

    path = Path(path)

    if not path.is_file():
        raise GeolocationError(
            f"Missing point-value output: {path}"
        )

    values = np.fromfile(
        path,
        dtype=">f4",
    ).astype(np.float64)

    if values.size != expected_count:
        raise GeolocationError(
            "Unexpected point-value output count: "
            f"{values.size} != {expected_count}"
        )

    return values


def _look_factor(
    geometry_values: dict[str, str],
    raw_values: dict[str, str],
    *,
    look_key: str,
    spacing_key: str,
) -> int:
    """
    Resolve one integer multilook factor.

    Prefer the explicit GAMMA MLI look count. Fall back to the pixel-spacing
    ratio for older/non-standard parameter files.
    """

    try:
        if look_key in geometry_values:
            factor = gamma_par_int(
                geometry_values,
                look_key,
            )
        else:
            ratio = (
                gamma_par_scalar(
                    geometry_values,
                    spacing_key,
                )
                /
                gamma_par_scalar(
                    raw_values,
                    spacing_key,
                )
            )
            factor = int(round(ratio))

            if (
                factor <= 0
                or
                abs(ratio - factor)
                >
                1.0e-6 * max(1.0, abs(ratio))
            ):
                raise GeolocationError(
                    f"Cannot infer integral {look_key}: "
                    f"{spacing_key} ratio={ratio:.12g}"
                )

    except GammaParError as exc:
        raise GeolocationError(
            f"Cannot resolve {look_key}: {exc}"
        ) from exc

    if factor <= 0:
        raise GeolocationError(
            f"Invalid {look_key}: {factor}"
        )

    return factor


def resolve_radar_look_factors(
    geometry_par: str | Path,
    reference_rslc_par: str | Path,
) -> tuple[int, int]:
    """
    Return (range_looks, azimuth_looks) for a radar geometry raster.
    """

    gpar = read_gamma_par(geometry_par)
    rpar = read_gamma_par(reference_rslc_par)

    return (
        _look_factor(
            gpar,
            rpar,
            look_key="range_looks",
            spacing_key="range_pixel_spacing",
        ),
        _look_factor(
            gpar,
            rpar,
            look_key="azimuth_looks",
            spacing_key="azimuth_pixel_spacing",
        ),
    )


def _axis_linear_indices(
    coordinate: np.ndarray,
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return lower/upper raster indices plus fractional coordinate.

    Interior points use ordinary linear interpolation. Points in a residual
    partial raw-look interval beyond the final MLI center use linear
    extrapolation from the final two MLI centers rather than being discarded.
    This preserves valid full-resolution edge points.
    """

    if size <= 0:
        raise GeolocationError(
            f"Invalid raster axis size: {size}"
        )

    coordinate = np.asarray(
        coordinate,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(coordinate)):
        raise GeolocationError(
            "Non-finite continuous raster coordinate."
        )

    if size == 1:
        lower = np.zeros(
            coordinate.size,
            dtype=np.int64,
        )
        upper = lower
        frac = np.zeros(
            coordinate.size,
            dtype=np.float64,
        )
        return lower, upper, frac

    lower = np.floor(
        coordinate
    ).astype(np.int64)

    lower = np.clip(
        lower,
        0,
        size - 2,
    )

    upper = lower + 1

    frac = (
        coordinate
        -
        lower.astype(
            np.float64,
            copy=False,
        )
    )

    return lower, upper, frac


def sample_radar_raster_at_points(
    *,
    source_raster: str | Path,
    geometry_par: str | Path,
    point_list: str | Path,
    reference_rslc_par: str | Path,
    output_path: str | Path,
    expected_count: int,
    data2pt: str | Path | None = None,
    chunk_points: int = 1_000_000,
) -> np.ndarray:
    """
    Sample a GAMMA radar-coordinate raster at original-resolution radar points.

    The point list is on the original RSLC grid while longitude/latitude/height
    rasters may be multilooked (for example 10x2). Older GAMMA data2pt sampling
    maps those raw coordinates to the nearest MLI cell. That collapses up to
    range_looks*azimuth_looks single-look points onto one map coordinate.

    Instead evaluate the MLI raster at continuous raw-to-MLI coordinates:

        u = raw_range_col / range_looks
        v = raw_azimuth_row / azimuth_looks

    with bilinear interpolation. Integer MLI-center coordinates remain exactly
    equal to the source raster. A final incomplete raw-look interval is linearly
    extrapolated from the final two MLI centers, so valid edge points are not
    silently dropped.

    Returned values are float64. A big-endian float32 diagnostic point file is
    still written to output_path for backward-compatible provenance.

    data2pt is accepted only for API compatibility and intentionally unused.
    """

    del data2pt

    if expected_count < 0:
        raise GeolocationError(
            f"expected_count must be non-negative: {expected_count}"
        )

    if chunk_points <= 0:
        raise GeolocationError(
            f"chunk_points must be positive: {chunk_points}"
        )

    source_raster = Path(source_raster).resolve()
    geometry_par = Path(geometry_par).resolve()
    point_list = Path(point_list).resolve()
    reference_rslc_par = Path(reference_rslc_par).resolve()
    output_path = Path(output_path).resolve()

    for label, path in (
        ("source raster", source_raster),
        ("geometry parameter", geometry_par),
        ("IPTA point list", point_list),
        ("reference RSLC parameter", reference_rslc_par),
    ):
        if not path.is_file():
            raise GeolocationError(
                f"Missing {label}: {path}"
            )

    try:
        gpar = read_gamma_par(geometry_par)
        rpar = read_gamma_par(reference_rslc_par)

        width = gamma_par_int(
            gpar,
            "range_samples",
        )
        length = gamma_par_int(
            gpar,
            "azimuth_lines",
        )

        raw_width = gamma_par_int(
            rpar,
            "range_samples",
        )
        raw_length = gamma_par_int(
            rpar,
            "azimuth_lines",
        )

    except GammaParError as exc:
        raise GeolocationError(
            f"Invalid GAMMA geometry parameters: {exc}"
        ) from exc

    range_looks, azimuth_looks = (
        resolve_radar_look_factors(
            geometry_par,
            reference_rslc_par,
        )
    )

    expected_raster_bytes = (
        width * length * 4
    )

    if (
        source_raster.stat().st_size
        !=
        expected_raster_bytes
    ):
        raise GeolocationError(
            "Radar raster byte size does not match geometry: "
            f"{source_raster.stat().st_size} != "
            f"{expected_raster_bytes}: {source_raster}"
        )

    expected_point_bytes = (
        expected_count * 8
    )

    if (
        point_list.stat().st_size
        !=
        expected_point_bytes
    ):
        raise GeolocationError(
            "Point-list byte size mismatch: "
            f"{point_list.stat().st_size} != "
            f"{expected_point_bytes}: {point_list}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raster = np.memmap(
        source_raster,
        dtype=">f4",
        mode="r",
        shape=(length, width),
    )

    points = np.memmap(
        point_list,
        dtype=">i4",
        mode="r",
        shape=(expected_count, 2),
    )

    values = np.empty(
        expected_count,
        dtype=np.float64,
    )

    diagnostic = np.memmap(
        output_path,
        dtype=">f4",
        mode="w+",
        shape=(expected_count,),
    )

    for start in range(
        0,
        expected_count,
        chunk_points,
    ):
        stop = min(
            start + chunk_points,
            expected_count,
        )

        cols = np.asarray(
            points[start:stop, 0],
            dtype=np.int64,
        )
        rows = np.asarray(
            points[start:stop, 1],
            dtype=np.int64,
        )

        if cols.size:
            if (
                cols.min() < 0
                or cols.max() >= raw_width
                or rows.min() < 0
                or rows.max() >= raw_length
            ):
                raise GeolocationError(
                    "Point list exceeds original RSLC grid at "
                    f"{start}:{stop}."
                )

        u = (
            cols.astype(
                np.float64,
                copy=False,
            )
            /
            float(range_looks)
        )

        v = (
            rows.astype(
                np.float64,
                copy=False,
            )
            /
            float(azimuth_looks)
        )

        x0, x1, fx = _axis_linear_indices(
            u,
            size=width,
        )
        y0, y1, fy = _axis_linear_indices(
            v,
            size=length,
        )

        z00 = np.asarray(
            raster[y0, x0],
            dtype=np.float64,
        )
        z01 = np.asarray(
            raster[y0, x1],
            dtype=np.float64,
        )
        z10 = np.asarray(
            raster[y1, x0],
            dtype=np.float64,
        )
        z11 = np.asarray(
            raster[y1, x1],
            dtype=np.float64,
        )

        top = (
            z00
            +
            fx * (z01 - z00)
        )
        bottom = (
            z10
            +
            fx * (z11 - z10)
        )
        block = (
            top
            +
            fy * (bottom - top)
        )

        values[start:stop] = block

        diagnostic[start:stop] = (
            block.astype(
                np.float32,
                copy=False,
            )
        )

    diagnostic.flush()

    return values


@dataclass(
    frozen=True,
    slots=True,
)
class FullResolutionPointGeometry:
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    height_m: np.ndarray
    valid_mask: np.ndarray

    point_list: Path

    full_support_count: int
    partial_support_count: int
    single_support_count: int
    zero_support_count: int


def _bilinear_weights(fx, fy):
    return (
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    )


def sample_full_resolution_point_geometry(
    *,
    rows,
    cols,
    geometry: GeometryInputs,
    height_raster: str | Path,
    work_dir: str | Path,
    chunk_points: int = 1_000_000,
) -> FullResolutionPointGeometry:
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
        raise GeolocationError(
            "rows/cols must be one-dimensional arrays with identical shape."
        )

    if chunk_points <= 0:
        raise GeolocationError("chunk_points must be positive.")

    n = int(rows.size)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    point_list = build_ipta_point_list(
        cols,
        rows,
        work_dir / "strict_points.plist",
    )

    height_raster = Path(height_raster).resolve()

    try:
        gpar = read_gamma_par(geometry.geometry_par)
        rpar = read_gamma_par(geometry.reference_rslc_par)

        width = gamma_par_int(gpar, "range_samples")
        length = gamma_par_int(gpar, "azimuth_lines")
        raw_width = gamma_par_int(rpar, "range_samples")
        raw_length = gamma_par_int(rpar, "azimuth_lines")

    except GammaParError as exc:
        raise GeolocationError(
            f"Invalid GAMMA geometry parameters: {exc}"
        ) from exc

    range_looks, azimuth_looks = resolve_radar_look_factors(
        geometry.geometry_par,
        geometry.reference_rslc_par,
    )

    expected_bytes = width * length * 4

    for label, path in (
        ("longitude raster", geometry.longitude_raster),
        ("latitude raster", geometry.latitude_raster),
        ("height raster", height_raster),
    ):
        path = Path(path)
        if not path.is_file():
            raise GeolocationError(f"Missing {label}: {path}")
        if path.stat().st_size != expected_bytes:
            raise GeolocationError(
                f"{label} byte size mismatch: "
                f"{path.stat().st_size} != {expected_bytes}: {path}"
            )

    lon_grid = np.memmap(
        geometry.longitude_raster,
        dtype=">f4",
        mode="r",
        shape=(length, width),
    )
    lat_grid = np.memmap(
        geometry.latitude_raster,
        dtype=">f4",
        mode="r",
        shape=(length, width),
    )
    hgt_grid = np.memmap(
        height_raster,
        dtype=">f4",
        mode="r",
        shape=(length, width),
    )

    lon = np.empty(n, dtype=np.float64)
    lat = np.empty(n, dtype=np.float64)
    hgt = np.empty(n, dtype=np.float64)
    valid_out = np.zeros(n, dtype=np.bool_)

    full_support = 0
    partial_support = 0
    single_support = 0
    zero_support = 0

    for start in range(0, n, chunk_points):
        stop = min(start + chunk_points, n)

        rr = np.asarray(rows[start:stop], dtype=np.int64)
        cc = np.asarray(cols[start:stop], dtype=np.int64)

        if cc.size:
            if (
                cc.min() < 0
                or cc.max() >= raw_width
                or rr.min() < 0
                or rr.max() >= raw_length
            ):
                raise GeolocationError(
                    f"Point coordinates exceed original RSLC at {start}:{stop}."
                )

        u = cc.astype(np.float64, copy=False) / float(range_looks)
        v = rr.astype(np.float64, copy=False) / float(azimuth_looks)

        u = np.clip(u, 0.0, float(width - 1))
        v = np.clip(v, 0.0, float(length - 1))

        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, length - 1)

        fx = u - x0
        fy = v - y0

        weights = _bilinear_weights(fx, fy)
        corner_index = (
            (y0, x0),
            (y0, x1),
            (y1, x0),
            (y1, x1),
        )

        m = stop - start

        out_lon = np.zeros(m, dtype=np.float64)
        out_lat = np.zeros(m, dtype=np.float64)
        out_hgt = np.zeros(m, dtype=np.float64)
        wsum = np.zeros(m, dtype=np.float64)
        support = np.zeros(m, dtype=np.uint8)

        for (iy, ix), w in zip(corner_index, weights):
            lo = np.asarray(lon_grid[iy, ix], dtype=np.float64)
            la = np.asarray(lat_grid[iy, ix], dtype=np.float64)
            hh = np.asarray(hgt_grid[iy, ix], dtype=np.float64)

            node_valid = (
                np.isfinite(lo)
                &
                np.isfinite(la)
                &
                np.isfinite(hh)
                &
                (lo > -180.0)
                &
                (lo < 180.0)
                &
                (la > -90.0)
                &
                (la < 90.0)
                &
                ~((lo == 0.0) & (la == 0.0))
            )

            active = node_valid & (w > 0.0)
            ww = np.where(active, w, 0.0)

            out_lon += ww * lo
            out_lat += ww * la
            out_hgt += ww * hh
            wsum += ww
            support += active.astype(np.uint8)

        good = wsum > 0.0

        out_lon[good] /= wsum[good]
        out_lat[good] /= wsum[good]
        out_hgt[good] /= wsum[good]

        out_lon[~good] = np.nan
        out_lat[~good] = np.nan
        out_hgt[~good] = np.nan

        lon[start:stop] = out_lon
        lat[start:stop] = out_lat
        hgt[start:stop] = out_hgt

        valid_out[start:stop] = (
            good
            &
            np.isfinite(out_lon)
            &
            np.isfinite(out_lat)
            &
            np.isfinite(out_hgt)
            &
            (out_lon > -180.0)
            &
            (out_lon < 180.0)
            &
            (out_lat > -90.0)
            &
            (out_lat < 90.0)
        )

        full_support += int(np.count_nonzero(support == 4))
        partial_support += int(np.count_nonzero((support >= 2) & (support <= 3)))
        single_support += int(np.count_nonzero(support == 1))
        zero_support += int(np.count_nonzero(support == 0))

    return FullResolutionPointGeometry(
        longitude_deg=lon,
        latitude_deg=lat,
        height_m=hgt,
        valid_mask=valid_out,
        point_list=point_list,
        full_support_count=full_support,
        partial_support_count=partial_support,
        single_support_count=single_support,
        zero_support_count=zero_support,
    )

def geolocate_points(
    *,
    rows,
    cols,
    geometry: GeometryInputs,
    work_dir: str | Path,
    data2pt: str | Path | None = None,
) -> PointGeolocation:
    """
    Geolocate original-resolution radar points.

    Multilooked longitude/latitude rasters are sampled continuously at the
    original single-look row/column coordinates. No scientific point
    multilooking or nearest-MLI-cell collapse is performed.
    """

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if rows.shape != cols.shape:
        raise GeolocationError(
            "rows and cols must have identical shape."
        )

    if rows.ndim != 1:
        raise GeolocationError(
            "rows and cols must be one-dimensional."
        )

    n_points = int(rows.size)

    work_dir = Path(work_dir)
    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_list = build_ipta_point_list(
        cols,
        rows,
        work_dir / "strict_points.plist",
    )

    lon_output = (
        work_dir
        / "longitude_deg.gamma_pt"
    )
    lat_output = (
        work_dir
        / "latitude_deg.gamma_pt"
    )

    longitude = sample_radar_raster_at_points(
        source_raster=
            geometry.longitude_raster,
        geometry_par=
            geometry.geometry_par,
        point_list=
            point_list,
        reference_rslc_par=
            geometry.reference_rslc_par,
        output_path=
            lon_output,
        expected_count=
            n_points,
        data2pt=
            data2pt,
    )

    latitude = sample_radar_raster_at_points(
        source_raster=
            geometry.latitude_raster,
        geometry_par=
            geometry.geometry_par,
        point_list=
            point_list,
        reference_rslc_par=
            geometry.reference_rslc_par,
        output_path=
            lat_output,
        expected_count=
            n_points,
        data2pt=
            data2pt,
    )

    valid = (
        np.isfinite(longitude)
        &
        np.isfinite(latitude)
        &
        (longitude > -180.0)
        &
        (longitude < 180.0)
        &
        (latitude > -90.0)
        &
        (latitude < 90.0)
    )

    return PointGeolocation(
        longitude_deg=longitude,
        latitude_deg=latitude,
        valid_mask=valid,
        point_list=point_list,
        longitude_gamma_pt=lon_output,
        latitude_gamma_pt=lat_output,
    )
