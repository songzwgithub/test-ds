from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np

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

    No machine-specific fallback path is used.
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
    Write the GAMMA/IPTA point list used by validated v4.

    Column 0 = range pixel  = col
    Column 1 = azimuth line = row

    Coordinates remain 0-based.

    Binary representation:
        big-endian signed int32 (>i4)
    """

    cols = np.asarray(
        cols
    )

    rows = np.asarray(
        rows
    )

    if cols.ndim != 1 or rows.ndim != 1:
        raise GeolocationError(
            "cols and rows must be one-dimensional arrays."
        )

    if cols.shape != rows.shape:
        raise GeolocationError(
            "cols and rows must have identical shape."
        )

    if not (
        np.issubdtype(
            cols.dtype,
            np.integer,
        )
        and
        np.issubdtype(
            rows.dtype,
            np.integer,
        )
    ):
        raise GeolocationError(
            "cols and rows must contain integer radar coordinates."
        )


    i32 = np.iinfo(
        np.int32
    )

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


    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    np.column_stack(
        (
            cols,
            rows,
        )
    ).astype(
        ">i4",
        copy=False,
    ).tofile(
        output
    )


    expected_bytes = (
        cols.size
        *
        8
    )

    actual_bytes = (
        output.stat().st_size
    )

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
    Read GAMMA data2pt output.

    Validated v4 semantics:
        big-endian float32 -> float64
    """

    path = Path(
        path
    )

    if not path.is_file():
        raise GeolocationError(
            f"Missing data2pt output: {path}"
        )


    values = np.fromfile(
        path,
        dtype=">f4",
    ).astype(
        np.float64
    )


    if values.size != expected_count:
        raise GeolocationError(
            "Unexpected data2pt output count: "
            f"{values.size} != {expected_count}"
        )

    return values


def sample_radar_raster_at_points(
    *,
    source_raster: str | Path,
    geometry_par: str | Path,
    point_list: str | Path,
    reference_rslc_par: str | Path,
    output_path: str | Path,
    expected_count: int,
    data2pt: str | Path | None = None,
) -> np.ndarray:
    """
    Sample one GAMMA radar-coordinate raster with data2pt.

    Command semantics are frozen from validated v4:

        data2pt
            source_raster
            geometry_par
            point_list
            reference_rslc_par
            output
            1
            2
    """

    executable = resolve_data2pt(
        data2pt
    )

    source_raster = Path(
        source_raster
    ).resolve()

    geometry_par = Path(
        geometry_par
    ).resolve()

    point_list = Path(
        point_list
    ).resolve()

    reference_rslc_par = Path(
        reference_rslc_par
    ).resolve()

    output_path = Path(
        output_path
    ).resolve()


    for label, path in (
        (
            "source raster",
            source_raster,
        ),
        (
            "geometry parameter",
            geometry_par,
        ),
        (
            "IPTA point list",
            point_list,
        ),
        (
            "reference RSLC parameter",
            reference_rslc_par,
        ),
    ):
        if not path.is_file():
            raise GeolocationError(
                f"Missing {label}: {path}"
            )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    command = [
        str(executable),
        str(source_raster),
        str(geometry_par),
        str(point_list),
        str(reference_rslc_par),
        str(output_path),
        "1",
        "2",
    ]


    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


    if process.returncode != 0:
        output = (
            process.stdout
            or ""
        )

        raise GeolocationError(
            "GAMMA data2pt failed with "
            f"return code {process.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"{output}"
        )


    return read_gamma_point_values(
        output_path,
        expected_count=
            expected_count,
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

    This function performs no multilooking and no scientific
    point-to-raster conversion.
    """

    rows = np.asarray(
        rows
    )

    cols = np.asarray(
        cols
    )

    if rows.shape != cols.shape:
        raise GeolocationError(
            "rows and cols must have identical shape."
        )

    if rows.ndim != 1:
        raise GeolocationError(
            "rows and cols must be one-dimensional."
        )


    n_points = int(
        rows.size
    )

    work_dir = Path(
        work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    point_list = build_ipta_point_list(
        cols,
        rows,
        work_dir
        / "strict_points.plist",
    )


    lon_output = (
        work_dir
        / "longitude_deg.gamma_pt"
    )

    lat_output = (
        work_dir
        / "latitude_deg.gamma_pt"
    )


    longitude = (
        sample_radar_raster_at_points(
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
    )


    latitude = (
        sample_radar_raster_at_points(
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
    )


    valid = (
        np.isfinite(
            longitude
        )
        &
        np.isfinite(
            latitude
        )
        &
        (
            longitude
            >
            -180.0
        )
        &
        (
            longitude
            <
            180.0
        )
        &
        (
            latitude
            >
            -90.0
        )
        &
        (
            latitude
            <
            90.0
        )
    )


    return PointGeolocation(
        longitude_deg=
            longitude,

        latitude_deg=
            latitude,

        valid_mask=
            valid,

        point_list=
            point_list,

        longitude_gamma_pt=
            lon_output,

        latitude_gamma_pt=
            lat_output,
    )
