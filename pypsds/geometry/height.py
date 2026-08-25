from __future__ import annotations

from pathlib import Path
from typing import Any

from .inputs import GeometryInputs
from .geolocation import sample_radar_raster_at_points


class HeightGeometryError(RuntimeError):
    """Invalid or unresolved radar-coordinate height geometry."""


def resolve_height_raster(
    cfg: dict[str, Any],
    paths,
    geometry: GeometryInputs,
) -> Path:
    """
    Resolve the GAMMA radar-coordinate terrain-height raster.

    Default validated convention:
        <paths.dem_dir>/<reference_date>.hgt

    Non-standard projects may set:
        geometry.height_raster
    """

    raw = (
        cfg.get("geometry", {})
        .get("height_raster")
    )

    dem_value = getattr(
        paths,
        "dem_dir",
        None,
    )

    if raw not in (None, "", "auto"):
        path = Path(raw).expanduser()

        if not path.is_absolute():
            if dem_value is None:
                base = Path(paths.data_dir)
            else:
                base = Path(dem_value)

            path = base / path

        path = path.resolve()

    else:
        if dem_value is None:
            raise HeightGeometryError(
                "paths.dem_dir is required to discover the "
                "radar-coordinate height raster."
            )

        path = (
            Path(dem_value).resolve()
            / f"{geometry.reference_date}.hgt"
        )

    if not path.is_file():
        raise HeightGeometryError(
            "Height raster not found: "
            f"{path}. Set geometry.height_raster explicitly "
            "for a non-standard GAMMA project."
        )

    size = path.stat().st_size

    if size != geometry.float32_expected_bytes:
        raise HeightGeometryError(
            "Height raster byte size does not match Geometry: "
            f"{size} != {geometry.float32_expected_bytes}: {path}"
        )

    return path


def sample_height_m(
    *,
    height_raster: str | Path,
    geometry: GeometryInputs,
    point_list: str | Path,
    output_path: str | Path,
    expected_count: int,
    data2pt: str | Path | None = None,
):
    """
    Sample validated GAMMA terrain height at strict radar points.

    Disk output follows data2pt (>f4); returned values are float64.
    """

    return sample_radar_raster_at_points(
        source_raster=height_raster,
        geometry_par=geometry.geometry_par,
        point_list=point_list,
        reference_rslc_par=geometry.reference_rslc_par,
        output_path=output_path,
        expected_count=expected_count,
        data2pt=data2pt,
    )
