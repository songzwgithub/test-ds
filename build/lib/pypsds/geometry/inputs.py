from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .gamma_par import (
    GammaParError,
    gamma_par_int,
    read_gamma_par,
)

from .lonlat import (
    RadarLonLatError,
    ensure_radar_lonlat,
)


@dataclass(
    frozen=True,
    slots=True,
)
class GeometryInputs:
    reference_date: str

    reference_rslc_par: Path

    geometry_par: Path
    longitude_raster: Path
    latitude_raster: Path

    radar_width: int
    radar_length: int

    float32_expected_bytes: int


class GeometryInputError(RuntimeError):
    """Geometry input discovery or validation failed."""


def _cfg(
    cfg: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:

    value: Any = cfg

    for key in keys:

        if not isinstance(
            value,
            dict,
        ):
            return default

        if key not in value:
            return default

        value = value[key]

    return value


def _normalize_reference_date(
    value: Any,
) -> str:

    if value in (
        None,
        "",
        "auto",
    ):
        raise GeometryInputError(
            "Geometry reference date is not defined. "
            "Set geometry.reference_date or "
            "phase_correction.geometric_reference_date."
        )

    text = str(value)

    if not re.fullmatch(
        r"\d{8}",
        text,
    ):
        raise GeometryInputError(
            f"Invalid Geometry reference date: {text!r}"
        )

    return text


def _resolve_file(
    value: Any,
    *,
    base: Path,
    label: str,
) -> Path | None:

    if value in (
        None,
        "",
        "auto",
    ):
        return None

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = base / path

    path = path.resolve()

    if not path.is_file():
        raise GeometryInputError(
            f"{label} does not exist: {path}"
        )

    return path


def _require_unique(
    label: str,
    candidates,
) -> Path:

    files = sorted(
        {
            Path(path).resolve()
            for path in candidates
            if Path(path).is_file()
        }
    )

    if not files:
        raise GeometryInputError(
            f"No candidate found for {label}."
        )

    if len(files) > 1:
        formatted = "\n".join(
            f"  - {path}"
            for path in files
        )

        raise GeometryInputError(
            f"Ambiguous {label}; "
            f"{len(files)} candidates found:\n"
            f"{formatted}"
        )

    return files[0]


def _optional_unique(
    label: str,
    candidates,
) -> Path | None:

    files = sorted(
        {
            Path(path).resolve()
            for path in candidates
            if Path(path).is_file()
        }
    )

    if not files:
        return None

    if len(files) > 1:
        formatted = "\n".join(
            f"  - {path}"
            for path in files
        )

        raise GeometryInputError(
            f"Ambiguous {label}; "
            f"{len(files)} candidates found:\n"
            f"{formatted}"
        )

    return files[0]


def _raster_dimensions(
    geometry_par: Path,
) -> tuple[int, int]:

    values = read_gamma_par(
        geometry_par
    )

    try:
        width = gamma_par_int(
            values,
            "range_samples",
        )

        length = gamma_par_int(
            values,
            "azimuth_lines",
        )

    except GammaParError as exc:
        raise GeometryInputError(
            f"Invalid Geometry parameter file "
            f"{geometry_par}: {exc}"
        ) from exc

    if width <= 0 or length <= 0:
        raise GeometryInputError(
            f"Invalid radar geometry dimensions "
            f"{length} x {width} in {geometry_par}"
        )

    return (
        width,
        length,
    )


def _matches_rasters(
    geometry_par: Path,
    *,
    longitude_raster: Path,
    latitude_raster: Path,
) -> bool:

    try:
        width, length = (
            _raster_dimensions(
                geometry_par
            )
        )

    except GeometryInputError:
        return False

    expected = (
        width
        *
        length
        *
        4
    )

    return (
        longitude_raster.stat().st_size
        ==
        expected
        and
        latitude_raster.stat().st_size
        ==
        expected
    )


def resolve_geometry_inputs(
    cfg: dict[str, Any],
    paths,
) -> GeometryInputs:

    reference_value = _cfg(
        cfg,
        "geometry",
        "reference_date",
    )

    if reference_value in (
        None,
        "",
        "auto",
    ):
        reference_value = _cfg(
            cfg,
            "phase_correction",
            "geometric_reference_date",
        )

    reference_date = (
        _normalize_reference_date(
            reference_value
        )
    )


    rslc_dir = Path(
        paths.rslc_dir
    ).resolve()


    dem_value = getattr(
        paths,
        "dem_dir",
        None,
    )

    dem_dir = (
        Path(dem_value).resolve()
        if dem_value is not None
        else None
    )


    reference_rslc_par = (
        _resolve_file(
            _cfg(
                cfg,
                "geometry",
                "reference_rslc_par",
            ),
            base=rslc_dir,
            label="reference RSLC parameter",
        )
    )

    if reference_rslc_par is None:
        reference_rslc_par = (
            _require_unique(
                "reference RSLC parameter",
                rslc_dir.glob(
                    f"{reference_date}*.rslc.par"
                ),
            )
        )


    explicit_geometry_par = _cfg(
        cfg,
        "geometry",
        "geometry_par",
    )

    explicit_lon = _cfg(
        cfg,
        "geometry",
        "longitude_raster",
    )

    explicit_lat = _cfg(
        cfg,
        "geometry",
        "latitude_raster",
    )


    if dem_dir is None:

        if any(
            value in (
                None,
                "",
                "auto",
            )
            for value in (
                explicit_geometry_par,
                explicit_lon,
                explicit_lat,
            )
        ):
            raise GeometryInputError(
                "paths.dem_dir is required for automatic "
                "Geometry discovery."
            )

        dem_dir = Path(
            paths.data_dir
        ).resolve()



    longitude_raster = (
        _resolve_file(
            explicit_lon,
            base=dem_dir,
            label="longitude radar raster",
        )
    )

    latitude_raster = (
        _resolve_file(
            explicit_lat,
            base=dem_dir,
            label="latitude radar raster",
        )
    )


    # Explicit longitude/latitude must be supplied as a pair.
    if (
        longitude_raster is None
    ) != (
        latitude_raster is None
    ):
        if (
            explicit_lon
            not in (
                None,
                "",
                "auto",
            )
            or
            explicit_lat
            not in (
                None,
                "",
                "auto",
            )
        ):
            raise GeometryInputError(
                "geometry.longitude_raster and "
                "geometry.latitude_raster must be supplied together."
            )


    # If no explicit pair was configured, discover an existing
    # radar-coordinate pair if available.
    if (
        longitude_raster is None
        and
        latitude_raster is None
    ):

        longitude_raster = (
            _optional_unique(
                "longitude radar raster",
                dem_dir.glob(
                    f"{reference_date}*.rdc.lon"
                ),
            )
        )

        latitude_raster = (
            _optional_unique(
                "latitude radar raster",
                dem_dir.glob(
                    f"{reference_date}*.rdc.lat"
                ),
            )
        )


    geometry_par = (
        _resolve_file(
            explicit_geometry_par,
            base=dem_dir,
            label="Geometry MLI parameter",
        )
    )


    if geometry_par is None:

        # ----------------------------------------------------
        # Existing complete lon/lat pair:
        # preserve the original v1.3 semantic selection.
        # ----------------------------------------------------
        if (
            longitude_raster is not None
            and
            latitude_raster is not None
        ):

            matching = [
                candidate
                for candidate in dem_dir.glob(
                    f"{reference_date}*.mli.par"
                )
                if _matches_rasters(
                    candidate,
                    longitude_raster=
                        longitude_raster,
                    latitude_raster=
                        latitude_raster,
                )
            ]

            geometry_par = (
                _require_unique(
                    "Geometry MLI parameter",
                    matching,
                )
            )


        # ----------------------------------------------------
        # lon/lat missing:
        #
        # Geometry must be known BEFORE lon/lat can be
        # generated. First prefer a unique reference MLI par.
        #
        # If several exist, use the existing RDC height raster
        # size as the independent geometry discriminator.
        # ----------------------------------------------------
        else:

            candidates = sorted(
                {
                    candidate.resolve()
                    for candidate in dem_dir.glob(
                        f"{reference_date}*.mli.par"
                    )
                    if candidate.is_file()
                }
            )

            if len(candidates) == 1:
                geometry_par = candidates[0]

            else:

                height_candidates = sorted(
                    {
                        candidate.resolve()
                        for candidate in dem_dir.glob(
                            f"{reference_date}*.hgt"
                        )
                        if candidate.is_file()
                    }
                )

                height_matching = []

                for candidate in candidates:

                    try:
                        candidate_width, candidate_length = (
                            _raster_dimensions(
                                candidate
                            )
                        )

                    except GeometryInputError:
                        continue

                    candidate_bytes = (
                        candidate_width
                        *
                        candidate_length
                        *
                        4
                    )

                    if any(
                        height.stat().st_size
                        ==
                        candidate_bytes
                        for height in height_candidates
                    ):
                        height_matching.append(
                            candidate
                        )

                geometry_par = (
                    _require_unique(
                        "Geometry MLI parameter",
                        (
                            height_matching
                            if height_matching
                            else candidates
                        ),
                    )
                )


    # --------------------------------------------------------
    # Existing lon/lat are reused.
    #
    # If either raster is missing, generate a fresh matched
    # pair in the pyPSDS output tree.
    #
    # IMPORTANT:
    # generated lon/lat follow geometry_par, e.g. 4x1.
    # They are NOT expanded into a full-scene 1x1 raster.
    #
    # Single-look PS/DS rows/cols are sampled later by
    # GAMMA data2pt.
    # --------------------------------------------------------
    if (
        longitude_raster is None
        or
        latitude_raster is None
    ):

        auto_generate = bool(
            _cfg(
                cfg,
                "geometry",
                "lonlat",
                "auto_generate",
                default=True,
            )
        )

        if not auto_generate:
            raise GeometryInputError(
                "Radar-coordinate longitude/latitude raster "
                "is missing and "
                "geometry.lonlat.auto_generate=false."
            )

        try:
            assets = (
                ensure_radar_lonlat(
                    cfg,
                    paths,
                    reference_date=
                        reference_date,
                    geometry_par=
                        geometry_par,
                )
            )

        except RadarLonLatError as exc:
            raise GeometryInputError(
                "Automatic radar-coordinate lon/lat "
                f"generation failed: {exc}"
            ) from exc

        longitude_raster = (
            assets.longitude_raster
        )

        latitude_raster = (
            assets.latitude_raster
        )


    width, length = (
        _raster_dimensions(
            geometry_par
        )
    )

    expected_bytes = (
        width
        *
        length
        *
        4
    )


    lon_bytes = (
        longitude_raster
        .stat()
        .st_size
    )

    lat_bytes = (
        latitude_raster
        .stat()
        .st_size
    )


    if lon_bytes != expected_bytes:
        raise GeometryInputError(
            "Longitude raster byte size does not match "
            f"Geometry dimensions: "
            f"{lon_bytes} != {expected_bytes}"
        )


    if lat_bytes != expected_bytes:
        raise GeometryInputError(
            "Latitude raster byte size does not match "
            f"Geometry dimensions: "
            f"{lat_bytes} != {expected_bytes}"
        )


    return GeometryInputs(
        reference_date=
            reference_date,

        reference_rslc_par=
            reference_rslc_par,

        geometry_par=
            geometry_par,

        longitude_raster=
            longitude_raster,

        latitude_raster=
            latitude_raster,

        radar_width=
            width,

        radar_length=
            length,

        float32_expected_bytes=
            expected_bytes,
    )
