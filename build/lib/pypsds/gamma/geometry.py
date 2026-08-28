from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from .par import first_number, parse_gamma_parameter_file


@dataclass(frozen=True, slots=True)
class RadarPixelGeometry:
    """Approximate radar-grid pixel spacing used to define physical SHP windows."""

    azimuth_spacing_m: float
    range_spacing_m: float
    ground_range_spacing_m: float
    incidence_angle_deg: float | None
    image_geometry: str | None

    def half_window_for_radius(
        self,
        radius_m: float,
        *,
        max_half_window_y: int | None = None,
        max_half_window_x: int | None = None,
        min_half_window_y: int = 1,
        min_half_window_x: int = 1,
    ) -> tuple[int, int]:
        """Convert an approximate ground search half-width in metres to radar pixels.

        The result is a rectangular *maximum search envelope*. Statistical
        homogeneity still determines which neighbors actually enter covariance
        estimation.
        """
        if radius_m <= 0:
            raise ValueError("radius_m must be > 0")
        hy = max(int(min_half_window_y), int(math.ceil(radius_m / self.azimuth_spacing_m)))
        hx = max(int(min_half_window_x), int(math.ceil(radius_m / self.ground_range_spacing_m)))
        if max_half_window_y is not None:
            hy = min(hy, int(max_half_window_y))
        if max_half_window_x is not None:
            hx = min(hx, int(max_half_window_x))
        return hy, hx


def geometry_from_par(path: str | Path) -> RadarPixelGeometry:
    p = Path(path)
    params = parse_gamma_parameter_file(p)
    az = first_number(params, ("azimuth_pixel_spacing", "azimuth_spacing"))
    rg = first_number(params, ("range_pixel_spacing", "range_spacing"))
    inc = first_number(params, ("incidence_angle", "center_incidence_angle"))
    geom_vals = params.get("image_geometry", [])
    image_geometry = geom_vals[0] if geom_vals else None

    if az is None or az <= 0:
        raise ValueError(f"Cannot determine positive azimuth pixel spacing from {p}")
    if rg is None or rg <= 0:
        raise ValueError(f"Cannot determine positive range pixel spacing from {p}")

    ground_rg = float(rg)
    if image_geometry and "SLANT" in image_geometry.upper() and inc is not None:
        s = math.sin(math.radians(float(inc)))
        if s > 1e-6:
            ground_rg = float(rg) / s

    return RadarPixelGeometry(
        azimuth_spacing_m=float(az),
        range_spacing_m=float(rg),
        ground_range_spacing_m=float(ground_rg),
        incidence_angle_deg=(None if inc is None else float(inc)),
        image_geometry=image_geometry,
    )
