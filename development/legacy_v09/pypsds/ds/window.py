from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from pypsds.gamma.geometry import RadarPixelGeometry, geometry_from_par


@dataclass(frozen=True, slots=True)
class ShpWindowSpec:
    half_window: tuple[int, int]
    mode: str
    geometry: RadarPixelGeometry | None
    search_radius_m: float | None
    search_shape: str = "rectangle"

    def offsets(self) -> np.ndarray:
        """Return dy/dx offsets for the actual maximum search support.

        For physical mode the default is an ellipse in approximate ground
        coordinates. This prevents a slant-range radar grid from turning a
        circular physical neighbourhood into a very large rectangular support.
        """
        hy, hx = self.half_window
        if self.search_shape != "ellipse" or self.geometry is None or self.search_radius_m is None:
            return np.asarray(
                [(dy, dx) for dy in range(-hy, hy + 1) for dx in range(-hx, hx + 1)],
                dtype=np.int16,
            )

        az = float(self.geometry.azimuth_spacing_m)
        rg = float(self.geometry.ground_range_spacing_m)
        r2 = float(self.search_radius_m) ** 2
        out: list[tuple[int, int]] = []
        for dy in range(-hy, hy + 1):
            for dx in range(-hx, hx + 1):
                d2 = (dy * az) ** 2 + (dx * rg) ** 2
                if d2 <= r2 + 1e-6:
                    out.append((dy, dx))
        return np.asarray(out, dtype=np.int16)


def _get(cfg: dict[str, Any], key: str, default=None):
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_shp_window(cfg: dict[str, Any], stack) -> ShpWindowSpec:
    """Resolve the SHP maximum search support for v0.5.

    Production default is a fixed *physical* radius with elliptical support in
    approximate ground coordinates. Statistical testing still decides which
    neighbours are SHPs; the ellipse only defines where candidates may occur.
    """
    mode = str(_get(cfg, "shp.window_mode", "physical_fixed")).lower()
    shape = str(_get(cfg, "shp.search_shape", "ellipse")).lower()
    if shape not in {"ellipse", "rectangle"}:
        raise ValueError("shp.search_shape must be ellipse or rectangle")

    if mode in {"fixed", "pixel_fixed"}:
        hy = int(_get(cfg, "shp.half_window_y", 5))
        hx = int(_get(cfg, "shp.half_window_x", 11))
        return ShpWindowSpec((hy, hx), mode, None, None, "rectangle")

    if mode == "adaptive":
        cands = _get(cfg, "shp.window_candidates", None)
        if cands:
            hy = max(int(v[0]) for v in cands)
            hx = max(int(v[1]) for v in cands)
        else:
            hy = int(_get(cfg, "shp.half_window_y", 5))
            hx = int(_get(cfg, "shp.half_window_x", 11))
        return ShpWindowSpec((hy, hx), mode, None, None, "rectangle")

    if mode not in {"physical", "physical_fixed"}:
        raise ValueError("shp.window_mode must be physical_fixed, fixed, or adaptive")

    if not stack.records:
        raise ValueError("Cannot infer physical SHP window without an acquisition")
    geom = geometry_from_par(stack.records[0].par)
    radius = float(_get(cfg, "shp.search_radius_m", 55.0))
    max_y = _get(cfg, "shp.max_half_window_y", None)
    max_x = _get(cfg, "shp.max_half_window_x", None)
    min_y = int(_get(cfg, "shp.min_half_window_y", 1))
    min_x = int(_get(cfg, "shp.min_half_window_x", 1))
    half = geom.half_window_for_radius(
        radius,
        max_half_window_y=(None if max_y in (None, "null") else int(max_y)),
        max_half_window_x=(None if max_x in (None, "null") else int(max_x)),
        min_half_window_y=min_y,
        min_half_window_x=min_x,
    )
    return ShpWindowSpec(half, mode, geom, radius, shape)
