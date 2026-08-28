from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .gamma_par import GammaParError, gamma_par_int, read_gamma_par


class RadarLonLatError(RuntimeError):
    """Radar-coordinate longitude/latitude generation failed."""


@dataclass(frozen=True, slots=True)
class RadarLonLatAssets:
    longitude_raster: Path
    latitude_raster: Path
    map_longitude_raster: Path
    map_latitude_raster: Path
    dem_parameter_file: Path
    lookup_table_file: Path
    geometry_par: Path
    map_width: int
    map_length: int
    radar_width: int
    radar_length: int
    generated_map_coordinates: bool
    generated_radar_coordinates: bool
    manifest: Path


def _cfg(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _resolve_optional(value: Any, *, base: Path, label: str) -> Path | None:
    if value in (None, "", "auto"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise RadarLonLatError(f"{label} does not exist: {path}")
    return path


def _which(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise RadarLonLatError(f"GAMMA command '{name}' is not available in PATH.")
    return Path(found).resolve()


def _run(command: list[str], *, log_path: Path, timeout_seconds: float) -> None:
    p = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=None if timeout_seconds <= 0.0 else timeout_seconds,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(command) + "\n")
        f.write(p.stdout or "")
        f.write(f"\nreturncode={p.returncode}\n")
    if p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-40:])
        raise RadarLonLatError(
            "GAMMA command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Last output:\n{tail}\nFull log: {log_path}"
        )


def _map_dims(path: Path) -> tuple[int, int]:
    v = read_gamma_par(path)
    try:
        width = gamma_par_int(v, "width")
        length = gamma_par_int(v, "nlines")
    except GammaParError as exc:
        raise RadarLonLatError(f"Invalid DEM parameter file {path}: {exc}") from exc
    if width <= 0 or length <= 0:
        raise RadarLonLatError(f"Invalid DEM dimensions: {length} x {width}")
    return width, length


def _radar_dims(path: Path) -> tuple[int, int]:
    v = read_gamma_par(path)
    try:
        width = gamma_par_int(v, "range_samples")
        length = gamma_par_int(v, "azimuth_lines")
    except GammaParError as exc:
        raise RadarLonLatError(f"Invalid radar geometry parameter file {path}: {exc}") from exc
    if width <= 0 or length <= 0:
        raise RadarLonLatError(f"Invalid radar dimensions: {length} x {width}")
    return width, length


def _unique(label: str, files) -> Path:
    paths = sorted({Path(x).resolve() for x in files if Path(x).is_file()})
    if not paths:
        raise RadarLonLatError(f"No candidate found for {label}.")
    if len(paths) > 1:
        raise RadarLonLatError(
            f"Ambiguous {label}; {len(paths)} candidates:\n" +
            "\n".join(f"  - {x}" for x in paths)
        )
    return paths[0]


def _lookup(cfg, dem_dir: Path, date: str) -> Path:
    explicit = _resolve_optional(
        _cfg(cfg, "geometry", "lonlat", "lookup_table"),
        base=dem_dir,
        label="lookup table",
    )
    if explicit is not None:
        return explicit
    fine = list(dem_dir.glob(f"*.{date}.lt_fine"))
    if fine:
        return _unique("refined lookup table", fine)
    return _unique("lookup table", dem_dir.glob(f"*.{date}.lt"))


def _segment(lookup: Path, date: str) -> str:
    marker = f".{date}."
    if marker not in lookup.name:
        raise RadarLonLatError(f"Cannot infer DEM segment from {lookup.name}")
    out = lookup.name.split(marker, 1)[0]
    if not out:
        raise RadarLonLatError(f"Empty DEM segment inferred from {lookup.name}")
    return out


def _dem_par(cfg, dem_dir: Path, segment: str) -> Path:
    explicit = _resolve_optional(
        _cfg(cfg, "geometry", "lonlat", "dem_parameter_file"),
        base=dem_dir,
        label="DEM parameter file",
    )
    if explicit is not None:
        return explicit
    path = dem_dir / f"{segment}.dem_par"
    if not path.is_file():
        raise RadarLonLatError(f"DEM segment parameter file not found: {path}")
    return path.resolve()


def _check(path: Path, width: int, length: int, label: str) -> None:
    expected = width * length * 4
    if not path.is_file():
        raise RadarLonLatError(f"Missing {label}: {path}")
    actual = path.stat().st_size
    if actual != expected:
        raise RadarLonLatError(
            f"{label} byte-size mismatch: {actual} != {expected} "
            f"({length} x {width} FLOAT)"
        )


def _stem(par: Path) -> str:
    n = par.name
    if n.endswith(".mli.par"):
        return n[:-8]
    if n.endswith(".par"):
        return n[:-4]
    return par.stem


def ensure_radar_lonlat(
    cfg: dict[str, Any],
    paths,
    *,
    reference_date: str,
    geometry_par: str | Path,
    force: bool = False,
) -> RadarLonLatAssets:
    """
    Ensure radar-coordinate lon/lat in the current auxiliary MLI geometry.

    These rasters are intentionally NOT expanded to the full 1x1 RSLC grid.
    The existing GAMMA data2pt path samples them later at single-look PS/DS
    point coordinates.
    """
    dem_value = getattr(paths, "dem_dir", None)
    if dem_value is None:
        raise RadarLonLatError("paths.dem_dir is required for lon/lat generation")
    dem_dir = Path(dem_value).resolve()
    geometry_par = Path(geometry_par).resolve()

    lookup = _lookup(cfg, dem_dir, reference_date)
    segment = _segment(lookup, reference_date)
    dem_par = _dem_par(cfg, dem_dir, segment)
    map_width, map_length = _map_dims(dem_par)
    radar_width, radar_length = _radar_dims(geometry_par)

    raw_out = _cfg(cfg, "geometry", "lonlat", "output_dir")
    if raw_out in (None, "", "auto"):
        out = Path(paths.output_dir) / "processing" / "geometry_assets"
    else:
        out = Path(str(raw_out)).expanduser()
        if not out.is_absolute():
            out = Path(paths.work_dir) / out
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Reuse valid map-coordinate lon/lat if the upstream DEM workflow already made them.
    dem_lon = dem_dir / f"{segment}.lon"
    dem_lat = dem_dir / f"{segment}.lat"
    reuse_map = False
    if not force and dem_lon.is_file() and dem_lat.is_file():
        try:
            _check(dem_lon, map_width, map_length, "map longitude")
            _check(dem_lat, map_width, map_length, "map latitude")
            reuse_map = True
        except RadarLonLatError:
            reuse_map = False

    if reuse_map:
        map_lon, map_lat = dem_lon.resolve(), dem_lat.resolve()
    else:
        map_lon, map_lat = out / f"{segment}.lon", out / f"{segment}.lat"

    timeout = float(_cfg(
        cfg, "phase_correction", "command_timeout_seconds", default=300.0
    ))
    log = out / "geometry_lonlat_gamma.log"
    generated_map = False

    if force or not (map_lon.is_file() and map_lat.is_file()):
        dem_coord = _which(str(_cfg(
            cfg, "geometry", "lonlat", "dem_coord_command", default="dem_coord"
        )))
        for p in (map_lon, map_lat):
            if p.exists(): p.unlink()
        # GAMMA: dem_coord DEM_par map.lon map.lat 0
        _run(
            [str(dem_coord), str(dem_par), str(map_lon), str(map_lat), "0"],
            log_path=log,
            timeout_seconds=timeout,
        )
        generated_map = True

    _check(map_lon, map_width, map_length, "map longitude")
    _check(map_lat, map_width, map_length, "map latitude")

    stem = _stem(geometry_par)
    radar_lon = out / f"{stem}.rdc.lon"
    radar_lat = out / f"{stem}.rdc.lat"
    generated_radar = False
    ready = False
    if not force and radar_lon.is_file() and radar_lat.is_file():
        try:
            _check(radar_lon, radar_width, radar_length, "radar longitude")
            _check(radar_lat, radar_width, radar_length, "radar latitude")
            ready = True
        except RadarLonLatError:
            ready = False

    if not ready:
        geocode = _which(str(_cfg(
            cfg, "geometry", "lonlat", "geocode_command", default="geocode"
        )))
        for p in (radar_lon, radar_lat):
            if p.exists(): p.unlink()
        # Same forward-geocode semantics already used by the RDC hgt workflow:
        # geocode LT input MAP_WIDTH output RADAR_WIDTH RADAR_LINES 2 0
        for src, dst in ((map_lon, radar_lon), (map_lat, radar_lat)):
            _run(
                [
                    str(geocode), str(lookup), str(src), str(map_width),
                    str(dst), str(radar_width), str(radar_length), "2", "0",
                ],
                log_path=log,
                timeout_seconds=timeout,
            )
        generated_radar = True

    _check(radar_lon, radar_width, radar_length, "radar longitude")
    _check(radar_lat, radar_width, radar_length, "radar latitude")

    manifest = out / "geometry_lonlat_manifest.json"
    payload = {
        "contract": "pyPSDS-GAMMA-v1.3-radar-lonlat-autogen",
        "reference_date": reference_date,
        "geometry_model": "auxiliary_multilook_radar_grid_sampled_to_singlelook_points",
        "longitude_file": str(radar_lon),
        "latitude_file": str(radar_lat),
        "map_longitude_file": str(map_lon),
        "map_latitude_file": str(map_lat),
        "dem_parameter_file": str(dem_par),
        "lookup_table_file": str(lookup),
        "radar_parameter_file": str(geometry_par),
        "map_width": map_width,
        "map_length": map_length,
        "radar_width": radar_width,
        "radar_length": radar_length,
        "generated_map_coordinates": generated_map,
        "generated_radar_coordinates": generated_radar,
        "sampling_to_psds_points": "GAMMA data2pt using single-look RSLC point coordinates",
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return RadarLonLatAssets(
        radar_lon.resolve(), radar_lat.resolve(), map_lon.resolve(), map_lat.resolve(),
        dem_par.resolve(), lookup.resolve(), geometry_par,
        map_width, map_length, radar_width, radar_length,
        generated_map, generated_radar, manifest.resolve(),
    )
