from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from pypsds.config import cfg_get
from pypsds.progress import log
from .par import first_int, parse_gamma_parameter_file


class PhaseCorrectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PhaseCorrectionAssets:
    reference_date: str
    reference_index: int
    reference_par: Path
    height_path: Path
    height_geometry_par: Path
    pslc_par: Path
    itab: Path
    pair_secondary_indices: tuple[int, ...]
    scratch_dir: Path


@dataclass(frozen=True, slots=True)
class CorrectionTileStats:
    n_points: int
    n_valid_height: int
    height_seconds: float
    simulation_seconds: float
    total_seconds: float
    phase_min: float
    phase_max: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_path(raw: Any, bases: list[Path]) -> Path | None:
    if raw in (None, "", "auto"):
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    for base in bases:
        q = (base / p).resolve()
        if q.exists():
            return q
    return (bases[0] / p).resolve()


def _par_shape(path: Path) -> tuple[int, int] | None:
    try:
        params = parse_gamma_parameter_file(path)
    except Exception:
        return None
    width = first_int(params, ("range_samples", "width", "range_samp_1"))
    length = first_int(params, ("azimuth_lines", "nlines", "az_samp_1"))
    if width is None or length is None:
        return None
    return length, width


def _discover_height(cfg: dict[str, Any], paths, stack) -> tuple[Path, Path]:
    bases = [paths.work_dir, paths.data_dir]
    path = _resolve_path(cfg_get(cfg, "phase_correction.radar_height.path", None), bases)
    par = _resolve_path(cfg_get(cfg, "phase_correction.radar_height.geometry_par", None), bases)

    search_dirs = []
    for d in (
        paths.work_dir,
        paths.data_dir / "DEM_prep",
        paths.data_dir / "DEM",
        paths.data_dir,
    ):
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)

    # ------------------------------------------------------------
    # Prefer the public Geometry height contract before broad
    # legacy discovery.
    #
    # Priority:
    #   1. phase_correction.radar_height.path
    #   2. geometry.height_raster
    #   3. <dem_dir>/<reference_date>.hgt
    #   4. legacy broad discovery
    #
    # This prevents auxiliary products such as
    # <reference_date>.pixel_area.hgt from being mistaken for
    # the terrain-height raster.
    # ------------------------------------------------------------

    if path is None:
        geometry_height = cfg_get(
            cfg,
            "geometry.height_raster",
            None,
        )

        if geometry_height not in (None, "", "auto"):
            dem_dir = getattr(paths, "dem_dir", None)

            bases = []

            if dem_dir is not None:
                bases.append(Path(dem_dir))

            bases.extend(
                [
                    Path(paths.work_dir),
                    Path(paths.data_dir),
                ]
            )

            path = _resolve_path(
                geometry_height,
                bases,
            )

    if path is None:
        reference_date = cfg_get(
            cfg,
            "reference_date",
            None,
        )

        if reference_date in (None, "", "auto"):
            reference_date = cfg_get(
                cfg,
                "geometry.reference_date",
                cfg_get(
                    cfg,
                    "phase_correction.geometric_reference_date",
                    None,
                ),
            )

        dem_dir = getattr(
            paths,
            "dem_dir",
            None,
        )

        if (
            reference_date not in (None, "", "auto")
            and dem_dir is not None
        ):
            canonical = (
                Path(dem_dir).resolve()
                / f"{reference_date}.hgt"
            )

            if canonical.is_file():
                path = canonical

    if path is None:
        candidates: list[Path] = []
        patterns = ("*hgt*", "*.hgt", "*dem*.rdc", "*height*")
        for d in search_dirs:
            for pattern in patterns:
                for p in d.glob(pattern):
                    if p.is_file() and not p.name.lower().endswith((".par", ".json", ".txt", ".png", ".bmp", ".ras", ".tif", ".tiff", ".jpg", ".jpeg")):
                        if p.stat().st_size >= 4:
                            candidates.append(p.resolve())
        # unique, stable order
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            path = candidates[0]
        elif not candidates:
            raise PhaseCorrectionError(
                "No radar-coordinate height raster was found. Set "
                "phase_correction.radar_height.path in pypsds.yaml."
            )
        else:
            names = "\n  - ".join(str(p) for p in candidates[:20])
            raise PhaseCorrectionError(
                "Multiple radar-height candidates were found; choose one explicitly with "
                f"phase_correction.radar_height.path:\n  - {names}"
            )

    if not path.is_file():
        raise PhaseCorrectionError(f"Radar height raster not found: {path}")

    if par is None:
        # data2pt needs the SLC/MLI parameter file defining the multilooked height geometry.
        # Match candidate parameter files by raster byte size (FLOAT = 4 bytes/pixel).
        matching: list[Path] = []
        for d in [path.parent, *search_dirs]:
            if not d.is_dir():
                continue
            for q in d.glob("*.par"):
                shp = _par_shape(q)
                if shp is not None and shp[0] * shp[1] * 4 == path.stat().st_size:
                    matching.append(q.resolve())
        matching = list(dict.fromkeys(matching))
        preferred = [q for q in matching if any(s in q.name.lower() for s in ("mli", "rmli", "hgt"))]
        pool = preferred or matching
        if len(pool) == 1:
            par = pool[0]
        elif not pool:
            raise PhaseCorrectionError(
                f"Could not discover an SLC/MLI parameter file matching {path}. "
                "Set phase_correction.radar_height.geometry_par explicitly."
            )
        else:
            names = "\n  - ".join(str(p) for p in pool[:20])
            raise PhaseCorrectionError(
                "Multiple height-geometry parameter files match the raster. Set "
                f"phase_correction.radar_height.geometry_par explicitly:\n  - {names}"
            )

    if not par.is_file():
        raise PhaseCorrectionError(f"Height geometry parameter file not found: {par}")

    shape = _par_shape(par)
    if shape is None:
        raise PhaseCorrectionError(f"Cannot read width/length from height geometry parameter file: {par}")
    expected = shape[0] * shape[1] * 4
    if path.stat().st_size != expected:
        raise PhaseCorrectionError(
            f"Height raster size mismatch: {path.stat().st_size} bytes != {expected} "
            f"for geometry {shape[0]}x{shape[1]} FLOAT"
        )
    return path, par


def _which(name_or_path: str) -> str:
    p = Path(name_or_path).expanduser()
    if p.parent != Path(".") or "/" in name_or_path:
        if not p.is_file() or not os.access(p, os.X_OK):
            raise PhaseCorrectionError(f"GAMMA executable not executable: {p}")
        return str(p.resolve())
    found = shutil.which(name_or_path)
    if found is None:
        raise PhaseCorrectionError(
            f"GAMMA command '{name_or_path}' not found in PATH. "
            "Load the GAMMA environment or set its explicit path in phase_correction.commands."
        )
    return found


def _run_command_once(
    cmd: list[str],
    *,
    log_file: Path,
    label: str,
) -> float:
    t0 = perf_counter()
    command_text = " ".join(cmd)

    raw_timeout = os.environ.get(
        "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS",
        "300",
    )

    try:
        timeout_seconds = float(
            raw_timeout
        )
    except ValueError as exc:
        raise PhaseCorrectionError(
            "invalid "
            "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS="
            f"{raw_timeout!r}"
        ) from exc

    if timeout_seconds <= 0:
        timeout_seconds = None

    log(
        f"GAMMA START {label}: "
        f"{command_text}"
    )

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed = (
            perf_counter()
            -
            t0
        )

        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            exc.stdout
            or exc.output
            or ""
        )

        if isinstance(
            output,
            bytes,
        ):
            output = output.decode(
                "utf-8",
                errors="replace",
            )

        with log_file.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                "\n===== "
                f"{label} TIMEOUT =====\n"
            )
            f.write(
                f"$ {command_text}\n"
            )
            f.write(str(output))
            f.write(
                "\ntimeout_seconds="
                f"{timeout_seconds} "
                f"elapsed={elapsed:.3f}s\n"
            )

        log(
            "GAMMA TIMEOUT "
            f"{label}: "
            f"elapsed={elapsed:.2f}s "
            f"limit={timeout_seconds}s"
        )

        raise PhaseCorrectionError(
            "GAMMA command timeout "
            f"({label}) after "
            f"{elapsed:.1f}s "
            f"(limit={timeout_seconds}s).\n"
            f"Command: {command_text}\n"
            f"Full log: {log_file}"
        ) from exc

    elapsed = (
        perf_counter()
        -
        t0
    )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_file.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"\n===== {label} =====\n"
            f"$ {command_text}\n"
        )
        f.write(
            proc.stdout
            or ""
        )
        f.write(
            "\nreturncode="
            f"{proc.returncode} "
            f"elapsed={elapsed:.3f}s\n"
        )

    if proc.returncode != 0:
        tail = "\n".join(
            (
                proc.stdout
                or ""
            ).splitlines()[-30:]
        )

        raise PhaseCorrectionError(
            "GAMMA command failed "
            f"({label}, "
            f"returncode={proc.returncode}).\n"
            f"Command: {command_text}\n"
            f"Last output:\n{tail}\n"
            f"Full log: {log_file}"
        )

    log(
        f"GAMMA END   {label}: "
        f"elapsed={elapsed:.2f}s"
    )

    return elapsed

def _run_command(
    cmd: list[str],
    *,
    log_file: Path,
    label: str,
) -> float:
    # FASTPATCH bounded retry wrapper around upstream GAMMA runner.
    raw_retries = os.environ.get(
        "PYPSDS_GAMMA_COMMAND_RETRIES",
        "0",
    )
    raw_backoff = os.environ.get(
        "PYPSDS_GAMMA_RETRY_BACKOFF_SECONDS",
        "1.0",
    )

    try:
        retries = max(
            0,
            int(raw_retries),
        )
        backoff = max(
            0.0,
            float(raw_backoff),
        )
    except ValueError as exc:
        raise PhaseCorrectionError(
            "invalid GAMMA retry configuration: "
            f"retries={raw_retries!r}, "
            f"backoff={raw_backoff!r}"
        ) from exc

    total_t0 = perf_counter()
    last_exc = None

    for attempt in range(
        retries + 1
    ):
        try:
            _run_command_once(
                cmd,
                log_file=log_file,
                label=label,
            )

        except PhaseCorrectionError as exc:
            last_exc = exc

            if attempt >= retries:
                raise

            delay = (
                backoff
                *
                (2 ** attempt)
            )

            log(
                f"GAMMA RETRY {label}: "
                f"attempt={attempt + 1}/"
                f"{retries + 1} failed; "
                f"sleep={delay:.2f}s"
            )

            if delay > 0:
                time.sleep(
                    delay
                )

        else:
            return (
                perf_counter()
                -
                total_t0
            )

    assert last_exc is not None
    raise last_exc


class GammaPointPhaseCorrectionProvider:
    """Tile-local GAMMA orbit/topography phase correction using IPTA point tools.

    The provider keeps the user's multilooked radar-coordinate height raster.
    `data2pt` samples that raster at the 1x1 RSLC point-list coordinates, and
    `phase_sim_orb_pt` computes orbit/terrain phase at those exact points.
    No full-resolution 1x1 DEM or full-resolution simulated-phase stack is made.
    """

    def __init__(self, cfg: dict[str, Any], paths, stack):
        self.cfg = cfg
        self.paths = paths
        self.stack = stack
        self.enabled = bool(cfg_get(cfg, "phase_correction.enabled", True))
        self.sign = float(cfg_get(cfg, "phase_correction.apply_sign", 1.0))
        self.keep_tile_files = bool(cfg_get(cfg, "phase_correction.keep_tile_files", False))

        self.command_timeout_seconds = float(
            cfg_get(
                cfg,
                "phase_correction.command_timeout_seconds",
                300.0,
            )
        )

        if (
            self.command_timeout_seconds
            <
            0.0
        ):
            raise PhaseCorrectionError(
                "phase_correction.command_timeout_seconds "
                "must be >= 0"
            )

        os.environ[
            "PYPSDS_GAMMA_COMMAND_TIMEOUT_SECONDS"
        ] = str(
            self.command_timeout_seconds
        )

        # FASTPATCH: retry transient external GAMMA failures.
        self.command_retries = max(
            0,
            int(
                cfg_get(
                    cfg,
                    "phase_correction.command_retries",
                    2,
                )
            ),
        )
        self.retry_backoff_seconds = max(
            0.0,
            float(
                cfg_get(
                    cfg,
                    "phase_correction.retry_backoff_seconds",
                    1.0,
                )
            ),
        )

        os.environ[
            "PYPSDS_GAMMA_COMMAND_RETRIES"
        ] = str(
            self.command_retries
        )
        os.environ[
            "PYPSDS_GAMMA_RETRY_BACKOFF_SECONDS"
        ] = str(
            self.retry_backoff_seconds
        )

        self.zero_height_is_valid = bool(
            cfg_get(cfg, "phase_correction.radar_height.zero_height_is_valid", False)
        )
        self.zero_height_epsilon_m = float(
            cfg_get(cfg, "phase_correction.radar_height.zero_height_epsilon_m", 0.001)
        )
        scratch_raw = cfg_get(cfg, "phase_correction.scratch_dir", "phase_correction")
        scratch = Path(str(scratch_raw)).expanduser()
        if not scratch.is_absolute():
            scratch = paths.output_dir / scratch
        self.scratch_dir = scratch.resolve()
        self.assets: PhaseCorrectionAssets | None = None

    def _resolve_reference(self) -> tuple[int, str]:
        raw = cfg_get(
            self.cfg,
            "phase_correction.geometric_reference_date",
            None,
        )
        if raw in (None, ""):
            raise PhaseCorrectionError(
                "phase_correction.geometric_reference_date is required. "
                "Set the actual GAMMA co-registration reference date (YYYYMMDD), "
                "or explicitly set 'auto' to opt in to temporal-reference fallback."
            )
        if str(raw).lower() == "auto":
            idx = int(
                cfg_get(
                    self.cfg,
                    "phase_linking.temporal_reference_index",
                    cfg_get(
                        self.cfg,
                        "phase_linking.reference_idx",
                        0,
                    ),
                )
            )
            if idx < 0:
                idx += len(self.stack.records)
            if not 0 <= idx < len(self.stack.records):
                raise PhaseCorrectionError("phase_linking.temporal_reference_index outside stack")
            log(
                "phase_correction.geometric_reference_date=auto: using the date at "
                f"phase_linking.reference_idx={idx} ({self.stack.dates[idx]}). "
                "Verify that this fallback matches the GAMMA co-registration reference."
            )
            return idx, self.stack.dates[idx]
        date = str(raw)
        if date not in self.stack.dates:
            raise PhaseCorrectionError(
                f"Geometric reference date {date} is not in RSLC_tab. Available: {self.stack.dates}"
            )
        return self.stack.dates.index(date), date

    def prepare(self, *, force: bool = False) -> PhaseCorrectionAssets:
        if not self.enabled:
            raise PhaseCorrectionError("Phase correction provider is disabled")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        (self.scratch_dir / "tiles").mkdir(parents=True, exist_ok=True)
        gamma_log = self.scratch_dir / "gamma.log"

        slc2pt = _which(str(cfg_get(self.cfg, "phase_correction.commands.SLC2pt", "SLC2pt")))
        data2pt = _which(str(cfg_get(self.cfg, "phase_correction.commands.data2pt", "data2pt")))
        phase_sim = _which(
            str(cfg_get(self.cfg, "phase_correction.commands.phase_sim_orb_pt", "phase_sim_orb_pt"))
        )
        self._commands = {"SLC2pt": slc2pt, "data2pt": data2pt, "phase_sim_orb_pt": phase_sim}

        ref_idx, ref_date = self._resolve_reference()
        ref_par = self.stack.records[ref_idx].par.resolve()
        height_path, height_par = _discover_height(self.cfg, self.paths, self.stack)

        pslc_par = self.scratch_dir / "pSLC_par"
        dummy_plist = self.scratch_dir / "dummy.plist"
        itab = self.scratch_dir / "reference_to_all.itab"
        gamma_rslc_tab = self.scratch_dir / "RSLC_tab.absolute"
        manifest_path = self.scratch_dir / "manifest.json"

        # GAMMA scripts resolve relative entries in RSLC_tab against the current
        # shell working directory. Python already resolved all records safely, so
        # production always emits an internal absolute-path tab for GAMMA commands.
        gamma_rslc_tab.write_text(
            "".join(f"{r.rslc.resolve()} {r.par.resolve()}\n" for r in self.stack.records),
            encoding="utf-8",
        )
        np.asarray([[0, 0]], dtype=">i4").tofile(dummy_plist)
        pair_secondary = tuple(i for i in range(len(self.stack.records)) if i != ref_idx)
        with itab.open("w", encoding="utf-8") as f:
            for rec, sec_idx in enumerate(pair_secondary, 1):
                f.write(f"{ref_idx + 1} {sec_idx + 1} {rec} 1\n")

        desired_manifest = {
            "version": "1.0.0",
            "source_rslc_tab": str(self.paths.rslc_tab),
            "gamma_absolute_tab": str(gamma_rslc_tab),
            "gamma_absolute_tab_sha256": _sha256(gamma_rslc_tab),
            "dates": list(self.stack.dates),
            "geometric_reference_date": ref_date,
            "geometric_reference_index": ref_idx,
            "height_path": str(height_path),
            "height_size": height_path.stat().st_size,
            "height_geometry_par": str(height_par),
            "height_geometry_par_sha256": _sha256(height_par),
            # apply_sign is deliberately not part of preparation cache:
            # SLC2pt/pSLC_par/itab do not depend on the sign used later.
            "commands": self._commands,
        }

        reuse = False
        if not force and manifest_path.is_file() and pslc_par.is_file() and pslc_par.stat().st_size > 0:
            try:
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                reuse = old == desired_manifest
            except Exception:
                reuse = False

        if reuse:
            log(f"Phase-correction preparation cache valid: {self.scratch_dir}")
        else:
            if pslc_par.exists():
                pslc_par.unlink()
            _run_command(
                [slc2pt, str(gamma_rslc_tab), str(dummy_plist), "-", str(pslc_par), "-", "-"],
                log_file=gamma_log,
                label="SLC2pt:create-pSLC_par",
            )
            if not pslc_par.is_file() or pslc_par.stat().st_size == 0:
                raise PhaseCorrectionError(f"SLC2pt did not create a valid pSLC_par: {pslc_par}")
            manifest_path.write_text(json.dumps(desired_manifest, indent=2), encoding="utf-8")
            log(f"Prepared phase-correction assets: {self.scratch_dir}")

        assets = PhaseCorrectionAssets(
            reference_date=ref_date,
            reference_index=ref_idx,
            reference_par=ref_par,
            height_path=height_path,
            height_geometry_par=height_par,
            pslc_par=pslc_par,
            itab=itab,
            pair_secondary_indices=pair_secondary,
            scratch_dir=self.scratch_dir,
        )
        self.assets = assets
        return assets

    def _ensure_assets(self) -> PhaseCorrectionAssets:
        return self.assets if self.assets is not None else self.prepare()

    def phase_sim_worker_count(
        self,
        n_pairs: int,
    ) -> int:
        """
        Number of independent phase_sim_orb_pt processes.

        phase_sim_orb_pt itself is effectively single-core for the
        present workload. Reference-secondary pairs are independent,
        therefore parallelism is applied across pair groups.

        Current 32-CPU benchmark:
            1  worker : 21.247 s
            8  workers:  4.001 s
            16 workers:  3.517 s

        16 workers reproduced the serial FLOAT output bit-exactly.

        Config:
            phase_correction.phase_sim_workers: auto | integer
        """

        if n_pairs <= 0:
            return 1

        # Runtime phase sources may explicitly provide a
        # machine-autotuned pair-parallel worker count.
        override = getattr(
            self,
            "_phase_sim_workers_override",
            None,
        )

        if override is not None:

            return min(
                max(
                    1,
                    int(override),
                ),
                n_pairs,
            )

        raw = cfg_get(
            self.cfg,
            "phase_correction.phase_sim_workers",
            "auto",
        )

        if raw in (
            None,
            "",
            "auto",
        ):

            cpu = max(
                1,
                int(
                    os.cpu_count()
                    or
                    1
                ),
            )

            # Current benchmark optimum on 32 CPUs.
            workers = min(
                16,
                cpu,
                n_pairs,
            )

        else:

            workers = min(
                max(
                    1,
                    int(raw),
                ),
                n_pairs,
            )

        return int(
            workers
        )


    def _prepare_cached_geometry(
        self,
        *,
        assets,
        global_row0: int,
        global_col0: int,
        rows: int,
        cols: int,
        plist: Path,
        label: str,
    ):
        """
        Prepare or reuse date-independent GAMMA geometry.

        `data2pt` depends on spatial coordinates, radar height and
        the geometric reference acquisition, but not on the current
        temporal ministack.

        P11B-3 therefore persists:

            phgt
            pmask

        per canonical spatial cell.

        The cached phgt remains the exact GAMMA big-endian FLOAT
        byte stream used by phase_sim_orb_pt.
        """

        global_row0 = int(
            global_row0
        )

        global_col0 = int(
            global_col0
        )

        rows = int(
            rows
        )

        cols = int(
            cols
        )

        global_row1 = (
            global_row0
            +
            rows
        )

        global_col1 = (
            global_col0
            +
            cols
        )

        npoints = (
            rows
            *
            cols
        )

        cache_root = (
            self.scratch_dir
            /
            "geometry_cache"
        )

        cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        cell_dir = (
            cache_root
            /
            (
                f"r{global_row0}_"
                f"{global_row1}_"
                f"c{global_col0}_"
                f"{global_col1}"
            )
        )

        cell_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        phgt = (
            cell_dir
            /
            "phgt"
        )

        pmask = (
            cell_dir
            /
            "pmask"
        )

        manifest_path = (
            cell_dir
            /
            "manifest.json"
        )

        # ---------------------------------------------------------
        # Geometry source identity
        # ---------------------------------------------------------

        source_signature = getattr(
            self,
            "_geometry_source_signature_cache",
            None,
        )

        if source_signature is None:

            height_stat = (
                assets.height_path.stat()
            )

            data2pt_cmd = Path(
                str(
                    self._commands[
                        "data2pt"
                    ]
                )
            )

            if data2pt_cmd.is_file():

                cmd_stat = (
                    data2pt_cmd.stat()
                )

                cmd_identity = {
                    "path":
                        str(
                            data2pt_cmd.resolve()
                        ),

                    "size":
                        int(
                            cmd_stat.st_size
                        ),

                    "mtime_ns":
                        int(
                            cmd_stat.st_mtime_ns
                        ),
                }

            else:

                cmd_identity = {
                    "path":
                        str(
                            self._commands[
                                "data2pt"
                            ]
                        )
                }

            source_signature = {
                "reference_date":
                    str(
                        assets.reference_date
                    ),

                "reference_par":
                    str(
                        assets.reference_par.resolve()
                    ),

                "reference_par_sha256":
                    _sha256(
                        assets.reference_par
                    ),

                "height_path":
                    str(
                        assets.height_path.resolve()
                    ),

                "height_size":
                    int(
                        height_stat.st_size
                    ),

                "height_mtime_ns":
                    int(
                        height_stat.st_mtime_ns
                    ),

                "height_geometry_par":
                    str(
                        assets
                        .height_geometry_par
                        .resolve()
                    ),

                "height_geometry_par_sha256":
                    _sha256(
                        assets.height_geometry_par
                    ),

                "zero_height_is_valid":
                    bool(
                        self.zero_height_is_valid
                    ),

                "zero_height_epsilon_m":
                    float(
                        self.zero_height_epsilon_m
                    ),

                "data2pt":
                    cmd_identity,
            }

            self._geometry_source_signature_cache = (
                source_signature
            )

        expected_manifest = {
            "format":
                "pyPSDS-GAMMA-phase-geometry-cache-v1",

            "global_row0":
                global_row0,

            "global_row1":
                global_row1,

            "global_col0":
                global_col0,

            "global_col1":
                global_col1,

            "rows":
                rows,

            "cols":
                cols,

            "npoints":
                npoints,

            "source":
                source_signature,
        }

        # ---------------------------------------------------------
        # Existing-cache validation
        # ---------------------------------------------------------

        cache_valid = False

        if (
            manifest_path.is_file()
            and
            phgt.is_file()
            and
            pmask.is_file()
        ):

            try:

                old_manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                cache_valid = (
                    old_manifest
                    ==
                    expected_manifest
                    and
                    phgt.stat().st_size
                    ==
                    npoints * 4
                    and
                    pmask.stat().st_size
                    ==
                    npoints
                )

            except Exception:

                cache_valid = False

        if cache_valid:

            mask_u8 = np.fromfile(
                pmask,
                dtype=np.uint8,
            )

            if (
                mask_u8.size
                ==
                npoints
                and
                np.all(
                    (
                        mask_u8
                        ==
                        0
                    )
                    |
                    (
                        mask_u8
                        ==
                        1
                    )
                )
            ):

                valid = (
                    mask_u8.astype(
                        np.bool_,
                        copy=False,
                    )
                )

                log(
                    "Geometry cache HIT "
                    f"r{global_row0}:"
                    f"{global_row1} "
                    f"c{global_col0}:"
                    f"{global_col1}"
                )

                return (
                    phgt,
                    pmask,
                    valid,
                    0.0,
                    True,
                )

        # ---------------------------------------------------------
        # MISS: run data2pt exactly once.
        # ---------------------------------------------------------

        log(
            "Geometry cache MISS "
            f"r{global_row0}:"
            f"{global_row1} "
            f"c{global_col0}:"
            f"{global_col1}"
        )

        pid = os.getpid()

        phgt_tmp = (
            cell_dir
            /
            f"phgt.tmp.{pid}"
        )

        pmask_tmp = (
            cell_dir
            /
            f"pmask.tmp.{pid}"
        )

        manifest_tmp = (
            cell_dir
            /
            f"manifest.json.tmp.{pid}"
        )

        for tmp in (
            phgt_tmp,
            pmask_tmp,
            manifest_tmp,
        ):

            if tmp.exists():
                tmp.unlink()

        try:

            t0 = perf_counter()

            _run_command(
                [
                    self._commands[
                        "data2pt"
                    ],
                    str(
                        assets.height_path
                    ),
                    str(
                        assets.height_geometry_par
                    ),
                    str(
                        plist
                    ),
                    str(
                        assets.reference_par
                    ),
                    str(
                        phgt_tmp
                    ),
                    "1",
                    "2",
                ],
                log_file=(
                    self.scratch_dir
                    /
                    "gamma.log"
                ),
                label=(
                    "data2pt:"
                    "geometry_cache:"
                    f"r{global_row0}_"
                    f"c{global_col0}"
                ),
            )

            height_seconds = (
                perf_counter()
                -
                t0
            )

            h = np.fromfile(
                phgt_tmp,
                dtype=">f4",
            )

            if h.size != npoints:

                raise PhaseCorrectionError(
                    "data2pt produced "
                    f"{h.size} heights; "
                    f"expected {npoints} "
                    f"for {label}"
                )

            h_native = h.astype(
                np.float32
            )

            finite = np.isfinite(
                h_native
            )

            if self.zero_height_is_valid:

                valid = finite

                zero = (
                    finite
                    &
                    (
                        h_native
                        ==
                        0.0
                    )
                )

                h_native[
                    zero
                ] = (
                    self.zero_height_epsilon_m
                )

                # Same production convention as before:
                # phgt consumed by GAMMA is >f4.
                h_native.astype(
                    ">f4"
                ).tofile(
                    phgt_tmp
                )

            else:

                valid = (
                    finite
                    &
                    (
                        h_native
                        !=
                        0.0
                    )
                )

            valid.astype(
                np.uint8
            ).tofile(
                pmask_tmp
            )

            if (
                phgt_tmp.stat().st_size
                !=
                npoints * 4
            ):

                raise PhaseCorrectionError(
                    "geometry-cache phgt "
                    "byte-size mismatch"
                )

            if (
                pmask_tmp.stat().st_size
                !=
                npoints
            ):

                raise PhaseCorrectionError(
                    "geometry-cache pmask "
                    "byte-size mismatch"
                )

            # Publish arrays first.
            os.replace(
                phgt_tmp,
                phgt,
            )

            os.replace(
                pmask_tmp,
                pmask,
            )

            # Manifest published last = valid-cache marker.
            manifest_tmp.write_text(
                json.dumps(
                    expected_manifest,
                    indent=2,
                    sort_keys=True,
                )
                +
                "\n",
                encoding="utf-8",
            )

            os.replace(
                manifest_tmp,
                manifest_path,
            )

            log(
                "Geometry cache CREATED "
                f"r{global_row0}:"
                f"{global_row1} "
                f"c{global_col0}:"
                f"{global_col1} "
                f"data2pt="
                f"{height_seconds:.2f}s"
            )

            return (
                phgt,
                pmask,
                np.asarray(
                    valid,
                    dtype=np.bool_,
                ),
                height_seconds,
                False,
            )

        finally:

            for tmp in (
                phgt_tmp,
                pmask_tmp,
                manifest_tmp,
            ):

                try:

                    if tmp.exists():
                        tmp.unlink()

                except OSError:

                    pass

    def correct_block(
        self,
        block: np.ndarray,
        *,
        global_row0: int,
        global_col0: int,
        tile_label: str | None = None,
        date_indices=None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        CorrectionTileStats,
    ]:
        """
        Apply GAMMA orbit/topographic phase correction.

        P11B-1
        -------
        date_indices=None
            Preserve the original all-date production path.

        date_indices=(...)
            block contains only the requested GLOBAL acquisition
            indices, in exactly that order.

            Only reference-secondary pairs required for those dates
            are simulated.  The geometrical reference acquisition
            itself has zero simulated phase.

        Scientific phase convention is unchanged.
        """

        assets = self._ensure_assets()

        full_ndate = len(
            self.stack.records
        )

        if date_indices is None:

            selected_indices = tuple(
                range(full_ndate)
            )

        else:

            selected_indices = tuple(
                int(x)
                for x in date_indices
            )

            if not selected_indices:
                raise ValueError(
                    "date_indices must not be empty"
                )

            if len(
                set(selected_indices)
            ) != len(
                selected_indices
            ):
                raise ValueError(
                    "date_indices contains duplicates"
                )

            bad = [
                x
                for x in selected_indices
                if (
                    x < 0
                    or
                    x >= full_ndate
                )
            ]

            if bad:
                raise ValueError(
                    "date_indices outside stack: "
                    f"{bad}"
                )

        local_ndate = len(
            selected_indices
        )

        if (
            block.ndim != 3
            or
            block.shape[0] != local_ndate
        ):
            raise ValueError(
                "correct_block expects "
                "(len(date_indices), rows, cols); "
                f"block={block.shape}, "
                f"dates={local_ndate}"
            )

        legacy_all_dates = (
            selected_indices
            ==
            tuple(
                range(full_ndate)
            )
        )

        rows = int(
            block.shape[1]
        )

        cols = int(
            block.shape[2]
        )

        npoints = (
            rows
            *
            cols
        )

        label = (
            tile_label
            or
            (
                f"r{global_row0}_"
                f"c{global_col0}_"
                f"{rows}x{cols}"
            )
        )

        t_total = perf_counter()

        fixed_dir = (
            self.scratch_dir
            /
            "tiles"
            /
            label
        )

        tmp_ctx = None

        if self.keep_tile_files:

            fixed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            tile_dir = fixed_dir

        else:

            tmp_ctx = (
                tempfile.TemporaryDirectory(
                    prefix=label + "_",
                    dir=(
                        self.scratch_dir
                        /
                        "tiles"
                    ),
                )
            )

            tile_dir = Path(
                tmp_ctx.name
            )

        try:

            rr, cc = np.meshgrid(
                np.arange(
                    global_row0,
                    global_row0 + rows,
                    dtype=np.int32,
                ),
                np.arange(
                    global_col0,
                    global_col0 + cols,
                    dtype=np.int32,
                ),
                indexing="ij",
            )

            # IPTA plist order:
            # range, azimuth.
            plist_arr = np.column_stack(
                (
                    cc.ravel(),
                    rr.ravel(),
                )
            ).astype(
                ">i4",
                copy=False,
            )

            plist = (
                tile_dir
                /
                "plist"
            )

            plist_arr.tofile(
                plist
            )

            phgt = (
                tile_dir
                /
                "phgt"
            )

            pmask = (
                tile_dir
                /
                "pmask"
            )

            psim = (
                tile_dir
                /
                "psim_unw"
            )

            tile_log = (
                self.scratch_dir
                /
                "gamma.log"
            )

            # ----------------------------------------------------
            # Geometry / height sampling.
            # ----------------------------------------------------

            (
                phgt,
                pmask,
                valid,
                height_seconds,
                geometry_cache_hit,
            ) = self._prepare_cached_geometry(
                assets=assets,
                global_row0=global_row0,
                global_col0=global_col0,
                rows=rows,
                cols=cols,
                plist=plist,
                label=label,
            )

            # ----------------------------------------------------
            # Build only the phase records requested by this block.
            #
            # selected_pairs:
            #   local output row,
            #   global stack secondary index
            # ----------------------------------------------------

            selected_pairs = tuple(
                (
                    local_pos,
                    global_idx,
                )
                for (
                    local_pos,
                    global_idx,
                )
                in enumerate(
                    selected_indices
                )
                if (
                    global_idx
                    !=
                    assets.reference_index
                )
            )

            sim_selected = np.zeros(
                (
                    local_ndate,
                    npoints,
                ),
                dtype=np.float32,
            )

            simulation_seconds = 0.0

            if (
                selected_pairs
                and
                np.any(valid)
            ):

                n_pairs = len(
                    selected_pairs
                )

                sim_workers = (
                    self.phase_sim_worker_count(
                        n_pairs
                    )
                )

                ref_one_based = (
                    assets.reference_index
                    +
                    1
                )

                # ------------------------------------------------
                # Serial route.
                #
                # For exact all-date legacy operation retain the
                # already validated original assets.itab.
                #
                # For a subset create an equivalent local-record
                # ITAB containing only requested pairs.
                # ------------------------------------------------

                if sim_workers == 1:

                    if legacy_all_dates:

                        active_itab = (
                            assets.itab
                        )

                    else:

                        active_itab = (
                            tile_dir
                            /
                            "requested_dates.itab"
                        )

                        with active_itab.open(
                            "w",
                            encoding="utf-8",
                        ) as f:

                            for (
                                rec,
                                (
                                    _local_pos,
                                    global_sec,
                                ),
                            ) in enumerate(
                                selected_pairs,
                                1,
                            ):

                                f.write(
                                    f"{ref_one_based} "
                                    f"{global_sec + 1} "
                                    f"{rec} "
                                    "1\n"
                                )

                    t0 = perf_counter()

                    _run_command(
                        [
                            self._commands[
                                "phase_sim_orb_pt"
                            ],
                            str(plist),
                            str(pmask),
                            str(
                                assets.pslc_par
                            ),
                            "-",
                            str(
                                active_itab
                            ),
                            "-",
                            str(phgt),
                            str(psim),
                            str(
                                assets.reference_par
                            ),
                            "-",
                            "0",
                        ],
                        log_file=tile_log,
                        label=(
                            f"phase_sim_orb_pt:"
                            f"{label}"
                        ),
                    )

                    simulation_seconds = (
                        perf_counter()
                        -
                        t0
                    )

                    raw = np.fromfile(
                        psim,
                        dtype=">f4",
                    )

                    expected = (
                        n_pairs
                        *
                        npoints
                    )

                    if raw.size != expected:

                        raise PhaseCorrectionError(
                            "phase_sim_orb_pt "
                            "output size mismatch "
                            f"for {label}: "
                            f"{raw.size} != "
                            f"{expected}. "
                            f"Full log: {tile_log}"
                        )

                    pair_phase = (
                        raw
                        .astype(
                            np.float32
                        )
                        .reshape(
                            n_pairs,
                            npoints,
                        )
                    )

                    for (
                        rec,
                        (
                            local_pos,
                            _global_sec,
                        ),
                    ) in enumerate(
                        selected_pairs
                    ):

                        sim_selected[
                            local_pos
                        ] = (
                            pair_phase[
                                rec
                            ]
                        )

                # ------------------------------------------------
                # Pair-parallel route.
                #
                # Same GAMMA command and pair convention as the
                # validated production implementation.  Only the
                # selected pair set is partitioned.
                # ------------------------------------------------

                else:

                    pair_positions = np.arange(
                        n_pairs,
                        dtype=np.int32,
                    )

                    chunks = [
                        x
                        for x
                        in np.array_split(
                            pair_positions,
                            sim_workers,
                        )
                        if x.size
                    ]

                    chunk_root = (
                        tile_dir
                        /
                        "phase_sim_chunks"
                    )

                    chunk_root.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    def run_chunk(
                        chunk_id,
                        positions,
                    ):

                        chunk_dir = (
                            chunk_root
                            /
                            f"chunk_{chunk_id:03d}"
                        )

                        chunk_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        chunk_itab = (
                            chunk_dir
                            /
                            "itab"
                        )

                        chunk_psim = (
                            chunk_dir
                            /
                            "psim"
                        )

                        chunk_log = (
                            chunk_dir
                            /
                            "gamma.log"
                        )

                        with chunk_itab.open(
                            "w",
                            encoding="utf-8",
                        ) as f:

                            for (
                                local_rec,
                                pair_pos,
                            ) in enumerate(
                                positions,
                                1,
                            ):

                                (
                                    _local_pos,
                                    global_sec,
                                ) = (
                                    selected_pairs[
                                        int(
                                            pair_pos
                                        )
                                    ]
                                )

                                f.write(
                                    f"{ref_one_based} "
                                    f"{global_sec + 1} "
                                    f"{local_rec} "
                                    "1\n"
                                )

                        elapsed = _run_command(
                            [
                                self._commands[
                                    "phase_sim_orb_pt"
                                ],
                                str(plist),
                                str(pmask),
                                str(
                                    assets.pslc_par
                                ),
                                "-",
                                str(
                                    chunk_itab
                                ),
                                "-",
                                str(phgt),
                                str(
                                    chunk_psim
                                ),
                                str(
                                    assets.reference_par
                                ),
                                "-",
                                "0",
                            ],
                            log_file=(
                                chunk_log
                            ),
                            label=(
                                f"phase_sim_orb_pt:"
                                f"{label}:"
                                f"chunk"
                                f"{chunk_id:03d}"
                            ),
                        )

                        raw_chunk = (
                            np.fromfile(
                                chunk_psim,
                                dtype=">f4",
                            )
                        )

                        expected_chunk = (
                            len(
                                positions
                            )
                            *
                            npoints
                        )

                        if (
                            raw_chunk.size
                            !=
                            expected_chunk
                        ):

                            raise (
                                PhaseCorrectionError(
                                    "phase_sim_orb_pt "
                                    f"chunk {chunk_id} "
                                    "output size "
                                    f"{raw_chunk.size} "
                                    "!= "
                                    f"{expected_chunk}"
                                )
                            )

                        arr = (
                            raw_chunk
                            .astype(
                                np.float32
                            )
                            .reshape(
                                len(
                                    positions
                                ),
                                npoints,
                            )
                        )

                        return (
                            positions,
                            arr,
                            elapsed,
                        )

                    t0 = perf_counter()

                    worker_seconds = []

                    with ThreadPoolExecutor(
                        max_workers=len(
                            chunks
                        ),
                        thread_name_prefix=(
                            "pypsds-phase-sim"
                        ),
                    ) as ex:

                        futures = [
                            ex.submit(
                                run_chunk,
                                chunk_id,
                                positions,
                            )
                            for (
                                chunk_id,
                                positions,
                            )
                            in enumerate(
                                chunks
                            )
                        ]

                        for fut in as_completed(
                            futures
                        ):

                            (
                                positions,
                                arr,
                                elapsed,
                            ) = fut.result()

                            worker_seconds.append(
                                elapsed
                            )

                            for (
                                local_rec,
                                pair_pos,
                            ) in enumerate(
                                positions
                            ):

                                (
                                    local_pos,
                                    _global_sec,
                                ) = (
                                    selected_pairs[
                                        int(
                                            pair_pos
                                        )
                                    ]
                                )

                                sim_selected[
                                    local_pos
                                ] = (
                                    arr[
                                        local_rec
                                    ]
                                )

                    simulation_seconds = (
                        perf_counter()
                        -
                        t0
                    )

                    log(
                        "phase_sim_orb_pt "
                        "parallel "
                        f"{label}: "
                        f"requested_dates="
                        f"{local_ndate}, "
                        f"pairs={n_pairs}, "
                        f"workers={len(chunks)}, "
                        f"wall="
                        f"{simulation_seconds:.2f}s, "
                        f"max_worker="
                        f"{max(worker_seconds):.2f}s"
                    )

            sim3 = sim_selected.reshape(
                local_ndate,
                rows,
                cols,
            )

            corrected = (
                block.astype(
                    np.complex64,
                    copy=False,
                )
                *
                np.exp(
                    1j
                    *
                    self.sign
                    *
                    sim3
                ).astype(
                    np.complex64
                )
            )

            valid2 = valid.reshape(
                rows,
                cols,
            )

            corrected[
                :,
                ~valid2
            ] = np.complex64(
                np.nan
                +
                1j
                *
                np.nan
            )

            finite_phase = (
                sim_selected[
                    :,
                    valid,
                ]
                if np.any(valid)
                else np.empty(
                    0,
                    np.float32,
                )
            )

            phase_min = (
                float(
                    np.nanmin(
                        finite_phase
                    )
                )
                if finite_phase.size
                else float("nan")
            )

            phase_max = (
                float(
                    np.nanmax(
                        finite_phase
                    )
                )
                if finite_phase.size
                else float("nan")
            )

            stats = CorrectionTileStats(
                n_points=npoints,
                n_valid_height=int(
                    valid.sum()
                ),
                height_seconds=(
                    height_seconds
                ),
                simulation_seconds=(
                    simulation_seconds
                ),
                total_seconds=(
                    perf_counter()
                    -
                    t_total
                ),
                phase_min=phase_min,
                phase_max=phase_max,
            )

            return (
                corrected,
                valid2,
                stats,
            )

        except Exception:

            # FASTPATCH: preserve final failed TemporaryDirectory bundle.
            if tmp_ctx is not None:
                try:
                    failed_root = (
                        self.scratch_dir
                        /
                        "failed_tiles"
                    )
                    failed_root.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    failed_dir = (
                        failed_root
                        /
                        (
                            f"{label}_pid{os.getpid()}_"
                            f"{time.time_ns()}"
                        )
                    )

                    if tile_dir.exists():
                        shutil.copytree(
                            tile_dir,
                            failed_dir,
                            dirs_exist_ok=False,
                        )
                        log(
                            "Preserved failed GAMMA tile bundle: "
                            f"{failed_dir}"
                        )

                except Exception as preserve_exc:
                    log(
                        "WARNING: failed to preserve GAMMA "
                        f"crash bundle: {preserve_exc}"
                    )

            raise

        finally:

            if tmp_ctx is not None:
                tmp_ctx.cleanup()


__all__ = [
    "CorrectionTileStats",
    "GammaPointPhaseCorrectionProvider",
    "PhaseCorrectionAssets",
    "PhaseCorrectionError",
]
