from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import cfg_get


@dataclass(
    frozen=True,
    slots=True,
)
class ProjectPaths:
    """
    Resolved filesystem contract for one pyPSDS-GAMMA project.

    Required processing inputs
    --------------------------
    work_dir
    data_dir
    rslc_dir
    rslc_tab

    Project-level auxiliary locations
    ---------------------------------
    dem_dir
    gacos_dir

    Managed output locations
    ------------------------
    output_dir
    scratch_dir
    products_dir

    Notes
    -----
    DEM and GACOS directories are optional at path-resolution
    time because the corresponding corrections may be disabled.
    Individual processing modules are responsible for requiring
    them only when their feature is enabled.
    """

    work_dir: Path
    data_dir: Path

    rslc_dir: Path
    rslc_tab: Path

    output_dir: Path

    dem_dir: Path | None
    gacos_dir: Path | None

    scratch_dir: Path
    products_dir: Path


def _resolve(
    value,
    *,
    base: Path,
) -> Path | None:

    if value in (
        None,
        "",
    ):
        return None

    p = Path(
        value
    ).expanduser()

    if not p.is_absolute():
        p = (
            base
            /
            p
        )

    return p.resolve()


def _discover_directory(
    *,
    configured,
    base: Path,
    candidates: tuple[str, ...],
) -> Path | None:
    """
    Resolve an optional project directory.

    Explicit configuration has priority.

    If the setting is null/empty, existing conventional directory
    names are discovered below ``base``.

    An explicitly configured path is returned even if it does not
    yet exist. This keeps path resolution separate from feature
    validation.
    """

    resolved = _resolve(
        configured,
        base=base,
    )

    if resolved is not None:
        return resolved

    for name in candidates:

        candidate = (
            base
            /
            name
        )

        if candidate.is_dir():
            return (
                candidate
                .resolve()
            )

    return None


def resolve_project_paths(
    cfg: dict[str, Any],
    config_path: Path,
) -> ProjectPaths:

    config_path = (
        Path(
            config_path
        )
        .expanduser()
        .resolve()
    )

    base = (
        config_path
        .parent
    )


    # ------------------------------------------------------------------
    # Project working directory
    # ------------------------------------------------------------------

    work_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.work_dir",
                None,
            ),
            base=base,
        )
        or
        base
    )


    # ------------------------------------------------------------------
    # Data root
    # ------------------------------------------------------------------

    data_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.data_dir",
                None,
            ),
            base=work_dir,
        )
        or
        work_dir
    )


    # ------------------------------------------------------------------
    # RSLC directory: required
    # ------------------------------------------------------------------

    rslc_dir = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_dir",
            None,
        ),
        base=data_dir,
    )

    if rslc_dir is None:

        for name in (
            "RSLC_cropped",
            "RSLC",
        ):

            candidate = (
                data_dir
                /
                name
            )

            if candidate.is_dir():

                rslc_dir = (
                    candidate
                    .resolve()
                )

                break

    if (
        rslc_dir is None
        or
        not rslc_dir.is_dir()
    ):

        raise FileNotFoundError(
            "Unable to locate RSLC directory "
            f"below {data_dir}"
        )


    # ------------------------------------------------------------------
    # RSLC_tab: required
    # ------------------------------------------------------------------

    rslc_tab = _resolve(
        cfg_get(
            cfg,
            "paths.rslc_tab",
            None,
        ),
        base=data_dir,
    )

    if rslc_tab is None:

        direct = (
            data_dir
            /
            "RSLC_tab"
        )

        if direct.is_file():

            rslc_tab = (
                direct
                .resolve()
            )

        else:

            hits = sorted(
                data_dir.glob(
                    "*RSLC*tab*"
                )
            )

            if hits:

                rslc_tab = (
                    hits[0]
                    .resolve()
                )

    if (
        rslc_tab is None
        or
        not rslc_tab.is_file()
    ):

        raise FileNotFoundError(
            "Unable to locate RSLC_tab "
            f"below {data_dir}"
        )


    # ------------------------------------------------------------------
    # Primary output directory
    # ------------------------------------------------------------------

    output_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.output_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            work_dir
            /
            "output"
        ).resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # DEM / geometry directory
    #
    # Optional until a module that requires DEM geometry is enabled.
    # ------------------------------------------------------------------

    dem_dir = _discover_directory(
        configured=cfg_get(
            cfg,
            "paths.dem_dir",
            None,
        ),
        base=data_dir,
        candidates=(
            "DEM_prep",
            "DEM",
            "dem",
        ),
    )


    # ------------------------------------------------------------------
    # GACOS directory
    #
    # Optional until corrections.atmosphere.mode == gacos.
    # ------------------------------------------------------------------

    gacos_dir = _discover_directory(
        configured=cfg_get(
            cfg,
            "paths.gacos_dir",
            None,
        ),
        base=data_dir,
        candidates=(
            "GACOS",
            "gacos",
        ),
    )


    # ------------------------------------------------------------------
    # Managed scratch directory
    # ------------------------------------------------------------------

    scratch_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.scratch_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            output_dir
            /
            ".scratch"
        ).resolve()
    )

    scratch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # Managed final-product directory
    # ------------------------------------------------------------------

    products_dir = (
        _resolve(
            cfg_get(
                cfg,
                "paths.products_dir",
                None,
            ),
            base=work_dir,
        )
        or
        (
            output_dir
            /
            "products"
        ).resolve()
    )

    products_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    return ProjectPaths(
        work_dir=work_dir,
        data_dir=data_dir,

        rslc_dir=rslc_dir,
        rslc_tab=rslc_tab,

        output_dir=output_dir,

        dem_dir=dem_dir,
        gacos_dir=gacos_dir,

        scratch_dir=scratch_dir,
        products_dir=products_dir,
    )


__all__ = [
    "ProjectPaths",
    "resolve_project_paths",
]
