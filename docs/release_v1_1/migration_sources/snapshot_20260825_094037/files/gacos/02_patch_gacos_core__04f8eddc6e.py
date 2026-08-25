from pathlib import Path

p = Path("pystamps/pipeline/gacos_correction.py")
text = p.read_text(encoding="utf-8")


def once(old, new, label):
    global text
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {n}"
        )
    text = text.replace(old, new, 1)


# ------------------------------------------------------------
# 1. Import central YAML config
# ------------------------------------------------------------

once(
    """from scipy import ndimage

from pystamps.io.mat import read_mat, read_mat_variables
""",
    """from scipy import ndimage

from pystamps.config import GacosConfig as PipelineGacosConfig
from pystamps.io.mat import read_mat, read_mat_variables
""",
    "central GacosConfig import",
)


# ------------------------------------------------------------
# 2. Rename internal resolved config to avoid name collision
# ------------------------------------------------------------

once(
    """@dataclass(slots=True)
class GacosConfig:
""",
    """@dataclass(slots=True)
class ResolvedGacosConfig:
""",
    "rename internal config",
)

text = text.replace(
    "config: GacosConfig,",
    "config: ResolvedGacosConfig,",
)


# ------------------------------------------------------------
# 3. Replace config resolver
# ------------------------------------------------------------

start = text.index(
    "def _resolve_gacos_dir(dataset_root: Path)"
)

end = text.index(
    "def _date_from_name(",
    start,
)

new_block = '''def _resolve_optional_path(
    dataset_root: Path,
    value: str | None,
) -> Path | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = dataset_root / path

    return path.resolve()


def _resolve_gacos_dir(
    dataset_root: Path,
    configured_dir: str | None = None,
    *,
    use_env: bool = True,
) -> Path:
    candidates: list[Path] = []

    # Explicit YAML config has highest priority.
    if (
        configured_dir is not None
        and str(configured_dir).strip()
    ):
        path = Path(
            str(configured_dir)
        ).expanduser()

        if not path.is_absolute():
            path = dataset_root / path

        path = path.resolve()

        if not path.is_dir():
            raise GacosCorrectionError(
                "Configured GACOS directory "
                f"does not exist: {path}"
            )

        return path

    # Retain legacy environment-variable mode only
    # for direct/standalone GACOS calls.
    if use_env:
        raw = os.environ.get(
            "PYSTAMPS_GACOS_DIR",
            "",
        ).strip()

        if raw:
            candidates.append(
                Path(raw).expanduser()
            )

    # Automatic discovery.
    candidates.extend(
        [
            dataset_root / "GACOS",
            dataset_root / "gacos",
            dataset_root.parent / "GACOS",
            dataset_root.parent / "gacos",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved.is_dir():
            return resolved

    raise GacosCorrectionError(
        "Unable to locate GACOS directory. "
        "Set gacos.gacos_dir in pystamps.yaml "
        "or place a GACOS/ directory inside or "
        "beside the pySTAMPS work directory."
    )


def _load_config(
    dataset_root: Path,
    settings: PipelineGacosConfig | None = None,
) -> ResolvedGacosConfig:

    # --------------------------------------------------------
    # Normal production path: pystamps.yaml
    # --------------------------------------------------------
    if settings is not None:

        if not bool(settings.enabled):
            raise GacosCorrectionError(
                "GACOS correction was invoked while "
                "gacos.enabled=false"
            )

        incidence_tif = _resolve_optional_path(
            dataset_root,
            settings.incidence_tif,
        )

        return ResolvedGacosConfig(
            gacos_dir=_resolve_gacos_dir(
                dataset_root,
                settings.gacos_dir,
                use_env=False,
            ),
            product_format=str(
                settings.product_format
            ).strip().lower(),
            product_unit=str(
                settings.product_unit
            ).strip().lower(),
            projection=str(
                settings.projection
            ).strip().lower(),
            sign=str(
                settings.sign
            ).strip().lower(),
            strict_dates=bool(
                settings.strict_dates
            ),
            rebuild=bool(
                settings.rebuild
            ),
            incidence_tif=incidence_tif,
            incidence_deg=(
                None
                if settings.incidence_deg is None
                else float(settings.incidence_deg)
            ),
            qa_ps=max(
                1000,
                int(settings.qa_ps),
            ),
            qa_ifg=max(
                10,
                int(settings.qa_ifg),
            ),
            chunk_ps=max(
                256,
                int(settings.chunk_ps),
            ),
            min_valid_fraction=float(
                settings.min_valid_fraction
            ),
        )

    # --------------------------------------------------------
    # Legacy direct-call environment-variable path.
    # Keep backward compatibility.
    # --------------------------------------------------------

    product_format = os.environ.get(
        "PYSTAMPS_GACOS_FORMAT",
        "auto",
    ).strip().lower()

    if product_format not in {
        "auto",
        "tif",
        "ztd",
    }:
        raise GacosCorrectionError(
            "PYSTAMPS_GACOS_FORMAT must be "
            "auto, tif, or ztd"
        )

    product_unit = os.environ.get(
        "PYSTAMPS_GACOS_UNIT",
        "auto",
    ).strip().lower()

    if product_unit not in {
        "auto",
        "m",
        "cm",
        "mm",
    }:
        raise GacosCorrectionError(
            "PYSTAMPS_GACOS_UNIT must be "
            "auto, m, cm, or mm"
        )

    projection = os.environ.get(
        "PYSTAMPS_GACOS_PROJECTION",
        "zenith",
    ).strip().lower()

    if projection not in {
        "zenith",
        "los",
    }:
        raise GacosCorrectionError(
            "PYSTAMPS_GACOS_PROJECTION must "
            "be zenith or los"
        )

    sign = os.environ.get(
        "PYSTAMPS_GACOS_SIGN",
        "auto",
    ).strip().lower()

    aliases = {
        "-": "subtract",
        "+": "add",
        "minus": "subtract",
        "plus": "add",
    }

    sign = aliases.get(
        sign,
        sign,
    )

    if sign not in {
        "auto",
        "subtract",
        "add",
    }:
        raise GacosCorrectionError(
            "PYSTAMPS_GACOS_SIGN must be "
            "auto, subtract, or add"
        )

    incidence_tif_raw = os.environ.get(
        "PYSTAMPS_GACOS_INCIDENCE_TIF",
        "",
    ).strip()

    incidence_tif = (
        Path(
            incidence_tif_raw
        ).expanduser().resolve()
        if incidence_tif_raw
        else None
    )

    incidence_deg_raw = os.environ.get(
        "PYSTAMPS_GACOS_INCIDENCE_DEG",
        "",
    ).strip()

    incidence_deg = (
        float(incidence_deg_raw)
        if incidence_deg_raw
        else None
    )

    return ResolvedGacosConfig(
        gacos_dir=_resolve_gacos_dir(
            dataset_root,
            use_env=True,
        ),
        product_format=product_format,
        product_unit=product_unit,
        projection=projection,
        sign=sign,
        strict_dates=_env_bool(
            "PYSTAMPS_GACOS_STRICT_DATES",
            True,
        ),
        rebuild=_env_bool(
            "PYSTAMPS_GACOS_REBUILD",
            False,
        ),
        incidence_tif=incidence_tif,
        incidence_deg=incidence_deg,
        qa_ps=max(
            1000,
            int(
                os.environ.get(
                    "PYSTAMPS_GACOS_QA_PS",
                    "30000",
                )
            ),
        ),
        qa_ifg=max(
            10,
            int(
                os.environ.get(
                    "PYSTAMPS_GACOS_QA_IFG",
                    "80",
                )
            ),
        ),
        chunk_ps=max(
            256,
            int(
                os.environ.get(
                    "PYSTAMPS_GACOS_CHUNK_PS",
                    "4096",
                )
            ),
        ),
        min_valid_fraction=float(
            os.environ.get(
                "PYSTAMPS_GACOS_MIN_VALID_FRACTION",
                "0.995",
            )
        ),
    )


'''

text = (
    text[:start]
    + new_block
    + text[end:]
)


# ------------------------------------------------------------
# 4. Public entrypoint accepts YAML settings
# ------------------------------------------------------------

once(
    '''def ensure_gacos_corrected_phuw(dataset_root: Path) -> Path:
    """Create or reuse phuw2_gacos.mat and return its path."""

    root = Path(dataset_root).expanduser().resolve()
    config = _load_config(root)
''',
    '''def ensure_gacos_corrected_phuw(
    dataset_root: Path,
    settings: PipelineGacosConfig | None = None,
) -> Path:
    """Create or reuse phuw2_gacos.mat and return its path."""

    root = Path(dataset_root).expanduser().resolve()
    config = _load_config(
        root,
        settings,
    )
''',
    "public GACOS entrypoint",
)


# ------------------------------------------------------------
# 5. strict_dates semantics
#
# Missing dates cannot safely be temporally interpolated here.
# Therefore strict_dates=false is explicitly rejected if dates
# are actually missing, rather than silently producing a mixed
# corrected/uncorrected stack.
# ------------------------------------------------------------

once(
    '''    if missing_dates:
        preview = ", ".join(missing_dates[:20])
        raise GacosCorrectionError(
            f"Missing {len(missing_dates)}/{n_image} GACOS acquisition dates: {preview}. "
            "All acquisition dates are required; temporal interpolation is intentionally not used."
        )
''',
    '''    if missing_dates:
        preview = ", ".join(
            missing_dates[:20]
        )

        if not config.strict_dates:
            raise GacosCorrectionError(
                "gacos.strict_dates=false does not "
                "enable temporal interpolation or "
                "partial atmospheric correction. "
                f"Missing {len(missing_dates)}/{n_image} "
                f"GACOS acquisition dates: {preview}"
            )

        raise GacosCorrectionError(
            f"Missing {len(missing_dates)}/{n_image} "
            f"GACOS acquisition dates: {preview}. "
            "All acquisition dates are required; "
            "temporal interpolation is intentionally "
            "not used."
        )
''',
    "strict_dates handling",
)


p.write_text(
    text,
    encoding="utf-8",
)

print("02 PATCH: PASS")
