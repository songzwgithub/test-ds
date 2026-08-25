from pathlib import Path


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {n}"
        )
    return text.replace(old, new, 1)


# ============================================================
# 1. pystamps/config.py
# ============================================================

p = Path("pystamps/config.py")
text = p.read_text(encoding="utf-8")

text = replace_once(
    text,
    '''    # auto | tif | ztd
    product_format: str = "auto"

''',
    "",
    "remove GacosConfig.product_format",
)

text = replace_once(
    text,
    '''        self.product_format = str(
            self.product_format
        ).strip().lower()

        if self.product_format not in {
            "auto", "tif", "ztd"
        }:
            raise ConfigError(
                "gacos.product_format must be "
                "auto, tif, or ztd"
            )

''',
    "",
    "remove product_format validation",
)

p.write_text(text, encoding="utf-8")


# ============================================================
# 2. production YAML x 2
# ============================================================

for name in (
    "config/production.yaml",
    "pystamps/data/production.yaml",
):
    p = Path(name)
    text = p.read_text(encoding="utf-8")

    old = '''# product_format:
#   auto -> automatic TIF/ZTD discovery
#   tif  -> GeoTIFF
#   ztd  -> *.ztd + matching *.rsc
'''

    new = '''# Product type is detected automatically:
#   *.tif / *.tiff -> GeoTIFF
#   *.ztd + *.rsc  -> raw GACOS ZTD
'''

    text = replace_once(
        text,
        old,
        new,
        f"{name} comments",
    )

    text = replace_once(
        text,
        "  product_format: auto\n",
        "",
        f"{name} product_format",
    )

    p.write_text(text, encoding="utf-8")


# ============================================================
# 3. gacos_correction.py
# ============================================================

p = Path(
    "pystamps/pipeline/gacos_correction.py"
)
text = p.read_text(encoding="utf-8")

text = replace_once(
    text,
    "    product_format: str\n",
    "",
    "ResolvedGacosConfig.product_format",
)

text = replace_once(
    text,
    '''            product_format=str(
                settings.product_format
            ).strip().lower(),
''',
    "",
    "central settings product_format",
)

# Remove old environment format configuration completely.
start_marker = '''    product_format = os.environ.get(
        "PYSTAMPS_GACOS_FORMAT",
        "auto",
    ).strip().lower()
'''

start = text.find(start_marker)

if start < 0:
    raise RuntimeError(
        "legacy product_format start not found"
    )

end_marker = '''    product_unit = os.environ.get(
'''

end = text.find(end_marker, start)

if end < 0:
    raise RuntimeError(
        "legacy product_format end not found"
    )

text = text[:start] + text[end:]

text = replace_once(
    text,
    "        product_format=product_format,\n",
    "",
    "legacy return product_format",
)

text = replace_once(
    text,
    '''def discover_products(gacos_dir: Path, product_format: str) -> dict[str, GacosProduct]:
''',
    '''def discover_products(
    gacos_dir: Path,
) -> dict[str, GacosProduct]:
''',
    "discover_products signature",
)

text = replace_once(
    text,
    '''        if product_format in {"auto", "tif"} and date in tif_by_date:
''',
    '''        # Deterministic priority when both representations
        # exist for the same acquisition: GeoTIFF first.
        if date in tif_by_date:
''',
    "TIF auto detection",
)

text = replace_once(
    text,
    '''        if product_format in {"auto", "ztd"} and date in ztd_by_date:
''',
    '''        if date in ztd_by_date:
''',
    "ZTD auto detection",
)

text = replace_once(
    text,
    '''    products = discover_products(config.gacos_dir, config.product_format)
''',
    '''    products = discover_products(
        config.gacos_dir
    )
''',
    "production discovery call",
)

text = replace_once(
    text,
    '''        "format": config.product_format,
''',
    "",
    "cache format",
)

text = replace_once(
    text,
    '''        "product_format_requested": config.product_format,
''',
    "",
    "debug format",
)

if "product_format" in text:
    raise RuntimeError(
        "product_format still remains in "
        "gacos_correction.py"
    )

p.write_text(text, encoding="utf-8")


# ============================================================
# 4. pyproject.toml
# ============================================================

p = Path("pyproject.toml")
text = p.read_text(encoding="utf-8")

if '"rasterio' not in text:
    text = replace_once(
        text,
        '''  "PyYAML>=6.0",
''',
        '''  "PyYAML>=6.0",
  "rasterio>=1.5",
  "pyproj>=3.7.2",
''',
        "mandatory geo dependencies",
    )

p.write_text(text, encoding="utf-8")

print("02c PATCH: PASS")
