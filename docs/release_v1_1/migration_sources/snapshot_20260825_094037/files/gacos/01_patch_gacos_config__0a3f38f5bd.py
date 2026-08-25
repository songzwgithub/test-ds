from pathlib import Path

ROOT = Path.cwd()


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def write(name, text):
    (ROOT / name).write_text(text, encoding="utf-8")


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {n}"
        )
    return text.replace(old, new, 1)


# ============================================================
# pystamps/config.py
# ============================================================

name = "pystamps/config.py"
text = read(name)

if "class GacosConfig:" in text:
    raise RuntimeError("GacosConfig already exists")

anchor = '''@dataclass(slots=True)
class ReferenceConfig:
'''

gacos = '''@dataclass(slots=True)
class GacosConfig:
    enabled: bool = False
    gacos_dir: str | None = None

    # auto | tif | ztd
    product_format: str = "auto"

    # auto | m | cm | mm
    product_unit: str = "auto"

    # zenith | los
    projection: str = "zenith"

    # auto | subtract | add
    sign: str = "auto"

    strict_dates: bool = True
    rebuild: bool = False

    incidence_tif: str | None = None
    incidence_deg: float | None = None

    qa_ps: int = 30000
    qa_ifg: int = 80
    chunk_ps: int = 4096
    min_valid_fraction: float = 0.995

    def __post_init__(self) -> None:
        self.product_format = str(
            self.product_format
        ).strip().lower()

        if self.product_format not in {
            "auto", "tif", "ztd"
        }:
            raise ConfigError(
                "gacos.product_format must be "
                "auto, tif, or ztd"
            )

        self.product_unit = str(
            self.product_unit
        ).strip().lower()

        if self.product_unit not in {
            "auto", "m", "cm", "mm"
        }:
            raise ConfigError(
                "gacos.product_unit must be "
                "auto, m, cm, or mm"
            )

        self.projection = str(
            self.projection
        ).strip().lower()

        if self.projection not in {
            "zenith", "los"
        }:
            raise ConfigError(
                "gacos.projection must be "
                "zenith or los"
            )

        aliases = {
            "-": "subtract",
            "+": "add",
            "minus": "subtract",
            "plus": "add",
        }

        self.sign = aliases.get(
            str(self.sign).strip().lower(),
            str(self.sign).strip().lower(),
        )

        if self.sign not in {
            "auto", "subtract", "add"
        }:
            raise ConfigError(
                "gacos.sign must be "
                "auto, subtract, or add"
            )

        if self.incidence_deg is not None:
            value = float(self.incidence_deg)
            if not 0.0 < value < 90.0:
                raise ConfigError(
                    "gacos.incidence_deg must "
                    "be between 0 and 90"
                )

        if int(self.qa_ps) <= 0:
            raise ConfigError(
                "gacos.qa_ps must be positive"
            )

        if int(self.qa_ifg) <= 0:
            raise ConfigError(
                "gacos.qa_ifg must be positive"
            )

        if int(self.chunk_ps) <= 0:
            raise ConfigError(
                "gacos.chunk_ps must be positive"
            )

        value = float(
            self.min_valid_fraction
        )

        if not 0.0 < value <= 1.0:
            raise ConfigError(
                "gacos.min_valid_fraction must "
                "be in (0, 1]"
            )


@dataclass(slots=True)
class ReferenceConfig:
'''

text = once(
    text,
    anchor,
    gacos,
    "insert GacosConfig",
)

text = once(
    text,
    '''    tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
''',
    '''    tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    gacos: GacosConfig = field(default_factory=GacosConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
''',
    "RunConfig.gacos",
)

text = once(
    text,
    '''    tools = ExternalToolsConfig(**_as_dict(raw, "tools"))
    reference = ReferenceConfig(**_as_dict(raw, "reference"))
''',
    '''    tools = ExternalToolsConfig(**_as_dict(raw, "tools"))
    gacos = GacosConfig(**_as_dict(raw, "gacos"))
    reference = ReferenceConfig(**_as_dict(raw, "reference"))
''',
    "load gacos",
)

text = once(
    text,
    '''        tools=tools,
        reference=reference,
''',
    '''        tools=tools,
        gacos=gacos,
        reference=reference,
''',
    "return gacos",
)

write(name, text)


# ============================================================
# production.yaml
# ============================================================

name = "config/production.yaml"
yaml_text = read(name)

if "\ngacos:\n" in yaml_text:
    raise RuntimeError(
        "production.yaml already contains gacos"
    )

block = '''
# Optional GACOS atmospheric correction.
#
# enabled: false
#   Completely skip GACOS. Stage 7/8 use phuw2.mat.
#
# enabled: true
#   Apply GACOS after Stage 6.
#   Stage 7/8 use phuw2_gacos.mat.
#
# product_format:
#   auto -> automatic TIF/ZTD discovery
#   tif  -> GeoTIFF
#   ztd  -> *.ztd + matching *.rsc
gacos:
  enabled: false
  gacos_dir: null

  product_format: auto
  product_unit: auto
  projection: zenith
  sign: auto

  strict_dates: true
  rebuild: false

  incidence_tif: null
  incidence_deg: null

  qa_ps: 30000
  qa_ifg: 80
  chunk_ps: 4096
  min_valid_fraction: 0.995


'''

yaml_text = once(
    yaml_text,
    "reference:\n",
    block + "reference:\n",
    "production.yaml gacos",
)

write(
    "config/production.yaml",
    yaml_text,
)

# pystamps -g 使用的是这个打包配置，
# 所以必须与 config/production.yaml 同步。
write(
    "pystamps/data/production.yaml",
    yaml_text,
)

print("01 PATCH: PASS")
