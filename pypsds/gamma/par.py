from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")
COMMENT_PREFIXES = ("#", "%", ";")


class GammaInputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RslcRecord:
    index: int
    date: str
    rslc: Path
    par: Path


def extract_date(value: str | Path) -> str:
    text = str(value)
    filename_hits = DATE_RE.findall(Path(text).name)
    if filename_hits:
        return filename_hits[-1]
    hits = DATE_RE.findall(text)
    if not hits:
        raise GammaInputError(f"No YYYYMMDD date found in: {value}")
    return hits[-1]


def parse_gamma_parameter_file(path: str | Path) -> dict[str, list[str]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise GammaInputError(f"GAMMA parameter file not found: {p}")
    out: dict[str, list[str]] = {}
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIXES):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            vals = value.strip().split()
        else:
            fields = line.split()
            if len(fields) < 2:
                continue
            key, vals = fields[0], fields[1:]
        out[key.strip()] = list(vals)
    return out


def first_number(params: dict[str, list[str]], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        for token in params.get(key, []):
            try:
                return float(token)
            except ValueError:
                pass
    return None


def first_int(params: dict[str, list[str]], keys: tuple[str, ...]) -> int | None:
    value = first_number(params, keys)
    if value is None:
        return None
    value = int(round(value))
    return value if value > 0 else None


def _resolve_list_path(raw: str, list_file: Path, fallback_dir: Path | None) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = list_file.parent / p
    p = p.resolve()
    if p.exists():
        return p
    if fallback_dir is not None:
        candidate = (fallback_dir / Path(raw).name).resolve()
        if candidate.exists():
            return candidate
    return p


def parse_rslc_tab(path: str | Path, rslc_dir: str | Path | None = None) -> list[RslcRecord]:
    tab = Path(path).expanduser().resolve()
    fallback = Path(rslc_dir).expanduser().resolve() if rslc_dir else None
    if not tab.is_file():
        raise GammaInputError(f"RSLC_tab not found: {tab}")

    records: list[RslcRecord] = []
    for lineno, raw in enumerate(tab.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        line = raw
        for marker in COMMENT_PREFIXES:
            line = line.split(marker, 1)[0]
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            raise GammaInputError(f"{tab}:{lineno} requires RSLC and .par columns")
        rslc = _resolve_list_path(fields[0], tab, fallback)
        par = _resolve_list_path(fields[1], tab, fallback)
        date = extract_date(fields[0])
        if not rslc.is_file():
            raise GammaInputError(f"RSLC missing for {date}: {rslc}")
        if not par.is_file():
            raise GammaInputError(f"RSLC parameter missing for {date}: {par}")
        records.append(RslcRecord(len(records), date, rslc, par))

    if not records:
        raise GammaInputError(f"No acquisitions found in {tab}")
    dates = [r.date for r in records]
    if len(set(dates)) != len(dates):
        raise GammaInputError("Duplicate acquisition dates in RSLC_tab")
    return records
