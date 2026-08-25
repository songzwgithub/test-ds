"""
Portable validated correction core.

Generated mechanically from the frozen authoritative production source.

Runtime project geometry/state previously stored in module globals is
represented as explicit function arguments. Numerical function bodies
are otherwise retained.
"""

import numpy as np
import re

NUM_RE = re.compile(
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)


def read_par(path):
    out = {}
    for line in path.read_text(errors='ignore').splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        out[k.strip().lower()] = v.strip()
    return out

def par_scalar(pars, names):
    for name in names:
        x = pars.get(name.lower())
        if x is None:
            continue
        m = NUM_RE.search(x)
        if m:
            return float(m.group(0))
    raise KeyError(' / '.join(names))

def parse_vector(text, labels):
    for line in text.splitlines():
        if not any((x in line for x in labels)):
            continue
        rhs = line.split(':', 1)[1] if ':' in line else line
        x = NUM_RE.findall(rhs)
        if len(x) >= 3:
            return np.asarray([float(v) for v in x[:3]], dtype=np.float64)
    return None

def parse_base(path):
    text = path.read_text(errors='ignore')
    B = parse_vector(text, ('initial_baseline(TCN)', 'initial_baseline'))
    Br = parse_vector(text, ('initial_baseline_rate', 'baseline_rate(TCN)'))
    if B is None or Br is None:
        raise RuntimeError(f'invalid .base: {path}')
    return (B, Br)

def geometry_factors(rr, cc, azimuth_looks, earth_radius, mean_azimuth, near_range, prf, range_looks, range_spacing, sar_to_earth):
    range_original = cc * range_looks + (range_looks - 1) / 2.0
    azimuth_original = rr * azimuth_looks + (azimuth_looks - 1) / 2.0
    slant_range = near_range + range_original * range_spacing
    arg = (sar_to_earth ** 2 + slant_range ** 2 - earth_radius ** 2) / (2.0 * sar_to_earth * slant_range)
    look = np.arccos(np.clip(arg, -1.0, 1.0))
    cs = np.cos(look)
    ss = np.sin(look)
    dt = (azimuth_original - mean_azimuth) / prf
    return (cs, ss, dt)

def fast_bsm(rr, cc, CR_SM, C_SM, NR_SM, N_SM, azimuth_looks, earth_radius, mean_azimuth, near_range, prf, range_looks, range_spacing, sar_to_earth):
    cs, ss, dt = geometry_factors(rr, cc, azimuth_looks=azimuth_looks, earth_radius=earth_radius, mean_azimuth=mean_azimuth, near_range=near_range, prf=prf, range_looks=range_looks, range_spacing=range_spacing, sar_to_earth=sar_to_earth)
    return cs[:, None] * C_SM[None, :] - ss[:, None] * N_SM[None, :] + (dt * cs)[:, None] * CR_SM[None, :] - (dt * ss)[:, None] * NR_SM[None, :]

__all__ = ['read_par', 'par_scalar', 'parse_vector', 'parse_base', 'geometry_factors', 'fast_bsm']
