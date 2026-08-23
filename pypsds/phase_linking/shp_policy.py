from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .temporal_plan import build_temporal_plan


@dataclass(frozen=True, slots=True)
class ShpPolicy:
    mode: str
    ndate: int
    effective_strategy: str
    max_solver_size: int
    base_half_row: int
    base_half_col: int
    half_row: int
    half_col: int
    window_capacity: int
    window_adapted: bool
    base_formal_min_shp: int
    formal_min_shp: int
    base_state_min_shp: int
    state_min_shp: int
    full_scm_rank_min_shp: int
    rank_guard: bool
    adaptive_window: bool

    def as_dict(self):
        return asdict(self)


def window_capacity(half_row: int, half_col: int) -> int:
    return (2 * int(half_row) + 1) * (2 * int(half_col) + 1) - 1


def _odd_ceil(x: float) -> int:
    n = max(1, int(math.ceil(float(x))))
    return n if n % 2 else n + 1


def expand_window_for_capacity(
    half_row: int,
    half_col: int,
    required_neighbors: int,
) -> tuple[int, int]:
    hr0, hc0 = int(half_row), int(half_col)
    required = int(required_neighbors)
    if window_capacity(hr0, hc0) >= required:
        return hr0, hc0
    wh0, ww0 = 2 * hr0 + 1, 2 * hc0 + 1
    scale = math.sqrt((required + 1) / float(wh0 * ww0))
    wh = max(wh0, _odd_ceil(wh0 * scale))
    ww = max(ww0, _odd_ceil(ww0 * scale))
    while wh * ww - 1 < required:
        if wh / wh0 <= ww / ww0:
            wh += 2
        else:
            ww += 2
    return (wh - 1) // 2, (ww - 1) // 2


def resolve_shp_policy(
    cfg: dict,
    dates: Sequence[str],
    *,
    base_half_row: int | None = None,
    base_half_col: int | None = None,
    base_formal_min_shp: int | None = None,
) -> ShpPolicy:
    shp = cfg.get("selection", {}).get("shp", {})
    temporal = cfg.get("phase_linking", {}).get("temporal", {})
    phase = cfg.get("phase_linking", {})

    mode = str(shp.get("policy", "solver_aware")).strip().lower()
    if mode not in {"fixed", "solver_aware"}:
        raise ValueError(f"unsupported selection.shp.policy={mode!r}")

    dates = tuple(str(x) for x in dates)
    ndate = len(dates)
    hr0 = int(shp.get("half_row", 5) if base_half_row is None else base_half_row)
    hc0 = int(shp.get("half_col", 11) if base_half_col is None else base_half_col)
    formal0 = int(
        shp.get("min_count", 48)
        if base_formal_min_shp is None
        else base_formal_min_shp
    )
    state0 = int(temporal.get("state_min_shp", 24))

    plan = build_temporal_plan(
        dates,
        strategy=str(temporal.get("strategy", "sequential")).lower(),
        ministack_size=int(temporal.get("ministack_size", 19)),
        max_num_compressed=int(temporal.get("max_num_compressed", 5)),
        reference_index=int(phase.get("temporal_reference_index", 0)),
    )
    if not plan.execution_ready:
        raise RuntimeError("temporal plan is not execution-ready")

    rank_guard = bool(shp.get("rank_guard", True))
    adaptive = bool(shp.get("adaptive_window", {}).get("enabled", True))

    if mode == "fixed":
        state_min = state0
        formal_min = formal0
        full_min = formal0
        rank_guard = False
        adaptive = False
    else:
        state_min = max(state0, int(plan.max_solver_size))
        if plan.effective_strategy == "full_scm":
            formal_min = max(formal0, state_min, ndate)
        else:
            formal_min = max(formal0, state_min)
        full_min = max(formal_min, ndate)

    required = formal_min
    fallback = bool(temporal.get("full_scm_fallback", True))
    if rank_guard and (plan.effective_strategy == "full_scm" or fallback):
        required = max(required, full_min)

    hr, hc = hr0, hc0
    if adaptive and window_capacity(hr, hc) < required:
        hr, hc = expand_window_for_capacity(hr, hc, required)

    cap = window_capacity(hr, hc)
    if cap < formal_min:
        raise RuntimeError(f"SHP window capacity={cap} < formal K={formal_min}")
    if rank_guard and (plan.effective_strategy == "full_scm" or fallback):
        if cap < full_min:
            raise RuntimeError(
                f"SHP window capacity={cap} < full-SCM K={full_min}"
            )

    return ShpPolicy(
        mode=mode,
        ndate=ndate,
        effective_strategy=str(plan.effective_strategy),
        max_solver_size=int(plan.max_solver_size),
        base_half_row=hr0,
        base_half_col=hc0,
        half_row=hr,
        half_col=hc,
        window_capacity=cap,
        window_adapted=(hr != hr0 or hc != hc0),
        base_formal_min_shp=formal0,
        formal_min_shp=formal_min,
        base_state_min_shp=state0,
        state_min_shp=state_min,
        full_scm_rank_min_shp=full_min,
        rank_guard=rank_guard,
        adaptive_window=adaptive,
    )


def split_fallback_by_rank(
    fallback_mask,
    original_shp_count,
    *,
    full_scm_min_shp: int,
    rank_guard: bool,
):
    fallback = np.asarray(fallback_mask, dtype=np.bool_)
    K = np.asarray(original_shp_count)
    if fallback.shape != K.shape:
        raise ValueError("fallback/original-K shape mismatch")
    if not rank_guard:
        return fallback.copy(), np.zeros(fallback.shape, dtype=np.bool_)
    supported = fallback & (K >= int(full_scm_min_shp))
    under = fallback & ~supported
    return supported.astype(bool, copy=False), under.astype(bool, copy=False)


def write_shp_policy_json(path, policy: ShpPolicy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(policy.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
