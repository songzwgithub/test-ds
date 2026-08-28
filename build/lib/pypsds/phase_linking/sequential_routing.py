from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SequentialRouting:
    formal_ds: np.ndarray
    sequential: np.ndarray
    fallback: np.ndarray

    formal_count: int
    sequential_count: int
    fallback_count: int

    formal_min_shp: int
    state_min_shp: int

    def validate(self) -> None:

        if self.formal_ds.dtype != np.bool_:
            raise TypeError("formal_ds must be bool")

        if self.sequential.dtype != np.bool_:
            raise TypeError("sequential must be bool")

        if self.fallback.dtype != np.bool_:
            raise TypeError("fallback must be bool")

        if not (
            self.formal_ds.shape
            ==
            self.sequential.shape
            ==
            self.fallback.shape
        ):
            raise ValueError(
                "routing mask shape mismatch"
            )

        overlap = (
            self.sequential
            &
            self.fallback
        )

        if np.any(overlap):
            raise RuntimeError(
                "sequential/fallback overlap"
            )

        union = (
            self.sequential
            |
            self.fallback
        )

        if not np.array_equal(
            union,
            self.formal_ds,
        ):
            raise RuntimeError(
                "sequential/fallback do not "
                "exactly cover formal DS"
            )

        if self.formal_count != int(
            np.count_nonzero(
                self.formal_ds
            )
        ):
            raise RuntimeError(
                "formal_count mismatch"
            )

        if self.sequential_count != int(
            np.count_nonzero(
                self.sequential
            )
        ):
            raise RuntimeError(
                "sequential_count mismatch"
            )

        if self.fallback_count != int(
            np.count_nonzero(
                self.fallback
            )
        ):
            raise RuntimeError(
                "fallback_count mismatch"
            )

        if (
            self.formal_count
            !=
            self.sequential_count
            +
            self.fallback_count
        ):
            raise RuntimeError(
                "routing count conservation failed"
            )


def build_sequential_routing(
    *,
    center_prior: np.ndarray,
    valid: np.ndarray,
    ps: np.ndarray,

    original_shp_count: np.ndarray,
    effective_shp_count: np.ndarray,

    formal_min_shp: int = 48,
    state_min_shp: int = 24,
) -> SequentialRouting:
    """
    Build the production routing masks.

    Formal DS definition:
        valid
        & ~PS
        & original_K >= formal_min_shp

    center_prior is retained only for API/backward-compatible
    diagnostics. It does NOT participate in formal DS eligibility.

    Sequential route:
        formal DS
        & K24-state effective_K >= formal_min_shp

    Full-SCM fallback:
        all remaining formal DS.

    state_min_shp does NOT change formal DS eligibility.
    """

    if formal_min_shp < 1:
        raise ValueError(
            "formal_min_shp must be >= 1"
        )

    if state_min_shp < 1:
        raise ValueError(
            "state_min_shp must be >= 1"
        )

    if state_min_shp > formal_min_shp:
        raise ValueError(
            "state_min_shp must not exceed "
            "formal_min_shp"
        )

    prior = np.asarray(
        center_prior,
        dtype=np.bool_,
    )

    valid = np.asarray(
        valid,
        dtype=np.bool_,
    )

    ps = np.asarray(
        ps,
        dtype=np.bool_,
    )

    original_k = np.asarray(
        original_shp_count,
    )

    effective_k = np.asarray(
        effective_shp_count,
    )

    shape = prior.shape

    for name, arr in (
        ("valid", valid),
        ("ps", ps),
        ("original_shp_count", original_k),
        ("effective_shp_count", effective_k),
    ):
        if arr.shape != shape:
            raise ValueError(
                f"{name} shape={arr.shape} "
                f"!= {shape}"
            )

    valid_nonps = (
        valid
        &
        ~ps
    )

    # --------------------------------------------------------
    # Formal DS is defined solely by:
    #
    #   1. valid non-PS center
    #   2. exact full-stack GLRT support K >= formal_min_shp
    #
    # A secondary candidate prior must never remove a pixel
    # which satisfies the formal SHP definition.
    # --------------------------------------------------------

    formal_ds = (
        valid_nonps
        &
        (
            original_k
            >=
            formal_min_shp
        )
    )

    sequential = (
        formal_ds
        &
        (
            effective_k
            >=
            formal_min_shp
        )
    )

    fallback = (
        formal_ds
        &
        ~sequential
    )

    out = SequentialRouting(
        formal_ds=np.asarray(
            formal_ds,
            dtype=np.bool_,
        ),

        sequential=np.asarray(
            sequential,
            dtype=np.bool_,
        ),

        fallback=np.asarray(
            fallback,
            dtype=np.bool_,
        ),

        formal_count=int(
            np.count_nonzero(
                formal_ds
            )
        ),

        sequential_count=int(
            np.count_nonzero(
                sequential
            )
        ),

        fallback_count=int(
            np.count_nonzero(
                fallback
            )
        ),

        formal_min_shp=int(
            formal_min_shp
        ),

        state_min_shp=int(
            state_min_shp
        ),
    )

    out.validate()

    return out


__all__ = [
    "SequentialRouting",
    "build_sequential_routing",
]
