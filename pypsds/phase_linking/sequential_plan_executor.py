from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sequential_multistage import (
    StageResult,
    run_sequential_stage,
)
from .temporal_plan import (
    TemporalPlan,
    TemporalStrategy,
)


@dataclass(slots=True)
class SequentialPlanResult:
    stage_results: tuple[StageResult, ...]

    compressed_paths: dict[str, Path]
    valid_paths: dict[str, Path]

    total_stage_seconds: float

    @property
    def stage_count(self) -> int:
        return len(
            self.stage_results
        )

    @property
    def final_compressed_path(
        self,
    ) -> Path | None:

        if not self.stage_results:
            return None

        return self.stage_results[
            -1
        ].compressed_path

    @property
    def final_valid_path(
        self,
    ) -> Path | None:

        if not self.stage_results:
            return None

        return self.stage_results[
            -1
        ].valid_path


def run_sequential_plan(
    *,
    plan: TemporalPlan,

    yxt: np.ndarray,
    scale2: np.ndarray,
    valid: np.ndarray,
    ps: np.ndarray,

    state_core: np.ndarray,
    expected_effective_k: np.ndarray,

    output_dir: Path,

    phase_sink=None,

    full_glrt_nslc: int,
    state_min_shp: int,

    half_row: int = 5,
    half_col: int = 11,
    alpha: float = 0.005,

    beta: float = 0.0,
    gamma_jitter: float = 1e-6,
    emi_mu: float = 0.99,

    emi_backend: str = "current_eigh",

    tile_rows: int = 256,
    tile_cols: int = 512,

    center_batch: int = 16000,
    support_block: int = 1024,

    static_support_cache=None,

    pl_workers: int = 16,
    pl_chunk_size: int = 512,

    formula_audit_points: int = 5000,
) -> SequentialPlanResult:
    """
    Execute one validated true-sequential TemporalPlan.

    This promotes the orchestration already exercised by the
    U3.3b multi-ministack audit:

        TemporalPlan
          -> compressed input registry
          -> run_sequential_stage()
          -> compressed output registry
          -> next stage

    The same phase_sink is passed to every stage, so real
    acquisition phases are emitted during the same PL pass
    that creates each compressed SLC.
    """

    if not plan.execution_ready:
        raise ValueError(
            "temporal plan is not executable"
        )

    if (
        plan.effective_strategy
        !=
        TemporalStrategy.SEQUENTIAL.value
    ):
        raise ValueError(
            "run_sequential_plan requires "
            "effective_strategy='sequential'"
        )

    if plan.exact_collapse:
        raise ValueError(
            "exact-collapse/full-SCM plans must "
            "use the full-SCM production path"
        )

    if not plan.stages:
        raise ValueError(
            "sequential plan has no stages"
        )

    if not getattr(
        yxt,
        "is_phase_source_proxy",
        False,
    ):

        yxt = np.asarray(
            yxt
        )

    if yxt.ndim != 3:
        raise ValueError(
            "yxt must have shape [H,W,N]"
        )

    H, W, ndate = yxt.shape

    if plan.ndate != ndate:
        raise ValueError(
            f"plan dates={plan.ndate} "
            f"but YXT dates={ndate}"
        )

    if full_glrt_nslc != ndate:
        raise ValueError(
            "full_glrt_nslc must equal "
            "the full YXT date count"
        )

    for name, arr in (
        ("scale2", scale2),
        ("valid", valid),
        ("ps", ps),
        ("state_core", state_core),
        (
            "expected_effective_k",
            expected_effective_k,
        ),
    ):
        if np.shape(arr) != (
            H,
            W,
        ):
            raise ValueError(
                f"{name} shape={np.shape(arr)} "
                f"!= {(H, W)}"
            )

    if phase_sink is not None and not callable(
        phase_sink
    ):
        raise TypeError(
            "phase_sink must be callable"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_core_bool = np.asarray(
        state_core,
        dtype=np.bool_,
    )

    # --------------------------------------------------------
    # Registry semantics are intentionally identical to the
    # validated U3.3b driver.
    # --------------------------------------------------------

    compressed_registry: dict[
        str,
        np.ndarray,
    ] = {}

    valid_registry: dict[
        str,
        np.ndarray,
    ] = {}

    compressed_paths: dict[
        str,
        Path,
    ] = {}

    valid_paths: dict[
        str,
        Path,
    ] = {}

    results = []

    total_stage_seconds = 0.0

    for stage in plan.stages:

        input_ids = tuple(
            x.ref_id
            for x
            in stage.compressed_inputs
        )

        missing = [
            ref_id
            for ref_id
            in input_ids
            if ref_id
            not in compressed_registry
        ]

        if missing:
            raise RuntimeError(
                "missing compressed state(s): "
                f"{missing}"
            )

        compressed_inputs = tuple(
            compressed_registry[
                ref_id
            ]
            for ref_id
            in input_ids
        )

        compressed_valids = tuple(
            valid_registry[
                ref_id
            ]
            for ref_id
            in input_ids
        )

        if compressed_valids:

            inputs_complete = all(
                bool(
                    np.all(
                        v[
                            state_core_bool
                        ]
                    )
                )
                for v
                in compressed_valids
            )

        else:

            inputs_complete = True

        result = run_sequential_stage(
            stage_index=(
                stage.stage_index
            ),

            compressed_input_ids=(
                input_ids
            ),

            compressed_inputs=(
                compressed_inputs
            ),

            yxt=yxt,

            real_indices=(
                stage.real_indices
            ),

            scale2=scale2,
            valid=valid,
            ps=ps,

            state_core=(
                state_core_bool
            ),

            expected_effective_k=(
                expected_effective_k
            ),

            output_dir=output_dir,

            full_glrt_nslc=(
                full_glrt_nslc
            ),

            state_min_shp=(
                state_min_shp
            ),

            inputs_complete=(
                inputs_complete
            ),

            half_row=half_row,
            half_col=half_col,
            alpha=alpha,

            beta=beta,

            gamma_jitter=(
                gamma_jitter
            ),

            emi_mu=emi_mu,

            emi_backend=emi_backend,

            tile_rows=tile_rows,
            tile_cols=tile_cols,

            center_batch=(
                center_batch
            ),

            support_block=(
                support_block
            ),

            static_support_cache=static_support_cache,

            pl_workers=pl_workers,

            pl_chunk_size=(
                pl_chunk_size
            ),

            formula_audit_points=(
                formula_audit_points
            ),

            phase_sink=phase_sink,
        )

        if (
            stage.compressed_output
            is
            None
        ):
            raise RuntimeError(
                "true sequential stage has "
                "no compressed output: "
                f"{stage.stage_index}"
            )

        ref_id = (
            stage
            .compressed_output
            .ref_id
        )

        compressed_path = Path(
            result.compressed_path
        )

        valid_path = Path(
            result.valid_path
        )

        if not compressed_path.is_file():
            raise FileNotFoundError(
                compressed_path
            )

        if not valid_path.is_file():
            raise FileNotFoundError(
                valid_path
            )

        compressed_registry[
            ref_id
        ] = np.load(
            compressed_path,
            mmap_mode="r",
        )

        valid_registry[
            ref_id
        ] = np.load(
            valid_path,
            mmap_mode="r",
        )

        compressed_paths[
            ref_id
        ] = compressed_path

        valid_paths[
            ref_id
        ] = valid_path

        results.append(
            result
        )

        total_stage_seconds += float(
            result.elapsed_seconds
        )

    # Flush final linked_phase writer, but deliberately
    # do NOT close it: the caller may still write full-SCM
    # fallback points into the same production cube.
    if (
        phase_sink is not None
        and
        hasattr(
            phase_sink,
            "flush",
        )
    ):
        phase_sink.flush()

    return SequentialPlanResult(
        stage_results=tuple(
            results
        ),

        compressed_paths=(
            compressed_paths
        ),

        valid_paths=(
            valid_paths
        ),

        total_stage_seconds=(
            total_stage_seconds
        ),
    )


__all__ = [
    "SequentialPlanResult",
    "run_sequential_plan",
]
