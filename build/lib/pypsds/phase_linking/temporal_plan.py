from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from enum import Enum
from typing import Sequence


class TemporalPlanningError(
    RuntimeError
):
    pass


class TemporalStrategy(
    str,
    Enum,
):
    FULL_SCM = "full_scm"
    SEQUENTIAL = "sequential"
    AUTO = "auto"


@dataclass(
    frozen=True,
    slots=True,
)
class CompressedReference:
    """
    Metadata for one future compressed-SLC state.

    Planning model inspired by the ministack/compressed-SLC
    bookkeeping used by Dolphin, but implemented independently
    for pyPSDS-GAMMA.

    No compressed complex data are produced at production planner.
    """

    ref_id: str

    source_stage: int

    reference_index: int
    reference_date: str

    start_index: int
    stop_index: int

    start_date: str
    end_date: str

    def as_dict(self):
        return asdict(
            self
        )


@dataclass(
    frozen=True,
    slots=True,
)
class MiniStackStage:
    stage_index: int

    real_indices: tuple[
        int,
        ...
    ]

    real_dates: tuple[
        str,
        ...
    ]

    compressed_inputs: tuple[
        CompressedReference,
        ...
    ]

    output_reference: str

    compressed_output: (
        CompressedReference
        |
        None
    )

    exact_full_scm: bool = False

    @property
    def real_count(self) -> int:
        return len(
            self.real_indices
        )

    @property
    def compressed_count(
        self,
    ) -> int:
        return len(
            self.compressed_inputs
        )

    @property
    def solver_size(self) -> int:
        return (
            self.real_count
            +
            self.compressed_count
        )

    def as_dict(self):

        return {
            "stage_index":
                self.stage_index,

            "real_indices":
                list(
                    self.real_indices
                ),

            "real_dates":
                list(
                    self.real_dates
                ),

            "real_count":
                self.real_count,

            "compressed_inputs":
                [
                    x.as_dict()
                    for x
                    in self.compressed_inputs
                ],

            "compressed_count":
                self.compressed_count,

            "solver_size":
                self.solver_size,

            "output_reference":
                self.output_reference,

            "compressed_output":
                (
                    None
                    if
                    self.compressed_output
                    is None
                    else
                    self.compressed_output
                    .as_dict()
                ),

            "exact_full_scm":
                self.exact_full_scm,
        }


@dataclass(
    frozen=True,
    slots=True,
)
class TemporalPlan:
    dates: tuple[
        str,
        ...
    ]

    requested_strategy: str
    effective_strategy: str

    reference_index: int

    ministack_size: int
    max_num_compressed: int

    stages: tuple[
        MiniStackStage,
        ...
    ]

    exact_collapse: bool

    execution_ready: bool

    decision_reason: str

    @property
    def ndate(self) -> int:
        return len(
            self.dates
        )

    @property
    def max_solver_size(
        self,
    ) -> int:

        if not self.stages:
            return 0

        return max(
            x.solver_size
            for x
            in self.stages
        )

    @property
    def max_compressed_inputs(
        self,
    ) -> int:

        if not self.stages:
            return 0

        return max(
            x.compressed_count
            for x
            in self.stages
        )

    def as_dict(self):

        return {
            "ndate":
                self.ndate,

            "dates":
                list(
                    self.dates
                ),

            "requested_strategy":
                self.requested_strategy,

            "effective_strategy":
                self.effective_strategy,

            "reference_index":
                self.reference_index,

            "ministack_size":
                self.ministack_size,

            "max_num_compressed":
                self.max_num_compressed,

            "stage_count":
                len(
                    self.stages
                ),

            "max_solver_size":
                self.max_solver_size,

            "max_compressed_inputs":
                self.max_compressed_inputs,

            "exact_collapse":
                self.exact_collapse,

            "execution_ready":
                self.execution_ready,

            "decision_reason":
                self.decision_reason,

            "stages":
                [
                    x.as_dict()
                    for x
                    in self.stages
                ],
        }


def _validate(
    dates: Sequence[str],
    *,
    reference_index: int,
    ministack_size: int,
    max_num_compressed: int,
):

    if len(
        dates
    ) < 2:

        raise ValueError(
            "Need at least two acquisitions"
        )

    if not (
        0
        <=
        reference_index
        <
        len(
            dates
        )
    ):

        raise ValueError(
            "reference_index outside stack"
        )

    if ministack_size < 2:

        raise ValueError(
            "ministack_size must be >= 2"
        )

    if max_num_compressed < 1:

        raise ValueError(
            "max_num_compressed must be >= 1"
        )


def _full_scm_plan(
    dates: tuple[
        str,
        ...
    ],
    *,
    requested_strategy: str,
    reference_index: int,
    ministack_size: int,
    max_num_compressed: int,
    exact_collapse: bool,
    reason: str,
) -> TemporalPlan:

    indices = tuple(
        range(
            len(
                dates
            )
        )
    )

    stage = MiniStackStage(
        stage_index=0,

        real_indices=indices,

        real_dates=dates,

        compressed_inputs=(),

        output_reference=(
            f"real:{reference_index}"
        ),

        compressed_output=None,

        exact_full_scm=True,
    )

    return TemporalPlan(
        dates=dates,

        requested_strategy=(
            requested_strategy
        ),

        effective_strategy=(
            TemporalStrategy
            .FULL_SCM
            .value
        ),

        reference_index=(
            reference_index
        ),

        ministack_size=(
            ministack_size
        ),

        max_num_compressed=(
            max_num_compressed
        ),

        stages=(
            stage,
        ),

        exact_collapse=(
            exact_collapse
        ),

        execution_ready=True,

        decision_reason=reason,
    )


def build_temporal_plan(
    dates: Sequence[str],
    *,
    strategy: (
        str
        |
        TemporalStrategy
    ) = TemporalStrategy.FULL_SCM,
    ministack_size: int = 30,
    max_num_compressed: int = 5,
    reference_index: int = 0,
) -> TemporalPlan:
    """
    Build temporal phase-linking execution metadata.

    Build execution metadata for full-SCM or sequential
    ministack phase linking.

    The sequential strategy uses validated compressed-SLC
    propagation between ministacks. AUTO remains deliberately
    unresolved until a general switching rule is established.
    """

    dates = tuple(
        str(x)
        for x
        in dates
    )

    strategy = (
        TemporalStrategy(
            strategy
        )
    )

    _validate(
        dates,
        reference_index=(
            reference_index
        ),
        ministack_size=(
            ministack_size
        ),
        max_num_compressed=(
            max_num_compressed
        ),
    )

    ndate = len(
        dates
    )

    # --------------------------------------------------------
    # AUTO intentionally does not select an algorithm yet.
    # --------------------------------------------------------

    if (
        strategy
        ==
        TemporalStrategy.AUTO
    ):

        return TemporalPlan(
            dates=dates,

            requested_strategy=(
                TemporalStrategy
                .AUTO
                .value
            ),

            effective_strategy=(
                "unresolved"
            ),

            reference_index=(
                reference_index
            ),

            ministack_size=(
                ministack_size
            ),

            max_num_compressed=(
                max_num_compressed
            ),

            stages=(),

            exact_collapse=False,

            execution_ready=False,

            decision_reason=(
                "AUTO is intentionally unresolved in production planner; "
                "the switching rule will be calibrated from "
                "full-scene full-SCM/sequential benchmarks."
            ),
        )

    # --------------------------------------------------------
    # Explicit full SCM.
    # --------------------------------------------------------

    if (
        strategy
        ==
        TemporalStrategy.FULL_SCM
    ):

        return _full_scm_plan(
            dates,
            requested_strategy=(
                strategy.value
            ),
            reference_index=(
                reference_index
            ),
            ministack_size=(
                ministack_size
            ),
            max_num_compressed=(
                max_num_compressed
            ),
            exact_collapse=False,
            reason=(
                "Explicit full_scm strategy."
            ),
        )

    # --------------------------------------------------------
    # Critical exact-collapse rule:
    #
    # sequential framework with M >= N MUST dispatch exactly
    # the already validated full-SCM implementation.
    # --------------------------------------------------------

    if (
        ministack_size
        >=
        ndate
    ):

        return _full_scm_plan(
            dates,
            requested_strategy=(
                TemporalStrategy
                .SEQUENTIAL
                .value
            ),
            reference_index=(
                reference_index
            ),
            ministack_size=(
                ministack_size
            ),
            max_num_compressed=(
                max_num_compressed
            ),
            exact_collapse=True,
            reason=(
                "ministack_size >= ndate: sequential "
                "framework collapses exactly to the existing "
                "full-SCM solver."
            ),
        )

    # --------------------------------------------------------
    # production planner sequential metadata currently assumes the first
    # acquisition is the globally preserved reference.
    #
    # Current validated pyPSDS stack uses reference_index=0.
    # Do not silently invent semantics for another reference.
    # --------------------------------------------------------

    if reference_index != 0:

        raise TemporalPlanningError(
            "Multi-ministack sequential planning currently requires "
            "reference_index=0. Support for an arbitrary "
            "temporal reference must be explicitly derived "
            "and validated before use."
        )

    stages = []

    compressed_history: list[
        CompressedReference
    ] = []

    stage_index = 0

    for start in range(
        0,
        ndate,
        ministack_size,
    ):

        stop = min(
            ndate,
            start
            +
            ministack_size,
        )

        real_indices = tuple(
            range(
                start,
                stop,
            )
        )

        real_dates = tuple(
            dates[i]
            for i
            in real_indices
        )

        compressed_inputs = tuple(
            compressed_history[
                -max_num_compressed:
            ]
        )

        if compressed_inputs:

            # Latest compressed state is the current
            # globally referenced state.
            output_reference = (
                "compressed:"
                +
                compressed_inputs[-1]
                .ref_id
            )

        else:

            output_reference = (
                "real:0"
            )

        comp = CompressedReference(
            ref_id=(
                f"c{stage_index:04d}"
            ),

            source_stage=(
                stage_index
            ),

            # ALWAYS_FIRST semantics:
            # retain global first acquisition as phase origin.
            reference_index=0,

            reference_date=(
                dates[0]
            ),

            start_index=(
                start
            ),

            stop_index=(
                stop
            ),

            start_date=(
                dates[
                    start
                ]
            ),

            end_date=(
                dates[
                    stop - 1
                ]
            ),
        )

        stage = MiniStackStage(
            stage_index=(
                stage_index
            ),

            real_indices=(
                real_indices
            ),

            real_dates=(
                real_dates
            ),

            compressed_inputs=(
                compressed_inputs
            ),

            output_reference=(
                output_reference
            ),

            compressed_output=(
                comp
            ),

            exact_full_scm=False,
        )

        stages.append(
            stage
        )

        compressed_history.append(
            comp
        )

        stage_index += 1

    return TemporalPlan(
        dates=dates,

        requested_strategy=(
            TemporalStrategy
            .SEQUENTIAL
            .value
        ),

        effective_strategy=(
            TemporalStrategy
            .SEQUENTIAL
            .value
        ),

        reference_index=0,

        ministack_size=(
            ministack_size
        ),

        max_num_compressed=(
            max_num_compressed
        ),

        stages=tuple(
            stages
        ),

        exact_collapse=False,

        execution_ready=True,

        decision_reason=(
            "Sequential ministack plan is executable with the "
            "validated EMI/compressed-SLC executor."
        ),
    )


__all__ = [
    "CompressedReference",
    "MiniStackStage",
    "TemporalPlan",
    "TemporalPlanningError",
    "TemporalStrategy",
    "build_temporal_plan",
]
