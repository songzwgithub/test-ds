from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from time import perf_counter
import os
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class OneAheadTilePrefetcher(
    Generic[T]
):
    # Bounded one-ahead asynchronous tile prefetch.
    #
    # Exactly one future tile may be resident in addition to the
    # current compute tile.  positions must be supplied in exact
    # execution order.

    def __init__(
        self,
        *,
        positions,
        loader: Callable[[int], T],
        enabled: bool,
    ):
        self.positions = tuple(
            int(x)
            for x in positions
        )

        if (
            tuple(
                sorted(
                    self.positions
                )
            )
            !=
            self.positions
        ):
            raise ValueError(
                "prefetch positions must be sorted"
            )

        if (
            len(
                set(
                    self.positions
                )
            )
            !=
            len(
                self.positions
            )
        ):
            raise ValueError(
                "prefetch positions must be unique"
            )

        self.loader = loader
        self.enabled = bool(
            enabled
        )

        self._cursor = 0
        self._executor = None
        self._future = None
        self._future_position = None

        # read_seconds:
        #     wall time spent inside the actual loader.
        #
        # blocking_seconds:
        #     raw caller-side time blocked on future.result().
        #     This includes thread wake/scheduling overhead and can
        #     therefore be slightly larger than read_seconds.
        #
        # wait_seconds:
        #     the portion of loader I/O that remained visible to the
        #     compute thread. It is capped at the measured loader time.
        #
        # scheduler_overhead_seconds:
        #     blocking time not attributable to loader I/O.
        self.read_seconds = 0.0
        self.blocking_seconds = 0.0
        self.wait_seconds = 0.0
        self.scheduler_overhead_seconds = 0.0
        self.completed = 0

    @property
    def overlap_seconds(
        self,
    ) -> float:
        return max(
            0.0,
            float(
                self.read_seconds
                -
                self.wait_seconds
            ),
        )

    def _timed_load(
        self,
        position: int,
    ):
        t0 = perf_counter()

        value = self.loader(
            int(position)
        )

        elapsed = (
            perf_counter()
            -
            t0
        )

        return (
            value,
            float(elapsed),
        )

    def _submit_current(
        self,
    ) -> None:
        if not self.enabled:
            return

        if (
            self._cursor
            >=
            len(
                self.positions
            )
        ):
            self._future = None
            self._future_position = None
            return

        if self._executor is None:
            self._executor = (
                ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=(
                        "pypsds-phase-prefetch"
                    ),
                )
            )

        position = self.positions[
            self._cursor
        ]

        self._future_position = (
            position
        )

        self._future = (
            self._executor.submit(
                self._timed_load,
                position,
            )
        )

    def start(
        self,
    ) -> None:
        if (
            self.enabled
            and
            self._future is None
            and
            self._cursor
            <
            len(
                self.positions
            )
        ):
            self._submit_current()

    def get(
        self,
        position: int,
    ) -> T:
        position = int(
            position
        )

        if not self.enabled:
            t0 = perf_counter()

            value = self.loader(
                position
            )

            elapsed = (
                perf_counter()
                -
                t0
            )

            self.read_seconds += float(
                elapsed
            )

            self.blocking_seconds += float(
                elapsed
            )

            self.wait_seconds += float(
                elapsed
            )

            self.completed += 1

            return value

        if self._future is None:
            self.start()

        if (
            self._future is None
            or
            self._future_position
            is None
        ):
            raise RuntimeError(
                "prefetch requested after queue exhausted"
            )

        if (
            int(
                self._future_position
            )
            !=
            position
        ):
            raise RuntimeError(
                "prefetch execution-order mismatch: "
                f"requested={position}, "
                f"expected={self._future_position}"
            )

        t0 = perf_counter()

        # FASTPATCH: do not wait forever on the experimental
        # asynchronous canonical streamer.
        raw_timeout = os.environ.get(
            "PYPSDS_PREFETCH_FUTURE_TIMEOUT_SECONDS",
            "900",
        )

        try:
            future_timeout = float(
                raw_timeout
            )
        except ValueError as exc:
            raise RuntimeError(
                "invalid "
                "PYPSDS_PREFETCH_FUTURE_TIMEOUT_SECONDS="
                f"{raw_timeout!r}"
            ) from exc

        if future_timeout <= 0:
            future_timeout = None

        try:
            (
                value,
                read_seconds,
            ) = self._future.result(
                timeout=future_timeout,
            )

        except FutureTimeoutError as exc:
            raise RuntimeError(
                "phase prefetch future timed out after "
                f"{future_timeout}s "
                f"at position={position}; "
                "restart from the durable sequential checkpoint"
            ) from exc

        blocked = (
            perf_counter()
            -
            t0
        )

        self.read_seconds += float(
            read_seconds
        )

        self.blocking_seconds += float(
            blocked
        )

        # future.result() includes scheduler/wakeup latency in addition
        # to any unfinished loader work.  Only the loader-duration part
        # is attributable to visible I/O.
        io_wait = min(
            float(
                blocked
            ),
            float(
                read_seconds
            ),
        )

        scheduler_overhead = max(
            0.0,
            float(
                blocked
                -
                read_seconds
            ),
        )

        self.wait_seconds += (
            io_wait
        )

        self.scheduler_overhead_seconds += (
            scheduler_overhead
        )

        self.completed += 1
        self._cursor += 1

        self._future = None
        self._future_position = None

        # Submit the next tile immediately so its I/O can overlap
        # support/coherence/EMI/compression for the current tile.
        self._submit_current()

        return value

    def close(
        self,
    ) -> None:
        if self._future is not None:
            try:
                self._future.result()

            finally:
                self._future = None
                self._future_position = None

        if self._executor is not None:
            self._executor.shutdown(
                wait=True,
                cancel_futures=False,
            )

            self._executor = None


__all__ = [
    "OneAheadTilePrefetcher",
]
