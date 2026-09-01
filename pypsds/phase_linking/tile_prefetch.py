from __future__ import annotations

import os
import threading
from time import perf_counter
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class _DaemonResult:
    def __init__(self, fn, *args):
        self._event = threading.Event()
        self._value = None
        self._exc = None

        def runner():
            try:
                self._value = fn(*args)
            except BaseException as exc:
                self._exc = exc
            finally:
                self._event.set()

        self._thread = threading.Thread(
            target=runner,
            name="pypsds-phase-prefetch",
            daemon=True,
        )
        self._thread.start()

    def done(self) -> bool:
        return bool(self._event.is_set())

    def result(self, timeout=None):
        if not self._event.wait(timeout):
            raise TimeoutError
        if self._exc is not None:
            raise self._exc
        return self._value


class OneAheadTilePrefetcher(Generic[T]):
    # Exactly one future tile may be resident. The worker is daemonized so a
    # stuck native/GAMMA-backed load cannot block interpreter shutdown forever.

    def __init__(self, *, positions, loader: Callable[[int], T], enabled: bool):
        self.positions = tuple(int(x) for x in positions)

        if tuple(sorted(self.positions)) != self.positions:
            raise ValueError("prefetch positions must be sorted")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("prefetch positions must be unique")

        self.loader = loader
        self.enabled = bool(enabled)
        self._cursor = 0
        self._future = None
        self._future_position = None

        self.read_seconds = 0.0
        self.blocking_seconds = 0.0
        self.wait_seconds = 0.0
        self.scheduler_overhead_seconds = 0.0
        self.completed = 0
        self.abandoned = 0

    @property
    def overlap_seconds(self) -> float:
        return max(0.0, float(self.read_seconds - self.wait_seconds))

    def _timed_load(self, position: int):
        t0 = perf_counter()
        value = self.loader(int(position))
        return value, float(perf_counter() - t0)

    def _submit_current(self) -> None:
        if not self.enabled:
            return
        if self._cursor >= len(self.positions):
            self._future = None
            self._future_position = None
            return

        position = self.positions[self._cursor]
        self._future_position = position
        self._future = _DaemonResult(self._timed_load, position)

    def start(self) -> None:
        if self.enabled and self._future is None and self._cursor < len(self.positions):
            self._submit_current()

    @staticmethod
    def _timeout_seconds():
        raw = os.environ.get(
            "PYPSDS_PREFETCH_FUTURE_TIMEOUT_SECONDS",
            "900",
        )
        try:
            timeout = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                "invalid PYPSDS_PREFETCH_FUTURE_TIMEOUT_SECONDS="
                f"{raw!r}"
            ) from exc
        return None if timeout <= 0 else timeout

    def get(self, position: int) -> T:
        position = int(position)

        if not self.enabled:
            t0 = perf_counter()
            value = self.loader(position)
            elapsed = perf_counter() - t0
            self.read_seconds += float(elapsed)
            self.blocking_seconds += float(elapsed)
            self.wait_seconds += float(elapsed)
            self.completed += 1
            return value

        if self._future is None:
            self.start()

        if self._future is None or self._future_position is None:
            raise RuntimeError("prefetch requested after queue exhausted")

        if int(self._future_position) != position:
            raise RuntimeError(
                "prefetch execution-order mismatch: "
                f"requested={position}, expected={self._future_position}"
            )

        t0 = perf_counter()
        timeout = self._timeout_seconds()

        try:
            value, read_seconds = self._future.result(timeout=timeout)
        except TimeoutError as exc:
            self.abandoned += 1
            self._future = None
            self._future_position = None
            raise RuntimeError(
                "phase prefetch future timed out after "
                f"{timeout}s at position={position}; "
                "daemon worker abandoned; restart from durable checkpoint"
            ) from exc

        blocked = perf_counter() - t0
        self.read_seconds += float(read_seconds)
        self.blocking_seconds += float(blocked)

        io_wait = min(float(blocked), float(read_seconds))
        scheduler_overhead = max(0.0, float(blocked - read_seconds))
        self.wait_seconds += io_wait
        self.scheduler_overhead_seconds += scheduler_overhead

        self.completed += 1
        self._cursor += 1
        self._future = None
        self._future_position = None

        # Submit next tile immediately so loader can overlap current CPU work.
        self._submit_current()
        return value

    def close(self) -> None:
        # Never wait indefinitely. A normal completed run has no outstanding
        # future. On an error an unfinished daemon worker is abandoned.
        if self._future is not None and not self._future.done():
            self.abandoned += 1
        self._future = None
        self._future_position = None


__all__ = ["OneAheadTilePrefetcher"]
