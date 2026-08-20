from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import time
from time import perf_counter
from typing import Callable


# ============================================================================
# Shared formatting / memory helpers
# ============================================================================

def _fmt_seconds(
    seconds: float | None,
) -> str:

    if (
        seconds is None
        or
        not (seconds >= 0)
    ):
        return "--:--:--"

    sec = int(
        round(seconds)
    )

    h, rem = divmod(
        sec,
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return (
        f"{h:02d}:"
        f"{m:02d}:"
        f"{s:02d}"
    )


def _rss_mb() -> float | None:
    """
    Current resident memory on Linux without psutil.
    """

    try:
        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.startswith(
                    "VmRSS:"
                ):
                    return (
                        float(
                            line.split()[1]
                        )
                        /
                        1024.0
                    )

    except OSError:
        pass

    return None


def current_rss_bytes() -> int:
    """
    Current resident memory in bytes.

    Returns zero if /proc is unavailable.
    """

    rss = _rss_mb()

    if rss is None:
        return 0

    return int(
        rss
        *
        1024**2
    )


def _mem_text() -> str:

    rss = _rss_mb()

    if rss is None:
        return "RSS=--"

    return (
        f"RSS={rss:.1f} MiB"
    )


# ============================================================================
# ORIGINAL production API
#
# Keep these interfaces stable because GAMMA/phase-source modules use them.
# ============================================================================

def log(
    message: str,
    *,
    sink: Callable[[str], None] = print,
) -> None:

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    sink(
        f"[{now}] {message}"
    )


@dataclass
class ProgressTracker:
    """
    Existing lightweight progress API.

    Kept for backward compatibility with the current production package.
    """

    label: str
    total: int

    sink: Callable[[str], None] = print

    start_time: float = field(
        default_factory=perf_counter
    )

    last_time: float = field(
        default_factory=perf_counter
    )

    def update(
        self,
        completed: int,
        *,
        detail: str = "",
    ) -> None:

        now = perf_counter()

        elapsed = (
            now
            -
            self.start_time
        )

        completed = max(
            0,
            min(
                int(completed),
                int(self.total),
            ),
        )

        rate = (
            completed
            /
            elapsed
            if
            elapsed > 0
            and
            completed > 0
            else
            0.0
        )

        remain = (
            (
                self.total
                -
                completed
            )
            /
            rate
            if rate > 0
            else None
        )

        pct = (
            100.0
            *
            completed
            /
            self.total
            if self.total
            else 100.0
        )

        suffix = (
            f" | {detail}"
            if detail
            else ""
        )

        log(
            (
                f"{self.label}: "
                f"{completed}/{self.total} "
                f"({pct:6.2f}%) "
                f"elapsed={_fmt_seconds(elapsed)} "
                f"ETA={_fmt_seconds(remain)} "
                f"rate={rate:.3f}/s "
                f"{_mem_text()}"
                f"{suffix}"
            ),
            sink=self.sink,
        )

        self.last_time = now

    def done(
        self,
        *,
        detail: str = "",
    ) -> None:

        self.update(
            self.total,
            detail=detail,
        )


class StepTimer:

    def __init__(
        self,
        name: str,
        *,
        sink: Callable[[str], None] = print,
    ):

        self.name = name
        self.sink = sink
        self.t0 = 0.0

    def __enter__(self):

        self.t0 = perf_counter()

        log(
            (
                f"START {self.name} "
                f"| PID={os.getpid()} "
                f"{_mem_text()}"
            ),
            sink=self.sink,
        )

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        elapsed = (
            perf_counter()
            -
            self.t0
        )

        status = (
            "OK"
            if exc_type is None
            else "FAILED"
        )

        log(
            (
                f"END   {self.name} "
                f"| status={status} "
                f"elapsed={_fmt_seconds(elapsed)} "
                f"{_mem_text()}"
            ),
            sink=self.sink,
        )

        return False


# ============================================================================
# P8A high-volume production reporter
# ============================================================================

@dataclass(slots=True)
class ProgressReporter:
    """
    Throttled progress reporter for large production jobs.

    Features
    --------
    - completion percentage
    - elapsed time
    - smoothed ETA
    - throughput
    - current RSS
    - optional JSONL progress stream

    Numerical processing is untouched: this class is reporting only.
    """

    label: str
    total: int

    unit: str = "item"

    min_interval: float = 10.0

    log_path: (
        str
        |
        Path
        |
        None
    ) = None

    _t0: float = field(
        init=False,
        repr=False,
    )

    _last_print: float = field(
        init=False,
        repr=False,
    )

    _last_done: int = field(
        init=False,
        repr=False,
    )

    _last_rate: (
        float
        |
        None
    ) = field(
        init=False,
        repr=False,
        default=None,
    )

    def __post_init__(self):

        self.total = max(
            0,
            int(
                self.total
            ),
        )

        self.min_interval = max(
            0.0,
            float(
                self.min_interval
            ),
        )

        self._t0 = perf_counter()

        self._last_print = (
            self._t0
        )

        self._last_done = 0

        self._last_rate = None

        if self.log_path is not None:

            self.log_path = Path(
                self.log_path
            )

            self.log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.log_path.write_text(
                "",
                encoding="utf-8",
            )

    def update(
        self,
        done: int,
        *,
        force: bool = False,
        detail: str = "",
    ) -> None:

        now = perf_counter()

        done = max(
            0,
            min(
                int(done),
                self.total,
            ),
        )

        if (
            not force
            and
            done < self.total
            and
            (
                now
                -
                self._last_print
            )
            <
            self.min_interval
        ):
            return

        elapsed = max(
            now
            -
            self._t0,
            1e-12,
        )

        overall_rate = (
            done
            /
            elapsed
            if done > 0
            else 0.0
        )

        delta_done = (
            done
            -
            self._last_done
        )

        delta_t = max(
            now
            -
            self._last_print,
            1e-12,
        )

        instant_rate = (
            delta_done
            /
            delta_t
            if delta_done > 0
            else overall_rate
        )

        if instant_rate > 0:

            if self._last_rate is None:

                rate = (
                    instant_rate
                )

            else:

                # Smoothed ETA for large heterogeneous tiles/batches.
                rate = (
                    0.30
                    *
                    instant_rate
                    +
                    0.70
                    *
                    self._last_rate
                )

            self._last_rate = rate

        else:

            rate = (
                self._last_rate
                if
                self._last_rate
                is not None
                else
                overall_rate
            )

        remaining = max(
            0,
            self.total
            -
            done,
        )

        eta = (
            remaining
            /
            rate
            if
            rate
            and
            rate > 0
            else None
        )

        pct = (
            100.0
            *
            done
            /
            self.total
            if self.total > 0
            else 100.0
        )

        rss_bytes = (
            current_rss_bytes()
        )

        rss_text = (
            f"{rss_bytes / 1024**3:.2f} GiB"
            if rss_bytes > 0
            else "n/a"
        )

        line = (
            f"[{self.label}] "
            f"{done:,}/"
            f"{self.total:,} "
            f"({pct:6.2f}%)"
            f" | elapsed "
            f"{_fmt_seconds(elapsed)}"
            f" | ETA "
            f"{_fmt_seconds(eta)}"
            f" | {rate:,.0f} "
            f"{self.unit}/s"
            f" | RSS {rss_text}"
        )

        if detail:

            line += (
                f" | {detail}"
            )

        print(
            line,
            flush=True,
        )

        if self.log_path is not None:

            rec = {
                "time_unix":
                    time.time(),

                "label":
                    self.label,

                "done":
                    done,

                "total":
                    self.total,

                "percent":
                    pct,

                "elapsed_seconds":
                    elapsed,

                "eta_seconds":
                    eta,

                "rate_per_second":
                    rate,

                "unit":
                    self.unit,

                "rss_bytes":
                    rss_bytes,

                "detail":
                    detail,
            }

            with self.log_path.open(
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )

        self._last_done = done

        self._last_print = now

    def finish(
        self,
        done: int | None = None,
        *,
        detail: str = "",
    ) -> None:

        if done is None:
            done = self.total

        done = max(
            0,
            min(
                int(done),
                self.total,
            ),
        )

        # Avoid duplicate 100% records when the final batch
        # already emitted the completion state.
        if done == self._last_done:
            return

        self.update(
            done,
            force=True,
            detail=detail,
        )


__all__ = [
    # Existing public contract.
    "ProgressTracker",
    "StepTimer",
    "log",

    # P8A additions.
    "ProgressReporter",
    "current_rss_bytes",
]
