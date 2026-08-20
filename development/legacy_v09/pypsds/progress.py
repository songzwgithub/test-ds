from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Callable
import os


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or not (seconds >= 0):
        return "--"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _rss_mb() -> float | None:
    """Current resident memory on Linux without adding a psutil dependency."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _mem_text() -> str:
    rss = _rss_mb()
    return "RSS=--" if rss is None else f"RSS={rss:.1f} MiB"


def log(message: str, *, sink: Callable[[str], None] = print) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sink(f"[{now}] {message}")


@dataclass
class ProgressTracker:
    label: str
    total: int
    sink: Callable[[str], None] = print
    start_time: float = field(default_factory=perf_counter)
    last_time: float = field(default_factory=perf_counter)

    def update(self, completed: int, *, detail: str = "") -> None:
        now = perf_counter()
        elapsed = now - self.start_time
        completed = max(0, min(int(completed), int(self.total)))
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        remain = (self.total - completed) / rate if rate > 0 else None
        pct = 100.0 * completed / self.total if self.total else 100.0
        suffix = f" | {detail}" if detail else ""
        log(
            f"{self.label}: {completed}/{self.total} ({pct:6.2f}%) "
            f"elapsed={_fmt_seconds(elapsed)} ETA={_fmt_seconds(remain)} "
            f"rate={rate:.3f}/s {_mem_text()}{suffix}",
            sink=self.sink,
        )
        self.last_time = now

    def done(self, *, detail: str = "") -> None:
        self.update(self.total, detail=detail)


class StepTimer:
    def __init__(self, name: str, *, sink: Callable[[str], None] = print):
        self.name = name
        self.sink = sink
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = perf_counter()
        log(f"START {self.name} | PID={os.getpid()} {_mem_text()}", sink=self.sink)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.t0
        status = "OK" if exc_type is None else "FAILED"
        log(
            f"END   {self.name} | status={status} elapsed={_fmt_seconds(elapsed)} {_mem_text()}",
            sink=self.sink,
        )
        return False


__all__ = ["ProgressTracker", "StepTimer", "log"]
