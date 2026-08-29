"""Canonical scientific visualization implementation.

Public callers should normally use :mod:`pypsds.monitoring.stage_visualization`.
This package contains the validated internal renderer chain with production
names only; development-version module names are intentionally absent.
"""

from .renderers import (
    VISUALIZATION_PROFILE,
    render_override,
)

__all__ = [
    "VISUALIZATION_PROFILE",
    "render_override",
]
