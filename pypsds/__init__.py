"""pyPSDS-GAMMA: CPU/RAM-oriented PS/DS InSAR processing for GAMMA."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pypsds-gamma")
except PackageNotFoundError:
    __version__ = "1.1.0.dev0"

__all__ = ["__version__"]
