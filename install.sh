#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

if [[ "${1:-}" == "--install-qemu-deps" ]]; then
  python -m pip install --no-cache-dir -r requirements-qemu.txt
fi

# Do not let pip replace a known-good NumPy/SciPy stack unless explicitly requested above.
python -m pip install --no-build-isolation --no-deps -e .

python - <<'PY'
import pypsds
print("pyPSDS-GAMMA import PASS")
print("version:", pypsds.__version__)
import numpy, scipy, numba, llvmlite
print("NumPy   :", numpy.__version__)
print("SciPy   :", scipy.__version__)
print("Numba   :", numba.__version__)
print("llvmlite:", llvmlite.__version__)
print("threads :", numba.get_num_threads())
PY
