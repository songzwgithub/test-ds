#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
if [[ "${1:-}" == "--editable" ]]; then
  python -m pip install --no-build-isolation -e .
else
  python -m pip install --no-build-isolation .
fi
python - <<'PY2'
import pypsds
from pypsds.pipeline import STAGES
print("pyPSDS-GAMMA import PASS")
print("version :", pypsds.__version__)
print("stages  :", len(STAGES))
PY2
