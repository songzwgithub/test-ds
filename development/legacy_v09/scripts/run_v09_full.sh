#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/pypsds_v09.yaml}

export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-32}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python scripts/00_check_environment.py
python scripts/01_check_gamma_stack.py --config "$CONFIG"
python scripts/02_build_moraine_center_prior_v09.py --config "$CONFIG"
python scripts/03_build_phase_cache_v09.py --config "$CONFIG" --workers 8
python scripts/04_run_psds_v09.py \
  --config "$CONFIG" \
  --center-mode moraine \
  --half-row 5 --half-col 11 \
  --alpha 0.005 --min-shp 48 --adi-max 0.25 \
  --beta 0.05 --batch-size 16000 \
  --pl-workers 16 --pl-chunk-size 512

cat <<'EOF'

Core v0.9 processing finished.
No universal final-DS threshold is applied automatically.
Run Step05 explicitly, for example:
  python scripts/05_select_ds_v09.py --config <yaml> --tc-min 0.75 --pair-min 0.20 --accept-evd
Then build PointPhaseStack with Step06.
EOF
