#!/bin/bash
set -euo pipefail


ROOT=$(pwd)

echo "============================================================"
echo "pyPSDS-GAMMA v1.1 structure initialization"
echo "ROOT: ${ROOT}"
echo "============================================================"


mkdir -p \
src/pypsds \
src/pypsds/preparation \
src/pypsds/estimation \
src/pypsds/correction \
src/pypsds/inversion \
src/pypsds/export \
src/pypsds/quality \
scripts \
configs \
docs \
tests \
archive/development_history



touch \
src/pypsds/__init__.py \
src/pypsds/preparation/__init__.py \
src/pypsds/estimation/__init__.py \
src/pypsds/correction/__init__.py \
src/pypsds/inversion/__init__.py \
src/pypsds/export/__init__.py \
src/pypsds/quality/__init__.py



cat > VERSION <<EOF
1.1.0
EOF



cat > CHANGELOG.md <<EOF
# Changelog

## v1.1.0

- Production workflow architecture
- Config-driven processing
- Reproducible pipeline
- Point-level PS/DS products
- Quality control framework

EOF



cat > configs/example.yaml <<EOF
project:
  name: example_area
  root: /path/to/project

sensor:
  type: Sentinel-1

processing:
  workers: 8

output:
  format:
    - parquet
    - geopackage

EOF



echo
echo "Structure created:"
echo

find . \
-maxdepth 3 \
-type d \
| sort


