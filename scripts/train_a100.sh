#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PY="${PY:-/mnt/pfs/guoyuchong/guoyuchong/.venv-gnn-jax/bin/python}"
exec "$PY" -u -m ddz.train --config configs/a100.json --require-a100 "$@"
