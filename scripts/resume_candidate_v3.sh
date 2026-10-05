#!/usr/bin/env bash
set -euo pipefail
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
exec ../.venv-gnn-jax/bin/python -u -m ddz.train_v3 \
  --config configs/a100_v3.json \
  --out runs/a100_candidate_set_v3 \
  --resume --require-a100 --memory-limit 88 --ppo-epochs 3
