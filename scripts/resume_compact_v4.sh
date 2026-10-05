#!/usr/bin/env bash
set -euo pipefail
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
exec ../.venv-gnn-jax/bin/python -u -m ddz.train_v4 \
  --config configs/a100_v4.json --out runs/a100_compact_complete_move_v4 \
  --resume --require-a100
