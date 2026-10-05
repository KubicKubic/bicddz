#!/usr/bin/env bash
set -euo pipefail
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
export JAX_PLATFORMS=cpu
exec ../.venv-gnn-jax/bin/python -u -m ddz.score_ladder_graph_v3 \
  --run-dir runs/a100_candidate_set_v3 \
  --output runs/a100_candidate_set_v3/score_ladder_graph_500_v2 \
  --deals 288 --chunk-deals 24 --bootstrap-rounds 2000 \
  --watch --poll-seconds 60
