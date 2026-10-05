#!/usr/bin/env bash
set -euo pipefail
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
eval_cpus=$(../.venv-gnn-jax/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[-16:-8])))')
exec taskset -c "$eval_cpus" nice -n 19 ../.venv-gnn-jax/bin/python -u -m ddz.score_ladder_graph_all_roles_v5 \
  --run-dir runs/a100_interaction_complete_move_v5 \
  --output runs/a100_interaction_complete_move_v5/score_ladder_all_roles_500_v1 \
  --deals 288 --chunk-deals 12 --bootstrap-rounds 2000 --watch --poll-seconds 60
