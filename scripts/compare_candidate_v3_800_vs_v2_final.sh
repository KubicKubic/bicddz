#!/usr/bin/env bash
set -euo pipefail
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
eval_cpus=$(../.venv-gnn-jax/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[-4:])))')
v3_step=${DDZ_V3_COMPARE_STEP:-800}
v2_step=${DDZ_V2_COMPARE_STEP:-1572}
if [[ "$v2_step" == 1572 ]]; then
  comparison_dir="runs/a100_candidate_set_v3/vs_v2_final_${v3_step}_all_roles_v1"
else
  comparison_dir="runs/a100_candidate_set_v3/vs_v2_${v2_step}_${v3_step}_all_roles_v1"
fi
exec taskset -c "$eval_cpus" nice -n 19 ../.venv-gnn-jax/bin/python -u -m ddz.compare_v3_v2_all_roles \
  --v3-run runs/a100_candidate_set_v3 --v3-step "$v3_step" \
  --v2-run runs/a100_complete_move_v2 --v2-step "$v2_step" \
  --output "$comparison_dir" \
  --deals "${DDZ_COMPARE_DEALS:-144}" --chunk-deals 12 --bootstrap-rounds 2000
