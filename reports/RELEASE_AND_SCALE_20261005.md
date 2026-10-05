# 2026-10-05 DDZ release and immediate training resize

## Model publication

- Raw relative step 47608, global step 74162.
- Raw, EMA and complete resume state match one atomic checkpoint, with hashes in `models/release.json`.
- EMA decay 0.999; retained rollout counter 8608.
- `.msgpack` files use Git LFS. Remote: https://github.com/KubicKubic/bicddz.git, branch `main`.
- Git authentication uses the user's external `id_ed25519_bicddz`, SSH port 443 and the local HTTP CONNECT proxy. Credentials are not stored in Git.
- Online client V8 verified against QOJ with this raw policy; 7 accepted actions and no fallback/automatic play at the recorded check.
- The earlier CPU/legal-move release verification remains retained at its original recorded global step; the new release additionally passed actual eight-GPU PPO and live QOJ execution.

## Immediate eight-GPU resize

- Complete source checkpoint: relative 47600 / global 74154. User explicitly requested immediate switching.
- Cancelled the previous pending boundary-48000 task; interrupted only the active DDZ segment, then recovered the same remote queue worker.
- Eight A100-SXM4-80GB devices, actual free memory query and real NCCL initialization verified.
- Environments 32768 → 65536; nominal global PPO minibatch 16384 → 32768; horizon stays 64.
- Fresh decisions per rollout 2097152 → 4194304; total target remains 200000 global updates.
- Exact retention verified for weights, Adam, RNG, each rank's existing environments and EMA.
- One logged but unsaved source update is excluded from the resumed metrics; original source artifacts remain retained.
- Eight actual new rollout rounds passed: invalid actions 0, nonfinite 0, replica difference 0, EMA rollout counter 8600 → 8608.
- Recorded peak 21.85 GiB per GPU; value explained variance 0.7151.
- Warmed mixed-history throughput over five timed rounds: 165869 fresh decisions/second. This short sample does not prove a throughput or playing-strength gain.
- Nominal short-history batch is 32768. The existing complete long-history branch retains its conservative effective global batch 8192; no history is truncated.
- Long training is running under `runs/v5_cluster8_env2x_batch2x_now_v2`; raw policy publication continues to the established local 500-step evaluator.
- Full evidence: `RESIZE_IMMEDIATE_20261005.json` and retained shared queue logs/manifests.
