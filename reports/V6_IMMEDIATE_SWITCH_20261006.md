# V6 immediate switch — 2026-10-06

The user requested immediate switching, superseding the old relative-50600 boundary.
The current replacement immutable FIFO item is `01791265643288461199-667d0d7633e34e678fb802e8aa160455`, under `runs/v6_cluster8_attention_8m_now_v3`. Previous failed items are retained under their original roots.

The owned V5 segment was interrupted and the queue worker was recovered in the same remote eight-GPU allocation. The retained complete checkpoint is relative **50000**, global **76554**, SHA-256 `c8c1d3acea1757e175cf9e6101cfa3b66af0193e45d8bf092a9180bcff30b1dc`. The 59 logged updates beyond that checkpoint were unsaved; their original metrics remain in `source_metrics_before_pause.jsonl` and are excluded from resumed training metrics. No optimizer or EMA progress is fabricated for them.

The migration preserves the full checkpoint's weights, inherited Adam coordinates and ages, EMA, rollout counter, all resident environments, RNG, arena state and adaptive value coefficient. The architecture and PPO settings remain the previously audited 8,014,192-parameter preset, 65,536 environments, horizon 64 and global minibatch 32,768; long-history minibatches retain their existing cap.

Immediate migration checks source checkpoint/status/log checksums, the interrupted item's identity, actual remote process absence, source `nranks=8` and NCCL proof, and eight-GPU resource headroom. Nine new pause/ownership rejection tests plus three existing continuation/supersession tests passed: **12 passed in 17.62 seconds**. Previous architecture audit proofs remain retained; they are reused, not recounted as new tests.

A V5 continuation to relative 51000 remains queued behind the migration. Only after CPU function preservation, CUDA probes, both full backward signatures and eight actual healthy PPO rounds pass does V6 cancel that continuation and publish the production pointer. Failure retains the failed V6 item and the V5 recovery path; no migration is retried silently.

**V6 passed eight actual eight-GPU PPO rounds and took over production at relative 50008 / global 76562.** The V5 fallback was cancelled only after acceptance. The campaign continues to the original global-200000 target, with 100-step saving and rollout-clock EMA.

## Reviewed startup revisions

The first candidate preserved raw/EMA outputs exactly in CPU FP32 and CUDA BF16, passed both backward signatures and completed eight actual eight-GPU rounds. Round 50006 triggered the existing KL stop, applying 21/512 minibatches (4.1015625%); the unchanged 99% acceptance requirement rejected the candidate before production promotion. Source V5 had seven partially stopped rounds among its prior 600 updates, with average acceptance 99.60%, so early stopping alone is not a numerical failure; the low acceptance still makes this candidate unsuitable for the startup gate.

The new candidate ramps newly born Adam-coordinate updates over 1024 applied optimizer steps. Mature coordinates retain their updates, old moments/ages remain intact and the global learning rate stays 1e-5. The runtime persists the warmup length and requires it on resume. This is a targeted startup treatment, not proven long-term improvement. The model, rollout budget, EMA rollout clock and original acceptance threshold remain the same. Maximum minibatch KL and the KL-stop flag are now recorded.

The second queue item stopped before V6 training because the interrupted wrapper had exited while its trainer was still finishing a round. The revised control derives the actual child PID from the active NCCL log, waits up to 180 seconds for exit, then rechecks retained checkpoint/status checksums and free GPU memory. It never launches a competing trainer. The CPU control, coordinate warmup, cold-restore and KL diagnostic checks passed 17 cases. The original first interruption's 59 unsaved rounds and the subsequent recovery attempts' four and two unsaved rounds are separately retained; these counts are separate attempts, not additional unique global-update progress.

## Accepted eight-GPU startup

All eight rounds applied 100% of their planned minibatches; no KL stop occurred. Illegal actions and nonfinite results were zero, raw and EMA replicas agreed, and both history training signatures were exercised. CPU FP32 and CUDA BF16 raw/EMA migration outputs remained exact on 128 actual states. The following figures describe startup engineering, not improved playing strength or long-term stability.

- Warmed value explained variance: 0.7139.
- Largest minibatch KL: 0.02332, below the unchanged 0.03 stop threshold.
- Peak GPU memory: 30.93 GiB per device, under the 64 GiB allocator budget.
- Five warmed rounds: rollout 8.47s, training 42.81s; about 81,552 fresh decisions/s. This small mixture of short/full-history rounds is not a matched throughput comparison; it is slower than the prior V5 recent average.
- Saved EMA counter advanced 11000→11008 with decay 0.999; the newborn-coordinate warmup clock is separately preserved in runtime.

The default `models/` bundle now contains accepted V6 raw weights, EMA and the full resume checkpoint at global 76562. Its prior bundle is retained under `runs/model_exports/previous_models_before_v6_76562_20261006`. The single local A100 evaluator followed the verified pointer and started the V6 step-50008 all-role DouZero BEST benchmark.
