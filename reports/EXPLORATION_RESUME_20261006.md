# V6 fixed exploration continuation — 2026-10-06

The user requested `random_action_prob=0.02` and continuation from the latest
complete checkpoint. The previously authorized global top 50% absolute raw
advantage selection is retained in the same continuation. Its earlier pending
item is superseded with an interrupted receipt; frozen inputs remain intact.

## Configuration and state

- V6: 8,014,192 parameters, eight GPUs, 65,536 environments, horizon 64;
  4,194,304 fresh decisions per PPO rollout.
- PPO: one epoch, global minibatch 32,768 (long-history 8,192), gamma 1,
  lambda 0.95, constant LR 1e-5, adaptive value coefficient.
- One Bernoulli choice per complete action: probability 0.02 uses the legal
  random decision tree, otherwise samples the learned policy. Random legal
  nodes are uniform; complete action leaves are not necessarily uniform.
- Rollout log probabilities and PPO scores use the exact complete-action mixture.
  Fresh rollouts are collected after checkpoint restoration.
- Weights, Adam moments and coordinate ages, resident environments, eight RNG
  streams, adaptive value state and EMA are retained. EMA decay remains 0.999,
  with one tick per completed rollout. Save interval remains 100 steps.
- GAE/returns/normalization use complete trajectories before top-50% raw-|adv|
  selection. EV remains a complete-rollout diagnostic.
- With exploration enabled, entropy is estimated by sampled mixture surprisal;
  its metric is not directly identical to the previous conditional-entropy metric.

## Verification and operation

40 focused CPU checks passed: 33 checkpoint/queue/evaluator checks, 6 complete
mixture and actual-game legality cases (including p=0.02), and one eight-CPU
replica own-seat GAE/top-50% PPO and adaptive-value gradient comparison against
a single global reference batch. CPU replicas are engineering evidence only.

The real continuation must pass eight fresh GPU rollout rounds, actual
`nranks=8` and NCCL logs, zero illegal actions/nonfinite values, replica agreement,
exact selected sample counts, optimizer acceptance and EMA clock continuity
before production promotion. The old continuation stays queued until acceptance.
Failure artifacts remain retained; retries require a new explicit queue item.

The local A100 controller release is
`runs/local_a100_v6_explore02_follow_500_65536_v1`; completed benchmarks and
65,536-deal, all-role, every-500-step BEST evaluation are preserved. The same
controller owns the single idle GPU workload while no useful local task runs.

Artifacts: `runs/v6_cluster8_random02_top50_now_v1` and
`runs/exploration_resume_audit_20261006_v1/TESTS_VERIFICATION.json`.
The final checkpoint identity and actual eight-GPU acceptance are recorded in
`EXPLORATION_RESUME_20261006.json` once the continuation passes.

## Accepted continuation

Actual eight-GPU verification passed at relative **50708 / global 77262**.
The source was **50700 / global 77254**; 66 logged, unsaved source rounds remain
archived and were not resumed. The old recovery successor was cancelled only
after acceptance. Production is running toward relative 51700, with automatic
1000-round successors continuing to the unchanged global 200000 target.

All eight fresh rounds used p=0.02 and global top-50% selection, with zero
illegal actions, nonfinite metrics or replica differences. EMA advanced exactly
11700→11708. Peak allocated memory was 33,195,328,768 bytes; maximum round KL
was 0.00361145. Full-rollout value explained variance averaged 0.6668 over the
four timed rounds. These are engineering/short-term health measurements, not
evidence of a playing-strength gain.

Excluding the first compile of each signature, five short-history rounds
averaged 22.58 seconds and one full-history round took 34.97 seconds. The
initial compiles took 172.38 and 132.74 seconds, respectively.
The local evaluator followed the accepted pointer and started the 65,536-deal
BEST benchmark at step 50708. The live raw checkpoint watcher also accepted
this checkpoint without a worker restart.
