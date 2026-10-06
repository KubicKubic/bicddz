# Restore model sampling and raise entropy — 2026-10-07

The user requested restoring the original action selection and increasing the
model's entropy regularization. The registered continuation disables the fixed
random-action branch (`ppo.random_action_prob: 0.02 -> 0`) and raises
`ppo.entropy_end: 0.002 -> 0.01`. The existing entropy schedule is already past
its endpoint, so the effective coefficient becomes 0.01, without a new warmup.
The p=0 path restores the original learned-policy sampling, full-action log
probabilities and original conditional entropy calculation.

## Preserved state and settings

V6 remains 8,014,192 parameters on eight A100 GPUs. Keep 65,536 environments,
horizon 64, 4,194,304 fresh decisions per rollout, global short-history batch
32,768/full-history batch 8,192, one epoch, gamma 1, lambda .95, LR 1e-5,
adaptive vf, global top-50% absolute raw advantage selection and global target
200000. EMA decay remains .999 with one tick per completed rollout; saving
remains every 100 steps. Retain weights, Adam moments/coordinate ages, resident
environments, eight RNG streams, adaptive vf and EMA state.

The source task is allowed to finish the next complete 100-step checkpoint
(relative 50800) before the user-directed pause, to minimize unsaved work.
Frozen old inputs and historical results remain retained. The same allocation
is recovered with one persistent queue worker; all training tasks are published
through the immutable FIFO helper. The old recovery successor stays queued
until eight fresh real GPU rounds pass.

## Verification and deployment

46 focused checks passed, including retained-state serialization, exact two-field
configuration changes, the effective entropy schedule, the original p=0 sampling
path, source ownership/process-exit gates and registered local evaluator handoff.
One test fixture initially omitted the production config; it was corrected and
that test alone was rerun. Unchanged model/optimizer and prior collective
engineering proofs are reused.

Promotion requires real nranks=8/NCCL evidence, eight fresh completed rollouts,
p=0 and entropy coefficient .01 in every row, exact global top-50% counts,
optimizer acceptance, finite/legal actions, replica agreement, memory headroom
and exactly eight rollout EMA ticks. These gates establish engineering health;
they do not establish an improvement in DouZero score.

The local 65,536-deal BEST benchmark in progress is preserved. The local controller
release waits for it to finish before taking GPU ownership. A frozen old
controller's expected registry rejection may be reviewed only against the
accepted entropy continuation and a completed, nonfailed benchmark; failed
benchmarks are never automatically retried. Completed curve history is retained.
The single controller continues every-500-step all-role evaluation and idle GPU
occupation outside useful local work.

Artifacts:
- `runs/v6_cluster8_entropy01_top50_now_v1`
- `runs/entropy_resume_audit_20261007_v1/TESTS_VERIFICATION.json`
- `runs/local_a100_v6_entropy01_follow_500_65536_v1`

The final acceptance and checkpoint identities are recorded in
`ENTROPY_RESUME_20261007.json` after verification.

## Accepted continuation

Actual eight-GPU verification passed at relative **50808 / global 77362**,
resuming the full checkpoint **50800 / global 77354**. One boundary round was
logged but unsaved; its original row is archived and not resumed. The recovery
successor was cancelled only after acceptance. Production continues toward
relative 51800, with automatic 1000-round successors to global 200000.

All eight rounds used p=0 and entropy coefficient .01, with exact top-50% sample
counts, all planned updates accepted, and zero illegal actions, nonfinite metrics
or replica differences. Both short-history and full 192-history paths ran.
EMA advanced exactly 11800→11808; peak allocated memory was 33,257,187,328 bytes.
Maximum round KL was .00326053; value explained variance averaged .71098 over
the four timed rounds. Logged original-policy entropy was approximately .067
nats; it is not directly comparable to the mixed-policy entropy metric.

The prior 65,536-deal/393,216-game BEST evaluation completed before replacing
its controller. The old registry rejected the new configuration after completion;
that specific registry failure was reviewed against the accepted handoff and
completed benchmark. No failed benchmark was retried. The new controller has
started the initial step-50808 benchmark and retains all completed history,
500-step evaluation and idle occupation. The QOJ raw watcher accepted step
50808 without restarting its worker.

The long-term production process independently wrote fresh rounds after acceptance.
At the report snapshot it was running at relative **50811 / global
77365**, with eight ranks; the current status is retained in the JSON
proof index.
