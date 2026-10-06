# Local A100 operation (explicit DDZ user instruction)

The user requires the local A100 to run an idle GPU workload at full utilization
whenever it is not running useful local GPU work. Retain this preference across
sessions. The current owner and artifacts are in `runs/local_a100_current.json`.

Use one local GPU controller. Stop and wait for its owned idle workload before
starting evaluation or another authorized GPU task, then restore idle occupation
when real work ends. Respect other active local GPU jobs and preserve stopped
processes and their retained GPU state. Do not launch competing occupation
workers or kill unrelated processes. Verify occupation using the driver-matched
NVML library (`LD_LIBRARY_PATH=/tmp/ddz_nvml`), since the default NVML library
currently mismatches the kernel driver.

DDZ user explicitly authorizes single local A100 evaluations against pinned
EdwardPooh DouZero ResNet 2.0 best weights. Eight-GPU training remains on the
persistent remote queue. Keep formal Q research under the root Q rules.

Continue the every-500-update DouZero benchmark while running large frozen
checkpoint verification. Use fresh validation deals, pair all six role/control
legs within each deal, and report all-role expected score with confidence
intervals. Do not count repeatedly reused common deals as independent samples.

## Model weight EMA (explicit DDZ user instruction)

Default EMA decay is 0.999. Its clock is completed PPO rollout rounds, not
optimizer minibatches or individual environment transitions: advance once per
round, including rounds with rejected optimizer minibatches. Save EMA weights,
decay and the rollout counter with each full checkpoint, and restore them on
resume. Keep separate EMA policy snapshots under the training/ema/ directory
so the existing raw-policy watcher does not mistake them for new checkpoints.

## Persistent QOJ rated-match bot (explicit user instruction)

The user requires Fortune to play QOJ rated matches continuously in tmux.
The session is `ddz_qoj_match`; deployment and live status are under
`runs/qoj_match_v1/`. Use `scripts/qoj_match.sh prepare` to freeze a verified
`models/` release; `launch_current.sh` then selects its immutable worker/code/model
under `releases/`. `prepared_deployment.json` records the next prepared release;
`deployment.json` records the release actually verified live. Inspect both and
the active process before restarting. The portable worker is `deploy/qoj/worker.py`
(client version 9); older V7/V8 files remain historical recovery artifacts.
The raw V5/V6 policy uses CPU inference. The launcher sources the
deployment's private `network.env`: the long-lived tmux server does not inherit
the editing process's proxy variables automatically. Respect its account lock
and inspect the current process before launching another QOJ client.
The Bearer key is in `/root/.config/ddz/qoj_api_key` with mode 0600; never print
it or include it in logs. The live server reports 9 rounds per match.
The client automatically moves to the next game and queues after each match.

The main tmux pane displays the live dashboard; the second pane tails
`console.log`. `dashboard.html` exposes the same read-only state with complete
model input tensors. Model values are expected final raw score deltas and must
be labeled with their game/version; do not present them as win probabilities.
Temporary network/protocol faults keep the compiled policy. Unknown POST
outcomes fetch authoritative state before deciding again. Same-version action
rejections and adapter faults use explicitly logged legal fallback decisions.
The supervisor restarts its owned worker on process exit or stale progress;
authentication rejection or account-lock conflict stops for inspection.

Canonicalize all masked history padding to zero so absolute cyclic seat labels leave the complete Observation identical. In the own-seat GAE deployment, only display the acting player's trained value directly; derive other final scores from the known team payout ratio, and hide their scores before landlord selection. Do not treat the raw other critic heads as independently supervised values.

User requires every new completed checkpoint to replace the live raw model.
V9's `ddz/qoj_checkpoint_watch.py` polls `runs/production_current.json` every
5 seconds, follows the atomic `latest.json` completion marker, and loads/checks/
warms weights in a background thread. Commit prepared weights only between
requests or decisions using a parameterized JIT; never restart for each
checkpoint. Keep the prior model on failure and record updates in events and
console logs. `active_model.json` is the persisted, checksum-bound last accepted
model, restored after worker restart; `deployment.json` and decision records
identify the actual serving step. Preserve immutable frozen code and old
releases, and do not alter the running trainer's frozen dependencies.

## User-directed environment and PPO batch doubling (2026-10-05)

The user subsequently required an immediate switch. The source run is
`runs/v5_cluster8_rollout4x_h64_v2`, interrupted under explicit user direction
with a complete checkpoint at relative update 47600 / global 74154. The previous
pending task for boundary 48000 was cancelled with a retained queue receipt.
The replacement FIFO task targets `runs/v5_cluster8_env2x_batch2x_now_v2`, with
envs 32768 to 65536 and global PPO minibatch 16384 to 32768. Keep horizon 64
and the existing 200000 global-update target. Do not rewrite live frozen code
or submit duplicate resize tasks.
Inspect the new run's `SUBMISSION.json`, `status.json`, `engineering_READY.json`
and the shared queue to distinguish pending from verified/running. Accept only
real eight-rank NCCL evidence and healthy completed rollout rounds. The resize
task retains training/EMA state and republishes new raw policy snapshots to
the original directory so the local 500-step evaluator continues unchanged.
`source_pause_receipt.json` records one logged but unsaved source rollout, which
is excluded from the resumed metrics. A stale source `running` status after the
wrapper interruption is accepted only with reviewed checkpoint identity,
eight-rank NCCL proof and a remote check that the original trainer is gone.
Old checkpoints and metrics remain intact. Never apply the old bridge STOP
command to a healthy queue for ordinary submissions; this interruption and
same-worker recovery were explicitly user-directed emergency control.

## V6 attention scaling (user instruction, 2026-10-06)

V6 contains 8,014,192 parameters. The latest user override is “现在就切换”.
Its frozen immediate campaign is `runs/v6_cluster8_attention_8m_now_v3`, task
`01791265643288461199-667d0d7633e34e678fb802e8aa160455`, resuming the complete
V5 checkpoint at relative 50000 / global 76554. V1/V2/V3/V4 were superseded
before execution with retained interrupted receipts; never run or resubmit them.
The active V5 segment was interrupted under explicit user direction and the
same remote queue allocation was recovered. Keep the 59 unsaved logged rounds
in `source_metrics_before_pause.jsonl`; exclude them from resumed metrics.
The checksum-bound pause receipt and actual remote process absence gate replace
the normal completed-boundary gate. One V5 fallback to relative 51000 remains
queued until V6 passes; never cancel it before actual eight-GPU acceptance.
The first immediate candidate failed the unchanged optimizer-acceptance gate
after a long-history round stopped at 4.1% of minibatches. V6 now warms newly
born Adam coordinates over 1024 applied optimizer steps; mature coordinates
retain their updates. Save `runtime.new_coordinate_warmup_steps=1024` and reject
unregistered V6 resumes. EMA remains on its separate completed-rollout clock.
The second item stopped at the source-exit gate before any V6 training. Preserve
both failed items. The third item's control fix derives the actual child PID
from the active NCCL log and waits for source exit before checking snapshots
and GPU memory; stale source status PIDs must never authorize overlapping jobs.
Inspect REQUEST, SUBMISSION, engineering_READY and queue state before reporting
V6 as training. CPU verification alone is not eight-GPU or playing-strength proof.
Do not alter frozen pending/live dependencies or submit duplicate migrations.

The preset adds early state/history layers and independent attention head banks:
state 6 layers, total attention width 320; history 4 layers, width 256; residual
width 192 and inherited FFNs/action heads retained. Old heads keep dimension 32.
Preserve existing weights, Adam coordinates and coordinate ages, all environments,
RNG, adaptive vf, EMA weights and rollout counter. PPO scale/hyperparameters and
200000 global-update target remain the existing configuration. Before promotion,
require CPU FP32 function preservation, real CUDA BF16 function tolerance,
eight completed actual PPO rounds, nranks=8/NCCL, finite/legal actions, replica
agreement and memory headroom. The old V5 successor stays queued until V6 passes;
a failed V6 task is retained and is never automatically retried.

Local GPU ownership now resides in
`runs/local_a100_v6_follow_500_65536_v1`; consult `runs/local_a100_current.json`
for the current process. It preserves the previous precision result/benchmark
cache, drains V5 checkpoints, then follows an accepted V6 production pointer.
Keep the single local controller, every-500-step BEST benchmark and idle occupation.
QOJ code `v5-74162-77a7cbc0c4` supports background V5->V6 Policy construction,
validation and warmup. The serving thread commits between requests/decisions;
further V6 updates replace weights in place. Persisted V6 activation can resume
from the frozen V5 fallback. Live weights remain V5 until V6 is actually promoted.
See `reports/V6_ATTENTION_SCALING_20261006.md` and its JSON proof index.

The V4 freeze incorporates the user's subsequent architecture audit. Its READY
binds 58 CPU engineering tests, a real 65,536-environment full checkpoint
migration/cold restore, and local A100 short/full-history backward signatures
(4096 x 88 / 1024 x 192). The local source was relative 49900 / global 76454;
both raw and EMA migration outputs were exact in CPU FP32 and CUDA BF16 on 128
actual states. Measured local backward peak was 26,399,646,976 bytes. This is
single-GPU engineering evidence; eight actual rollout rounds and real NCCL
still gate promotion. The migration restores validated resident states without
redealing a discarded template, uses atomic checkpoint publication, verifies
the exact old continuation before cancellation, and checks explicit history
query masks, coordinate ages, finite metrics and EMA replica agreement.
See `reports/V6_ARCHITECTURE_AUDIT_20261006.md`; preserved artifacts are in
`runs/v6_audit_20261006_v1`. The local controller PID changed during owned GPU
engineering; use its current pointer, never historical process ids.

## Absolute-advantage sample trimming (user instruction, 2026-10-06)

The user requests retaining only the largest 50% of absolute advantages.
`runs/v6_cluster8_adv_top50_v1` is frozen and queued after the currently owned
segment at relative 51000 / global 77554. `ppo.adv_keep_fraction=0.5` ranks raw
advantages globally across all eight ranks over valid learner decisions.
Compute GAE/returns and normalization on complete trajectories first; selected
decisions train policy and value, and adaptive vf uses the same selection.
EV remains a complete-rollout diagnostic; EMA remains on the rollout clock.
Do not change live frozen code or duplicate the handoff. Its eight-round
nranks=8/NCCL, exact-count, optimizer, replica and memory gates must pass before
promotion; preserve the old successor until then. CPU engineering proof alone
does not establish GPU speed or playing-strength gains.
