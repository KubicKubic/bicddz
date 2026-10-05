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
(client version 8); older V7 files remain historical recovery artifacts.
The fixed raw V5 policy uses CPU inference. The launcher sources the
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
