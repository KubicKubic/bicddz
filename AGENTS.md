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

**Latest user override (2026-10-06): resume continuous QOJ rated play.**
The user subsequently instructed “你需要让它持续进行对战” and “你自己启动就行了”.
This supersedes the earlier temporary-offline request. The old OFFLINE receipt
is archived under `runs/qoj_match_v1/`; `ONLINE.json` records this resumption.
Keep the single tmux supervisor and owned worker running, with raw checkpoint
updates. Do not stop them under the superseded offline instruction.
Eight-GPU training and the local DouZero evaluator continue normally.

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
the normal completed-boundary gate. V6 passed actual eight-GPU acceptance at
relative 50008 / global 76562; the V5 fallback was cancelled only after that
acceptance. Historical failed and interrupted items remain retained.
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
`runs/local_a100_v6_trim_follow_500_65536_v1`; consult `runs/local_a100_current.json`
for the current process. It preserves the previous precision result/benchmark
cache, drains V5 checkpoints, then follows an accepted V6 production pointer.
Keep the single local controller, every-500-step BEST benchmark and idle occupation.
QOJ code `v5-74162-77a7cbc0c4` supports background V5->V6 Policy construction,
validation and warmup. The serving thread commits between requests/decisions;
further V6 updates replace weights in place. Persisted V6 activation can resume
from the frozen V5 fallback. The actual serving checkpoint is recorded in
`runs/qoj_match_v1/active_model.json`; never infer it from an old deployment bundle.
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

The local controller additionally accepts this registered one-field V6 sampling
handoff after its eight-rank proof and exact retained-state receipt. The initial
trimming proof's older schema obtains model parameter count from actual trainer
status; require 8,014,192 and nranks=8. Preserve the completed benchmark cache,
the initial handoff point and the single idle workload. Do not restart for each
checkpoint. See `reports/REPO_CLEANUP_20261006.md` for the verified controller update.

## Fixed exploration continuation (user instruction, 2026-10-06)

The latest user requested `random_action_prob=0.02` and resumption from the
latest complete checkpoint. The frozen continuation is
`runs/v6_cluster8_random02_top50_now_v1`, queue item
`01791299912048312550-410742529dc0409c9dbd74c212a7f4c8`. It retains the previously
authorized global top-50%-absolute-raw-advantage selection in the same run.
The old pending `v6_cluster8_adv_top50_v1` task was superseded with an interrupted
receipt before execution; never run or replay it. The existing allocation was
paused under this explicit checkpoint-restart instruction, and training tasks
were submitted through the immutable queue helper before restoring its worker.

The resume source is relative 50700 / global 77254. Preserve the 66 logged but
unsaved source rounds in `source_metrics_before_pause.jsonl`; those updates are
excluded from resumed metrics. The checksum-bound pause proof and remote trainer
process-absence gate are required. Retain all weights, Adam moments/coordinate
ages, environment states, eight RNG streams, adaptive vf runtime and EMA. EMA
remains decay 0.999 on the completed-rollout clock. Training remains V6 with
8,014,192 parameters, 65,536 environments, horizon 64, LR 1e-5, gamma 1,
lambda .95, one PPO epoch, save every 100, and global target 200000.

Use one Bernoulli per complete action and the exact complete-action mixture
probability in PPO. The random branch is uniform at legal tree nodes, not
necessarily across complete-action leaves. Positive-p entropy is sampled
mixture surprisal, so its metric differs from the previous conditional entropy.
Do not apply training exploration automatically to greedy DouZero/QOJ play.
Require eight fresh real GPU rounds, nranks=8/NCCL, exact crop counts, optimizer
acceptance, finite/legal decisions, replica agreement, memory and EMA continuity
before production promotion. Retain the old recovery successor until acceptance;
failures require explicit review and a new queue item.

The current single local controller release is
`runs/local_a100_v6_explore02_follow_500_65536_v1`; use the live pointer for its PID.
It accepts the registered exploration/crop transition with retained-state and
actual eight-rank proof, preserves completed benchmarks and every-500-step BEST
all-role evaluation, and owns idle occupation during unused local A100 time.
See `reports/EXPLORATION_RESUME_20261006.md` and its JSON proof index for actual
acceptance status; CPU engineering checks alone are not GPU or strength proof.

## Restore original sampling and stronger entropy (user override, 2026-10-07)

The latest user rejects fixed random-action exploration and requests the original
scheme with stronger natural model entropy. This supersedes the previous p=.02
instruction: set `ppo.random_action_prob=0` and `ppo.entropy_end=.01` (previously
.002). The saturated original entropy schedule gives effective .01 without
resetting its clock. Use the original p=0 learned-policy sampler, complete-action
log probabilities and conditional entropy calculation. Retain the independently
authorized global top-50% raw absolute advantage selection and all other settings.

The frozen continuation is `runs/v6_cluster8_entropy01_top50_now_v1`, prepared
for the complete relative 50800 checkpoint before pausing its currently owned
p=.02 segment. Inspect SUBMISSION/REQUEST/source_pause_receipt and the queue for
the actual pinned step and task id. Never alter frozen source dependencies or
replay a failed task. The same allocation's queue worker is recovered only for
this explicit configuration restart; training submissions use the immutable
queue helper. Keep the source recovery successor until eight actual fresh
nranks=8/NCCL rounds pass p=0, entropy=.01, exact crop counts, optimizer, replica,
finite/legal, memory and rollout-EMA continuity gates. Preserve all source state
and any logged but unsaved rows with an explicit receipt.

The local controller replacement
`runs/local_a100_v6_entropy01_follow_500_65536_v1` waits for the current BEST
benchmark to complete, retains all completed results, and continues 500-step
all-role benchmarks and idle occupation. Do not stop an active evaluation to
update its controller. A known old-controller registry failure can be reviewed
only against the accepted entropy transition and a completed, nonfailed
benchmark; this does not authorize retrying an evaluation failure. Use
`runs/local_a100_current.json` for actual ownership and curve paths. See
`reports/ENTROPY_RESUME_20261007.md` and its JSON proof index for accepted status.

## Pin the highest scored checkpoint online (user override, 2026-10-07)

The user requests uploading the highest scoring checkpoint to Git LFS and
replacing the online bot with it. This supersedes automatic latest-checkpoint
replacement for QOJ. The selected raw V6 checkpoint is relative 50808 / global
77362, with the highest observed completed all-role BEST expected-score mean
among 106 recorded versions: .1371841431, 95% interval [.1268920898, .1473541260],
65536 deals / 393216 games. This is an observed ranking, not a significance claim
against the runner-up. Its SHA-256 is
`a86407046f8ec41b73024e592e10de5e3ba74546d2b29436624aa4a6fd29605f`.

`models/BEST.json` points to the immutable `models/best_77362` LFS policy bundle;
the raw policy was evaluated, and the matching EMA snapshot is retained separately.
This bundle is a policy snapshot, with no fabricated full optimizer/environment
resume state. The earlier complete bundle at `models/` remains retained.
Prepare this pinned release with `ddz.qoj_deployment prepare --models
models/best_77362` and no `--checkpoint-pointer`. Do not use the latest-following
default `scripts/qoj_match.sh prepare` for this pin. The current launcher uses
the immutable `v6-77362-574012aa7f` release; its worker records the actual pinned
weights and ignores a prior, newer `active_model.json` on startup. Keep the single
tmux rated-match supervisor running; training, every-500-step local evaluation
and local idle occupation continue. Do not reenable latest-following without a
subsequent user instruction. See `reports/BEST_QOJ_DEPLOYMENT_20261007.md` and
the checksum-bound JSON deployment evidence.
