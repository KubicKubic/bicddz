# Highest scored policy uploaded and deployed — 2026-10-07

Selected the maximum completed all-role expected-score mean among the 106
versions recorded in the current DouZero BEST curve. The winner is V6 relative
50808 / global 77362, with **+.1371841431** and 95% interval
**[+.1268920898, +.1473541260]**, from 65,536 deals / 393,216 games. The runner-up
is relative 51500 at +.1368026733; the small mean difference does not establish
that the winner is significantly stronger. Selection uses the completed observed
mean requested by the user. The fixed-bid-three protocol measures all three
playing roles; it does not measure bidding strength.

## Published weights

`models/BEST.json` points to `models/best_77362/`. Its raw policy SHA-256 is
`a86407046f8ec41b73024e592e10de5e3ba74546d2b29436624aa4a6fd29605f`, exactly
matching the evaluated snapshot. The matching EMA policy, training config,
original evaluation summary and checksum manifest are included. Both policy
objects have been uploaded to origin via Git LFS. This is a policy snapshot;
no full training resume state is claimed for this historical step. The prior
complete raw/EMA/optimizer/environment bundle at `models/` remains retained.

## Live bot

Prepared the immutable CPU release `v6-77362-574012aa7f` without a checkpoint
watch pointer, warmed the actual frozen weights, and switched the owned tmux
supervisor and worker after verifying ownership. Both old processes exited
before the same tmux pane was respawned; the account lock was free. The bot
authenticated and resumed the current game with the selected policy.

The actual deployment, active-model marker and fresh model decision agree on
global step 77362 and the evaluated SHA-256. The new worker reports
`checkpoint_watch_enabled=false`; subsequent training checkpoints cannot replace
this pin. It records static activation on startup and ignores a previous newer
latest-policy resume marker. Continuous rated play and the existing dashboard
continue. Eight-GPU training and the single local A100 evaluator/idle controller
were left running.

## Verification

Six focused checks passed: consistent export/frozen deployment, pinned-worker
startup with a newer stale resume marker, disabled automatic updates, correct
decision/deployment metadata and preserved latest-following behavior when
explicitly configured. One new test initially replaced a watcher class with a
function, invalidating `isinstance`; the fixture was corrected and that single
case rerun.

The frozen V6 policy has 8,014,192 finite parameters. Actual CPU warmup took
2.426 seconds, including bidding and playing forward verification. Fresh online
decisions used the selected model, with accepted actions and no own-seat
autoplay at verification. Full selection, CPU and live evidence is indexed in
`BEST_QOJ_DEPLOYMENT_20261007.json`; retained switching artifacts are in
`runs/qoj_best_77362_20261007/`.
