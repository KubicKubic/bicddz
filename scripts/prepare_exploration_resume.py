"""User-directed checkpoint restart: freeze p=.02 plus the authorized crop."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flax import serialization
from ddz.exploration_campaign import revised_config, TRANSITION
from ddz.scale_rollout_campaign import read, write, sha
from ddz.train_v2 import atomic


def owned_source(active, old, production):
    tokens = shlex.split(active['command'])
    if (not any(module in tokens for module in
                ('ddz.cluster_distributed_job', 'ddz.scale_architecture_v6',
                 'ddz.exploration_campaign', 'ddz.entropy_campaign'))
            or '--root' not in tokens
            or Path(tokens[tokens.index('--root') + 1]).resolve() != old.resolve()
            or production.get('state') != 'running'
            or production.get('queue_id') != active['id']):
        raise RuntimeError('Active queue item is not the owned DDZ production task')


def supersede_trim(queue, record, previous, root):
    path = queue / 'pending' / (record['id'] + '.json')
    tokens = shlex.split(record['command'])
    if (read(path) != record or record.get('status') != 'pending'
            or record['command_sha256'] != hashlib.sha256(record['command'].encode()).hexdigest()
            or 'ddz.trim_rollout_campaign' not in tokens or '--root' not in tokens
            or Path(tokens[tokens.index('--root') + 1]).resolve() != previous.resolve()
            or (queue / 'running' / path.name).exists()
            or (queue / 'interrupted' / path.name).exists()):
        raise RuntimeError('Reviewed trimming item was changed or claimed')
    reason = 'USER_REQUESTED_LATEST_CHECKPOINT_RESUME_WITH_P_0_02_AND_RETAINED_TOP50'
    write(queue / 'interrupted' / path.name, {**record, 'status': 'interrupted',
        'finished_at_ns': time.time_ns(), 'exit_code': None,
        'reason': reason, 'replacement_root': str(root)})
    path.unlink()
    write(previous / 'SUPERSEDED.json', {'queue_id': record['id'], 'never_started': True,
        'reason': reason, 'replacement_root': str(root), 'time': time.time()})


def wait_for_pause(queue, run, log, seconds=180):
    """Allow an orphaned graceful trainer to finish before binding its files."""
    begin = time.monotonic(); deadline = begin + seconds
    signature = None; unchanged = begin
    while True:
        state = read(queue / 'status.json')
        fingerprint = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in
                            (run / 'latest.json', run / 'status.json', log))
        now = time.monotonic()
        if fingerprint != signature:
            signature = fingerprint; unchanged = now
        stopped = read(run / 'status.json')['state'] == 'stopped'
        if (state['phase'] == 'STOPPED'
                and not list((queue / 'running').glob('*.json'))
                and not (queue / 'worker.lock').exists() and now - begin >= 12
                and now - unchanged >= (10 if stopped else 75)):
            return
        if now > deadline:
            raise RuntimeError('Source pause is unresolved; do not start a competing worker')
        time.sleep(2)


def checkpoint_ready(latest, minimum):
    return latest['iteration'] >= minimum


def wait_for_checkpoint(queue, run, task_id, minimum, seconds=1800):
    deadline=time.monotonic()+seconds
    while not checkpoint_ready(read(run/'latest.json'),minimum):
        state=read(queue/'status.json')
        if (state['phase']!='RUNNING' or state['active_id']!=task_id or
            time.time()-state['updated_at_ns']/1e9>180):
            raise RuntimeError('Source ownership or worker heartbeat changed while waiting for save')
        if time.monotonic()>deadline:
            raise RuntimeError('Checkpoint wait expired; source task has not been interrupted')
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--supersede-root', type=Path)
    ap.add_argument('--tests-proof', type=Path, required=True)
    ap.add_argument('--transition',choices=('random02','entropy01'),default='random02')
    ap.add_argument('--checkpoint-at',type=int,default=0)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    entropy=args.transition=='entropy01'
    previous=args.supersede_root.resolve() if args.supersede_root else None
    if not entropy and previous is None:raise ValueError('Random02 requires the exact queued crop root')
    if entropy:
        from ddz.entropy_campaign import revised_config as revise, TRANSITION as transition
        module='ddz.entropy_campaign'
        instruction='改回原来的方案，然后提高更自然的熵奖励。'
    else:
        revise=revised_config;transition=TRANSITION;module='ddz.exploration_campaign'
        instruction='把它增加到 0.02，然后从最新 ckpt 接续训练。'
    if root.exists():
        raise RuntimeError('Resume campaign already retained; inspect before another submission')
    tests = read(args.tests_proof)
    if not tests.get('passed'):
        raise RuntimeError('Exploration, migration and evaluator verification required')
    for path, expected in tests['files_sha256'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError('Tested dependency changed: ' + path)
    queue = repo.parent / 'Q/tools/cluster/cluster_scheduler.queue'; cluster = queue.parent
    state = read(queue / 'status.json')
    if (state['phase'] != 'RUNNING' or state['running_count'] != 1
            or time.time() - state['updated_at_ns'] / 1e9 > 180):
        raise RuntimeError('Healthy owned eight-GPU queue worker required')
    active = read(queue / 'running' / (state['active_id'] + '.json'))
    run = Path(read(repo / 'runs/production_current.json')['run']); old = run.parent.parent
    owned_source(active, old, read(run.parent / 'queue_job_status.json'))
    request = read(old/'production/READY.json') if entropy else read(previous / 'REQUEST.json')
    if not entropy and (request['old_root'] != str(old) or request['active_source_queue_id'] != active['id']):
        raise RuntimeError('Queued crop source ownership differs')
    for path, expected in request['files_sha256'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError('Previously frozen crop input changed: ' + path)
    pending = [read(p) for p in (queue / 'pending').glob('*.json')]
    if len(pending) != (0 if entropy else 1):
        raise RuntimeError('Expected exactly the reviewed unclaimed trimming task')
    log = old / 'production' / (active['id'] + '.log')
    content = log.read_text()
    pids = set(re.findall(r'^[^:\n]+:(\d+):\d+ .*NCCL.*nranks[ =]+8\b', content, re.MULTILINE))
    if len(pids) != 1:
        raise RuntimeError('Actual trainer PID and eight-rank NCCL identity required')
    cfg = read(run / 'config.json'); target = revise(cfg)
    root.mkdir(); (root / 'production').mkdir()
    shutil.copytree(repo / 'ddz', root / 'code/ddz',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(args.tests_proof, root / 'TESTS_VERIFICATION.json')
    os.link(old / 'production/source.msgpack', root / 'production/source.msgpack')
    write(root / 'production/config.json', target)
    write(root / 'USER_RESTART_REQUEST.json', {
        'user_instruction': instruction,
        'retained_authorization': 'global top 50% absolute-advantage selection',
        'source_claim': active, 'old_worker': state, 'time': time.time()})
    if args.checkpoint_at:
        write(root/'WAITING_FOR_CHECKPOINT.json',{'minimum_iteration':args.checkpoint_at,
            'initial_latest':read(run/'latest.json'),'source_queue_id':active['id'],'time':time.time()})
        print(json.dumps({'state':'waiting_for_complete_checkpoint','minimum_iteration':args.checkpoint_at}),flush=True)
        wait_for_checkpoint(queue,run,active['id'],args.checkpoint_at)
    if read(queue/'status.json')['active_id']!=active['id']:
        raise RuntimeError('Source ownership changed before user-directed pause')
    # Only the explicitly requested restart uses the existing allocation bridge.
    # Every training task below is published through the immutable FIFO helper.
    subprocess.run([str(cluster / 'stop_scheduler_task.sh')], check=True)
    wait_for_pause(queue, run, log)
    interrupted = read(queue / 'interrupted' / (active['id'] + '.json'))
    if interrupted['task_pid'] != active['task_pid']:
        raise RuntimeError('Interrupted source process identity changed')
    if not entropy:supersede_trim(queue, pending[0], previous, root)
    saved = serialization.msgpack_restore((run / 'latest.msgpack').read_bytes())
    at = int(saved['iteration']); status = read(run / 'status.json')
    if (read(run / 'latest.json')['iteration'] != at or status['iteration'] < at
            or status['nranks'] != 8 or saved['config'] != cfg
            or saved['key'].shape != (8, 2) or 'ema' not in saved):
        raise RuntimeError('Latest complete checkpoint and completion marker differ')
    metrics = (run / 'metrics.jsonl').read_text()
    all_rows = [json.loads(line) for line in metrics.splitlines()]
    rows = [r for r in all_rows if r['iteration'] <= at][-600:]
    if (len(rows) < 8 or rows[-1]['iteration'] != at or any(
            r['nranks'] != 8 or r['invalid_actions'] or r['nonfinite']
            or r['fresh_decisions'] != cfg['envs'] * cfg['horizon']
            or r.get('replica_parameter_max_difference', 0)
            or r.get('ema_replica_parameter_max_difference', 0)
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in r.values())
            for r in rows)):
        raise RuntimeError('Latest retained checkpoint lacks healthy eight-rank rollout evidence')
    content = log.read_text()
    nccl = [line for line in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b', line)]
    if not nccl or 'nranks=8;' not in content:
        raise RuntimeError('Actual source eight-rank NCCL proof missing')
    proof = {'passed': True, 'queue_id': active['id'], 'nranks': 8, 'NCCL_evidence': nccl[:8],
        'source_log': str(log), 'source_log_sha256': sha(log),
        'accepted_checkpoint_iteration': at, 'validated_updates': len(rows)}
    write(root / 'source_health_proof.json', proof)
    pause = {'retained_checkpoint_iteration': at,
        'retained_global_iteration': cfg['global_source_iteration'] + at,
        'interrupted_queue_id': active['id'], 'remote_training_pid': int(next(iter(pids))),
        'retained_checkpoint_sha256': sha(run / 'latest.msgpack'),
        'source_status_sha256': sha(run / 'status.json'),
        'source_health_proof': str(root / 'source_health_proof.json'),
        'last_logged_iteration': all_rows[-1]['iteration'],
        'unsaved_logged_updates_not_resumed': sum(r['iteration'] > at for r in all_rows),
        'remote_process_absence_required_before_migration': True, 'time': time.time()}
    write(root / 'source_pause_receipt.json', pause)
    (root / 'source_metrics_before_pause.jsonl').write_text(metrics)
    filtered = '\n'.join(line for line in metrics.splitlines()
                         if json.loads(line)['iteration'] <= at) + '\n'
    atomic(run / 'metrics.jsonl', filtered.encode())
    files = [p for p in (root / 'code').rglob('*') if p.is_file()]
    files += [root / name for name in ('TESTS_VERIFICATION.json', 'production/config.json',
        'production/source.msgpack', 'source_pause_receipt.json', 'source_health_proof.json')]
    files.append(run / 'config.json')
    peak = max(r.get('gpu_peak_memory_bytes', 0) for r in rows)
    if not peak:
        raise RuntimeError('Measured source memory evidence missing')
    ready = {'cpu_verification': {'passed': True, 'proof': str(root / 'TESTS_VERIFICATION.json')},
        'files_sha256': {str(p): sha(p) for p in files},
        'resource_gate': {'source_peak_bytes': peak, 'estimated_peak_bytes': int(peak * 1.1) + 2**30},
        'boundary_source_policy': 'checksum-bound user-directed latest checkpoint restart'}
    write(root / 'READY.json', ready)
    write(root / 'REQUEST.json', {'old_root': str(old), 'at_iteration': at, 'queue': str(queue),
        'source_pause_receipt': str(root / 'source_pause_receipt.json'),
        'source_exit_wait_seconds': 180, 'active_source_queue_id': active['id'],
        'transition': transition, 'resource_gate': ready['resource_gate'],
        'files_sha256': {**ready['files_sha256'], str(root / 'READY.json'): sha(root / 'READY.json')},
        'preserve_global_update_target': cfg['updates'], 'time': time.time()})
    helper = cluster / 'submit_scheduler_task.sh'
    command = 'cd ' + shlex.quote(str(root / 'code')) + ' && ' + shlex.join([
        'env', 'JAX_PLATFORMS=cpu', 'CUDA_VISIBLE_DEVICES=', 'OPENBLAS_NUM_THREADS=1',
        'OMP_NUM_THREADS=2', sys.executable, '-u', '-m', module, '--root', str(root)])
    submitted = subprocess.run([str(helper), command], check=True, text=True, capture_output=True)
    write(root / 'SUBMISSION.json', {'state': 'queued', 'command': command,
        'submission': submitted.stdout, 'at_iteration': at,
        'random_action_prob':target['ppo']['random_action_prob'],
        'entropy_end':target['ppo']['entropy_end'],
        'adv_keep_fraction': .5, 'time': time.time()})
    fallback_target = min(at + 1000, cfg['continuation_updates'])
    fallback = 'cd ' + shlex.quote(str(old / 'code')) + ' && ' + shlex.join([
        sys.executable, '-u', '-m', 'ddz.cluster_distributed_job', '--root', str(old),
        '--phase', 'production', '--steps', str(fallback_target), '--resume'])
    recovery = subprocess.run([str(helper), fallback], check=True, text=True, capture_output=True)
    write(root / 'SOURCE_FALLBACK.json', {'command': fallback, 'submission': recovery.stdout,
        'target': fallback_target, 'cancel_only_after_eight_rank_acceptance': True, 'time': time.time()})
    bridge = 'RUN: ' + shlex.join(['env', 'Q_CLUSTER_TASK=1', '/root/miniforge/envs/wm/bin/python',
        str(cluster / 'cluster_scheduler_queue.py'), 'worker', '--gpus', '8']) + '\n'
    atomic(cluster / 'cluster_scheduler.command', bridge.encode())
    (cluster / 'cluster_scheduler.command').chmod(0o644)
    write(root / 'worker_recovery_receipt.json', {'user_directed_checkpoint_restart': True,
        'old_worker_stopped': True, 'no_unresolved_running_claims': True,
        'same_allocation_queue_worker_restart': True, 'time': time.time()})
    print(json.dumps({'state': 'resume_queued_and_worker_recovering', **pause}, indent=2), flush=True)


if __name__ == '__main__':
    main()
