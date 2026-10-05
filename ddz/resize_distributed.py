"""Double environments and PPO batches at an accepted eight-GPU boundary."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from flax import serialization
from .scale_rollout_campaign import fresh_environments, read, sha, summarize, write
from .switch_distributed_gae import cancel_old_followups, equal


def revised_config(old):
    cfg = copy.deepcopy(old)
    for field in ('envs', 'per_gpu_envs', 'per_gpu_minibatch'):
        cfg[field] *= 2
    cfg['ppo']['minibatch'] *= 2
    return cfg


def expand_checkpoint(saved, cfg, fresh, at):
    old = saved['config']
    if cfg != revised_config(old):
        raise RuntimeError('Only environment count and PPO minibatch may double')
    n = old['envs']
    if n % 8 or old['per_gpu_envs'] != n // 8 or np.shape(saved['key']) != (8, 2):
        raise RuntimeError('Requires a complete eight-rank checkpoint')
    if saved['iteration'] != at or 'ema' not in saved:
        raise RuntimeError('Checkpoint boundary or retained EMA is missing')
    if saved['ema']['decay'] != cfg['ema_decay']:
        raise RuntimeError('EMA decay differs')

    def grow(original, extra):
        original, extra = np.asarray(original), np.asarray(extra)
        if original.shape[0] != n or extra.shape != original.shape or extra.dtype != original.dtype:
            raise RuntimeError('Environment expansion shape or dtype differs')
        shape = (8, n // 8) + original.shape[1:]
        result = np.concatenate((original.reshape(shape), extra.reshape(shape)), axis=1)
        np.testing.assert_array_equal(result[:, :n // 8], original.reshape(shape))
        return result.reshape((n * 2,) + original.shape[1:])

    if saved['env'].keys() != fresh.keys():
        raise RuntimeError('Environment schema differs')
    result = copy.deepcopy(saved)
    result['config'] = copy.deepcopy(cfg)
    result['env'] = {k: grow(v, fresh[k]) for k, v in saved['env'].items()}
    result['runtime']['arena'] = {k: grow(v, np.zeros_like(v))
                                 for k, v in saved['runtime']['arena'].items()}
    result['runtime']['distributed'].update(
        per_gpu_envs=cfg['per_gpu_envs'], per_gpu_minibatch=cfg['per_gpu_minibatch'])
    result['runtime'].setdefault('config_migrations', []).append({
        'iteration': at, 'reason': 'user requested double environments and PPO minibatch',
        'old_envs': n, 'new_envs': n * 2,
        'old_minibatch': old['ppo']['minibatch'], 'new_minibatch': cfg['ppo']['minibatch'],
        'horizon': cfg['horizon'], 'time': time.time()})
    for field in ('train', 'key', 'iteration', 'ema'):
        if not equal(saved[field], result[field]):
            raise RuntimeError('Inherited state changed: ' + field)
    for field in ('vf_coef', 'stable_updates', 'source_sha256'):
        if saved['runtime'][field] != result['runtime'][field]:
            raise RuntimeError('Inherited runtime changed: ' + field)
    return result


def verify_source(request, old_run, at):
    status = read(old_run / 'status.json')
    pause_path = request.get('source_pause_receipt')
    if pause_path:
        pause = read(pause_path)
        claim = read(Path(request['queue']) / 'interrupted' /
                     (pause['interrupted_queue_id'] + '.json'))
        # An interrupted wrapper can close stdout before its trainer publishes
        # a final status. In that reviewed case use the complete saved state,
        # exclude unsaved metrics, and verify the old process is gone below.
        allowed_state = status['state'] == 'stopped' or (
            pause.get('reviewed_stale_running_status') and status['state'] == 'running')
        if (claim['status'] != 'interrupted' or pause['retained_checkpoint_iteration'] != at or
            not allowed_state or status['iteration'] != at or status['nranks'] != 8 or
            sha(old_run / 'latest.msgpack') != pause['retained_checkpoint_sha256']):
            raise RuntimeError('Reviewed pause/checkpoint identity differs')
        process = Path(f"/proc/{pause['remote_training_pid']}/cmdline")
        if process.exists() and b'ddz.train_distributed_v5' in process.read_bytes():
            raise RuntimeError('Paused source trainer is still alive; no competing training')
        proof = read(pause['source_health_proof'])
    else:
        proof = read(old_run.parent / 'queue_job_status.json')
        if (status['state'] != 'completed' or status['iteration'] != at or
            status['nranks'] != 8 or proof.get('steps', at) != at):
            raise RuntimeError('Source boundary differs')
    if not proof.get('passed') or proof.get('nranks') != 8 or not proof.get('NCCL_evidence'):
        raise RuntimeError('Source requires accepted eight-rank NCCL proof')
    return bool(pause_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    request = read(root / 'REQUEST.json')
    if os.environ.get('Q_CLUSTER_TASK') != '1':
        raise RuntimeError('Persistent eight-GPU queue required')
    for name, expected in request['files_sha256'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen input changed: ' + name)
    old = Path(request['old_root'])
    old_run = old / 'production/training'
    phase = root / 'production'
    run = phase / 'training'
    at = request['at_iteration']
    if run.exists():
        raise RuntimeError('Migration already started; inspect before an explicit new task')
    paused = verify_source(request, old_run, at)
    # Read hardware memory after the old task has released its GPU allocator.
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    resource = {'query_exit_code': query.returncode, 'rows': query.stdout.splitlines(),
                'source_peak_bytes': request['resource_gate']['source_peak_bytes']}
    if query.returncode:
        raise RuntimeError('Remote GPU capacity query failed; no unverified resize')
    capacities = []
    free = []
    for line in resource['rows']:
        parts = [v.strip() for v in line.split(',')]
        capacities.append(int(parts[2]) * 2**20)
        free.append((int(parts[2]) - int(parts[3])) * 2**20)
    estimate = request['resource_gate']['estimated_peak_bytes']
    if len(capacities) != 8 or min(free) < estimate or min(capacities) * .8 < estimate:
        raise RuntimeError('Eight-GPU memory headroom gate failed')
    write(root / 'resource_verified.json', {**resource, 'passed': True,
          'estimated_peak_bytes': estimate, 'time': time.time()})
    raw = (old_run / 'latest.msgpack').read_bytes()
    saved = serialization.msgpack_restore(raw)
    cfg = read(phase / 'config.json')
    if saved['runtime']['source_sha256'] != sha(phase / 'source.msgpack'):
        raise RuntimeError('Source weight identity differs')
    fresh = fresh_environments(saved, cfg, at)
    migrated = expand_checkpoint(saved, cfg, fresh, at)
    payload = serialization.msgpack_serialize(migrated)
    if not equal(migrated, serialization.msgpack_restore(payload)):
        raise RuntimeError('Serialization changed migrated state')
    cancelled = cancel_old_followups(Path(request['queue']), old)
    if len(cancelled) != 1 and not (paused and not cancelled):
        raise RuntimeError('Old DDZ follow-up count differs; inspect queue')
    for task in cancelled:
        path = Path(request['queue']) / 'interrupted' / (task + '.json')
        record = read(path)
        record['reason'] = 'USER_DOUBLED_DDZ_ENVS_AND_PPO_MINIBATCH'
        write(path, record)
    run.mkdir()
    (run / 'latest.msgpack').write_bytes(payload)
    write(run / 'config.json', cfg)
    with (run / 'metrics.jsonl').open('w') as stream:
        for line in (old_run / 'metrics.jsonl').read_text().splitlines():
            if json.loads(line)['iteration'] <= at:
                stream.write(line + '\n')
    for snapshot in old_run.glob('policy_*.msgpack'):
        (run / snapshot.name).symlink_to(snapshot.resolve())
    (run / 'ema').mkdir()
    for snapshot in (old_run / 'ema').glob('policy_*.msgpack'):
        (run / 'ema' / snapshot.name).symlink_to(snapshot.resolve())
    write(run / 'latest.json', read(old_run / 'latest.json'))
    write(root / 'migration_receipt.json', {
        'iteration': at, 'global_iteration': cfg['global_source_iteration'] + at,
        'source_checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
        'migrated_checkpoint_sha256': sha(run / 'latest.msgpack'),
        'weights_adam_rng_existing_rank_envs_ema_bit_exact': True,
        'ema_updates_preserved': saved['ema']['updates'], 'cancelled_old_followups': cancelled,
        'old_envs': saved['config']['envs'], 'new_envs': cfg['envs'],
        'old_minibatch': saved['config']['ppo']['minibatch'],
        'new_minibatch': cfg['ppo']['minibatch'], 'time': time.time()})
    write(root / 'status.json', {'state': 'verifying_eight_gpu_resize', 'time': time.time()})
    del fresh, migrated, payload
    log = root / 'engineering.log'
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'NCCL_DEBUG': 'INFO',
           'PYTHONPATH': str(root / 'code'), 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
           'XLA_PYTHON_CLIENT_MEM_FRACTION': '.80',
           'JAX_COMPILATION_CACHE_DIR': str(phase / 'jax_cache')}
    argv = [sys.executable, '-u', '-m', 'ddz.train_distributed_v5',
            '--config', str(phase / 'config.json'), '--source', str(phase / 'source.msgpack'),
            '--out', str(run), '--steps', str(at + 8), '--resume']
    with log.open('w') as stream:
        process = subprocess.Popen(argv, cwd=root / 'code', env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError('Resize failed; retained logs require explicit review before retry')
    nccl = [line for line in log.read_text().splitlines() if re.search(r'NCCL.*nranks[ =]+8\b', line)]
    if not nccl or 'nranks=8;' not in log.read_text():
        raise RuntimeError('Real eight-rank NCCL evidence missing')
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['iteration'] > at]
    result = summarize(rows, cfg['envs'] * cfg['horizon'])
    if len(rows) != 8 or rows[-1]['iteration'] != at + 8:
        raise RuntimeError('Eight verification rollout rounds required')
    final = serialization.msgpack_restore((run / 'latest.msgpack').read_bytes())
    if final['ema']['updates'] != saved['ema']['updates'] + 8 or final['ema']['decay'] != .999:
        raise RuntimeError('Rollout-clock EMA continuity failed')
    if result['gpu_peak_memory_bytes'] >= min(capacities) * .8:
        raise RuntimeError('Measured peak leaves insufficient allocator headroom')
    result.update(NCCL_evidence=nccl[:8], global_envs=cfg['envs'],
                  global_minibatch=cfg['ppo']['minibatch'], log=str(log), time=time.time())
    write(root / 'engineering_READY.json', result)
    # Keep the established local 500-step evaluator watching the same directory.
    write(root / 'policy_publication.json', {'run_dir': str(run), 'destination': str(old_run)})
    for snapshot in run.glob('policy_*.msgpack'):
        destination = old_run / snapshot.name
        if not destination.exists():
            destination.symlink_to(snapshot.resolve())
    write(root / 'status.json', {'state': 'verified_and_continuing', **result})
    write(root.parent / 'production_current.json', {
        'state': 'running', 'run': str(run), 'nranks': 8,
        'global_iteration': cfg['global_source_iteration'] + at + 8,
        'proof': str(root / 'engineering_READY.json'), 'time': time.time()})
    target = min(at + 1000, cfg['continuation_updates'])
    code = subprocess.run([sys.executable, '-u', '-m', 'ddz.cluster_distributed_job',
                           '--root', str(root), '--phase', 'production',
                           '--steps', str(target), '--resume'], cwd=root / 'code', env=env).returncode
    if code:
        raise RuntimeError('Continuation failed; inspect before a new queue task')


if __name__ == '__main__':
    main()
