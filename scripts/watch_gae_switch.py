"""Local handoff after the queued DDZ GAE migration, without GPU work."""
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OLD = BASE / 'runs/v5_cluster8_v1'
NEW = BASE / 'runs/v5_cluster8_own_gae_v1'
QUEUE = BASE.parent / 'Q/tools/cluster/cluster_scheduler.queue'
TASK = '01790957906009769746-9b66bab0535a40da8dd601d15b23d9a9'
RUN = NEW / 'production/training'
OLD_RUN = OLD / 'production/training'
NAME = 'score_ladder_all_roles_500_v1'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        for state in ('failed', 'interrupted'):
            if (QUEUE / state / (TASK + '.json')).exists():
                raise RuntimeError('Migration queue item ' + state + '; inspect its log')
        phase_status = NEW / 'production/queue_job_status.json'
        if phase_status.exists() and read(phase_status)['state'] == 'failed':
            raise RuntimeError('Own-seat production failed; no automatic retry')
        status_path = RUN / 'status.json'
        if status_path.exists() and read(status_path)['iteration'] >= 1004:
            break
        time.sleep(45)
    else:
        raise TimeoutError('Migration still pending after two hours')

    receipt = read(NEW / 'migration_receipt.json')
    assert receipt['weights_optimizer_env_rng_bit_exact']
    cfg = read(RUN / 'config.json')
    assert cfg['ppo']['gae_clock'] == 'own'
    status = read(status_path)
    assert status['nranks'] == 8
    rows = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    new_rows = [row for row in rows if row['iteration'] > 1000]
    assert len(new_rows) >= 4 and new_rows[0]['iteration'] == 1001
    assert new_rows[0]['replica_parameter_max_difference'] == 0
    for row in new_rows:
        assert row['nranks'] == 8 and row['fresh_decisions'] == 524288
        assert row['invalid_actions'] == 0 and row['nonfinite'] == 0
        assert all(math.isfinite(v) for v in row.values())
    previous = next(row for row in rows if row['iteration'] == 1000)
    assert new_rows[0]['optimizer_step'] - previous['optimizer_step'] in (32, 64)
    log_path = NEW / 'production' / (TASK + '.log')
    log = log_path.read_text()
    nccl = [line for line in log.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b', line)]
    assert nccl and 'nranks=8;' in log

    # This PID belongs to the editing container; never signal it remotely.
    old_pid = 1895609
    process_cmd = Path(f'/proc/{old_pid}/cmdline')
    if process_cmd.exists():
        args = process_cmd.read_bytes().split(b'\0')
        expected = str(OLD_RUN).encode()
        if b'ddz.score_ladder_graph_all_roles_v5' in args and expected in args:
            os.kill(old_pid, signal.SIGTERM)
            for _ in range(100):
                if not process_cmd.exists() or not process_cmd.read_bytes():
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Old CPU watcher did not stop; inspect before handoff')
        else:
            raise RuntimeError('Old watcher PID identity differs; inspect before handoff')

    ladder = RUN / NAME
    ladder.mkdir(exist_ok=True)
    cached = 0
    for directory in ('matches', 'chunks'):
        target = ladder / directory
        target.mkdir(exist_ok=True)
        for path in sorted((OLD_RUN / NAME / directory).glob('*.json')):
            destination = target / path.name
            if not destination.exists():
                destination.symlink_to(path)
                cached += 1
    write(ladder / 'training_origin.json', {
        'global_source_iteration': 26554,
        'own_gae_switch_iteration': 1000,
        'own_gae_switch_global_iteration': 27554,
        'original_ladder': str(OLD_RUN / NAME),
        'reused_cache_files': cached,
        'migration_receipt': str(NEW / 'migration_receipt.json')})
    cpus = sorted(os.sched_getaffinity(0))[-16:-8]
    if not cpus:
        cpus = sorted(os.sched_getaffinity(0))
    argv = ['taskset', '-c', ','.join(map(str, cpus)), 'nice', '-n', '19',
            sys.executable, '-u', '-m', 'ddz.score_ladder_graph_all_roles_v5',
            '--run-dir', str(RUN), '--output', str(ladder), '--deals', '1536',
            '--chunk-deals', '48', '--bootstrap-rounds', '2000',
            '--watch', '--poll-seconds', '60']
    environment = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(NEW / 'code')}
    with (ladder / 'watcher.log').open('a') as stream:
        child = subprocess.Popen(argv, cwd=NEW / 'code', env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    if child.poll() is not None:
        raise RuntimeError('New CPU watcher failed to start; inspect watcher.log')
    verified = {
        'state': 'running', 'run': str(RUN), 'gae_clock': 'own', 'nranks': 8,
        'remote_pid': status['pid'], 'global_iteration': status['global_iteration'],
        'migration_receipt': str(NEW / 'migration_receipt.json'),
        'first_verified_own_updates': len(new_rows), 'invalid_actions': 0, 'nonfinite': 0,
        'replica_parameter_max_difference': 0, 'NCCL_evidence': nccl[:8],
        'global_envs': 8192, 'fresh_decisions_per_update': 524288,
        'global_minibatch': 16384, 'ladder_pid': child.pid,
        'ladder': str(ladder), 'time': time.time()}
    write(NEW / 'switch_health_verification.json', verified)
    write(BASE / 'runs/production_current.json', verified)
    print(json.dumps(verified), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(NEW / 'local_handoff_failure.json', {'error': repr(error), 'time': time.time()})
        raise
