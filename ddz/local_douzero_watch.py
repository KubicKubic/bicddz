"""One local GPU: persistent 500-update benchmark plus idle-only occupation."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def pending_steps(run, start, output, extra_steps=()):
    snapshots={int(p.stem.split('_')[1]) for p in Path(run).glob('policy_*.msgpack')}
    requested={int(step) for step in extra_steps}
    steps = sorted(step for step in snapshots
                   if (step>=start and step%500==0) or step in requested)
    return [step for step in steps if not (Path(output)/f'step_{step:07d}/BEST/summary.json').exists()]


def active_cuda_processes(exclude=()):
    """Respect unrelated local GPU jobs; stopped processes retain their state."""
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) in exclude:
            continue
        try:
            state = next(line for line in (process/'status').read_text().splitlines()
                         if line.startswith('State:')).split()[1]
            if state in ('T', 't', 'Z'):
                continue
            if any('nvidia' in os.readlink(fd) for fd in (process/'fd').iterdir()):
                found.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
            continue
    return found


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def merge_history(cfg,rows):
    if not cfg.get('history_ratings'):return rows
    history=read(cfg['history_ratings'])
    for field in ('seed','equal_role_score','bidding','interval_updates'):
        if history['protocol'].get(field)!=cfg['protocol'].get(field):
            raise RuntimeError('Historical curve protocol mismatch: '+field)
    merged={row['step']:row for row in history['rows']}
    for row in rows:
        old=merged.get(row['step'])
        if old is not None and old['checkpoint_sha256']!=row['checkpoint_sha256']:
            raise RuntimeError('Historical curve checkpoint identity changed')
        # A more precise assessment replaces the old point for the SAME policy.
        # Distinct snapshots retain their own sample counts and intervals.
        merged[row['step']]=row
    return [merged[step] for step in sorted(merged)]


def update_curve(cfg):
    output = Path(cfg['output'])
    rows = []
    for path in sorted(output.glob('step_*/BEST/summary.json')):
        data = read(path)
        identity, summary = data['identity'], data['summary']
        step = identity['checkpoint_step']
        if (identity['seed'] != cfg['seed'] or summary['deals'] != cfg['deals']
                or identity['resnet_commit'] != cfg['best_commit']
                or identity.get('hardware', {}).get('policy_devices') != 1):
            raise RuntimeError('Benchmark curve protocol mismatch: ' + str(path))
        score = summary['equal_role_expected_score']
        win = summary['equal_role_team_win_rate']
        rows.append({'step':step, 'global_step':cfg['global_source_iteration']+step,
            'expected_score':score['mean'], 'low':score['ci95'][0], 'high':score['ci95'][1],
            'win_rate':win['mean'], 'deals':summary['deals'], 'games':summary['games'],
            'roles':summary['roles'], 'checkpoint_sha256':identity['checkpoint_sha256'],
            'result':str(path)})
    rows=merge_history(cfg,rows)
    write(output/'ratings.json', {'protocol':cfg['protocol'], 'rows':rows, 'time':time.time()})
    with (output/'ratings.csv.tmp').open('w', newline='') as stream:
        fields = ('step','global_step','expected_score','low','high','win_rate','deals','games')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(output/'ratings.csv.tmp', output/'ratings.csv')
    if rows:
        subprocess.run([cfg['burn_python'], '-m', 'ddz.render_douzero_trend',
                        str(output/'ratings.json'), str(output/'douzero_best_curve.png')],
                       cwd=cfg['code_dir'], env={**os.environ,'PYTHONPATH':cfg['code_dir']}, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    cfg = read(args.config)
    for path, expected in cfg['files_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen evaluation dependency changed: ' + path)
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    write(output/'watcher.json', {'pid':os.getpid(), 'config':str(args.config.resolve()), 'time':time.time()})
    burn = evaluation = None
    env = {**os.environ, 'JAX_PLATFORMS':'cuda', 'CUDA_VISIBLE_DEVICES':'0',
           'XLA_PYTHON_CLIENT_MEM_FRACTION':'.12', 'OMP_NUM_THREADS':'2',
           'OPENBLAS_NUM_THREADS':'1', 'PYTHONPATH':cfg['code_dir']+':'+cfg['torch_cpu_path'],
           'LD_LIBRARY_PATH':'/tmp/ddz_nvml:'+os.environ.get('LD_LIBRARY_PATH','')}
    try:
        while not stop[0]:
            exclude = {os.getpid()}
            if burn is not None:
                exclude.add(burn.pid)
            foreign = active_cuda_processes(exclude)
            pending = pending_steps(cfg['run_dir'], cfg['start_step'], output)
            if foreign:
                stop_child(burn)
                burn = None
                write(output/'status.json', {'state':'waiting_for_local_gpu','foreign_pids':foreign,
                      'pending_steps':pending, 'time':time.time()})
                time.sleep(cfg['poll_seconds'])
                continue
            if pending:
                stop_child(burn)
                burn = None
                step = pending[0]
                destination = output/f'step_{step:07d}'
                if (destination/'FAILED.json').exists():
                    raise RuntimeError('Failed evaluation requires reviewed resume: ' + str(destination))
                argv = [sys.executable, '-u', '-m', 'ddz.compare_efficiency_douzero',
                    '--run-dir', cfg['run_dir'], '--step', str(step),
                    '--douzero-src', cfg['douzero_src'], '--resnet-src', cfg['resnet_src'],
                    '--weights-root', cfg['weights_root'], '--out', str(destination),
                    '--deals', str(cfg['deals']), '--chunk-deals', str(cfg['chunk_deals']),
                    '--seed', str(cfg['seed']), '--require-a100', '--jax-cache', str(output/'jax_cache')]
                with (output/f'step_{step:07d}.log').open('a') as stream:
                    evaluation = subprocess.Popen(argv, cwd=cfg['code_dir'], env=env,
                                                  stdout=stream, stderr=subprocess.STDOUT)
                write(output/'status.json', {'state':'evaluating','step':step,
                    'global_step':cfg['global_source_iteration']+step,'pid':evaluation.pid,
                    'queued_steps':pending[1:],'time':time.time()})
                while evaluation.poll() is None and not stop[0]:
                    time.sleep(2)
                if stop[0]:
                    break
                code = evaluation.returncode
                evaluation = None
                if code:
                    destination.mkdir(exist_ok=True)
                    write(destination/'FAILED.json', {'exit_code':code, 'time':time.time()})
                    raise RuntimeError('Local benchmark failed; inspect retained log')
                update_curve(cfg)
                print(json.dumps({'completed_step':step,'curve':str(output/'douzero_best_curve.png')}), flush=True)
                continue
            if burn is None or burn.poll() is not None:
                if burn is not None:
                    raise RuntimeError('Idle GPU occupation failed; inspect retained log')
                burn_env = {**os.environ, 'PYTHONPATH':cfg['code_dir'], 'CUDA_VISIBLE_DEVICES':'0',
                            'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']}
                with (output/'idle_gpu.log').open('a') as stream:
                    burn = subprocess.Popen([cfg['burn_python'], '-u', '-m', 'ddz.gpu_idle'],
                        cwd=cfg['code_dir'], env=burn_env, stdout=stream, stderr=subprocess.STDOUT)
                write(output/'status.json', {'state':'idle_gpu_occupied', 'burn_pid':burn.pid,
                                            'latest_completed':read(output/'ratings.json')['rows'][-1]['step']
                                            if (output/'ratings.json').exists() else None,'time':time.time()})
            time.sleep(cfg['poll_seconds'])
    except Exception as error:
        write(output/'status.json', {'state':'failed','error':repr(error),'time':time.time()})
        raise
    finally:
        stop_child(evaluation)
        stop_child(burn)
    write(output/'status.json', {'state':'stopped','time':time.time()})


if __name__ == '__main__':
    main()
