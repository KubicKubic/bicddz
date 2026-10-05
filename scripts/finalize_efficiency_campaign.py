"""Finish the frozen campaign without repeated interactive status polling.

The existing controller owns all experiments. This helper waits for its final
result, checks the declared promotion gates, renders the report, and continues
the verified policy or the saved original trainer. It starts no extra contrasts.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import copy
import shutil
import shlex


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def command_line(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except FileNotFoundError:
        return b''


def start(argv, cwd, env, log):
    with log.open('a') as stream:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)


def handoff_cluster(root,cluster,project,manifest,result,qualified):
    """Use the verified eight-GPU runner; keep production queue chunks bounded."""
    from flax import serialization
    ready_path=cluster/'engineering_READY.json'
    while not ready_path.exists():
        progress_path=cluster/'engineering'/'queue_job_status.json'
        if progress_path.exists() and read(progress_path)['state']=='failed':
            raise RuntimeError('Eight-GPU engineering task failed; no automatic retry')
        time.sleep(45)
    # Two further updates exercise full eight-device checkpoint resume, rather
    # than repeating the original engineering run or its CPU parity tests.
    if read(ready_path)['tested_updates']==32:
        resume_command='cd '+shlex.quote(str(cluster/'code'))+' && '+shlex.join([
            sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(cluster),
            '--phase','engineering','--steps','34','--resume'])
        helper=project.parent/'Q'/'tools'/'cluster'/'submit_scheduler_task.sh'
        submission=subprocess.run([str(helper),resume_command],check=True,text=True,capture_output=True)
        write(root/'cluster_resume_submission.json',{'command':resume_command,
              'submission':submission.stdout,'time':time.time()})
        while read(ready_path)['tested_updates']<34:
            progress=read(cluster/'engineering'/'queue_job_status.json')
            if progress['state']=='failed':raise RuntimeError('Eight-GPU resume verification failed')
            time.sleep(45)
    engineering=read(ready_path)
    if not engineering['passed'] or engineering['nranks']!=8 or not engineering['NCCL_evidence']:
        raise RuntimeError('Eight-GPU NCCL/health gate failed')
    if qualified:
        if result['winner']!='own_gae':raise RuntimeError('Unreviewed winner architecture')
        original=root/'training'/'own_gae_rng0'/'latest.msgpack'
        global_iteration=manifest['source_iteration']+manifest['confirm_steps']
        decision='Continue verified own_gae on eight GPUs; keep original saved checkpoint'
    else:
        original=project/'runs'/'a100_interaction_complete_move_v5'/'latest.msgpack'
        global_iteration=read(original.parent/'status.json')['iteration']
        decision='Continue saved original on eight GPUs; unconfirmed efficiency options disabled'
    phase=cluster/'production'
    phase.mkdir(exist_ok=False)
    shutil.copy2(original,phase/'source.msgpack')
    source=serialization.msgpack_restore((phase/'source.msgpack').read_bytes())
    cfg=copy.deepcopy(source['config'])
    cfg.update(envs=8192,per_gpu_envs=1024,per_gpu_minibatch=2048,
               global_source_iteration=global_iteration,continuation_updates=100000-global_iteration)
    cfg['ppo']['minibatch']=16384
    write(phase/'config.json',cfg)
    engineering_manifest=read(cluster/'engineering'/'READY.json')
    frozen_files=[phase/'source.msgpack',phase/'config.json',*sorted((cluster/'code').rglob('*.py'))]
    for path,expected in engineering_manifest['files_sha256'].items():
        if sha(Path(path))!=expected:raise RuntimeError('Eight-GPU frozen dependency changed')
    write(phase/'READY.json',{'version':1,'frozen_at':time.time(),
          'cpu_verification':engineering_manifest['cpu_verification'],
          'engineering_verification':engineering,'strength_result_sha256':sha(root/'result.json'),
          'files_sha256':{str(path):sha(path) for path in frozen_files}})
    target=min(1000,cfg['continuation_updates'])
    command='cd '+shlex.quote(str(cluster/'code'))+' && '+shlex.join([
            sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(cluster),
            '--phase','production','--steps',str(target)])
    helper=project.parent/'Q'/'tools'/'cluster'/'submit_scheduler_task.sh'
    submission=subprocess.run([str(helper),command],check=True,text=True,capture_output=True)
    receipt={'state':'queued','decision':decision,'run':str(phase/'training'),'nranks':8,
             'global_source_iteration':global_iteration,'source':str(original),
             'source_sha256':sha(phase/'source.msgpack'),'global_envs':8192,
             'fresh_decisions_per_update':524288,'global_minibatch':16384,
             'queue_chunk_updates':1000,'command':command,'submission':submission.stdout,
             'strength_verification':str(root/'final_verification.json'),
             'engineering_verification':str(ready_path),'time':time.time()}
    write(root/'production_continuation.json',receipt)
    write(project/'runs'/'production_current.json',receipt)
    # Wait for a real first production update before starting the CPU ladder.
    run=phase/'training';deadline=time.monotonic()+1800
    while not (run/'status.json').exists():
        status_path=phase/'queue_job_status.json'
        if status_path.exists() and read(status_path)['state']=='failed':raise RuntimeError('Production queue task failed')
        if time.monotonic()>deadline:raise RuntimeError('Production not started; retained queue receipt, no duplicate launch')
        time.sleep(30)
    progress=read(run/'status.json')
    if progress['nranks']!=8:raise RuntimeError('Production world size mismatch')
    ladder=run/'score_ladder_all_roles_500_v1';ladder.mkdir(exist_ok=True)
    write(ladder/'continuation_origin.json',{'step_zero_source_v5_iteration':global_iteration,
          'axis':'eight-GPU updates since preserved source; 524288 new decisions per update'})
    cpus=','.join(map(str,sorted(os.sched_getaffinity(0))[-16:-8]))
    argv=['taskset','-c',cpus,'nice','-n','19',sys.executable,'-u','-m',
          'ddz.score_ladder_graph_all_roles_v5','--run-dir',str(run),'--output',str(ladder),
          '--deals','1536','--chunk-deals','48','--bootstrap-rounds','2000','--watch','--poll-seconds','60']
    watcher=start(argv,cluster/'code',{**os.environ,'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'4',
          'OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(cluster/'code')},root/'production_ladder.log')
    receipt.update(state='running',remote_pid=progress['pid'],ladder_pid=watcher.pid,
                   ladder=str(ladder),time=time.time())
    write(root/'production_continuation.json',receipt);write(project/'runs'/'production_current.json',receipt)
    write(root/'finalizer_status.json',{'state':'complete','nranks':8,'confirmed_benefit':qualified,
          'report':str(root/'report.html'),'figure':str(root/'comparison.png'),
          'production_run':str(run),'time':time.time()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--cluster-root', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    project = root.parent.parent
    lock = (root / 'finalizer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'production_continuation.json').exists():
        raise RuntimeError('Continuation already recorded; do not launch another trainer')
    write(root / 'finalizer_status.json', {'state': 'waiting', 'pid': os.getpid(),
          'time': time.time(), 'additional_experiments': 0})
    while True:
        status = read(root / 'campaign_status.json')
        if status['state'] == 'failed':
            raise RuntimeError('Campaign failed; preserve failure for explicit review')
        if status['state'] == 'complete':
            break
        if b'ddz.efficiency_campaign' not in command_line(status['pid']):
            raise RuntimeError('Campaign controller exited without final results')
        time.sleep(45)
    manifest = read(root / 'manifest.json')
    if sha(root / 'source.msgpack') != manifest['source_sha256']:
        raise RuntimeError('Frozen source changed')
    for name, expected in manifest['config_sha256'].items():
        if sha(root / 'configs' / (name + '.json')) != expected:
            raise RuntimeError('Frozen configuration changed: ' + name)
    for name, expected in manifest['frozen_code_sha256'].items():
        if sha(root / 'code' / name) != expected:
            raise RuntimeError('Frozen code changed: ' + name)
    for name, expected in manifest['opponent_sha256'].items():
        if sha(Path(name)) != expected:
            raise RuntimeError('Frozen reference policy changed')
    if not read(root / 'engineering_verification.json')['passed']:
        raise RuntimeError('Engineering verification failed')
    result = read(root / 'result.json')
    winner = result['winner']
    confirmation = result['confirmation']
    if set(confirmation) != {'0', '1'}:
        raise RuntimeError('Both independent training continuations are required')
    for seed, evaluation in confirmation.items():
        if evaluation['games'] != 12 * manifest['confirmation_deals'] or evaluation['invalid_actions']:
            raise RuntimeError('Incomplete or invalid held-out evaluation')
        run = root / 'training' / f'{winner}_rng{seed}'
        baseline = root / 'training' / f'baseline_rng{seed}'
        filename = f"policy_{manifest['confirm_steps']:07d}.msgpack"
        if evaluation['identity']['checkpoint_sha256'] != [sha(run / filename), sha(baseline / filename)]:
            raise RuntimeError('Held-out checkpoint identity mismatch')
    totals = {'updates': 0, 'fresh_decisions': 0, 'invalid_actions': 0, 'nonfinite': 0}
    for metrics in sorted((root / 'training').glob('*/metrics.jsonl')):
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            totals['updates'] += 1
            for key in ('fresh_decisions', 'invalid_actions', 'nonfinite'):
                totals[key] += row[key]
    if totals['invalid_actions'] or totals['nonfinite']:
        raise RuntimeError('Training health gate failed')
    qualified = all(c['results']['forced']['ci95'][0] > 0 and
                    c['results']['natural']['ci95'][1] >= 0 for c in confirmation.values())
    qualified = qualified and result['strong_douzero_paired_change']['ci95'][0] > 0
    if qualified != result['confirmed_benefit']:
        raise RuntimeError('Promotion decision disagrees with frozen gates')
    write(root / 'final_verification.json', {'passed': True, 'totals': totals,
          'confirmed_benefit': qualified, 'additional_experiments': 0,
          'result_sha256': sha(root / 'result.json'), 'time': time.time()})
    subprocess.run(['/root/miniforge/envs/wm/bin/python', '-m', 'ddz.report_efficiency',
                    '--root', str(root)], cwd=project, check=True)
    if args.cluster_root:
        handoff_cluster(root,args.cluster_root.resolve(),project,manifest,result,qualified)
        return
    # A recorded live trainer is an ownership conflict, never a reason to kill it.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        cmd = command_line(proc.name)
        if b'ddz.train_v5' in cmd or b'ddz.train_efficiency' in cmd:
            raise RuntimeError('A DDZ trainer already owns the device')
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'CUDA_VISIBLE_DEVICES': '0',
           'XLA_PYTHON_CLIENT_MEM_FRACTION': '.35', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '8'}
    if qualified:
        if winner != 'own_gae':
            raise RuntimeError('Only the reviewed own_gae continuation is eligible')
        run = root / 'training' / 'own_gae_rng0'
        argv = [sys.executable, '-u', '-m', 'ddz.train_efficiency', '--config',
                str(root / 'configs' / 'own_gae.json'), '--source', str(root / 'source.msgpack'),
                '--out', str(run), '--steps', '74000', '--training-seed', '0',
                '--resume', '--require-a100']
        env['PYTHONPATH'] = str(root / 'code')
        cwd = root / 'code'
        start_iteration = manifest['confirm_steps']
        decision = 'Continue verified own_gae; retain original checkpoint at 26554'
    else:
        run = project / 'runs' / 'a100_interaction_complete_move_v5'
        argv = read(root / 'production_pause.json')['resume_command']
        argv[0] = sys.executable
        cwd = project
        env['PYTHONPATH'] = str(project)
        start_iteration = read(run / 'status.json')['iteration']
        decision = 'Resume saved original; no sample-efficiency change met all benefit gates'
    trainer = start(argv, cwd, env, root / 'production_continuation.log')
    receipt = {'state': 'starting', 'decision': decision, 'pid': trainer.pid,
               'command': argv, 'cwd': str(cwd), 'run': str(run),
               'start_iteration': start_iteration, 'source_iteration': manifest['source_iteration'],
               'verification': str(root / 'final_verification.json'), 'time': time.time()}
    write(root / 'production_continuation.json', receipt)
    deadline = time.monotonic() + 1200
    while True:
        if trainer.poll() is not None:
            raise RuntimeError('Continuation exited; inspect production_continuation.log')
        progress = read(run / 'status.json')
        if progress['pid'] == trainer.pid and progress['iteration'] >= start_iteration + 3:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Continuation health check timed out; do not launch a duplicate')
        time.sleep(20)
    if qualified:
        ladder = run / 'score_ladder_all_roles_500_efficiency_v1'
        ladder.mkdir(exist_ok=True)
        write(ladder / 'continuation_origin.json', {
            'step_zero_source_v5_iteration': manifest['source_iteration'],
            'axis': 'updates since frozen source checkpoint; not training from scratch'})
        cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[-16:-8]))
        ladder_argv = ['taskset', '-c', cpus, 'nice', '-n', '19', sys.executable,
                       '-u', '-m', 'ddz.score_ladder_graph_all_roles_v5', '--run-dir', str(run),
                       '--output', str(ladder), '--deals', '1536', '--chunk-deals', '48',
                       '--bootstrap-rounds', '2000', '--watch', '--poll-seconds', '60']
        watcher = start(ladder_argv, cwd, {**env, 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '4'},
                        root / 'production_ladder.log')
        receipt.update(ladder_pid=watcher.pid, ladder=str(ladder), ladder_command=ladder_argv)
    receipt.update(state='running', verified_iteration=progress['iteration'], time=time.time())
    write(root / 'production_continuation.json', receipt)
    write(project / 'runs' / 'production_current.json', receipt)
    write(root / 'finalizer_status.json', {'state': 'complete', 'pid': os.getpid(),
          'confirmed_benefit': qualified, 'report': str(root / 'report.html'),
          'figure': str(root / 'comparison.png'), 'production_pid': trainer.pid, 'time': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Keep failure observable; never retry or start an alternative trainer silently.
        print('FINALIZER FAILED:', repr(error), flush=True)
        if '--root' in sys.argv:
            failed_root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            if failed_root.exists():
                write(failed_root / 'finalizer_status.json', {'state': 'failed',
                      'pid': os.getpid(), 'error': str(error), 'time': time.time()})
        raise
