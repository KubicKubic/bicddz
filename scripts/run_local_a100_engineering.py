"""Temporarily yield the single idle owner and always restore its controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ddz.local_douzero_watch import active_cuda_processes,read,write


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ',1)[1][0]!='Z'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--attempt',type=int,required=True)
    ap.add_argument('--memory-fraction',type=float,default=.375);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];root=args.audit.resolve()
    log=root/f'cuda_engineering_attempt_{args.attempt}.log'
    handoff=root/f'LOCAL_GPU_HANDOFF_{args.attempt}.json'
    if log.exists() or handoff.exists():raise RuntimeError('Engineering attempt already retained; choose a new explicit attempt')
    if not 0<args.memory_fraction<=.375:raise RuntimeError('Respect retained local GPU state and reviewed engineering cap')
    if not read(root/'RESUME_VERIFICATION.json')['passed']:raise RuntimeError('Real complete CPU resume proof required')
    pointer=repo/'runs/local_a100_current.json';owner=read(pointer);cfg=read(owner['config'])
    state=read(owner['status']);pid=owner['watcher_pid'];burn=state.get('burn_pid')
    if state['state']!='idle_gpu_occupied' or not burn:raise RuntimeError('Wait for useful GPU work to finish')
    if (b'ddz.local_a100_controller' not in Path(f'/proc/{pid}/cmdline').read_bytes() or
        str(Path(owner['config'])).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes() or
        b'ddz.gpu_idle' not in Path(f'/proc/{burn}/cmdline').read_bytes() or
        int(next(x for x in Path(f'/proc/{burn}/status').read_text().splitlines() if x.startswith('PPid:')).split()[1])!=pid):
        raise RuntimeError('Owned controller/idle process identity differs')
    if active_cuda_processes({pid,burn,os.getpid()}):raise RuntimeError('Respect unrelated active local GPU work')
    receipt={'old_controller_pid':pid,'old_burn_pid':burn,'state':'stopping_idle','time':time.time()}
    write(handoff,receipt)
    stop=[False];child=[None]
    def terminate(*_):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].terminate()
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    os.kill(pid,signal.SIGTERM)
    end=time.monotonic()+30
    while (alive(pid) or alive(burn)) and time.monotonic()<end:time.sleep(.1)
    if alive(pid) or alive(burn):raise RuntimeError('Previous controller/idle must stop before GPU engineering')
    try:
        environment={**os.environ,'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':'0',
            'XLA_PYTHON_CLIENT_MEM_FRACTION':str(args.memory_fraction),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2',
            'PYTHONPATH':str(repo),'LD_LIBRARY_PATH':'/tmp/ddz_nvml:'+os.environ.get('LD_LIBRARY_PATH','')}
        with log.open('w') as stream:
            child[0]=subprocess.Popen([sys.executable,'-u','-m','ddz.audit_v6_cuda','--audit',str(root)],
                cwd=repo,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            write(handoff,{**receipt,'state':'cuda_engineering','engineering_pid':child[0].pid,
                           'memory_fraction':args.memory_fraction})
            try:code=child[0].wait(timeout=600)
            except subprocess.TimeoutExpired:
                child[0].terminate()
                try:child[0].wait(timeout=15)
                except subprocess.TimeoutExpired:child[0].kill();child[0].wait(timeout=10)
                raise RuntimeError('CUDA engineering timed out')
        if code or stop[0]:raise RuntimeError('CUDA engineering failed/interrupted; inspect retained log')
    finally:
        environment={**os.environ,'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':'',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','PYTHONPATH':cfg['code_dir']}
        controller_root=Path(owner['root'])
        with (controller_root/'controller.log').open('a') as stream:
            restored=subprocess.Popen([sys.executable,'-u','-m','ddz.local_a100_controller','--config',owner['config']],
                cwd=cfg['code_dir'],env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write(pointer,{**owner,'watcher_pid':restored.pid,'engineering_handoff':str(handoff),'time':time.time()})
        end=time.monotonic()+30
        while time.monotonic()<end:
            if restored.poll() is not None:raise RuntimeError('Restored controller exited; inspect retained log')
            watcher=read(controller_root/'watcher.json');current=read(owner['status'])
            if watcher['pid']==restored.pid and current['state'] in ('idle_gpu_occupied','regular_evaluation'):
                if current['state']=='regular_evaluation' or (current.get('burn_pid')!=burn and alive(current['burn_pid'])):break
            time.sleep(.2)
        else:raise RuntimeError('Restored controller did not regain GPU ownership')
        write(handoff,{**receipt,'state':'controller_restored','new_controller_pid':restored.pid,
            'idle_or_evaluation_restored':True,'time':time.time()})
        print(json.dumps({'local_controller_restored':restored.pid,'status':current['state']}),flush=True)


if __name__=='__main__':main()
