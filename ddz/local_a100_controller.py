"""Serialize regular benchmarks, high-precision verification and idle GPU work."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .local_douzero_watch import (read,write,pending_steps,active_cuda_processes,
                                 stop_child,update_curve)
from .douzero_precision import completed_blocks,aggregate


def follow_architecture(cfg,regular,root):
    """Drain benchmarks, then follow a verified architecture or sample handoff."""
    pointer=cfg.get('follow_production_pointer')
    if not pointer:return regular
    current=read(pointer);run=Path(current['run'])
    candidate=read(run/'config.json')
    if candidate.get('model_family')!='V6' or str(run)==regular['run_dir']:return regular
    proof=read(current['proof'])
    original=read(Path(regular['run_dir'])/'config.json')
    trimming=original.get('model_family')=='V6'
    parameters=proof.get('parameters')
    if trimming:
        campaign=run.parent.parent
        registration=read(campaign/'REQUEST.json');migration=read(campaign/'migration_receipt.json')
        exploration=registration.get('transition')=='random_action_02_and_advantage_trim'
        if exploration:
            from .exploration_campaign import revised_config
            if proof.get('random_action_prob')!=.02:
                raise RuntimeError('Exploration handoff lacks its accepted probability proof')
        else:
            from .trim_rollout_campaign import revised_config
        try:expected=revised_config(original)
        except ValueError as error:raise RuntimeError('Unregistered sample-selection handoff') from error
        if (candidate!=expected or
            Path(registration['old_root']).resolve()!=Path(regular['run_dir']).parent.parent.resolve() or
            not migration.get('weights_adam_env_rng_vf_ema_retained') or
            migration.get('iteration')!=registration.get('at_iteration')):
            raise RuntimeError('Unregistered sample-selection handoff or changed retained state')
        # Initial trimming acceptance proves exact counts and synchronization;
        # its older frozen schema records parameter count in trainer status.
        if parameters is None:
            status=read(run/'status.json')
            if proof.get('adv_keep_fraction')!=.5 or status.get('nranks')!=8:
                raise RuntimeError('Trimming parameter-count evidence missing')
            parameters=status.get('parameters')
    else:
        from .upgrade_v6 import revised_config
        if candidate!=revised_config(original):raise RuntimeError('Unregistered production architecture handoff')
    if (current.get('nranks')!=8 or not proof.get('passed') or proof.get('nranks')!=8 or
        not proof.get('NCCL_evidence') or parameters!=8_014_192):
        raise RuntimeError('V6 local evaluator handoff lacks accepted eight-rank proof')
    receipt=Path(root)/'architecture_handoff.json'
    if receipt.exists():
        record=read(receipt)
        if record['new_run']!=str(run):raise RuntimeError('Recorded V6 evaluator run differs')
        first=record['initial_step']
    else:
        first=read(run/'latest.json')['iteration']
        write(receipt,{'old_run':regular['run_dir'],'new_run':str(run),
            'initial_step':first,'proof':current['proof'],
            'transition':registration['transition'] if trimming and exploration else
                         'advantage_trim' if trimming else 'V5_to_V6','time':time.time()})
    result={**regular,'run_dir':str(run),'initial_steps':[first]}
    result['protocol']={**regular['protocol'],'architecture_transition':{
        'family':'V6','parameters':8_014_192,'first_global_step':candidate['global_source_iteration']+first}}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    args = ap.parse_args()
    cfg = read(args.config)
    regular = read(cfg['regular_config'])
    precision = cfg['precision']
    root = Path(cfg['root'])
    regular_root = Path(regular['output'])
    precision_root = Path(precision['root'])
    for path,expected in cfg['files_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen controller dependency changed: '+path)
    # Share the old watcher's GPU ownership lock: two controllers cannot run.
    lock = (regular_root/'watcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop = [False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    write(root/'watcher.json',{'pid':os.getpid(),'config':str(args.config),'time':time.time()})
    burn = evaluation = None

    def environment(code):
        return {**os.environ,'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':'0',
            'JAX_COMPILATION_CACHE_DIR':str(cfg.get('jax_cache',root/'shared_jax_cache')),
            'XLA_PYTHON_CLIENT_MEM_FRACTION':str(cfg.get('gpu_memory_fraction','.12')),
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
            'PYTHONPATH':str(code)+':'+regular['torch_cpu_path'],
            'LD_LIBRARY_PATH':'/tmp/ddz_nvml:'+os.environ.get('LD_LIBRARY_PATH','')}

    def status(value):
        write(root/'status.json',{**value,'pid':os.getpid(),'time':time.time()})

    def idle():
        nonlocal burn
        if burn is not None and burn.poll() is not None:
            raise RuntimeError('Idle occupation child failed')
        if burn is None:
            with (root/'idle_gpu.log').open('a') as stream:
                burn = subprocess.Popen([regular['burn_python'],'-u','-m','ddz.gpu_idle'],
                    cwd=cfg['code_dir'],env={**os.environ,'PYTHONPATH':cfg['code_dir'],
                         'CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':'/tmp/ddz_nvml'},
                    stdout=stream,stderr=subprocess.STDOUT)

    def run_evaluation(options,step,destination,code,offset=0,large=False):
        nonlocal evaluation,burn
        stop_child(burn)
        burn = None
        destination = Path(destination)
        destination.mkdir(parents=True,exist_ok=True)
        if (destination/'FAILED.json').exists():
            raise RuntimeError('Inspect failed evaluation before retry: '+str(destination))
        argv = [sys.executable,'-u','-m',cfg.get('evaluation_module','ddz.compare_efficiency_douzero'),
            '--run-dir',options['run_dir'],'--step',str(step),'--douzero-src',regular['douzero_src'],
            '--resnet-src',regular['resnet_src'],'--weights-root',regular['weights_root'],
            '--out',str(destination),'--deals',str(options['deals']),
            '--chunk-deals',str(options['chunk_deals']),'--seed',str(options['seed']),
            '--require-a100','--jax-cache',str(cfg.get('jax_cache',root/'shared_jax_cache'))]
        if large:
            argv += ['--deal-offset',str(offset)]
        with (destination/'evaluation.log').open('a') as stream:
            evaluation = subprocess.Popen(argv,cwd=code,env=environment(code),
                                          stdout=stream,stderr=subprocess.STDOUT)
        value = {'state':'large_validation' if large else 'regular_evaluation','step':step,
                 'evaluation_pid':evaluation.pid,'global_deal_offset':offset,'output':str(destination)}
        status(value)
        write(regular_root/'status.json',{**value,'time':time.time()})
        while evaluation.poll() is None and not stop[0]:
            time.sleep(2)
        if stop[0]:
            return False
        code = evaluation.returncode
        evaluation = None
        if code:
            write(destination/'FAILED.json',{'exit_code':code,'time':time.time()})
            raise RuntimeError('Local evaluation failed; retained log: '+str(destination))
        return True

    try:
        while not stop[0]:
            excluded = {os.getpid()}
            if burn is not None:
                excluded.add(burn.pid)
            foreign = active_cuda_processes(excluded)
            if foreign:
                stop_child(burn)
                burn = None
                status({'state':'waiting_for_local_gpu','foreign_pids':foreign})
                time.sleep(10)
                continue
            pending = pending_steps(regular['run_dir'],regular['start_step'],regular_root,
                                    regular.get('initial_steps',()))
            if not pending:
                regular=follow_architecture(cfg,regular,root)
                pending=pending_steps(regular['run_dir'],regular['start_step'],regular_root,
                                      regular.get('initial_steps',()))
            if pending:
                step = pending[0]
                if not run_evaluation(regular,step,regular_root/f'step_{step:07d}',regular['code_dir']):
                    break
                idle()  # GPU occupation while the CPU renders/publishes results.
                update_curve(regular)
                print(json.dumps({'regular_completed':step}),flush=True)
                continue
            final_path = precision_root/'precision_result.json'
            done = final_path.exists() and read(final_path)['state']=='complete'
            if not done:
                paths = completed_blocks(precision)
                deals = len(paths)*precision['block_deals']
                stage = 1
                target = precision['initial_deals']
                if final_path.exists():
                    previous = read(final_path)
                    stage = previous['stage']+1
                    target = previous['deals']*2
                write(precision_root/'progress.json',{'state':'running','completed_deals':deals,
                    'completed_games':6*deals,'target_deals':target,'stage':stage,'time':time.time()})
                if deals >= target:
                    idle()  # Bootstrap aggregation runs on CPU.
                    result = aggregate(precision,paths[:target//precision['block_deals']],stage)
                    result['time'] = time.time()
                    write(precision_root/f'stage_{stage}.json',result)
                    write(final_path,result)
                    print(json.dumps({'precision_stage':stage,'state':result['state'],
                         'deals':deals,'interval_width':result['interval_width']}),flush=True)
                    continue
                options = {'run_dir':precision['run_dir'],'deals':precision['block_deals'],
                    'chunk_deals':precision['chunk_deals'],'seed':precision['seed'],
                    'output':str(precision_root)}
                destination = precision_root/'blocks'/f'deal_{deals:07d}'
                if not run_evaluation(options,precision['step'],destination,cfg['code_dir'],deals,True):
                    break
                print(json.dumps({'precision_deals_completed':deals+precision['block_deals']}),flush=True)
                continue
            idle()
            status({'state':'idle_gpu_occupied','burn_pid':burn.pid,'precision_complete':True})
            write(regular_root/'status.json',{'state':'idle_gpu_occupied','burn_pid':burn.pid,
                                             'controller':str(root),'time':time.time()})
            time.sleep(10)
    except Exception as error:
        status({'state':'failed','error':repr(error)})
        # Keep the user's idle-GPU preference even while an evaluation failure
        # awaits review. Do not silently rerun or skip the failed experiment.
        stop_child(evaluation)
        evaluation = None
        stop_child(burn)
        burn = None
        while not stop[0]:
            excluded = {os.getpid()}
            if burn is not None:
                excluded.add(burn.pid)
            foreign = active_cuda_processes(excluded)
            if foreign:
                stop_child(burn)
                burn = None
            else:
                try:
                    idle()
                except Exception as idle_error:
                    stop_child(burn)
                    burn = None
                    status({'state':'failed_waiting_review','error':repr(error),
                            'idle_error':repr(idle_error)})
                    time.sleep(10)
                    continue
            status({'state':'failed_waiting_review','error':repr(error),
                    'burn_pid':burn.pid if burn is not None else None,'foreign_pids':foreign})
            time.sleep(10)
    finally:
        stop_child(evaluation)
        stop_child(burn)
    status({'state':'stopped'})


if __name__=='__main__':
    main()
