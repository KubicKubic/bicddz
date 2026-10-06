"""Replace the local controller at idle, retain benchmarks, follow V6 safely."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ddz.scale_rollout_campaign import read,write,sha


def running(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ',1)[1][0]!='Z'


def reviewed_entropy_handoff(owner,status,campaign):
    """Review an old registry failure only after its evaluation completed."""
    if not campaign or status.get('state')!='failed_waiting_review':return False
    if 'Unregistered sample-selection handoff' not in status.get('error',''):return False
    old=Path(owner['root']);root=Path(campaign)
    evidence=read(root/'engineering_READY.json');request=read(root/'REQUEST.json')
    handoff=read(old/'architecture_handoff.json');source=Path(handoff['new_run'])
    from ddz.entropy_campaign import revised_config,TRANSITION
    if (request.get('transition')!=TRANSITION or
        Path(request['old_root']).resolve()!=source.parent.parent.resolve() or
        read(root/'production/config.json')!=revised_config(read(source/'config.json')) or
        not evidence.get('passed') or evidence.get('nranks')!=8 or not evidence.get('NCCL_evidence') or
        evidence.get('random_action_prob')!=0. or abs(evidence.get('entropy_coefficient',0.)-.01)>1e-12 or
        not (old/f"step_{handoff['initial_step']:07d}/BEST/summary.json").exists() or
        any((p/'FAILED.json').exists() for p in old.glob('step_*'))):
        raise RuntimeError('Registry review requires accepted entropy handoff and completed, nonfailed evaluation')
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--cpu-proof',type=Path,required=True)
    ap.add_argument('--wait-idle',action='store_true')
    ap.add_argument('--reviewed-handoff',type=Path);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];root=args.root.resolve()
    if root.exists():raise RuntimeError('Controller release already exists; inspect retained state')
    proof=read(args.cpu_proof)
    if not proof.get('passed'):raise RuntimeError('CPU controller verification required')
    for name,expected in proof.get('files_sha256',{}).items():
        if sha(Path(name))!=expected:raise RuntimeError('Verified controller dependency changed: '+name)
    pointer=repo/'runs/local_a100_current.json';owner=read(pointer)
    old_cfg=read(owner['config']);old=Path(owner['root'])
    status=read(old/'status.json')
    deadline=time.monotonic()+3600
    while args.wait_idle and status['state'] not in ('idle_gpu_occupied','failed_waiting_review'):
        if read(pointer)['watcher_pid']!=owner['watcher_pid']:
            raise RuntimeError('Local owner changed while waiting; do not replace another controller')
        if time.monotonic()>deadline:raise RuntimeError('Idle handoff wait expired; active evaluation preserved')
        time.sleep(2);status=read(old/'status.json')
    reviewed=reviewed_entropy_handoff(owner,status,args.reviewed_handoff)
    if status['state']!='idle_gpu_occupied' and not reviewed:
        raise RuntimeError('Wait for current local evaluation to finish before controller handoff')
    for name,expected in proof.get('files_sha256',{}).items():
        if sha(Path(name))!=expected:raise RuntimeError('Verified dependency changed during idle wait: '+name)
    if not read(Path(old_cfg['precision']['root'])/'precision_result.json')['state']=='complete':
        raise RuntimeError('Preserve the active precision evaluation before handoff')
    pid=owner['watcher_pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'ddz.local_a100_controller' not in cmd or str(Path(owner['config'])).encode() not in cmd:
        raise RuntimeError('Local controller process identity differs')
    burn=status['burn_pid'];burn_cmd=Path(f'/proc/{burn}/cmdline').read_bytes()
    if b'ddz.gpu_idle' not in burn_cmd:raise RuntimeError('Owned idle workload identity differs')
    root.mkdir()
    shutil.copytree(repo/'ddz',root/'code/ddz',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    regular=read(old_cfg['regular_config'])
    old_run=regular['run_dir']
    actual=read(repo/'runs/production_current.json')['run']
    regular.update(output=str(root),code_dir=str(root/'code'),run_dir=actual)
    if args.reviewed_handoff and Path(actual).parent.parent.resolve()==args.reviewed_handoff.resolve():
        regular['initial_steps']=[read(Path(actual)/'latest.json')['iteration']]
    history=root/'historical_ratings.json'
    shutil.copyfile(old/'ratings.json',history);regular['history_ratings']=str(history)
    # Cache only completed immutable results; never abort or inherit half an evaluation.
    for path in old.glob('step_*'):
        if (path/'BEST/summary.json').exists():(root/path.name).symlink_to(path.resolve(),target_is_directory=True)
    write(root/'regular_CONFIG.json',regular)
    cfg=copy.deepcopy(old_cfg)
    cfg.update(root=str(root),code_dir=str(root/'code'),regular_config=str(root/'regular_CONFIG.json'),
               follow_production_pointer=str(repo/'runs/production_current.json'))
    files=[p for p in (root/'code').rglob('*') if p.is_file()]
    files += [history,root/'regular_CONFIG.json']
    bindings={str(p):sha(p) for p in files}
    # Retain publisher BEST weights and the completed precision checkpoint proofs.
    for name,digest in old_cfg['files_sha256'].items():
        path=Path(name)
        if not path.is_relative_to(old):
            if sha(path)!=digest:raise RuntimeError('External frozen evaluation input changed')
            bindings[name]=digest
    cfg['files_sha256']=bindings
    write(root/'controller_CONFIG.json',cfg)
    environment={**os.environ,'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(root/'code')}
    # Verify the release fully before stopping the existing idle owner.
    subprocess.run([sys.executable,'-c',
        'from ddz.local_a100_controller import follow_architecture; from ddz.model_efficiency import EfficientMoveTransformer'],
        cwd=root/'code',env=environment,check=True)
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while (running(pid) or running(burn)) and time.monotonic()<deadline:time.sleep(.1)
    if running(pid) or running(burn):raise RuntimeError('Old owned controller/idle worker remains alive')
    with (root/'controller.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m','ddz.local_a100_controller','--config',str(root/'controller_CONFIG.json')],
            cwd=root/'code',env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if child.poll() is not None:raise RuntimeError('New controller exited; inspect controller.log')
        if (root/'status.json').exists() and read(root/'status.json')['state'] in ('idle_gpu_occupied','regular_evaluation'):
            break
        time.sleep(.2)
    else:raise RuntimeError('New controller did not publish active GPU status')
    receipt={'replaced_controller_pid':pid,'replaced_idle_pid':burn,'new_controller_pid':child.pid,
             'old_root':str(old),'new_root':str(root),'old_run':old_run,'current_training_run':actual,
             'production_handoff':'drain completed results then follow verified eight-rank production transitions',
             'idle_occupation_preserved':True,'time':time.time()}
    if reviewed:receipt['reviewed_registry_failure']=status['error']
    write(root/'migration_receipt.json',receipt)
    update={**owner,'watcher_pid':child.pid,'root':str(root),'config':str(root/'controller_CONFIG.json'),
        'status':str(root/'status.json'),'regular_curve':str(root/'douzero_best_curve.png'),
        'training_run':actual,'replaced_controller_pid':pid,'follow_production_pointer':cfg['follow_production_pointer'],
        'time':time.time()}
    # Retain a usable curve path immediately while the new controller is idle.
    shutil.copyfile(old/'douzero_best_curve.png',root/'douzero_best_curve.png')
    shutil.copyfile(old/'ratings.json',root/'ratings.json')
    write(pointer,update)
    print(__import__('json').dumps(receipt,indent=2))


if __name__=='__main__':main()
