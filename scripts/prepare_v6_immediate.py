"""Explicit user emergency: retain V5 state, replace boundary migration, recover queue."""
import argparse
from collections import deque
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_v6 import supersede_pending
from ddz.scale_rollout_campaign import read,write,sha
from ddz.upgrade_v6 import revised_config
from flax import serialization


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--supersede-root',type=Path,required=True)
    ap.add_argument('--reviewed-failed',action='store_true')
    ap.add_argument('--tests-proof',type=Path,required=True);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];root=args.root.resolve();previous=args.supersede_root.resolve()
    if root.exists():raise RuntimeError('Immediate handoff already retained; inspect before any retry')
    tests=read(args.tests_proof)
    if not tests.get('passed'):raise RuntimeError('Immediate pause/ownership tests required')
    for name,expected in tests['files_sha256'].items():
        if sha(name)!=expected:raise RuntimeError('Tested immediate handoff dependency changed')
    queue=repo.parent/'Q/tools/cluster/cluster_scheduler.queue';cluster=queue.parent
    status=read(queue/'status.json')
    if status['phase']!='RUNNING' or status['running_count']!=1 or time.time()-status['updated_at_ns']/1e9>30:
        raise RuntimeError('Healthy owned queue worker required')
    active=read(queue/'running'/(status['active_id']+'.json'));tokens=shlex.split(active['command'])
    old_run=Path(read(repo/'runs/production_current.json')['run']);old=old_run.parent.parent
    if ('ddz.cluster_distributed_job' not in tokens or '--root' not in tokens or
        Path(tokens[tokens.index('--root')+1]).resolve()!=old.resolve()):
        raise RuntimeError('Active item is not the owned DDZ production run')
    request=read(previous/'REQUEST.json')
    if args.reviewed_failed:
        failed_submission=json.loads(read(previous/'SUBMISSION.json')['submission'].splitlines()[0])
        failed=read(queue/'failed'/(failed_submission['id']+'.json'))
        if (failed['status']!='failed' or not (previous/'FAILED.json').exists() or
            Path(request['old_root']).resolve()!=old.resolve()):
            raise RuntimeError('Exact reviewed failure and unchanged V5 recovery source required')
    elif request['active_source_queue_id']!=active['id']:
        raise RuntimeError('Reviewed V6 source ownership changed')
    for name,expected in request['files_sha256'].items():
        if sha(name)!=expected:raise RuntimeError('Previously frozen dependency changed: '+name)
    matches=[read(p) for p in (queue/'pending').glob('*.json')]
    bad_pending=bool(matches) if args.reviewed_failed else (len(matches)!=1 or
        Path(shlex.split(matches[0]['command'])[-1])!=previous)
    if bad_pending:
        raise RuntimeError('Expected only the reviewed, unclaimed V6 boundary item')
    cfg=read(old_run/'config.json');target=read(previous/'production/config.json')
    if target!=revised_config(cfg):raise RuntimeError('Audited architecture/protocol differs')
    log=old/'production'/(active['id']+'.log');content=log.read_text()
    child_pids=set(re.findall(r'^[^:\n]+:(\d+):\d+ .*NCCL.*nranks[ =]+8\b',content,re.MULTILINE))
    if len(child_pids)!=1:raise RuntimeError('Actual active trainer PID/NCCL identity missing')
    actual_training_pid=int(next(iter(child_pids)))
    root.mkdir();(root/'production').mkdir()
    shutil.copytree(previous/'code',root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('scale_architecture_v6.py','optimizer_efficiency.py','train_efficiency.py',
                 'train_distributed_v5.py','ppo_efficiency.py'):
        shutil.copyfile(repo/'ddz'/name,root/'code/ddz'/name)
    for name in ('CPU_VERIFICATION.json','TESTS_VERIFICATION.json','RESUME_VERIFICATION.json','CUDA_VERIFICATION.json'):
        shutil.copyfile(previous/name,root/name)
    shutil.copyfile(args.tests_proof,root/'IMMEDIATE_TESTS_VERIFICATION.json')
    write(root/'production/config.json',target)
    write(root/'EMERGENCY_REQUEST.json',{'user_instruction':'现在就切换','source_claim':active,
        'old_worker':status,'previous_v6_root':str(previous),'time':time.time()})
    if not args.reviewed_failed:
        supersede_pending(queue,matches[0]['id'],root,'USER_REQUESTED_IMMEDIATE_V6_SWITCH')
    else:
        write(root/'REVIEWED_FAILURE.json',{'failed_root':str(previous),'failure':read(previous/'FAILED.json'),
            'resolution':'1024-step warmup of newborn Adam coordinates; original acceptance gates retained',
            'explicit_new_queue_item':True,'time':time.time()})
    # The bridge is used only for this explicitly authorized interruption and recovery.
    subprocess.run([str(cluster/'stop_scheduler_task.sh')],check=True)
    stop_time=time.monotonic();deadline=stop_time+180
    while True:
        state=read(queue/'status.json')
        if (state['phase']=='STOPPED' and not list((queue/'running').glob('*.json')) and
            not (queue/'worker.lock').exists() and time.monotonic()-stop_time>=12):break
        if time.monotonic()>deadline:raise RuntimeError('Old worker/claim not resolved; no competing launch')
        time.sleep(2)
    interrupted=read(queue/'interrupted'/(active['id']+'.json'))
    if interrupted['task_pid']!=active['task_pid']:raise RuntimeError('Interrupted source identity changed')
    saved=serialization.msgpack_restore((old_run/'latest.msgpack').read_bytes());at=int(saved['iteration'])
    latest=read(old_run/'latest.json');source_status=read(old_run/'status.json')
    if (latest['iteration']!=at or source_status['iteration']<at or source_status['nranks']!=8 or
        saved['config']!=cfg or saved['key'].shape!=(8,2) or 'ema' not in saved):
        raise RuntimeError('Complete source checkpoint/marker/protocol differs')
    metrics=(old_run/'metrics.jsonl').read_text();all_rows=[json.loads(x) for x in metrics.splitlines()]
    rows=[r for r in all_rows if r['iteration']<=at][-600:]
    if (len(rows)<8 or rows[-1]['iteration']!=at or any(r['nranks']!=8 or r['invalid_actions'] or
        r['nonfinite'] or r['fresh_decisions']!=cfg['envs']*cfg['horizon'] or
        r.get('replica_parameter_max_difference',0) or r.get('ema_replica_parameter_max_difference',0) or
        not all(isinstance(v,(int,float)) and math.isfinite(v) for v in r.values()) for r in rows)):
        raise RuntimeError('Retained source rollout health gate failed')
    content=log.read_text()
    nccl=[line for line in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b',line)]
    if not nccl or 'nranks=8;' not in content:raise RuntimeError('Actual source eight-rank NCCL proof missing')
    proof={'passed':True,'queue_id':active['id'],'nranks':8,'NCCL_evidence':nccl[:8],
        'source_log':str(log),'source_log_sha256':sha(log),'accepted_checkpoint_iteration':at,
        'validated_updates':len(rows),'invalid_actions':0,'nonfinite':0,
        'retained_adam_step':int(saved['train']['step']),'retained_ema_updates':int(saved['ema']['updates'])}
    write(root/'source_health_proof.json',proof)
    pause={'retained_checkpoint_iteration':at,'retained_global_iteration':cfg['global_source_iteration']+at,
        'interrupted_queue_id':active['id'],'remote_training_pid':actual_training_pid,
        'retained_checkpoint_sha256':sha(old_run/'latest.msgpack'),
        'source_status_sha256':sha(old_run/'status.json'),'source_health_proof':str(root/'source_health_proof.json'),
        'last_logged_iteration':all_rows[-1]['iteration'],'unsaved_logged_updates_not_resumed':sum(r['iteration']>at for r in all_rows),
        'remote_process_absence_required_before_migration':True,'time':time.time()}
    write(root/'source_pause_receipt.json',pause)
    # Preserve every historical row; exclude only unsaved updates from future resume metrics.
    (root/'source_metrics_before_pause.jsonl').write_text(metrics)
    filtered='\n'.join(x for x in metrics.splitlines() if json.loads(x)['iteration']<=at)+'\n'
    from ddz.train_v2 import atomic
    atomic(old_run/'metrics.jsonl',filtered.encode())
    ready=read(previous/'READY.json');ready['boundary_source_policy']='checksum-bound explicit user emergency pause'
    files=[p for p in (root/'code').rglob('*') if p.is_file()]
    files += [root/name for name in ('CPU_VERIFICATION.json','TESTS_VERIFICATION.json','RESUME_VERIFICATION.json',
        'CUDA_VERIFICATION.json','IMMEDIATE_TESTS_VERIFICATION.json','source_health_proof.json',
        'source_pause_receipt.json','production/config.json')]
    files += [old_run/'config.json']
    ready['files_sha256']={str(p):sha(p) for p in files};write(root/'READY.json',ready)
    request={**request,'at_iteration':at,'source_pause_receipt':str(root/'source_pause_receipt.json'),
        'active_source_queue_id':active['id'],'optimizer_new_coordinate_warmup_steps':1024,
        'source_exit_wait_seconds':180,
        'files_sha256':{**ready['files_sha256'],str(root/'READY.json'):sha(root/'READY.json')},
        'user_instruction':'现在就切换','time':time.time()};write(root/'REQUEST.json',request)
    helper=cluster/'submit_scheduler_task.sh'
    command='cd '+shlex.quote(str(root/'code'))+' && '+shlex.join(['env','JAX_PLATFORMS=cpu',
        'CUDA_VISIBLE_DEVICES=','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2',sys.executable,'-u','-m',
        'ddz.scale_architecture_v6','--root',str(root)])
    submitted=subprocess.run([str(helper),command],check=True,text=True,capture_output=True)
    write(root/'SUBMISSION.json',{'state':'queued','command':command,'submission':submitted.stdout,
        'parameters':8014192,'at_iteration':at,'time':time.time()})
    fallback_target=min(at+1000,cfg['continuation_updates'])
    fallback='cd '+shlex.quote(str(old/'code'))+' && '+shlex.join([sys.executable,'-u','-m',
        'ddz.cluster_distributed_job','--root',str(old),'--phase','production','--steps',str(fallback_target),'--resume'])
    recovery=subprocess.run([str(helper),fallback],check=True,text=True,capture_output=True)
    write(root/'V5_FALLBACK.json',{'command':fallback,'submission':recovery.stdout,'target':fallback_target,
        'cancel_only_after_v6_acceptance':True,'time':time.time()})
    bridge='RUN: '+shlex.join(['env','Q_CLUSTER_TASK=1','/root/miniforge/envs/wm/bin/python',
        str(cluster/'cluster_scheduler_queue.py'),'worker','--gpus','8'])+'\n'
    atomic(cluster/'cluster_scheduler.command',bridge.encode())
    (cluster/'cluster_scheduler.command').chmod(0o644)
    write(root/'worker_recovery_receipt.json',{'user_directed_immediate_switch':True,
        'old_worker_stopped':True,'old_queue_id':active['id'],'no_unresolved_running_claims':True,
        'source_checkpoint_iteration':at,'same_remote_queue_worker_restart':True,'time':time.time()})
    print(json.dumps({'state':'immediate_v6_queued_and_worker_recovering',**pause},indent=2),flush=True)


if __name__=='__main__':main()
