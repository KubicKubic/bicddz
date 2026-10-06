"""Queue-boundary advantage trimming, retaining the complete training state."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from flax import serialization
from .scale_rollout_campaign import read,write,sha,summarize
from .scale_architecture_v6 import expected_followup, verify_source
from .switch_distributed_gae import equal,cancel_old_followups
from .train_v2 import atomic


def revised_config(old):
    cfg=copy.deepcopy(old)
    if cfg['ppo'].get('adv_keep_fraction',1.)!=1.:
        raise ValueError('Source already trims advantages; inspect rather than repeat migration')
    cfg['ppo']['adv_keep_fraction']=.5
    return cfg


def migrate(saved,cfg,at):
    if cfg!=revised_config(saved['config']):
        raise ValueError('Only ppo.adv_keep_fraction may change')
    if saved['iteration']!=at or np.shape(saved['key'])!=(8,2) or 'ema' not in saved:
        raise ValueError('Complete eight-rank boundary checkpoint with EMA required')
    if saved['ema']['decay']!=cfg.get('ema_decay',.999):
        raise ValueError('EMA decay differs')
    result={**saved,'config':copy.deepcopy(cfg),'runtime':copy.deepcopy(saved['runtime'])}
    result['runtime'].setdefault('config_migrations',[]).append({
        'field':'ppo.adv_keep_fraction','from':1.,'to':.5,'iteration':at,
        'ranking':'global raw absolute advantage over valid learner decisions',
        'reason':'explicit user instruction','time':time.time()})
    for name in ('train','env','key','iteration','ema'):
        if not equal(saved[name],result[name]):raise ValueError('Retained state differs: '+name)
    return result


def verify_trim_rows(rows):
    for row in rows:
        eligible=int(row['available_train_samples']);kept=int(row['retained_train_samples'])
        if (kept!=eligible//2 or row['adv_keep_fraction']!=.5 or kept<=0 or
            row['applied_train_samples']!=kept or
            not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row.values())):
            raise RuntimeError('Exact global 50% selection or selected-sample update gate failed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();request=read(root/'REQUEST.json')
    migration=migrate
    exploration=request.get('transition')=='random_action_02_and_advantage_trim'
    if exploration:
        from .exploration_campaign import migrate as migration
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Persistent remote queue required')
    for path,expected in request['files_sha256'].items():
        if sha(Path(path))!=expected:raise RuntimeError('Frozen input changed: '+path)
    old=Path(request['old_root']);old_run=old/'production/training';at=request['at_iteration']
    verify_source(request,old_run)
    queue=Path(request['queue']);phase=root/'production';run=phase/'training'
    if run.exists():raise RuntimeError('Migration already began; explicit review required')
    # Source has completed and released its allocator. Check all eight devices.
    query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    devices=[list(map(int,line.split(','))) for line in query.stdout.splitlines()]
    needed=request['resource_gate']['estimated_peak_bytes']
    if (len(devices)!=8 or {d[0] for d in devices}!=set(range(8)) or
        min((d[1]-d[2])*2**20 for d in devices)<needed):
        raise RuntimeError('Eight-device free-memory gate failed')
    write(root/'remote_resource_check.json',{'devices_mib':devices,'required_bytes':needed,'time':time.time()})
    raw=(old_run/'latest.msgpack').read_bytes();source_hash=sha(old_run/'latest.msgpack')
    saved=serialization.msgpack_restore(raw);cfg=read(phase/'config.json')
    migrated=migration(saved,cfg,at)
    if saved['runtime']['source_sha256']!=sha(phase/'source.msgpack'):
        raise RuntimeError('Original source checkpoint identity differs')
    target=min(at+1000,cfg['continuation_updates'])
    expected=expected_followup(queue,old,target)
    run.mkdir()
    atomic(run/'latest.msgpack',serialization.msgpack_serialize(migrated))
    write(run/'config.json',cfg)
    for subdir,name in ((run,f'policy_{at:07d}.msgpack'),(run/'ema',f'policy_{at:07d}.msgpack')):
        subdir.mkdir(exist_ok=True)
        source=(old_run/name) if subdir==run else (old_run/'ema'/name)
        os.link(source,subdir/name)
    write(run/'latest.json',{'iteration':at,'time':time.time(),'policy':f'policy_{at:07d}.msgpack'})
    shutil.copyfile(old_run/'metrics.jsonl',run/'metrics.jsonl')
    write(root/'migration_receipt.json',{'source_checkpoint_sha256':source_hash,
        'migrated_checkpoint_sha256':sha(run/'latest.msgpack'),'iteration':at,
        'weights_adam_env_rng_vf_ema_retained':True,'ema_updates':saved['ema']['updates'],
        'optimizer_step':int(saved['train']['step']),'time':time.time()})
    write(phase/'READY.json',{'cpu_verification':{'passed':True},'files_sha256':request['files_sha256']})
    log=root/'engineering.log'
    environment={**os.environ,'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':','.join(map(str,range(8))),
        'NCCL_DEBUG':'INFO','PYTHONPATH':str(root/'code'),'OPENBLAS_NUM_THREADS':'1',
        'OMP_NUM_THREADS':'8','XLA_PYTHON_CLIENT_MEM_FRACTION':'.80'}
    argv=[sys.executable,'-u','-m','ddz.train_distributed_v5','--config',str(phase/'config.json'),
          '--source',str(phase/'source.msgpack'),'--out',str(run),'--steps',str(at+8),'--resume']
    with log.open('w') as stream:
        process=subprocess.Popen(argv,cwd=root/'code',env=environment,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:raise RuntimeError('Trim verification failed; old successor retained, no automatic retry')
    content=log.read_text();nccl=[line for line in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b',line)]
    if not nccl or 'nranks=8;' not in content:raise RuntimeError('Actual nranks=8 and NCCL evidence required')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row['iteration']>at]
    if [r['iteration'] for r in rows]!=list(range(at+1,at+9)):
        raise RuntimeError('Exactly eight fresh rollout updates required')
    result=summarize(rows,cfg['envs']*cfg['horizon']);verify_trim_rows(rows)
    if exploration and any(row.get('random_action_prob')!=.02 for row in rows):
        raise RuntimeError('Fresh rollout/update exploration configuration differs')
    final=serialization.msgpack_restore((run/'latest.msgpack').read_bytes())
    if final['ema']['updates']!=saved['ema']['updates']+8 or final['ema']['decay']!=saved['ema']['decay']:
        raise RuntimeError('EMA rollout clock continuity failed')
    if result['gpu_peak_memory_bytes']>=.8*min(d[1]*2**20 for d in devices):
        raise RuntimeError('Actual peak exceeds allocator budget')
    if expected_followup(queue,old,target)!=expected:raise RuntimeError('Old successor changed')
    cancelled=cancel_old_followups(queue,old)
    if cancelled!=[expected]:raise RuntimeError('Cancellation set changed')
    path=queue/'interrupted'/(expected+'.json');record=read(path)
    write(path,{**record,'reason':'USER_REQUESTED_RANDOM_ACTION_02_AND_ADV_TOP50' if exploration
                                else 'USER_REQUESTED_GLOBAL_ABS_ADV_TOP_50_PERCENT'})
    result.update(NCCL_evidence=nccl[:8],log=str(log),adv_keep_fraction=.5,
                  parameters=read(run/'status.json')['parameters'],
                  ema_replica_parameter_max_difference=0,time=time.time())
    if exploration:result.update(random_action_prob=.02,transition=request['transition'])
    write(root/'engineering_READY.json',result)
    write(root.parent/'production_current.json',{'state':'running','run':str(run),'nranks':8,
        'global_iteration':cfg['global_source_iteration']+at+8,
        'proof':str(root/'engineering_READY.json'),'time':time.time()})
    result=subprocess.run([sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(root),
        '--phase','production','--steps',str(target),'--resume'],cwd=root/'code',env=environment)
    if result.returncode:raise RuntimeError('Trim production failed; explicit new queue item required')


if __name__=='__main__':
    try:main()
    except Exception as error:
        root_arg=Path(sys.argv[sys.argv.index('--root')+1]) if '--root' in sys.argv else None
        if root_arg and root_arg.exists():write(root_arg/'FAILED.json',{'error':str(error),'time':time.time()})
        raise
