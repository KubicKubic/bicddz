"""User-directed equal-budget eight-GPU rollout comparison and continuation.

Runs only at the existing production queue boundary. Both arms inherit the
same full checkpoint. Completed trials are retained and never retried silently.
"""
import argparse,copy,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from flax import serialization
from .switch_distributed_gae import equal,cancel_old_followups


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.chmod(0o644);os.replace(temp,path)


def revised_config(old,factor):
    if factor not in (2,4):raise ValueError('Compare environment multipliers 2 and 4 only')
    cfg=copy.deepcopy(old)
    cfg['envs']*=factor;cfg['per_gpu_envs']*=factor
    cfg['horizon']*=4//factor
    cfg['updates']*=2
    cfg['continuation_updates']=cfg['updates']-cfg['global_source_iteration']
    cfg['ppo']['lr']=cfg['ppo']['lr_min']=1e-5
    cfg['ema_decay']=0.999
    return cfg


def expand_checkpoint(saved,cfg,fresh,at):
    old=saved['config'];n=old['envs'];new=cfg['envs'];b=n//8;added=(new-n)//8
    if cfg!=revised_config(old,new//n):raise RuntimeError('Configuration differs outside the authorized scale, budget, LR and EMA changes')
    if saved['iteration']!=at or np.shape(saved['key'])!=(8,2):raise RuntimeError('Eight-rank checkpoint identity differs')
    if new<=n or n%8 or new%8:raise RuntimeError('Invalid eight-rank environment scale')
    def grow(original,extra):
        original=np.asarray(original);extra=np.asarray(extra)
        if original.shape[0]!=n or extra.shape!=(new-n,)+original.shape[1:] or extra.dtype!=original.dtype:
            raise RuntimeError('Environment expansion shape or dtype differs')
        result=np.concatenate((original.reshape((8,b)+original.shape[1:]),
                               extra.reshape((8,added)+extra.shape[1:])),axis=1).reshape((new,)+original.shape[1:])
        if not np.array_equal(result.reshape((8,b+added)+original.shape[1:])[:,:b],
                              original.reshape((8,b)+original.shape[1:]),equal_nan=True):
            raise RuntimeError('Existing rank environment state changed')
        return result
    if saved['env'].keys()!=fresh.keys():raise RuntimeError('Environment schema differs')
    result=copy.deepcopy(saved);result['config']=copy.deepcopy(cfg)
    result['env']={k:grow(v,fresh[k]) for k,v in saved['env'].items()}
    result['runtime']['arena']={k:grow(v,np.zeros((new-n,)+np.shape(v)[1:],dtype=np.asarray(v).dtype))
                                for k,v in saved['runtime']['arena'].items()}
    result['runtime']['distributed']['per_gpu_envs']=cfg['per_gpu_envs']
    result['runtime'].setdefault('config_migrations',[]).append({
        'iteration':at,'reason':'explicit user instruction: compare equal 4x rollout budgets',
        'old_envs':n,'new_envs':new,'old_horizon':old['horizon'],'new_horizon':cfg['horizon'],
        'new_lr':1e-5,'new_updates':cfg['updates'],'time':time.time()})
    if 'ema' not in result:
        result['ema']={'params':copy.deepcopy(saved['train']['params']),'decay':0.999,'updates':0}
    if result['ema']['decay']!=cfg['ema_decay']:raise RuntimeError('EMA decay changed')
    for name in ('train','key','iteration'):
        if not equal(saved[name],result[name]):raise RuntimeError('Inherited state changed: '+name)
    for name in ('vf_coef','stable_updates','source_sha256'):
        if saved['runtime'][name]!=result['runtime'][name]:raise RuntimeError('Inherited runtime changed: '+name)
    return result


def fresh_environments(saved,cfg,at):
    import jax
    from . import env_v4 as env
    added=(cfg['envs']-saved['config']['envs'])//8
    keys=[]
    for rank,key in enumerate(saved['key']):
        independent=jax.random.fold_in(jax.random.fold_in(key,0x524F4C4C),at)
        keys.append(jax.random.split(jax.random.fold_in(independent,rank),added))
    state=env.batch_reset(np.asarray(keys).reshape((-1,2)))
    return {k:np.asarray(v) for k,v in serialization.to_state_dict(state).items()}


def summarize(rows,expected):
    if len(rows)<8:raise RuntimeError('Eight fresh updates required')
    for row in rows:
        if (row['nranks']!=8 or row['fresh_decisions']!=expected or row['invalid_actions'] or
            row['nonfinite'] or row.get('replica_parameter_max_difference',0)!=0 or
            row.get('ema_replica_parameter_max_difference',0)!=0 or
            abs(row['learning_rate']-1e-5)>1e-11 or row['applied']<0.99):
            raise RuntimeError('Scale, LR, health, optimizer acceptance or replica gate failed')
    # Exclude the first encounter of each training history signature as well
    # as the initial three updates, so a later long-history compile is untimed.
    first={}
    for i,row in enumerate(rows):first.setdefault(row['train_memory_length'],i)
    warm=[r for i,r in enumerate(rows) if i>=3 and first[r['train_memory_length']]!=i]
    if len(warm)<2:raise RuntimeError('Insufficient warmed updates after compilation')
    elapsed=sum(r['seconds'] for r in warm)
    return {'passed':True,'updates':len(rows),'timed_updates':len(warm),
        'decisions_per_second':sum(r['fresh_decisions'] for r in warm)/elapsed,
        'eligible_decisions_per_second':sum(r['learner_decisions']*r['actor_eligible_fraction'] for r in warm)/elapsed,
        'rollout_seconds_mean':float(np.mean([r['rollout_seconds'] for r in warm])),
        'train_seconds_mean':float(np.mean([r['update_seconds'] for r in warm])),
        'actor_eligible_fraction_mean':float(np.mean([r['actor_eligible_fraction'] for r in warm])),
        'value_explained_variance_mean':float(np.mean([r['value_explained_variance'] for r in warm])),
        'kl_max':max(r['kl'] for r in rows),'clip_fraction_mean':float(np.mean([r['clip_fraction'] for r in warm])),
        'gpu_peak_memory_bytes':max(r.get('gpu_peak_memory_bytes',0) for r in rows),
        'nranks':8,'invalid_actions':0,'nonfinite':0,'replica_parameter_max_difference':0}


def choose(results):
    valid=[r for r in results if r.get('passed')]
    if not valid:raise RuntimeError('No rollout arm passed; inspect retained logs before a new task')
    # With less than 3% measured throughput difference prefer the longer trace.
    best=max(valid,key=lambda r:r['eligible_decisions_per_second'])
    longer=[r for r in valid if r['factor']==2]
    if longer and longer[0]['eligible_decisions_per_second']>=best['eligible_decisions_per_second']/1.03:
        best=longer[0]
    return best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();campaign=args.root.resolve();request=read(campaign/'REQUEST.json')
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Persistent eight-GPU queue required')
    for path,expected in request['files_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Frozen campaign input changed: '+path)
    old=Path(request['old_root']);old_run=old/'production'/'training';at=request['at_iteration']
    if request.get('source_pause_receipt'):
        pause=read(request['source_pause_receipt']);previous_proof=read(pause['last_completed_proof'])
        if pause['retained_checkpoint_iteration']!=at or read(Path(request['queue'])/'interrupted'/(pause['interrupted_queue_id']+'.json'))['status']!='interrupted':
            raise RuntimeError('Reviewed emergency pause identity differs')
        pid=pause['remote_training_pid'];process=Path(f'/proc/{pid}/cmdline')
        if process.exists() and b'ddz.train_distributed_v5' in process.read_bytes():
            raise RuntimeError('Interrupted source trainer remains alive; inspect before recovery')
    else:
        status=read(old_run/'status.json')
        if status['state']!='completed' or status['iteration']!=at or status['nranks']!=8:
            raise RuntimeError('Previous eight-GPU segment did not complete at the requested boundary')
        previous_proof=read(old/'production'/'queue_job_status.json')
    if not previous_proof.get('passed') or previous_proof.get('nranks')!=8 or not previous_proof.get('NCCL_evidence'):
        raise RuntimeError('Source checkpoint lacks accepted eight-rank NCCL proof')
    # Cancel only the old DDZ follow-up while this FIFO item owns the queue.
    cancelled=cancel_old_followups(Path(request['queue']),old)
    if not cancelled and not request.get('source_pause_receipt'):
        raise RuntimeError('Expected old DDZ continuation missing; inspect queue state')
    for item in cancelled:
        receipt=Path(request['queue'])/'interrupted'/(item+'.json');record=read(receipt)
        record['reason']='USER_CHANGED_DDZ_ROLLOUT_BUDGET_LR_AND_ADDED_EMA';write(receipt,record)
    write(campaign/'status.json',{'state':'migrating','cancelled_old_followups':cancelled,'time':time.time()})
    raw=Path(request.get('source_full_checkpoint',old_run/'latest.msgpack')).read_bytes();saved=serialization.msgpack_restore(raw)
    results=[]
    for arm in request['arms']:
        root=Path(arm['root']);phase=root/'production';run=phase/'training';cfg=read(phase/'config.json')
        if run.exists():raise RuntimeError('Trial already exists; never retry silently')
        if sha(phase/'source.msgpack')!=saved['runtime']['source_sha256']:raise RuntimeError('Source weight identity differs')
        fresh=fresh_environments(saved,cfg,at);migrated=expand_checkpoint(saved,cfg,fresh,at)
        payload=serialization.msgpack_serialize(migrated);restored=serialization.msgpack_restore(payload)
        if not equal(migrated,restored):raise RuntimeError('Checkpoint serialization changed migrated state')
        run.mkdir();(run/'latest.msgpack').write_bytes(payload);write(run/'config.json',cfg)
        # An interrupted segment may contain logged but unsaved updates.
        # Preserve its original log, copy only history belonging to this state.
        with (run/'metrics.jsonl').open('w') as stream:
            for line in (old_run/'metrics.jsonl').read_text().splitlines():
                if json.loads(line)['iteration']<=at:stream.write(line+'\n')
        for snapshot in old_run.glob('policy_*.msgpack'):(run/snapshot.name).symlink_to(snapshot)
        write(run/'latest.json',{'iteration':at,'policy':f'policy_{at:07d}.msgpack','time':time.time()})
        write(root/'migration_receipt.json',{'at_iteration':at,'source_checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
            'migrated_checkpoint_sha256':sha(run/'latest.msgpack'),'old_envs':saved['config']['envs'],
            'new_envs':cfg['envs'],'horizon':cfg['horizon'],'existing_envs_weights_adam_rng_bit_exact':True,
            'ema_initialized_from_latest_policy':True,'time':time.time()})
        del fresh,migrated,payload,restored
        write(campaign/'status.json',{'state':'benchmarking','factor':arm['factor'],'run':str(run),'time':time.time()})
        log=root/'benchmark.log';env={**os.environ,'JAX_PLATFORMS':'cuda','NCCL_DEBUG':'INFO',
            'JAX_COMPILATION_CACHE_DIR':str(phase/'jax_cache'),
            'XLA_PYTHON_CLIENT_MEM_FRACTION':'.80','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'8','PYTHONPATH':str(root/'code')}
        argv=[sys.executable,'-u','-m','ddz.train_distributed_v5','--config',str(phase/'config.json'),
              '--source',str(phase/'source.msgpack'),'--out',str(run),'--steps',str(at+8),'--resume']
        with log.open('w') as stream:
            process=subprocess.Popen(argv,cwd=root/'code',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
        result={'factor':arm['factor'],'root':str(root),'exit_code':code,'log':str(log)}
        try:
            if code:raise RuntimeError('Trial process failed; explicit new task needed for any retry')
            content=log.read_text();nccl=[x for x in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b',x)]
            if not nccl or 'nranks=8;' not in content:raise RuntimeError('Real eight-rank NCCL evidence missing')
            rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            rows=[r for r in rows if r['iteration']>at]
            result.update(summarize(rows,cfg['envs']*cfg['horizon']),NCCL_evidence=nccl[:8])
            final=serialization.msgpack_restore((run/'latest.msgpack').read_bytes())
            if final['ema']['updates']!=saved.get('ema',{}).get('updates',0)+8 or final['ema']['decay']!=0.999:
                raise RuntimeError('EMA checkpoint continuity gate failed')
            write(root/'engineering_READY.json',result)
        except Exception as error:
            result.update(passed=False,error=str(error));write(root/'FAILED.json',result)
        write(root/'benchmark_result.json',result);results.append(result)
    winner=choose(results);selected=Path(winner['root'])
    selection={'selected_root':str(selected),'results':results,'selection':'highest warmed eligible decision throughput; prefer horizon128 within 3%',
        'scope':'engineering throughput and PPO health, not a claim of improved playing strength',
        'at_iteration':at,'target_global_updates':200000,'ema_decay':0.999,'time':time.time()}
    write(campaign/'selection.json',selection)
    write(selected/'policy_publication.json',{'run_dir':str(selected/'production'/'training'),
        'destination':str(old_run),'selection':str(campaign/'selection.json'),'selection_sha256':sha(campaign/'selection.json')})
    write(campaign/'status.json',{'state':'selected_and_continuing','selected_root':str(selected),'time':time.time()})
    write(old.parent/'production_current.json',{'state':'running','run':str(selected/'production'/'training'),
        'nranks':8,'global_iteration':saved['config']['global_source_iteration']+at+8,'selection':str(campaign/'selection.json'),'time':time.time()})
    cfg=read(selected/'production'/'config.json')
    target=min(at+1000,cfg['continuation_updates'])
    code=subprocess.run([sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(selected),
         '--phase','production','--steps',str(target),'--resume'],cwd=selected/'code',
         env={**os.environ,'PYTHONPATH':str(selected/'code')}).returncode
    if code:raise RuntimeError('Selected continuation failed; inspect logs before an explicit new queue task')


if __name__=='__main__':main()
