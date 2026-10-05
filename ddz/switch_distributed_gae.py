"""Queue-boundary one-field GAE migration with exact training-state retention."""
import argparse,copy,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from flax import serialization


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.chmod(tmp,0o644);os.replace(tmp,path)


def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return np.array_equal(a,b,equal_nan=True) if isinstance(a,np.ndarray) else a==b


def migrate(saved,new_cfg,source_sha,at):
    expected=copy.deepcopy(saved['config']);expected['ppo']['gae_clock']='own'
    if new_cfg!=expected:raise RuntimeError('Only ppo.gae_clock may change')
    if saved['config']['ppo'].get('gae_clock','public')!='public':raise RuntimeError('Source GAE already changed')
    if saved['iteration']!=at or saved['runtime']['source_sha256']!=source_sha:
        raise RuntimeError('Checkpoint iteration/source identity differs')
    if saved['config']['envs']!=8192 or np.asarray(saved['key']).shape!=(8,2):
        raise RuntimeError('Requires a complete eight-replica checkpoint')
    result=copy.deepcopy(saved);result['config']=copy.deepcopy(new_cfg)
    result['runtime'].setdefault('config_migrations',[]).append({
        'field':'ppo.gae_clock','from':'public','to':'own','iteration':at,
        'reason':'explicit user instruction','time':time.time()})
    result['runtime']['stable_updates']=0
    for key in ('train','env','key','iteration'):
        if not equal(saved[key],result[key]):raise RuntimeError('Training state changed: '+key)
    if saved['runtime']['vf_coef']!=result['runtime']['vf_coef']:raise RuntimeError('Value coefficient changed')
    return result


def old_followup(record,old):
    tokens=shlex.split(record['command'])
    if '-m' not in tokens or '--root' not in tokens or '--phase' not in tokens:return False
    return (tokens[tokens.index('-m')+1]=='ddz.cluster_distributed_job'
            and Path(tokens[tokens.index('--root')+1]).resolve()==old.resolve()
            and tokens[tokens.index('--phase')+1]=='production')


def cancel_old_followups(queue,old):
    cancelled=[]
    for path in sorted((queue/'pending').glob('*.json')):
        record=read(path)
        if not old_followup(record,old):continue
        if record['status']!='pending' or record['command_sha256']!=hashlib.sha256(record['command'].encode()).hexdigest():
            raise RuntimeError('Pending item identity mismatch')
        destination=queue/'interrupted'/path.name
        if destination.exists():raise RuntimeError('Cancellation already recorded')
        record.update(status='interrupted',finished_at_ns=time.time_ns(),exit_code=None,
                      reason='USER_SWITCHED_DDZ_PUBLIC_GAE_TO_OWN_GAE')
        write(destination,record);path.unlink();cancelled.append(record['id'])
    return cancelled


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--new-root',type=Path,required=True)
    args=ap.parse_args();new=args.new_root.resolve();request=read(new/'migration_REQUEST.json')
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Queue worker required')
    for path,expected in request['files_sha256'].items():
        if sha(Path(path))!=expected:raise RuntimeError('Frozen migration input changed: '+path)
    old=Path(request['old_root']);at=request['at_iteration'];phase=new/'production';run=phase/'training'
    if (run/'latest.msgpack').exists():raise RuntimeError('Migration already performed; never replay silently')
    status=read(old/'production'/'training'/'status.json')
    if status['state']!='completed' or status['iteration']!=at or status['nranks']!=8:
        raise RuntimeError('Previous eight-GPU queue segment did not complete normally')
    old_run=old/'production'/'training';old_checkpoint=old_run/'latest.msgpack'
    raw=old_checkpoint.read_bytes();saved=serialization.msgpack_restore(raw)
    cfg=read(phase/'config.json');source_sha=sha(phase/'source.msgpack')
    migrated=migrate(saved,cfg,source_sha,at)
    payload=serialization.msgpack_serialize(migrated)
    restored=serialization.msgpack_restore(payload)
    for key in ('train','env','key','iteration'):
        if not equal(saved[key],restored[key]):raise RuntimeError('Serialization changed state: '+key)
    run.mkdir(parents=True,exist_ok=False)
    temporary=run/'latest.msgpack.tmp'
    with temporary.open('wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,run/'latest.msgpack');write(run/'config.json',cfg)
    shutil.copy2(old_run/'metrics.jsonl',run/'metrics.jsonl')
    for snapshot in sorted(old_run.glob('policy_*.msgpack')):os.symlink(snapshot,run/snapshot.name)
    write(run/'latest.json',{'iteration':at,'policy':f'policy_{at:07d}.msgpack','time':time.time()})
    # Bind the runtime checkpoint separately; preserve the frozen READY manifest.
    write(phase/'migration_state_binding.json',{'iteration':at,
          'migration_checkpoint_sha256':sha(run/'latest.msgpack'),
          'migration_from_checkpoint_sha256':hashlib.sha256(raw).hexdigest()})
    queue=Path(request['queue']);cancelled=cancel_old_followups(queue,old)
    if not cancelled:raise RuntimeError('Expected old production follow-up was not found; inspect queue')
    receipt={'state':'migrated','at_iteration':at,'global_iteration':cfg['global_source_iteration']+at,
             'old_root':str(old),'new_root':str(new),'old_checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
             'new_checkpoint_sha256':sha(run/'latest.msgpack'),'changed_field':'ppo.gae_clock',
             'old_value':'public','new_value':'own','weights_optimizer_env_rng_bit_exact':True,
             'learning_rate_origin_preserved':True,'vf_coef_preserved':True,
             'cancelled_old_followups':cancelled,'time':time.time()}
    write(new/'migration_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    target=min(at+1000,cfg['continuation_updates'])
    argv=[sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(new),
          '--phase','production','--steps',str(target),'--resume']
    code=subprocess.run(argv,cwd=new/'code',env={**os.environ,'PYTHONPATH':str(new/'code')}).returncode
    if code:raise RuntimeError('Own-seat GAE queue task failed; explicit reviewed retry required')


if __name__=='__main__':main()
