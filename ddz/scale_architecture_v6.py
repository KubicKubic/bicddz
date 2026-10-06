"""Queue-boundary V5 -> V6 migration, eight-rank health gate and continuation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from flax import serialization
from .scale_rollout_campaign import read, write, sha, summarize
from .switch_distributed_gae import cancel_old_followups, equal
from .upgrade_v6 import revised_config, grow_parameters, grow_births, inherited_births


def migrate_checkpoint(saved, cfg, train, source_hash, source_path):
    if cfg != revised_config(saved['config']):
        raise ValueError('Migration differs from the authorized architecture-only preset')
    if np.shape(saved['key'])!=(8,2) or 'ema' not in saved:
        raise ValueError('Complete eight-rank checkpoint with EMA required')
    result={**saved,'config':copy.deepcopy(cfg),'train':train,'runtime':copy.deepcopy(saved['runtime'])}
    result['ema']={**saved['ema'],'params':grow_parameters(train['params'],saved['ema']['params'],
        saved['config']['model'],cfg['model'])}
    result['runtime']['coordinate_births']=grow_births(train['params'],saved['train']['params'],
        inherited_births(saved['train']['params'],saved['runtime']),int(saved['train']['step']))
    result['runtime'].update(source_sha256=source_hash,source_checkpoint=str(source_path),
        source_iteration=saved['iteration'],source_adam_step=int(saved['train']['step']))
    result['runtime'].setdefault('config_migrations',[]).append({
        'iteration':saved['iteration'],'reason':'user requested attention depth and width scaling to approximately 8M',
        'from_family':'V5','to_family':'V6','old_model':saved['config']['model'],
        'new_model':cfg['model'],'time':time.time()})
    if (result['train']['step']!=saved['train']['step'] or
        result['ema']['updates']!=saved['ema']['updates'] or
        result['ema']['decay']!=saved['ema']['decay']):
        raise ValueError('Training or rollout EMA clock changed')
    from flax.traverse_util import flatten_dict
    flat_params=flatten_dict(train['params'])
    for key,value in flatten_dict(saved['train']['params']).items():
        target=flat_params[key]
        if not np.array_equal(np.asarray(target)[tuple(slice(0,n) for n in np.shape(value))],value):
            raise ValueError('Inherited weight coordinates changed')
    def check_optimizer(old,new):
        if isinstance(old,dict):
            for key,value in old.items():check_optimizer(value,new[key])
        else:
            old=np.asarray(old);new=np.asarray(new)
            if not np.array_equal(new[tuple(slice(0,n) for n in old.shape)],old):
                raise ValueError('Inherited Adam state changed')
    check_optimizer(saved['train']['opt_state'],train['opt_state'])
    for key in ('env','key','iteration'):
        if not equal(saved[key],result[key]):raise ValueError('Inherited state changed: '+key)
    for key in ('arena','vf_coef','stable_updates'):
        if not equal(saved['runtime'][key],result['runtime'][key]):
            raise ValueError('Inherited runtime changed: '+key)
    return result


def function_probe(saved,migrated,precision_modes=('fp32','bf16')):
    """Compare complete-action distributions and values on resident public states."""
    import jax
    import jax.numpy as j
    from . import env_v4 as env, policy_v5 as policy
    from .model_efficiency import EfficientMoveTransformer
    from .env_v2 import State
    count=len(saved['env']['turn'])
    indices=np.linspace(0,count-1,min(128,count),dtype=np.int32)
    states=State(**{k:j.asarray(np.asarray(v)[indices]) for k,v in saved['env'].items()})
    observations=jax.vmap(env.observe)(states)
    choose=jax.jit(jax.vmap(policy.greedy_one))
    result={}
    for kind,p,q in [('raw',saved['train']['params'],migrated['train']['params']),
                     ('ema',saved['ema']['params'],migrated['ema']['params'])]:
        # Match the trainer/serving signatures: weights are dynamic arguments,
        # never compile-time constants that can fold the new zero paths away.
        p=jax.tree_util.tree_map(j.asarray,p);q=jax.tree_util.tree_map(j.asarray,q)
        result[kind]={}
        for precision in precision_modes:
            a=EfficientMoveTransformer(**{**saved['config']['model'],'bf16':precision=='bf16'})
            b=EfficientMoveTransformer(**{**migrated['config']['model'],'bf16':precision=='bf16'})
            old=jax.jit(lambda params,obs:a.apply({'params':params},obs))(p,observations)
            new=jax.jit(lambda params,obs:b.apply({'params':params},obs))(q,observations)
            lp,lq=jax.nn.log_softmax(old[0]),jax.nn.log_softmax(new[0])
            kl=np.asarray(j.sum(j.exp(lp)*(lp-lq),axis=-1))
            old_actions=choose(states,old[0],old[1]);new_actions=choose(states,new[0],new[1])
            scorer=jax.jit(jax.vmap(policy.score_one))
            old_lp,_=scorer(states,old[0],old[1],old_actions)
            new_lp,_=scorer(states,new[0],new[1],old_actions)
            value_error=float(np.max(np.abs(np.asarray(old[2]-new[2]))))
            flips=int(np.sum(np.asarray(old_actions)!=np.asarray(new_actions)))
            lp_error=float(np.max(np.abs(np.asarray(old_lp-new_lp))))
            measured={'states':len(indices),'body_kl_max':float(kl.max()),'body_kl_mean':float(kl.mean()),
                'complete_action_flips':flips,'selected_complete_logprob_max_error':lp_error,
                'value_max_error':value_error}
            result[kind][precision]=measured
            print(json.dumps({'migration_probe':kind,'precision':precision,**measured}),flush=True)
            if precision=='fp32':
                passed=not flips and float(kl.max())<2e-5 and value_error<.0002 and lp_error<.0002
            else:
                # CPU BF16 fusion/reduction differs between graph sizes even
                # with every inherited tensor unchanged. Report this separately;
                # the FP32 check verifies the mathematical warm-start function.
                # PPO uses mean KL. Require migration mean KL below one tenth
                # of the existing .03 PPO target, and <=2% greedy changes.
                passed=(flips/len(indices)<=.02 and float(kl.mean())<.003
                        and value_error<1. and lp_error<.2)
            if not passed:raise RuntimeError(f'{kind} {precision} migration function gate failed: {measured}')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();request=read(root/'REQUEST.json')
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Persistent eight-GPU queue required')
    # The migration parent must not retain any CUDA allocator while its child trains.
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('Migration parent requires CPU JAX')
    for name,expected in request['files_sha256'].items():
        if sha(name)!=expected:raise RuntimeError('Frozen dependency changed: '+name)
    old=Path(request['old_root']);old_run=old/'production/training'
    status=read(old_run/'status.json');proof=read(old/'production/queue_job_status.json')
    at=request['at_iteration']
    if (status['state']!='completed' or status['iteration']!=at or status['nranks']!=8 or
        not proof.get('passed') or proof.get('nranks')!=8 or not proof.get('NCCL_evidence')):
        raise RuntimeError('Expected source queue boundary lacks accepted eight-rank NCCL proof')
    phase=root/'production';run=phase/'training'
    if run.exists():raise RuntimeError('Migration already exists; explicit review required before any retry')
    query=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    capacities=[];free=[]
    for line in query.stdout.splitlines():
        parts=[p.strip() for p in line.split(',')]
        capacities.append(int(parts[2])*2**20);free.append((int(parts[2])-int(parts[3]))*2**20)
    estimate=request['resource_gate']['estimated_peak_bytes']
    if len(capacities)!=8 or min(free)<estimate or min(capacities)*.8<estimate:
        raise RuntimeError('Eight-GPU capacity/free-memory gate failed')
    write(root/'resource_verified.json',{'passed':True,'rows':query.stdout.splitlines(),
        'estimated_peak_bytes':estimate,'time':time.time()})
    raw=(old_run/'latest.msgpack').read_bytes();saved=serialization.msgpack_restore(raw)
    if saved['iteration']!=at:raise RuntimeError('Full source checkpoint boundary differs')
    cfg=read(phase/'config.json')
    if cfg!=revised_config(saved['config']):raise RuntimeError('Frozen target config differs')
    from .train_efficiency import create
    model,ts=create(cfg,saved)
    train=serialization.to_state_dict(ts)
    migrated=migrate_checkpoint(saved,cfg,train,hashlib.sha256(raw).hexdigest(),phase/'source.msgpack')
    probe=function_probe(saved,migrated)
    params=sum(x.size for x in __import__('jax').tree_util.tree_leaves(train['params']))
    if params!=8_014_192:raise RuntimeError('Audited 8M parameter budget differs')
    # Keep the old production follow-up until every pre-training migration gate passes.
    source_hash=hashlib.sha256(raw).hexdigest()
    (phase/'source.msgpack').write_bytes(raw)
    run.mkdir();(run/'ema').mkdir()
    (run/'latest.msgpack').write_bytes(serialization.msgpack_serialize(migrated))
    (run/f'policy_{at:07d}.msgpack').write_bytes(serialization.msgpack_serialize(train['params']))
    (run/'ema'/f'policy_{at:07d}.msgpack').write_bytes(serialization.msgpack_serialize(migrated['ema']['params']))
    write(run/'latest.json',{'iteration':at,'policy':f'policy_{at:07d}.msgpack','time':time.time()})
    write(run/'config.json',cfg)
    with (run/'metrics.jsonl').open('w') as stream:
        for line in (old_run/'metrics.jsonl').read_text().splitlines():
            if json.loads(line)['iteration']<=at:stream.write(line+'\n')
    write(root/'migration_receipt.json',{'iteration':at,'global_iteration':cfg['global_source_iteration']+at,
        'source_checkpoint_sha256':source_hash,'migrated_checkpoint_sha256':sha(run/'latest.msgpack'),
        'parameters':params,'function_probe':probe,'env_rng_arena_clocks_retained':True,
        'inherited_weights_and_adam_coordinates_bit_exact':True,'ema_updates_preserved':saved['ema']['updates'],
        'new_coordinate_adam_age_starts_at':int(saved['train']['step']),'time':time.time()})
    write(root/'status.json',{'state':'verifying_eight_gpu_v6','parameters':params,'time':time.time()})
    ready={'cpu_verification':{'passed':True},'files_sha256':{**request['files_sha256'],
        str(phase/'source.msgpack'):source_hash}}
    write(phase/'READY.json',ready)
    log=root/'engineering.log'
    env={**os.environ,'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':','.join(map(str,range(8))),
        'NCCL_DEBUG':'INFO','PYTHONPATH':str(root/'code'),'OMP_NUM_THREADS':'8',
        'OPENBLAS_NUM_THREADS':'1','XLA_PYTHON_CLIENT_MEM_FRACTION':'.80'}
    argv=[sys.executable,'-u','-m','ddz.train_distributed_v5','--config',str(phase/'config.json'),
          '--source',str(phase/'source.msgpack'),'--out',str(run),'--steps',str(at+8),'--resume']
    with log.open('w') as stream:
        process=subprocess.Popen(argv,cwd=root/'code',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:raise RuntimeError('V6 verification failed; old continuation retained, explicit review required')
    content=log.read_text();nccl=[line for line in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b',line)]
    if not nccl or 'nranks=8;' not in content:raise RuntimeError('Real eight-rank NCCL evidence missing')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row['iteration']>at]
    result=summarize(rows,cfg['envs']*cfg['horizon'])
    if len(rows)!=8 or rows[-1]['iteration']!=at+8:raise RuntimeError('Eight verification rounds required')
    final=serialization.msgpack_restore((run/'latest.msgpack').read_bytes())
    if final['ema']['updates']!=saved['ema']['updates']+8 or final['ema']['decay']!=.999:
        raise RuntimeError('Rollout-clock EMA continuity failed')
    if result['gpu_peak_memory_bytes']>=min(capacities)*.8:raise RuntimeError('Measured peak exceeds allocator budget')
    result.update(parameters=params,NCCL_evidence=nccl[:8],log=str(log),global_envs=cfg['envs'],
                  global_minibatch=cfg['ppo']['minibatch'],time=time.time())
    # Cancel only the old DDZ successor while this FIFO item owns the worker.
    cancelled=cancel_old_followups(Path(request['queue']),old)
    if len(cancelled)!=1:raise RuntimeError('Old continuation count differs; inspect before promotion')
    for task in cancelled:
        path=Path(request['queue'])/'interrupted'/(task+'.json');record=read(path)
        record['reason']='USER_SCALED_DDZ_ATTENTION_TO_8M';write(path,record)
    write(root/'engineering_READY.json',result)
    write(root/'status.json',{'state':'verified_and_continuing','cancelled_old_followups':cancelled,**result})
    write(root.parent/'production_current.json',{'state':'running','run':str(run),'nranks':8,
        'global_iteration':cfg['global_source_iteration']+at+8,
        'proof':str(root/'engineering_READY.json'),'time':time.time()})
    # Publish to the new architecture's directory only. Old frozen evaluators
    # must not receive incompatible V6 snapshots under their V5 configuration.
    target=min(at+1000,cfg['continuation_updates'])
    code=subprocess.run([sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(root),
        '--phase','production','--steps',str(target),'--resume'],cwd=root/'code',env=env).returncode
    if code:raise RuntimeError('V6 continuation failed; explicit new task required')


if __name__=='__main__':
    try:main()
    except Exception as error:
        # The queue retains the failed item. This file helps local controllers
        # distinguish a reviewed failure from a still-pending handoff.
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            write(root/'FAILED.json',{'error':repr(error),'time':time.time(),'automatic_retry':False})
        raise
