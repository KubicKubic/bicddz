"""Persistent complete-move PPO training with atomic checkpoints."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.35')
import argparse
import copy
import json
import signal
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import optax
from flax import serialization
from flax.training.train_state import TrainState
from . import env_v2 as env
from .model_v2 import FullAttentionMoveTransformer
from .ppo_v2 import make_rollout,make_update,make_gradient_diagnostic


def learning_rate_schedule(cfg, runtime=None):
    """Keep learning-rate progress continuous when a resumed run changes batch size."""
    p=cfg['ppo']; steps=cfg['envs']*cfg['horizon']//p['minibatch']*p['epochs']
    base=optax.warmup_cosine_decay_schedule(p['lr']*.1,p['lr'],max(1,p['warmup']*steps),
                                           max(p['warmup']+1,cfg['updates'])*steps,p['lr_min'])
    if not runtime or 'schedule_real_anchor_step' not in runtime:
        return base
    real=runtime['schedule_real_anchor_step']
    virtual=runtime['schedule_virtual_anchor_step']
    scale=runtime['schedule_step_scale']
    return lambda count: base(virtual+(count-real)*scale)


def create(cfg, runtime=None):
    model=FullAttentionMoveTransformer(**cfg['model'])
    obs=jax.vmap(env.observe)(jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(cfg['seed']),1)))
    params=model.init(jax.random.PRNGKey(cfg['seed']+1),obs)['params']
    p=cfg['ppo']; lr=learning_rate_schedule(cfg,runtime)
    tx=optax.chain(optax.clip_by_global_norm(p['max_grad_norm']),optax.adamw(lr,weight_decay=p['weight_decay']))
    return model,TrainState.create(apply_fn=model.apply,params=params,tx=tx)


def warm_start_v1(params,path):
    """Copy matching v1 weights; initialize new event attention/wing heads."""
    from flax.core import FrozenDict,freeze,unfreeze
    from flax.traverse_util import flatten_dict,unflatten_dict
    old=flatten_dict(serialization.msgpack_restore(Path(path).read_bytes()))
    new=flatten_dict(unfreeze(params))
    copied=[]
    for path_key,weight in new.items():
        if path_key not in old:
            continue
        source=np.asarray(old[path_key]); target=np.asarray(weight).copy()
        name='.'.join(path_key)
        if source.shape==target.shape:
            new[path_key]=j.asarray(source); copied.append(name)
        elif path_key==('actor','kernel') and source.shape[:-1]==target.shape[:-1]:
            new[path_key]=j.asarray(source[..., :target.shape[-1]])
            copied.append(f'{name}[:{target.shape[-1]}]')
        elif path_key==('actor','bias') and source.shape[0]>=target.shape[0]:
            new[path_key]=j.asarray(source[:target.shape[0]])
            copied.append(f'{name}[:{target.shape[0]}]')
        elif path_key==('context_in','kernel') and source.shape[0]+4==target.shape[0]:
            target[:7]=source[:7]; target[11:]=source[7:]
            new[path_key]=j.asarray(target); copied.append('context_in.kernel[role insertion]')
        elif path_key==('event_16','embedding') and source.shape[0]+1==target.shape[0]:
            target[:3]=source; new[path_key]=j.asarray(target)
            copied.append('event_16.embedding[:3]')
    restored=unflatten_dict(new)
    return (freeze(restored) if isinstance(params,FrozenDict) else restored),copied


def atomic(path,data):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def checkpoint(out,ts,states,key,iteration,cfg,runtime,ema=None):
    # Single rolling full state + immutable lightweight periodic model snapshots.
    payload={'train':serialization.to_state_dict(ts),'env':serialization.to_state_dict(states),
             'key':np.asarray(key),'iteration':iteration,'config':cfg,'runtime':runtime}
    if ema is not None:payload['ema']=ema
    atomic(out/'latest.msgpack',serialization.msgpack_serialize(jax.device_get(payload)))
    atomic(out/f'policy_{iteration:07d}.msgpack',serialization.to_bytes(ts.params))
    if ema is not None:
        (out/'ema').mkdir(exist_ok=True)
        atomic(out/'ema'/f'policy_{iteration:07d}.msgpack',serialization.msgpack_serialize(jax.device_get(ema['params'])))
    atomic(out/'latest.json',json.dumps({'iteration':iteration,'time':time.time(),
        'policy':f'policy_{iteration:07d}.msgpack'}).encode())


def resume_config_migration(saved, current):
    """Allow explicit, recorded single-field cadence or GAE-lambda changes."""
    if saved == current:
        return None
    compatible = copy.deepcopy(saved)
    changes = []
    for field, parent, key in (('save_every', compatible, 'save_every'),
                               ('ppo.lambda', compatible['ppo'], 'lambda')):
        new = current['ppo'][key] if field == 'ppo.lambda' else current[key]
        if parent[key] != new:
            changes.append({'field': field, 'from': parent[key], 'to': new})
            parent[key] = new
    if compatible != current:
        raise ValueError('resume config differs outside save_every or ppo.lambda')
    if len(changes) != 1:
        raise ValueError('change one resume config field at a time')
    return changes[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/a100.json')
    ap.add_argument('--out',required=True); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--minibatch',type=int,help='PPO minibatch size; retained in runtime on resume')
    ap.add_argument('--train-memory-limit',type=int,help='skip masked history padding in PPO minibatches')
    ap.add_argument('--require-a100',action='store_true'); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--init-from-v1',help='v1 policy checkpoint for compatible weight transfer')
    args=ap.parse_args(); cfg=json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(envs=8,horizon=16,updates=2,save_every=1,eval_every=0)
        cfg['ppo'].update(minibatch=32,epochs=1,warmup=1)
    if not args.resume and args.minibatch is not None:
        cfg['ppo']['minibatch']=args.minibatch
    if cfg['save_every']<=0:
        raise ValueError('save_every must be positive')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Exclusive process lock prevents two writers/resumes from sharing a run.
    import fcntl
    lock=(out/'train.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:
        raise ValueError('existing run: explicitly use --resume')
    if args.resume and args.init_from_v1:
        raise ValueError('warm start is only valid for a new run')
    obj=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    migration=resume_config_migration(obj['config'],cfg) if obj is not None else None
    runtime=dict(obj['runtime']) if obj is not None else {'vf_coef':cfg['ppo']['value_coef'],'stable_updates':0}
    if migration is not None:
        migration['at_iteration']=int(obj['iteration'])
        runtime.setdefault('config_migrations',[]).append(migration)
    base_mb=cfg['ppo']['minibatch']
    previous_mb=runtime.get('train_minibatch',base_mb)
    mb=args.minibatch if args.minibatch is not None else previous_mb
    if mb<=0 or cfg['envs']*cfg['horizon']%mb:
        raise ValueError('rollout size must divide minibatch exactly')
    if mb!=previous_mb:
        if obj is None:
            raise ValueError('unexpected minibatch transition without a checkpoint')
        real=int(obj['train']['step'])
        virtual=runtime.get('schedule_virtual_anchor_step',real)
        if 'schedule_real_anchor_step' in runtime:
            virtual+=((real-runtime['schedule_real_anchor_step'])*runtime['schedule_step_scale'])
        runtime.update(schedule_real_anchor_step=real,schedule_virtual_anchor_step=float(virtual),
                       schedule_step_scale=mb/base_mb)
    runtime['train_minibatch']=mb
    memory_limit=args.train_memory_limit if args.train_memory_limit is not None else runtime.get('train_memory_limit',env.HISTORY)
    if not 1<=memory_limit<=env.HISTORY:
        raise ValueError('train memory limit must be within the history capacity')
    runtime['train_memory_limit']=memory_limit
    update_ppo=dict(cfg['ppo'],minibatch=mb)
    jax.config.update('jax_compilation_cache_dir',str(out.resolve()/'jax_cache'))
    devices=jax.devices()
    if args.require_a100 and (len(devices)!=1 or 'A100' not in devices[0].device_kind):
        raise RuntimeError(f'requires one A100; found {devices}')
    print(json.dumps({'devices':[str(d) for d in devices],'device_kind':devices[0].device_kind}),flush=True)
    model,ts=create(cfg,runtime)
    if args.init_from_v1:
        params,copied=warm_start_v1(ts.params,args.init_from_v1)
        ts=ts.replace(params=params)
        runtime['warm_start']={'path':str(Path(args.init_from_v1).resolve()),'copied':copied}
        print(json.dumps({'warm_start_layers':copied}),flush=True)
    params=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    print(json.dumps({'parameters':params,'batch_decisions':cfg['envs']*cfg['horizon'],
                      'reward':'raw terminal deltas only','config':cfg}),flush=True)
    if params<1_500_000:
        raise ValueError(f'unexpectedly small model: {params}')
    key=jax.random.PRNGKey(cfg['seed']+2); key,k=jax.random.split(key)
    states=env.batch_reset(jax.random.split(k,cfg['envs'])); start=0
    if obj is not None:
        ts=serialization.from_state_dict(ts,obj['train'])
        states=serialization.from_state_dict(states,obj['env']); key=j.array(obj['key']); start=int(obj['iteration'])
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode())
    atomic(out/'runtime_config.json',json.dumps({'save_every':cfg['save_every'],'gae_lambda':cfg['ppo']['lambda'],'train_minibatch':mb,'train_memory_limit':memory_limit,'schedule':{
        k:runtime[k] for k in ('schedule_real_anchor_step','schedule_virtual_anchor_step','schedule_step_scale')
        if k in runtime}},indent=2).encode())
    atomic(out/'pid',str(os.getpid()).encode())
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True
        print('Graceful stop requested; saving after the current update.',flush=True)
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    rollout=make_rollout(model,cfg['horizon'],memory_limit)
    update=make_update(model,update_ppo,memory_limit)
    diagnose=make_gradient_diagnostic(model,cfg['ppo'])
    if start==0: checkpoint(out,ts,states,key,0,cfg,runtime)
    metrics=(out/'metrics.jsonl').open('a',buffering=1)
    for it in range(start+1,cfg['updates']+1):
        begin=time.monotonic()
        states,key,tr,last,stats=rollout(ts.params,states,key)
        jax.block_until_ready(stats); rollout_secs=time.monotonic()-begin
        key,uk=jax.random.split(key)
        p=cfg['ppo']; frac=min(1,it/p['entropy_updates'])
        entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*frac
        ts,m=update(ts,tr,last,uk,j.float32(entropy),j.float32(runtime['vf_coef'])); jax.block_until_ready(m)
        elapsed=time.monotonic()-begin; stats=np.asarray(stats)
        row={k:float(v) for k,v in m.items()}
        row['vf_coef']=runtime['vf_coef']
        row['train_minibatch']=mb
        row['train_memory_limit']=memory_limit
        row['save_every']=cfg['save_every']
        row['gae_lambda']=p['lambda']
        stable=row['nonfinite']==0 and row['kl']<2*p['target_kl'] and row['applied']>0.1
        runtime['stable_updates']=runtime['stable_updates']+1 if stable else 0
        if it%p.get('vf_balance_every',10)==0:
            key,dk=jax.random.split(key)
            norms={k:float(v) for k,v in diagnose(ts.params,tr,last,dk).items()}
            row.update(norms)
            row['weighted_value_grad_l2']=runtime['vf_coef']*norms['value_grad_l2']
            row['weighted_value_shared_grad_l2']=runtime['vf_coef']*norms['value_shared_grad_l2']
            old=runtime['vf_coef']
            if it>=p.get('vf_balance_after',20) and runtime['stable_updates']>=5:
                ratio=norms['policy_grad_l2']/max(norms['value_grad_l2'],1e-12)
                if np.isfinite(ratio) and norms['policy_grad_l2']>1e-9:
                    # Geometric smoothing with at most 2x change per diagnostic.
                    target=np.clip(ratio,p.get('vf_coef_min',.001),p.get('vf_coef_max',10.))
                    runtime['vf_coef']=float(np.clip(np.sqrt(old*target),old/2,old*2))
            row['vf_coef_next']=runtime['vf_coef']
            row['balanced_gradient_ratio_next']=runtime['vf_coef']*norms['value_grad_l2']/max(norms['policy_grad_l2'],1e-12)
        games=int(stats[:,0].sum())
        row.update(iteration=it,time=time.time(),seconds=elapsed,rollout_seconds=rollout_secs,
            update_seconds=elapsed-rollout_secs,
            decisions_per_second=cfg['envs']*cfg['horizon']/elapsed,games=games,
            invalid_actions=int(stats[:,1].sum()),landlord_wins=int(stats[:,2].sum()),
            mean_abs_landlord_score=float(stats[:,3].sum()/max(games,1)),
            history_max=int(stats[:,4].max()),entropy_coef=entropy)
        row['seconds']=time.monotonic()-begin
        row['decisions_per_second']=cfg['envs']*cfg['horizon']/row['seconds']
        memory=devices[0].memory_stats() or {}
        row['gpu_peak_bytes']=int(memory.get('peak_bytes_in_use',0))
        row['gpu_bytes_in_use']=int(memory.get('bytes_in_use',0))
        metrics.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
        if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('training invariant failed; inspect metrics')
        if cfg['eval_every'] and it%cfg['eval_every']==0:
            from .evaluate_v2 import evaluate
            result=evaluate(model,ts.params,cfg['eval_games'],cfg['seed']+it)
            with (out/'eval.jsonl').open('a') as f: f.write(json.dumps(dict(iteration=it,**result))+'\n')
            print(json.dumps({'evaluation':result}),flush=True)
        if it%cfg['save_every']==0 or stop[0] or it==cfg['updates']:
            checkpoint(out,ts,states,key,it,cfg,runtime)
        atomic(out/'status.json',json.dumps({'iteration':it,'pid':os.getpid(),'time':time.time(),
               'train_minibatch':mb,'train_memory_limit':memory_limit,'save_every':cfg['save_every'],
               'gae_lambda':p['lambda'],
               'state':'stopped' if stop[0] else 'completed' if it==cfg['updates'] else 'running','parameters':params}).encode())
        if stop[0]: break
    metrics.close()

if __name__=='__main__': main()
