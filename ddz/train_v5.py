"""Persistent, device-resident PPO with compact complete actions."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.35')
import argparse
import copy
import hashlib
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
from . import env_v4 as env
from .model_v5 import InteractionMoveTransformer
from .ppo_v5 import make_rollout,make_update,make_gradient_diagnostic
from .train_v2 import atomic,checkpoint,learning_rate_schedule
from .train_v3 import balanced_vf_coefficient
from .upgrade_v5 import warm_start_v4,upgrade_state
from .optimizer_v5 import new_coordinate_mask,age_corrected_adam
from .policy_v5 import validate_random_action_prob
from flax.traverse_util import flatten_dict


def create(cfg,optimizer_origin=None):
    validate_random_action_prob(cfg['ppo'].get('random_action_prob',0.))
    model=InteractionMoveTransformer(**cfg['model'])
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(cfg['seed']),1))
    params=model.init(jax.random.PRNGKey(cfg['seed']+1),jax.vmap(env.observe)(states))['params']
    p=cfg['ppo']
    if optimizer_origin is None:
        adam=optax.adamw(learning_rate_schedule(cfg),weight_decay=p['weight_decay'])
    else:
        mask=new_coordinate_mask(params,optimizer_origin['source_shapes'])
        adam=optax.chain(age_corrected_adam(mask,optimizer_origin['source_step']),
            optax.add_decayed_weights(p['weight_decay']),
            optax.scale_by_learning_rate(learning_rate_schedule(cfg)))
    tx=optax.chain(optax.clip_by_global_norm(p['max_grad_norm']),adam)
    return model,TrainState.create(apply_fn=model.apply,params=params,tx=tx)


def load_v5_fork(path,cfg):
    """Read one coherent checkpoint; allow only the exploration probability to change."""
    path=Path(path)
    raw=path.read_bytes()
    obj=serialization.msgpack_restore(raw)
    required={'train','env','key','iteration','config','runtime'}
    if not isinstance(obj,dict) or not required.issubset(obj):
        raise ValueError('--fork-from-v5 requires a full V5 training checkpoint, not policy weights')
    saved=copy.deepcopy(obj['config']);current=copy.deepcopy(cfg)
    previous=validate_random_action_prob(saved['ppo'].pop('random_action_prob',0.))
    prob=validate_random_action_prob(current['ppo'].pop('random_action_prob',0.))
    if saved!=current:
        raise ValueError('V5 fork may change only ppo.random_action_prob')
    if 'actor_candidate_output' not in obj['train']['params']:
        raise ValueError('--fork-from-v5 requires a full V5 training checkpoint')
    if int(obj['iteration'])>=cfg['updates']:
        raise ValueError('fork checkpoint has already reached configured updates')
    return obj,{'checkpoint':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
        'iteration':int(obj['iteration']),'previous_random_action_prob':previous,
        'random_action_prob':prob,'optimizer_environment_rng':'preserved'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/a100_v5.json')
    ap.add_argument('--out',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--init-from-v4')
    ap.add_argument('--upgrade-from-v4')
    ap.add_argument('--fork-from-v5',help='fork a full V5 checkpoint into a new run; only random_action_prob may change')
    ap.add_argument('--require-a100',action='store_true')
    ap.add_argument('--stop-after',type=int)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    p=cfg['ppo'];n=cfg['envs']*cfg['horizon'];memory=cfg.get('memory_limit',88)
    random_action_prob=validate_random_action_prob(p.get('random_action_prob',0.))
    if n%p['minibatch'] or not 1<=memory<=env.HISTORY:
        raise ValueError('minibatch must divide rollout and memory must fit history capacity')
    if sum(bool(x) for x in (args.init_from_v4,args.upgrade_from_v4,args.fork_from_v5))>1:
        raise ValueError('choose one initialization mode')
    if args.resume and (args.init_from_v4 or args.upgrade_from_v4 or args.fork_from_v5):
        raise ValueError('warm start or fork only applies to new runs')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:raise ValueError('existing run requires --resume')
    obj=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    if obj is not None and obj['config']!=cfg:raise ValueError('resume configuration mismatch')
    fork=None
    if args.fork_from_v5:
        obj,fork=load_v5_fork(args.fork_from_v5,cfg)
    jax.config.update('jax_compilation_cache_dir',str(out.resolve()/'jax_cache'))
    devices=jax.devices()
    if args.require_a100 and (len(devices)!=1 or 'A100' not in devices[0].device_kind):
        raise RuntimeError(f'requires one A100, found {devices}')
    origin=serialization.msgpack_restore(Path(args.upgrade_from_v4).read_bytes()) if args.upgrade_from_v4 else None
    optimizer_origin=obj['runtime'].get('optimizer_origin') if obj is not None else None
    if origin is not None:
        source_policy=Path(args.upgrade_from_v4).parent/f"policy_{origin['iteration']:07d}.msgpack"
        source_params=serialization.msgpack_restore(source_policy.read_bytes())
        optimizer_origin={'source_step':int(origin['train']['step']),
            'source_shapes':{'.'.join(k):list(v.shape) for k,v in flatten_dict(source_params).items()}}
    model,ts=create(cfg,optimizer_origin)
    runtime={'vf_coef':p['value_coef'],'stable_updates':0,'vf_adaptive':True}
    if args.init_from_v4:
        params,copied=warm_start_v4(ts.params,args.init_from_v4);ts=ts.replace(params=params)
        runtime['warm_start']={'checkpoint':str(Path(args.init_from_v4).resolve()),'copied_parameters':copied}
    if args.upgrade_from_v4:
        source_policy=Path(args.upgrade_from_v4).parent/f"policy_{origin['iteration']:07d}.msgpack"
        ts,copied=upgrade_state(ts,origin,source_policy)
        runtime=dict(origin['runtime'])
        runtime['source_iteration']=int(origin['iteration'])
        runtime['optimizer_origin']=optimizer_origin
        runtime['upgrade']={'checkpoint':str(source_policy.resolve()),'copied_parameters':copied,
            'optimizer':'grown Adam moments; old bias counters preserved; new coordinates use their own Adam age','environment':'preserved',
            'source_fresh_decisions':int(origin['iteration'])*origin['config']['envs']*origin['config']['horizon']}
    parameters=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    if parameters>5_500_000:raise ValueError('parameter cap exceeded')
    key=jax.random.PRNGKey(cfg['seed']+2);key,k=jax.random.split(key)
    states=env.batch_reset(jax.random.split(k,cfg['envs']));start=0
    if origin is not None:
        if origin['config']['envs']!=cfg['envs']:raise ValueError('upgrade requires same environment count')
        states=serialization.from_state_dict(states,origin['env']);key=j.asarray(origin['key'])
    if obj is not None:
        ts=serialization.from_state_dict(ts,obj['train'])
        states=serialization.from_state_dict(states,obj['env']);key=j.asarray(obj['key'])
        runtime=obj['runtime'];start=int(obj['iteration'])
    finish=min(cfg['updates'],args.stop_after or cfg['updates'])
    if finish<=start:raise ValueError('--stop-after/configured updates must exceed the checkpoint iteration')
    if fork is not None:runtime['fork_from_v5']=fork
    runtime['exploration']={'random_action_prob':random_action_prob,
        'random_distribution':'uniform_legal_body_then_uniform_legal_wings',
        'branch_scope':'complete_action','ppo_policy':'mixture',
        'evaluation':'unchanged_greedy'}
    print(json.dumps({'devices':[str(x) for x in devices],'device_kind':devices[0].device_kind,
        'parameters':parameters,'batch_decisions':n,'config':cfg}),flush=True)
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode());atomic(out/'pid',str(os.getpid()).encode())
    rollout=make_rollout(model,cfg['horizon'],memory,random_action_prob)
    update_short=make_update(model,p,memory,fixed_memory_length=True)
    long_ppo={**p,'minibatch':min(1024,p['minibatch'])}
    update_long=make_update(model,long_ppo,env.HISTORY,fixed_memory_length=True)
    diagnose=make_gradient_diagnostic(model,p,memory)
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True;print('Graceful stop requested; saving after this update.',flush=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if start==0 or fork is not None:checkpoint(out,ts,states,key,start,cfg,runtime)
    metrics=(out/'metrics.jsonl').open('a',buffering=1)
    for it in range(start+1,finish+1):
        begin=time.monotonic()
        states,key,tr,last,stats=rollout(ts.params,states,key);jax.block_until_ready(stats)
        rollout_seconds=time.monotonic()-begin
        stats=np.asarray(stats)
        short_history=int(stats[:,4].max())<=memory
        update=update_short if short_history else update_long
        actual_minibatch=p['minibatch'] if short_history else long_ppo['minibatch']
        key,uk=jax.random.split(key)
        entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*min(1,(it+runtime.get('source_iteration',0))/p['entropy_updates'])
        used_vf=runtime['vf_coef']
        ts,m=update(ts,tr,last,uk,j.float32(entropy),j.float32(used_vf));jax.block_until_ready(m)
        update_seconds=time.monotonic()-begin-rollout_seconds
        row={k:float(v) for k,v in m.items()};stats=np.asarray(stats)
        stable=row['nonfinite']==0 and row['kl']<2*p['target_kl'] and row['applied']>.1
        runtime['stable_updates']=runtime['stable_updates']+1 if stable else 0
        if it%p.get('vf_balance_every',10)==0:
            key,dk=jax.random.split(key);norms={k:float(v) for k,v in diagnose(ts.params,tr,last,dk).items()}
            row.update(norms)
            if it>=p.get('vf_balance_after',20) and runtime['stable_updates']>=5:
                runtime['vf_coef']=balanced_vf_coefficient(used_vf,norms['policy_grad_l2'],norms['value_grad_l2'],p)
            row.update(vf_coef_next=runtime['vf_coef'],weighted_value_grad_l2=used_vf*norms['value_grad_l2'])
        games=int(stats[:,0].sum());elapsed=time.monotonic()-begin
        row.update(iteration=it,time=time.time(),seconds=elapsed,rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,diagnostic_seconds=max(0,elapsed-rollout_seconds-update_seconds),
            fresh_decisions=n,processed_decisions=n*p['epochs']*row['applied'],
            decisions_per_second=n/elapsed,games=games,games_per_second=games/elapsed,
            invalid_actions=int(stats[:,1].sum()),landlord_wins=int(stats[:,2].sum()),
            mean_abs_landlord_score=float(stats[:,3].sum()/max(games,1)),
            history_max=int(stats[:,4].max()),vf_coef=used_vf,vf_adaptive=True,
            entropy_coef=entropy,learning_rate=float(learning_rate_schedule(cfg)(ts.step)),
            optimizer_step=int(ts.step),train_minibatch=actual_minibatch,
            random_action_prob=random_action_prob,
            train_memory_length=memory if short_history else env.HISTORY)
        if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('training invariant failed')
        metrics.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        if cfg['eval_every'] and it%cfg['eval_every']==0:
            from .evaluate_v5 import evaluate
            result=evaluate(model,ts.params,cfg['eval_games'],cfg['seed']+it)
            with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=it,**result))+'\n')
        if it%cfg['save_every']==0 or stop[0] or it==finish:
            checkpoint(out,ts,states,key,it,cfg,runtime)
        atomic(out/'status.json',json.dumps({'iteration':it,'pid':os.getpid(),'time':time.time(),
            'state':'completed' if it==cfg['updates'] else 'stopped' if stop[0] or it==finish else 'running','parameters':parameters,
            'architecture':'interaction_complete_move_v5'}).encode())
        if stop[0]:break
    metrics.close()


if __name__=='__main__':main()
