"""Isolated, reproducible V5 efficiency experiments from one immutable checkpoint."""
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
from flax.traverse_util import flatten_dict,unflatten_dict
from . import env_v4 as env
from .model_efficiency import EfficientMoveTransformer
from .optimizer_efficiency import coordinate_births,birth_corrected_adam,sampled_lr,V6_NEW_COORDINATE_WARMUP_STEPS
from .upgrade_v5 import grow_optimizer
from .ppo_efficiency import ArenaState,make_rollout,make_update,make_diagnostic
from .train_v2 import atomic,checkpoint
from .train_v3 import balanced_vf_coefficient


def create(cfg,source):
    model=EfficientMoveTransformer(**cfg['model'])
    one=env.batch_reset(jax.random.split(jax.random.PRNGKey(cfg['seed']),1))
    # Parameter shapes are independent of history length. Avoid initializing a
    # full-history executable merely to obtain the resume tree template.
    target=model.init(jax.random.PRNGKey(cfg['seed']+1),jax.vmap(env.observe)(one),memory_length=4)['params']
    if cfg.get('model_family')=='V6' and cfg['model']!=source['config']['model']:
        from .upgrade_v6 import grow_parameters, grow_births, inherited_births
        params=grow_parameters(target,source['train']['params'],source['config']['model'],cfg['model'])
        births=grow_births(params,source['train']['params'],
            inherited_births(source['train']['params'],source['runtime']),int(source['train']['step']))
        params=jax.tree_util.tree_map(j.asarray,params)
    else:
        flat=flatten_dict(target);old=flatten_dict(source['train']['params'])
        for key,value in old.items():
            if key not in flat or value.shape!=flat[key].shape:raise ValueError(f'base architecture changed {key}')
            flat[key]=j.asarray(value)
        params=unflatten_dict(flat)
        if 'coordinate_births' in source['runtime']:
            from .upgrade_v6 import grow_births,inherited_births
            births=grow_births(params,source['train']['params'],
                inherited_births(source['train']['params'],source['runtime']),int(source['train']['step']))
        else:
            births=coordinate_births(params,source['train']['params'],
                source['runtime'].get('optimizer_origin'),int(source['train']['step']))
    p=cfg['ppo']
    # Keep the original Adam/decay/schedule state schema for exact migration.
    # The actual learning rate is passed by fresh-data progress to make_update.
    warmup=V6_NEW_COORDINATE_WARMUP_STEPS if cfg.get('model_family')=='V6' else 0
    tx=optax.chain(optax.clip_by_global_norm(p['max_grad_norm']),
        optax.chain(birth_corrected_adam(births,warmup),optax.add_decayed_weights(p['weight_decay']),
                    optax.scale_by_learning_rate(lambda _:1.)))
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=tx)
    template=serialization.to_state_dict(ts)
    template['opt_state']=grow_optimizer(source['train']['opt_state'],template['opt_state'])
    template['step']=source['train']['step']
    ts=serialization.from_state_dict(ts,template)
    return model,ts


def pool_parameters(template,paths):
    pool=[]
    for path in paths:
        source=serialization.msgpack_restore(Path(path).read_bytes())
        if not set(source)<=set(template):raise ValueError('opponent architecture mismatch')
        params={**template,**jax.tree_util.tree_map(j.asarray,source)}
        if jax.tree_util.tree_structure(params)!=jax.tree_util.tree_structure(template):raise ValueError('opponent structure mismatch')
        for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(template)):
            if a.shape!=b.shape:raise ValueError('opponent parameter shape mismatch')
        pool.append(params)
    return jax.tree_util.tree_map(lambda *xs:j.stack(xs),*pool)


def validate_config(cfg,source_cfg):
    allowed={'epochs','gae_clock','belief_coef','optional_actor_only','random_action_prob','adv_keep_fraction'}
    for key in set(cfg['ppo'])|set(source_cfg['ppo']):
        if key not in allowed and cfg['ppo'].get(key)!=source_cfg['ppo'].get(key):
            raise ValueError(f'unmatched PPO option {key}')
    for key,value in source_cfg['model'].items():
        if cfg['model'].get(key)!=value:raise ValueError(f'unmatched base architecture {key}')
    if set(cfg['model'])-set(source_cfg['model'])-{'belief_head','team_value'}:
        raise ValueError('unknown architecture extension')
    for key in ('seed','envs','horizon','updates','memory_limit'):
        if cfg.get(key)!=source_cfg.get(key):raise ValueError(f'unmatched protocol {key}')
    p=cfg['ppo']
    if p.get('gae_clock','public') not in ('public','own'):raise ValueError('invalid GAE clock')
    if p['epochs'] not in (1,2,3,4):raise ValueError('unsupported epochs')
    if p.get('belief_coef',0)<0 or (p.get('belief_coef',0)>0 and not cfg['model'].get('belief_head')):
        raise ValueError('belief loss requires belief head')
    from .policy_v5 import validate_random_action_prob
    validate_random_action_prob(p.get('random_action_prob',0.))
    from .advantage_sampling import validate_keep_fraction
    validate_keep_fraction(p.get('adv_keep_fraction',1.))
    if not 0<=cfg.get('pool_probability',0)<=1:raise ValueError('invalid pool probability')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--require-a100',action='store_true')
    ap.add_argument('--training-seed',type=int,default=0)
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    raw=Path(args.source).read_bytes();source=serialization.msgpack_restore(raw)
    source_hash=hashlib.sha256(raw).hexdigest();validate_config(cfg,source['config'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:raise ValueError('existing experiment requires --resume')
    saved=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    if saved is not None and (saved['config']!=cfg or saved['runtime']['source_sha256']!=source_hash
            or saved['runtime']['training_seed']!=args.training_seed):raise ValueError('resume identity mismatch')
    jax.config.update('jax_compilation_cache_dir',str(Path(cfg.get('compilation_cache',out.resolve()/'jax_cache')).resolve()))
    device=jax.devices()[0]
    if args.require_a100 and 'A100' not in device.device_kind:raise RuntimeError('requires A100')
    model,ts=create(cfg,source)
    params_count=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    if params_count>cfg.get('parameter_limit',5_500_000):raise ValueError('parameter cap exceeded')
    b=cfg['envs'];p=cfg['ppo'];memory=cfg.get('memory_limit',88)
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(0),b))
    states=serialization.from_state_dict(states,source['env'])
    key=j.asarray(source['key'])
    if args.training_seed:key=jax.random.fold_in(key,args.training_seed)
    arena=ArenaState(states,j.zeros(b,j.int32),j.zeros(b,j.int32),j.zeros(b,j.bool_),j.zeros(b,j.bool_))
    runtime=copy.deepcopy(source['runtime'])
    runtime.update(source_sha256=source_hash,source_checkpoint=str(Path(args.source).resolve()),
        source_iteration=int(source['iteration']),source_adam_step=int(source['train']['step']),
        training_seed=args.training_seed,stable_updates=0)
    start=0
    if saved is not None:
        ts=serialization.from_state_dict(ts,saved['train'])
        states=serialization.from_state_dict(states,saved['env']);key=j.asarray(saved['key'])
        runtime=saved['runtime'];start=int(saved['iteration'])
        a=runtime['arena'];arena=ArenaState(states,*[j.asarray(a[k]) for k in ('pool_id','focus','complement','use_pool')])
    paths=cfg.get('opponent_pool',[])
    pool_hashes={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    if saved is not None and runtime.get('opponent_pool_sha256',{})!=pool_hashes:
        raise ValueError('frozen opponent pool changed since checkpoint')
    if cfg.get('opponent_pool_sha256',pool_hashes)!=pool_hashes:
        raise ValueError('frozen opponent pool does not match config hashes')
    runtime['opponent_pool_sha256']=pool_hashes
    if not paths:
        # A valid unused frozen pool retains a common rollout signature.
        pool=jax.tree_util.tree_map(lambda x:x[None],ts.params)
    else:pool=pool_parameters(ts.params,paths)
    if cfg.get('pool_probability',0)>0 and not paths:raise ValueError('pool probability requires frozen opponents')
    print(json.dumps({'device':device.device_kind,'parameters':params_count,'source_iteration':source['iteration'],'config':cfg}),flush=True)
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode());atomic(out/'pid',str(os.getpid()).encode())
    rollout=make_rollout(model,cfg['horizon'],memory,cfg.get('pool_probability',0),
        p.get('random_action_prob',0.),cfg.get('pool_bucket',64))
    update_short=make_update(model,p,memory)
    long_p={**p,'minibatch':min(p['minibatch'],1024)}
    update_long=make_update(model,long_p,env.HISTORY)
    diagnostic_short=make_diagnostic(model,p,memory);diagnostic_long=make_diagnostic(model,p,env.HISTORY)
    lr=sampled_lr(cfg,int(source['train']['step']))
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def save(it):
        runtime['arena']={k:np.asarray(getattr(arena,k)) for k in ('pool_id','focus','complement','use_pool')}
        checkpoint(out,ts,arena.games,key,it,cfg,runtime)
    if start==0:save(0)
    with (out/'metrics.jsonl').open('a',buffering=1) as f:
        for it in range(start+1,args.steps+1):
            t=time.monotonic();arena,key,tr,last,stats=rollout(ts.params,arena,key,pool);jax.block_until_ready(stats)
            roll_s=time.monotonic()-t;short=int(stats[:,3].max())<=memory
            update=update_short if short else update_long;actual_mb=p['minibatch'] if short else long_p['minibatch']
            key,uk=jax.random.split(key);used_vf=runtime['vf_coef'];used_lr=float(lr(it-1))
            entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*min(1,
                (int(source['iteration'])+it+source['runtime'].get('source_iteration',0))/p['entropy_updates'])
            ts,m=update(ts,tr,last,uk,j.float32(entropy),j.float32(used_vf),j.float32(used_lr));jax.block_until_ready(m)
            train_s=time.monotonic()-t-roll_s;row={k:float(v) for k,v in m.items()}
            stable=row['nonfinite']==0 and row['kl']<2*p['target_kl'] and row['applied']>.1
            runtime['stable_updates']=runtime['stable_updates']+1 if stable else 0
            if it%p.get('vf_balance_every',10)==0:
                key,dk=jax.random.split(key)
                diagnose=diagnostic_short if short else diagnostic_long
                norms={k:float(v) for k,v in diagnose(ts.params,tr,last,dk).items()};row.update(norms)
                if it>=p.get('vf_balance_after',20) and runtime['stable_updates']>=5:
                    runtime['vf_coef']=balanced_vf_coefficient(used_vf,norms['policy_grad_l2'],norms['value_grad_l2'],p)
            elapsed=time.monotonic()-t;n=b*cfg['horizon']
            row.update(iteration=it,source_iteration=source['iteration'],time=time.time(),seconds=elapsed,
                rollout_seconds=roll_s,update_seconds=train_s,fresh_decisions=n,
                learner_decisions=int(stats[:,2].sum()),processed_decisions=actual_mb*row['applied_minibatches'],
                decisions_per_second=n/elapsed,games=int(stats[:,0].sum()),invalid_actions=int(stats[:,1].sum()),
                vf_coef=used_vf,learning_rate=used_lr,entropy_coef=entropy,history_max=int(stats[:,3].max()),
                optimizer_step=int(ts.step),train_minibatch=actual_mb,train_memory_length=memory if short else env.HISTORY)
            if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):raise RuntimeError('training invariant failed')
            f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if it%cfg['save_every']==0 or stop[0] or it==args.steps:save(it)
            atomic(out/'status.json',json.dumps({'state':'stopped' if stop[0] else 'completed' if it==args.steps else 'running',
                'iteration':it,'pid':os.getpid(),'time':time.time(),'parameters':params_count}).encode())
            if stop[0]:break


if __name__=='__main__':main()
