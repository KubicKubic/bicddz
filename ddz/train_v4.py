"""Persistent, device-resident PPO with compact complete actions."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.35')
import argparse
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
from .model_v4 import CompactMoveTransformer
from .ppo_v4 import make_rollout,make_update,make_gradient_diagnostic
from .train_v2 import atomic,checkpoint,learning_rate_schedule
from .train_v3 import warm_start_v2,balanced_vf_coefficient


def create(cfg):
    model=CompactMoveTransformer(**cfg['model'])
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(cfg['seed']),1))
    params=model.init(jax.random.PRNGKey(cfg['seed']+1),jax.vmap(env.observe)(states))['params']
    p=cfg['ppo']
    tx=optax.chain(optax.clip_by_global_norm(p['max_grad_norm']),
        optax.adamw(learning_rate_schedule(cfg),weight_decay=p['weight_decay']))
    return model,TrainState.create(apply_fn=model.apply,params=params,tx=tx)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/a100_v4.json')
    ap.add_argument('--out',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--init-from-v2')
    ap.add_argument('--require-a100',action='store_true')
    ap.add_argument('--stop-after',type=int)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    p=cfg['ppo'];n=cfg['envs']*cfg['horizon'];memory=cfg.get('memory_limit',88)
    if n%p['minibatch'] or not 1<=memory<=env.HISTORY:
        raise ValueError('minibatch must divide rollout and memory must fit history capacity')
    if args.resume and args.init_from_v2:raise ValueError('warm start only applies to new runs')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:raise ValueError('existing run requires --resume')
    obj=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    if obj is not None and obj['config']!=cfg:raise ValueError('resume configuration mismatch')
    jax.config.update('jax_compilation_cache_dir',str(out.resolve()/'jax_cache'))
    devices=jax.devices()
    if args.require_a100 and (len(devices)!=1 or 'A100' not in devices[0].device_kind):
        raise RuntimeError(f'requires one A100, found {devices}')
    model,ts=create(cfg)
    runtime={'vf_coef':p['value_coef'],'stable_updates':0,'vf_adaptive':True}
    if args.init_from_v2:
        params,copied=warm_start_v2(ts.params,args.init_from_v2);ts=ts.replace(params=params)
        runtime['warm_start']={'checkpoint':str(Path(args.init_from_v2).resolve()),'copied_parameters':copied}
    parameters=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    if parameters>4_000_000:raise ValueError('parameter cap exceeded')
    key=jax.random.PRNGKey(cfg['seed']+2);key,k=jax.random.split(key)
    states=env.batch_reset(jax.random.split(k,cfg['envs']));start=0
    if obj is not None:
        ts=serialization.from_state_dict(ts,obj['train'])
        states=serialization.from_state_dict(states,obj['env']);key=j.asarray(obj['key'])
        runtime=obj['runtime'];start=int(obj['iteration'])
    print(json.dumps({'devices':[str(x) for x in devices],'device_kind':devices[0].device_kind,
        'parameters':parameters,'batch_decisions':n,'config':cfg}),flush=True)
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode());atomic(out/'pid',str(os.getpid()).encode())
    rollout=make_rollout(model,cfg['horizon'],memory)
    update=make_update(model,p,memory)
    diagnose=make_gradient_diagnostic(model,p,memory)
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True;print('Graceful stop requested; saving after this update.',flush=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if start==0:checkpoint(out,ts,states,key,0,cfg,runtime)
    metrics=(out/'metrics.jsonl').open('a',buffering=1)
    finish=min(cfg['updates'],args.stop_after or cfg['updates'])
    for it in range(start+1,finish+1):
        begin=time.monotonic()
        states,key,tr,last,stats=rollout(ts.params,states,key);jax.block_until_ready(stats)
        rollout_seconds=time.monotonic()-begin
        key,uk=jax.random.split(key)
        entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*min(1,it/p['entropy_updates'])
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
            optimizer_step=int(ts.step))
        if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('training invariant failed')
        metrics.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        if cfg['eval_every'] and it%cfg['eval_every']==0:
            from .evaluate_v2 import evaluate
            result=evaluate(model,ts.params,cfg['eval_games'],cfg['seed']+it)
            with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=it,**result))+'\n')
        if it%cfg['save_every']==0 or stop[0] or it==finish:
            checkpoint(out,ts,states,key,it,cfg,runtime)
        atomic(out/'status.json',json.dumps({'iteration':it,'pid':os.getpid(),'time':time.time(),
            'state':'completed' if it==cfg['updates'] else 'stopped' if stop[0] or it==finish else 'running','parameters':parameters,
            'architecture':'compact_complete_move_v4'}).encode())
        if stop[0]:break
    metrics.close()


if __name__=='__main__':main()
