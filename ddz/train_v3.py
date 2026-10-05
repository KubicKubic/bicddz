"""Persistent PPO training over exact, variable-size complete-move sets."""
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
from flax.traverse_util import flatten_dict,unflatten_dict

from . import env_v2 as env
from . import candidates_v3 as moves
from .model_v3 import CandidateSetTransformer
from .ppo_v3 import make_rollout,make_update,make_gradient_diagnostic


def create(cfg):
    model=CandidateSetTransformer(**cfg['model'])
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(cfg['seed']),1))
    ids,mask,_=moves.batch_legal(states)
    params=model.init(jax.random.PRNGKey(cfg['seed']+1),jax.vmap(env.observe)(states),
                      j.asarray(ids),j.asarray(mask))['params']
    p=cfg['ppo']
    scale=p.get('optimizer_steps_per_update_estimate',64)
    lr=optax.warmup_cosine_decay_schedule(p['lr']*.1,p['lr'],
        max(1,p['warmup']*scale),max(p['warmup']+1,cfg['updates'])*scale,p['lr_min'])
    tx=optax.chain(optax.clip_by_global_norm(p['max_grad_norm']),
                   optax.adamw(lr,weight_decay=p['weight_decay']))
    return model,TrainState.create(apply_fn=model.apply,params=params,tx=tx)


def warm_start_v2(params,path):
    old=flatten_dict(serialization.msgpack_restore(Path(path).read_bytes()))
    new=flatten_dict(params)
    copied=[]
    for name,value in new.items():
        if name in old and np.asarray(old[name]).shape==np.asarray(value).shape:
            new[name]=j.asarray(old[name]);copied.append('.'.join(name))
    return unflatten_dict(new),copied


def balanced_vf_coefficient(old,policy_grad_l2,value_grad_l2,cfg):
    """Smoothly equalize unweighted policy/value gradient magnitudes."""
    minimum=float(cfg.get('vf_coef_min',.001))
    maximum=float(cfg.get('vf_coef_max',10.))
    if not (0<minimum<=maximum and old>0 and np.isfinite(old) and
            np.isfinite(policy_grad_l2) and np.isfinite(value_grad_l2)):
        return float(old)
    if policy_grad_l2<=1e-9 or value_grad_l2<=1e-12:
        return float(old)
    target=float(np.clip(policy_grad_l2/value_grad_l2,minimum,maximum))
    smoothed=np.sqrt(float(old)*target)
    return float(np.clip(smoothed,max(minimum,float(old)/2),
                         min(maximum,float(old)*2)))


def atomic(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def checkpoint(out,ts,states,key,iteration,cfg,runtime):
    payload={'train':serialization.to_state_dict(ts),
             'env':serialization.to_state_dict(states),'key':np.asarray(key),
             'iteration':iteration,'config':cfg,'runtime':runtime}
    atomic(out/'latest.msgpack',serialization.msgpack_serialize(jax.device_get(payload)))
    atomic(out/f'policy_{iteration:07d}.msgpack',serialization.to_bytes(ts.params))
    atomic(out/'latest.json',json.dumps({'iteration':iteration,'time':time.time(),
        'policy':f'policy_{iteration:07d}.msgpack'}).encode())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/a100_v3.json')
    ap.add_argument('--out',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--init-from-v2')
    ap.add_argument('--require-a100',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--memory-limit',type=int)
    ap.add_argument('--ppo-epochs',type=int)
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(envs=8,horizon=8,updates=2,save_every=1,eval_every=0)
        cfg['ppo'].update(candidate_tokens_per_minibatch=1024,max_minibatch=32,warmup=1)
    if args.resume and args.init_from_v2:
        raise ValueError('warm start applies only to a new run')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:
        raise ValueError('existing run: use --resume')
    obj=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    if obj is not None and obj['config']!=cfg:
        raise ValueError('resume configuration mismatch')
    runtime=dict(obj['runtime']) if obj is not None else {'vf_coef':cfg['ppo']['value_coef']}
    runtime.setdefault('ppo_epochs',cfg['ppo'].get('epochs',1))
    runtime.setdefault('stable_updates',0)
    runtime['vf_adaptive']=True
    if args.ppo_epochs is not None:
        runtime['ppo_epochs']=args.ppo_epochs
    if not 1<=runtime['ppo_epochs']<=8:
        raise ValueError('PPO epochs must be between one and eight')
    common_memory=args.memory_limit or runtime.get('memory_limit',88)
    if not 1<=common_memory<=env.HISTORY:
        raise ValueError('memory limit must fit history capacity')
    runtime['memory_limit']=common_memory
    jax.config.update('jax_compilation_cache_dir',str(out.resolve()/'jax_cache'))
    devices=jax.devices()
    if args.require_a100 and (len(devices)!=1 or 'A100' not in devices[0].device_kind):
        raise RuntimeError(f'requires one A100; found {devices}')
    print(json.dumps({'devices':[str(x) for x in devices],
        'device_kind':devices[0].device_kind}),flush=True)
    model,ts=create(cfg)
    if args.init_from_v2:
        params,copied=warm_start_v2(ts.params,args.init_from_v2)
        ts=ts.replace(params=params)
        runtime['warm_start']={'checkpoint':str(Path(args.init_from_v2).resolve()),
                               'copied_parameters':copied}
        print(json.dumps({'warm_start_copied':len(copied)}),flush=True)
    parameters=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    if parameters>4_000_000:
        raise ValueError(f'parameter cap exceeded: {parameters}')
    print(json.dumps({'parameters':parameters,'complete_moves':moves.N_PLAY,
        'batch_decisions':cfg['envs']*cfg['horizon'],'config':cfg}),flush=True)
    key=jax.random.PRNGKey(cfg['seed']+2);key,k=jax.random.split(key)
    states=env.batch_reset(jax.random.split(k,cfg['envs']));start=0
    if obj is not None:
        ts=serialization.from_state_dict(ts,obj['train'])
        states=serialization.from_state_dict(states,obj['env'])
        key=j.asarray(obj['key']);start=int(obj['iteration'])
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode())
    atomic(out/'pid',str(os.getpid()).encode())
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True
        print('Graceful stop requested; saving after this update.',flush=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    rollout=make_rollout(model,cfg['horizon'],common_memory)
    update=make_update(model,{**cfg['ppo'],'epochs':runtime['ppo_epochs']},common_memory)
    diagnose=make_gradient_diagnostic(model,cfg['ppo'],common_memory)
    if start==0:checkpoint(out,ts,states,key,0,cfg,runtime)
    metrics=(out/'metrics.jsonl').open('a',buffering=1)
    for iteration in range(start+1,cfg['updates']+1):
        begin=time.monotonic()
        states,key,tr,last,stats=rollout(ts.params,states,key)
        jax.block_until_ready(stats)
        rollout_seconds=time.monotonic()-begin
        key,uk=jax.random.split(key)
        p=cfg['ppo'];frac=min(1,iteration/p['entropy_updates'])
        entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*frac
        ts,m=update(ts,tr,last,uk,j.float32(entropy),j.float32(runtime['vf_coef']))
        stats=np.asarray(stats)
        row={k:float(v) for k,v in m.items()}
        stable=(row['nonfinite']==0 and row['kl']<2*p['target_kl']
                and row['applied']>0.1)
        runtime['stable_updates']=runtime['stable_updates']+1 if stable else 0
        if runtime['vf_adaptive'] and iteration%p.get('vf_balance_every',10)==0:
            key,dk=jax.random.split(key)
            norms=diagnose(ts.params,tr,last,dk)
            row.update(norms)
            old=runtime['vf_coef']
            row['weighted_value_grad_l2']=old*norms['value_grad_l2']
            row['weighted_value_shared_grad_l2']=old*norms['value_shared_grad_l2']
            if (iteration>=p.get('vf_balance_after',20)
                    and runtime['stable_updates']>=5):
                runtime['vf_coef']=balanced_vf_coefficient(
                    old,norms['policy_grad_l2'],norms['value_grad_l2'],p)
            row['vf_coef_next']=runtime['vf_coef']
            row['balanced_gradient_ratio_next']=(runtime['vf_coef']*
                norms['value_grad_l2']/max(norms['policy_grad_l2'],1e-12))
        row.update(iteration=iteration,time=time.time(),
            seconds=time.monotonic()-begin,rollout_seconds=rollout_seconds,
            update_seconds=time.monotonic()-begin-rollout_seconds,
            decisions_per_second=cfg['envs']*cfg['horizon']/(time.monotonic()-begin),
            games=int(stats[:,0].sum()),invalid_actions=int(stats[:,1].sum()),
            landlord_wins=int(stats[:,2].sum()),
            mean_abs_landlord_score=float(stats[:,3].sum()/max(1,stats[:,0].sum())),
            history_max=int(stats[:,4].max()),candidate_max=int(stats[:,5].max()),
            entropy_coef=entropy,
            vf_coef=(old if 'vf_coef_next' in row else runtime['vf_coef']),
            vf_adaptive=runtime['vf_adaptive'],
            vf_balance_stable_updates=runtime['stable_updates'],
            ppo_epochs=runtime['ppo_epochs'],
            memory_limit=common_memory)
        metrics.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('training invariant failed')
        if cfg['eval_every'] and iteration%cfg['eval_every']==0:
            from .evaluate_v3 import evaluate
            result=evaluate(model,ts.params,cfg['eval_games'],cfg['seed']+iteration,
                            common_memory)
            with (out/'eval.jsonl').open('a') as f:
                f.write(json.dumps(dict(iteration=iteration,**result))+'\n')
            print(json.dumps({'evaluation':result}),flush=True)
        if iteration%cfg['save_every']==0 or stop[0] or iteration==cfg['updates']:
            checkpoint(out,ts,states,key,iteration,cfg,runtime)
        atomic(out/'status.json',json.dumps({'iteration':iteration,'pid':os.getpid(),
            'time':time.time(),'state':'stopped' if stop[0] else
            'completed' if iteration==cfg['updates'] else 'running',
            'parameters':parameters,'memory_limit':common_memory}).encode())
        if stop[0]:break
    metrics.close()


if __name__=='__main__':main()
