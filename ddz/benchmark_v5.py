"""Matched V4/V5 A100 benchmarks with an identical frozen V4 policy."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.35')
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from . import env_v4 as env


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',choices=['v4','v5'],required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--config')
    ap.add_argument('--allow-rounding-drift',action='store_true')
    ap.add_argument('--output',required=True)
    ap.add_argument('--rounds',type=int,default=6)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_compilation_cache_dir',str(out.resolve()/'jax_cache'))
    if args.version=='v4':
        from .train_v4 import create
        from .ppo_v4 import make_rollout,make_update
    else:
        from .train_v5 import create
        from .ppo_v5 import make_rollout,make_update
        from .upgrade_v5 import warm_start_v4
    cfg=json.loads(Path(args.config or f'configs/a100_{args.version}.json').read_text())
    device=jax.devices()[0]
    if 'A100' not in device.device_kind:raise RuntimeError('requires A100')
    model,ts=create(cfg)
    params=serialization.msgpack_restore(Path(args.checkpoint).read_bytes()) if args.version=='v4' else warm_start_v4(ts.params,args.checkpoint)[0]
    if args.version=='v5':
        from .train_v4 import create as create4
        from .compare_v3_v2_all_roles import greedy_v2_one
        from .policy_v5 import greedy
        cfg4=json.loads(Path('configs/a100_v4.json').read_text())
        old_model,_=create4(cfg4)
        old_params=serialization.msgpack_restore(Path(args.checkpoint).read_bytes())
        payload=serialization.msgpack_restore((Path(args.checkpoint).parent/'latest.msgpack').read_bytes())
        initial=env.batch_reset(jax.random.split(jax.random.PRNGKey(0),cfg['envs']))
        source_states=serialization.from_state_dict(initial,payload['env'])
        source_states=jax.tree_util.tree_map(lambda x:x[:128],source_states)
        observations=jax.vmap(env.observe)(source_states)
        old_out=jax.jit(lambda p:old_model.apply({'params':p},observations))(old_params)
        new_out=jax.jit(lambda p:model.apply({'params':p},observations))(params)
        a=jax.jit(jax.vmap(greedy_v2_one))(source_states,*old_out[:2])
        b=jax.jit(greedy)(source_states,*new_out[:2]);jax.block_until_ready(b)
        legal=np.asarray(observations.legal[:,:313])
        proof={'states':128,'source_iteration':int(payload['iteration']),
            'greedy_action_mismatches':int(np.sum(np.asarray(a)!=np.asarray(b))),
            'legal_body_max_abs_difference':float(np.abs(np.asarray(old_out[0]-new_out[0]))[legal].max()),
            'wing_max_abs_difference':float(j.max(j.abs(old_out[1]-new_out[1].base))),
            'value_max_abs_difference':float(j.max(j.abs(old_out[2]-new_out[2])))}
        (out/'migration_policy_parity.json').write_text(json.dumps(proof,indent=2))
        print(json.dumps({'migration':proof}),flush=True)
        if proof['greedy_action_mismatches'] and not args.allow_rounding_drift:raise RuntimeError('migration changed deployed actions')
    ts=ts.replace(params=params);key=jax.random.PRNGKey(510001)
    states=env.batch_reset(jax.random.split(key,cfg['envs']))
    rollout=make_rollout(model,cfg['horizon'],cfg['memory_limit'])
    update=make_update(model,cfg['ppo'],cfg['memory_limit'],fixed_memory_length=True) if args.version=='v5' else make_update(model,cfg['ppo'],cfg['memory_limit'])
    long_update=make_update(model,{**cfg['ppo'],'minibatch':1024},env.HISTORY,fixed_memory_length=True) if args.version=='v5' else update
    count=sum(x.size for x in jax.tree_util.tree_leaves(params));rows=[]
    for i in range(args.rounds):
        start=time.monotonic();states,key,tr,last,stats=rollout(ts.params,states,key);jax.block_until_ready(stats)
        roll_seconds=time.monotonic()-start;key,k=jax.random.split(key)
        chosen_update=long_update if int(stats[:,4].max())>cfg['memory_limit'] else update
        ts,metrics=chosen_update(ts,tr,last,k,j.float32(.002),j.float32(.05));jax.block_until_ready(metrics)
        elapsed=time.monotonic()-start
        row={'version':args.version,'round':i,'cold':i==0,'parameters':count,
             'rollout_seconds':roll_seconds,'update_seconds':elapsed-roll_seconds,
             'seconds':elapsed,'fresh_decisions':cfg['envs']*cfg['horizon'],
             'processed_decisions':cfg['envs']*cfg['horizon']*float(metrics['applied']),
             'decisions_per_second':cfg['envs']*cfg['horizon']/elapsed,
             'invalid_actions':int(stats[:,1].sum()),'history_max':int(stats[:,4].max()),
             'train_memory_length':cfg['memory_limit'] if int(stats[:,4].max())<=cfg['memory_limit'] else env.HISTORY,
             'train_minibatch':1024 if args.version=='v5' and int(stats[:,4].max())>cfg['memory_limit'] else cfg['ppo']['minibatch'],
             'metrics':{k:float(v) for k,v in metrics.items()},'device':device.device_kind,
             'memory':device.memory_stats()}
        rows.append(row);(out/f'{args.version}.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
        if row['invalid_actions'] or float(metrics['nonfinite']) or not all(np.isfinite(float(v)) for v in metrics.values()):
            raise RuntimeError('benchmark invariant failed')


if __name__=='__main__':main()
