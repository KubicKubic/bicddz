"""Cold and warm measured A100 throughput; no live-run weights are modified."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.35')
import argparse,json,time
from pathlib import Path
import jax
import jax.numpy as j
from . import env_v4 as env
from .train_v4 import create,warm_start_v2
from .ppo_v4 import make_rollout,make_update


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--envs',type=int,nargs='+',default=[512,1024]);ap.add_argument('--rounds',type=int,default=4)
    args=ap.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_compilation_cache_dir',str(output.resolve()/'jax_cache'))
    device=jax.devices()[0]
    if 'A100' not in device.device_kind:raise RuntimeError('benchmark requires A100')
    results=[]
    for batch in args.envs:
        cfg=json.loads(Path('configs/a100_v4.json').read_text());cfg['envs']=batch
        model,ts=create(cfg);params,_=warm_start_v2(ts.params,'runs/a100_complete_move_v2/policy_0001572.msgpack')
        ts=ts.replace(params=params);key=jax.random.PRNGKey(410004)
        states=env.batch_reset(jax.random.split(key,batch))
        roll=make_rollout(model,cfg['horizon'],cfg['memory_limit'])
        update=make_update(model,cfg['ppo'],cfg['memory_limit'])
        for i in range(args.rounds):
            begin=time.monotonic();states,key,tr,last,stats=roll(ts.params,states,key);jax.block_until_ready(stats)
            rollout=time.monotonic()-begin;key,k=jax.random.split(key)
            ts,metrics=update(ts,tr,last,k,j.float32(.02),j.float32(.03));jax.block_until_ready(metrics)
            elapsed=time.monotonic()-begin
            row={'envs':batch,'horizon':cfg['horizon'],'round':i,'cold':i==0,
                'rollout_seconds':rollout,'update_seconds':elapsed-rollout,
                'decisions_per_second':batch*cfg['horizon']/elapsed,
                'games_per_second':int(j.sum(stats[:,0]))/elapsed,
                'invalid_actions':int(j.sum(stats[:,1])),'nonfinite':float(metrics['nonfinite']),
                'applied':float(metrics['applied']),'device':device.device_kind,'memory':device.memory_stats()}
            results.append(row);print(json.dumps(row),flush=True)
            (output/'benchmark.json').write_text(json.dumps(results,indent=2))
            if row['invalid_actions'] or row['nonfinite']:raise RuntimeError('benchmark invariant failed')
        del ts,params,tr,states
    print('Benchmark complete.',flush=True)


if __name__=='__main__':main()
