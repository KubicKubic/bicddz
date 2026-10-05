"""Measure warm GPU rollout and PPO throughput, including memory peaks."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.35')
import argparse,json,time
from pathlib import Path
import jax
import jax.numpy as j
from ddz import env
from ddz.train import create
from ddz.ppo import make_rollout,make_update
p=argparse.ArgumentParser(); p.add_argument('--envs',type=int,default=256); p.add_argument('--horizon',type=int,default=16)
p.add_argument('--minibatch',type=int,default=512)
p.add_argument('--train-memory-limit',type=int,default=env.HISTORY); a=p.parse_args()
c=json.loads(Path('configs/a100.json').read_text()); c.update(envs=a.envs,horizon=a.horizon)
c['ppo'].update(minibatch=a.minibatch)
m,ts=create(c); s=env.batch_reset(jax.random.split(jax.random.PRNGKey(9),a.envs)); key=jax.random.PRNGKey(1)
r=make_rollout(m,a.horizon); u=make_update(m,c['ppo'],a.train_memory_limit)
for i in range(3):
    t=time.monotonic(); s,key,tr,last,stats=r(ts.params,s,key); jax.block_until_ready(stats); rt=time.monotonic()-t
    t=time.monotonic(); ts,metrics=u(ts,tr,last,key,j.float32(.02),j.float32(.5)); jax.block_until_ready(metrics); ut=time.monotonic()-t
    print(json.dumps(dict(iteration=i,rollout_seconds=rt,update_seconds=ut,
         decisions_per_second=a.envs*a.horizon/(rt+ut),memory=jax.devices()[0].memory_stats())),flush=True)
