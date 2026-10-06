"""CPU engineering proof for the full 8M warm start, without a formal fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import jax
from flax import serialization
from . import env_v4 as env
from .model_efficiency import EfficientMoveTransformer
from .upgrade_v6 import revised_config,grow_parameters
from .scale_architecture_v6 import function_probe
from .scale_rollout_campaign import read,write,sha
from .api_v5 import Policy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--step',type=int,required=True);ap.add_argument('--states',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if jax.default_backend()!='cpu':raise RuntimeError('This engineering verifier uses CPU only')
    args.out.mkdir(parents=True,exist_ok=False)
    cfg=read(args.run/'config.json');target_cfg=revised_config(cfg)
    states=serialization.msgpack_restore(args.states.read_bytes())
    one=env.State(**{k:v[:1] for k,v in states.items()})
    obs=jax.vmap(env.observe)(one)
    model=EfficientMoveTransformer(**target_cfg['model'])
    template=model.init(jax.random.PRNGKey(cfg['seed']+1),obs,memory_length=4)['params']
    paths={'raw':args.run/f'policy_{args.step:07d}.msgpack',
           'ema':args.run/'ema'/f'policy_{args.step:07d}.msgpack'}
    old={k:serialization.msgpack_restore(path.read_bytes()) for k,path in paths.items()}
    grown={k:grow_parameters(template,p,cfg['model'],target_cfg['model']) for k,p in old.items()}
    saved={'config':cfg,'env':states,'train':{'params':old['raw']},'ema':{'params':old['ema']}}
    migrated={'config':target_cfg,'train':{'params':grown['raw']},'ema':{'params':grown['ema']}}
    write(args.out/'config.json',target_cfg)
    for kind,p in grown.items():
        (args.out/f'{kind}_policy.msgpack').write_bytes(serialization.msgpack_serialize(p))
    print('Probing full-size raw and EMA complete decisions',flush=True)
    probe=function_probe(saved,migrated)
    print('Warming portable CPU serving policy',flush=True)
    api=Policy(args.out/'config.json',args.out/'raw_policy.msgpack')
    state=jax.tree_util.tree_map(lambda x:x[0],one)
    api.prepare_params(api.params,target_cfg)
    samples=[]
    for _ in range(12):
        begin=time.monotonic();jax.block_until_ready(api.forward(state));samples.append(time.monotonic()-begin)
    parameters=sum(x.size for x in jax.tree_util.tree_leaves(grown['raw']))
    result={'passed':True,'scope':'CPU engineering verification, not playing-strength or eight-GPU throughput evidence',
        'parameters':parameters,'source_step':args.step,
        'global_source_step':cfg['global_source_iteration']+args.step,
        'backend':'CPU FP32 identity reference and BF16 compiled tolerance check',
        'source_sha256':{k:sha(path) for k,path in paths.items()},'states_sha256':sha(args.states),
        'actual_state_count':len(states['turn']),'actual_history_max':int(np.max(states['hist_len'])),
        'synthetic_history_lengths_tested':[0,2,89,192],
        'function_probe':probe,'cpu_serving_forward_seconds_median':float(np.median(samples)),
        'config':target_cfg,'time':time.time()}
    if parameters!=8_014_192:raise RuntimeError('8M budget differs')
    write(args.out/'CPU_VERIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='config'},indent=2),flush=True)


if __name__=='__main__':main()
