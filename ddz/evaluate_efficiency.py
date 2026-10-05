"""Unique common deals, natural bidding and six complementary role legs."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from . import env_v4 as env
from .model_efficiency import EfficientMoveTransformer
from .policy_v5 import greedy
from .elo_ladder import atomic_json

PROTOCOL='efficiency_unique_common_deals_natural_and_all_roles_v1'


def unique_deals(seed,offset,count,forced_bid=False):
    ids=j.arange(count,dtype=j.uint32)+j.asarray(offset,j.uint32)
    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(seed),i))(ids)
    reset_keys=jax.vmap(lambda k:jax.random.fold_in(k,0))(keys)
    states=env.batch_reset(reset_keys)._replace(turn=(ids%3).astype(j.int32))
    if forced_bid:
        step_keys=jax.vmap(lambda k:jax.random.fold_in(k,1))(keys)
        states,_,_,bad=env.batch_step(states,j.full(count,env.encode_bid(3)),step_keys)
    else:bad=j.zeros(count,j.bool_)
    return states,bad,keys


def make_evaluator(model_a,model_b,chunk,forced,memory_limit=88):
    size=6*chunk
    role=j.repeat(j.arange(3),2*chunk)
    complement=j.tile(j.repeat(j.array([False,True]),chunk),3)
    def choose(model,p,s):
        obs=jax.vmap(env.observe)(s)
        l,c,_=jax.lax.cond(j.max(s.hist_len)<=memory_limit,
            lambda _:model.apply({'params':p},obs,memory_length=memory_limit),
            lambda _:model.apply({'params':p},obs),None)
        return greedy(s,l,c)
    @jax.jit
    def evaluate(pa,pb,seed,offset):
        s,bad,keys=unique_deals(seed,offset,chunk,forced)
        first=s.landlord if forced else s.turn
        focus=j.tile(first,6)+role;focus=focus%3
        s=jax.tree_util.tree_map(lambda x:j.concatenate((x,)*6),s)
        finished=j.zeros(size,j.bool_);scores=j.zeros(size);invalid=j.tile(bad,6).astype(j.int32)
        def tick(carry):
            s,finished,scores,invalid,t=carry
            a=choose(model_a,pa,s);b=choose(model_b,pb,s)
            owns=j.where(complement,s.turn!=focus,s.turn==focus)
            step_keys=jax.vmap(lambda k:jax.random.fold_in(k,t+2))(keys)
            ns,reward,done,bad=env.batch_step(s,j.where(owns,a,b),j.tile(step_keys,(6,1)))
            terminal=done&~finished
            score=reward[j.arange(size),focus]*j.where(complement,-1.,1.)
            scores=j.where(terminal,score,scores)
            return ns,finished|done,scores,invalid+(bad&~finished).astype(j.int32),t+1
        s,finished,scores,invalid,t=jax.lax.while_loop(lambda c:(c[-1]<256)&~j.all(c[1]),tick,
            (s,finished,scores,invalid,j.int32(0)))
        return scores.reshape(3,2,chunk),finished,invalid,t
    return evaluate


def summarize(raw,forced,seed,bootstrap=10000):
    paired=raw.mean(axis=1)
    if forced:paired=paired/np.array([2.,1.,1.])[:,None]
    values=paired.mean(axis=0);rng=np.random.default_rng(seed)
    ix=rng.integers(len(values),size=(bootstrap,len(values)))
    sampled=values[ix].mean(-1)
    return {'mean':float(values.mean()),'ci95':np.percentile(sampled,[2.5,97.5]).tolist(),
        'roles':{name:{'mean':float(paired[r].mean()),
            'ci95':np.percentile(paired[r][ix].mean(-1),[2.5,97.5]).tolist()}
            for r,name in enumerate(('landlord','landlord_next','door') if forced else
                                    ('first_bidder','second_bidder','third_bidder'))},
        'per_deal':values.tolist(),'role_per_deal':paired.tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-a',type=Path,required=True);ap.add_argument('--step-a',type=int,required=True)
    ap.add_argument('--run-b',type=Path,required=True);ap.add_argument('--step-b',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--deals',type=int,default=2048)
    ap.add_argument('--chunk',type=int,default=64);ap.add_argument('--seed',type=int,default=512001)
    args=ap.parse_args()
    if args.deals<=0 or args.chunk<=0 or args.deals%args.chunk:
        raise ValueError('positive deals must divide into positive complete chunks')
    args.out.mkdir(exist_ok=True,parents=True)
    models=[];params=[];hashes=[]
    for run,step in ((args.run_a,args.step_a),(args.run_b,args.step_b)):
        cfg=json.loads((run/'config.json').read_text());models.append(EfficientMoveTransformer(**cfg['model']))
        raw=(run/f'policy_{step:07d}.msgpack').read_bytes();params.append(serialization.msgpack_restore(raw));hashes.append(hashlib.sha256(raw).hexdigest())
    identity={'protocol':PROTOCOL,'checkpoint_sha256':hashes,'seed':args.seed,'deals':args.deals,'chunk':args.chunk,
        'decision':'greedy body then conditional wings','bootstrap_unit':'whole common deal across all six legs',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    all_results={};started=time.monotonic()
    for forced in (False,True):
        name='forced' if forced else 'natural';folder=args.out/name;folder.mkdir(exist_ok=True)
        evaluate=make_evaluator(*models,args.chunk,forced);chunks=[]
        for offset in range(0,args.deals,args.chunk):
            path=folder/f'chunk_{offset:05d}.json'
            if path.exists():
                o=json.loads(path.read_text())
                if o['identity']!=identity or o['offset']!=offset:raise ValueError('cached evaluation identity mismatch')
                raw=np.asarray(o['raw'])
            else:
                scores,done,bad,ticks=evaluate(*params,args.seed,offset);jax.block_until_ready(scores)
                if not np.all(done) or np.any(bad):raise RuntimeError('invalid or unfinished evaluation')
                raw=np.asarray(scores);atomic_json(path,{'identity':identity,'offset':offset,'raw':raw.tolist(),
                    'invalid_actions':0,'all_finished':True,'ticks':int(ticks)})
            chunks.append(raw)
            atomic_json(args.out/'status.json',{'state':'evaluating','scope':name,'completed_deals':offset+args.chunk,
                'deals':args.deals,'seconds':time.monotonic()-started,'time':time.time()})
        all_results[name]=summarize(np.concatenate(chunks,axis=2),forced,args.seed+int(forced))
    atomic_json(args.out/'summary.json',{'identity':identity,'results':all_results,'games':args.deals*12,
        'invalid_actions':0,'seconds':time.monotonic()-started})
    atomic_json(args.out/'status.json',{'state':'complete','deals':args.deals,'games':args.deals*12,'time':time.time()})
    print(json.dumps({k:{x:v[x] for x in ('mean','ci95')} for k,v in all_results.items()}),flush=True)


if __name__=='__main__':main()
