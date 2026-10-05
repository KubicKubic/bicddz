"""Resident, deterministic, paired-deal comparisons over every seat and role."""
import jax
import jax.numpy as j
import numpy as np
from . import env_v4 as env
from .compare_v3_v2_all_roles import greedy_v2_one
from .policy_v5 import greedy
from .model_v5 import InteractionMoveTransformer

ROLE_NAMES=('landlord','landlord_next','door')


def paired_scores(scores,deals,forced_bid):
    paired=np.asarray(scores,np.float64).reshape(3,2,deals).mean(axis=1)
    if forced_bid:paired=paired/np.asarray([2.,1.,1.])[:,None]
    return paired,paired.mean(axis=0)


def make_role_evaluator(model,chunk_deals,forced_bid,memory_limit=88,reference_model=None):
    reference_model=reference_model or model
    games=6*chunk_deals
    role=j.repeat(j.arange(3),2*chunk_deals)
    complement=j.tile(j.repeat(j.array([False,True]),chunk_deals),3)
    leg_deal=j.tile(j.arange(chunk_deals),6)
    def choose(m,p,s):
        obs=jax.vmap(env.observe)(s)
        body,wings,_=jax.lax.cond(j.max(s.hist_len)<=memory_limit,
            lambda _:m.apply({'params':p},obs,memory_length=memory_limit),
            lambda _:m.apply({'params':p},obs),None)
        return greedy(s,body,wings) if isinstance(m,InteractionMoveTransformer) else jax.vmap(greedy_v2_one)(s,body,wings)
    @jax.jit
    def evaluate(params_a,params_b,seed,deal_offset):
        key=jax.random.PRNGKey(seed);key,dk,bk=jax.random.split(key,3)
        s=env.batch_reset(jax.random.split(dk,chunk_deals))
        first=(j.arange(chunk_deals)+deal_offset)%3;s=s._replace(turn=first)
        invalid=j.zeros(chunk_deals,j.int32)
        if forced_bid:
            s,_,_,bad=env.batch_step(s,j.full((chunk_deals,),env.encode_bid(3)),jax.random.split(bk,chunk_deals))
            invalid=bad.astype(j.int32)
        s=jax.tree_util.tree_map(lambda x:j.concatenate((x,)*6),s)
        focus=(first[leg_deal]+role)%3
        def condition(carry):
            _,_,finished,_,_,decision=carry
            return (decision<256)&~j.all(finished)
        def tick(carry):
            s,key,finished,scores,invalid,decision=carry
            a=choose(model,params_a,s);b=choose(reference_model,params_b,s)
            owns=j.where(complement,s.turn!=focus,s.turn==focus)
            key,sk=jax.random.split(key)
            ns,reward,done,bad=env.batch_step(s,j.where(owns,a,b),
                j.tile(jax.random.split(sk,chunk_deals),(6,1)))
            terminal=done&~finished;focus_reward=reward[j.arange(games),focus]
            scores=j.where(terminal,j.where(complement,-focus_reward,focus_reward),scores)
            invalid=invalid+(bad&~finished).astype(j.int32)
            return ns,key,finished|done,scores,invalid,decision+1
        result=jax.lax.while_loop(condition,tick,(s,key,j.zeros(games,j.bool_),
            j.zeros(games),j.tile(invalid,6),j.int32(0)))
        return result[3].reshape(3,2,chunk_deals),result[2],result[4],result[5]
    return evaluate
