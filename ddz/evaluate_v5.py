"""Batched score evaluation against fixed baselines, balanced over all seats."""
import jax
import jax.numpy as j
import numpy as np
from . import env_v4 as env
from . import policy_v5 as policy
from .env_v2 import legal_wings
from .actions import N_BODY,BID_OFFSET,WING_OFFSET,COUNTS

def baseline(s,key):
    legal=env.legal(s)
    # Fixed legal shedding heuristic. Prefer finish / many cards, save bombs,
    # pass over teammate. This is only an evaluation opponent, never a reward.
    from .actions import TYPE,RANK,WINGS,WUNIT
    size=j.array(COUNTS).sum(axis=-1)+j.array(WINGS)*j.array(WUNIT)
    score=3*size-j.array(RANK)*.1-j.where(j.array(TYPE)>=13,12.,0.)
    score=j.where(size==j.sum(s.hands[s.turn]),100.,score)
    same=(s.turn!=s.landlord)&(s.last_seat!=s.landlord)&(s.last_seat>=0)
    score=score.at[0].set(j.where(same,90.,-10.))
    scores=j.concatenate((score,j.array([1.,0.,-.5,-1.]),-j.arange(15)*.1))
    index=j.argmax(j.where(legal,scores,-1e9)).astype(j.int32)
    def bid(_): return env.encode_bid(index-N_BODY)
    def move(_):
        from .actions import WINGS,WUNIT
        body=index
        state=s._replace(pending=body,wing_left=j.array(WINGS)[body],
                         wing_min=j.int32(0),wings=j.zeros(15,j.int32))
        def add(i,carry):
            state,ranks=carry
            def wing(carry):
                state,ranks=carry
                choices=legal_wings(state)
                rank=j.argmax(j.where(choices,-j.arange(15),-100)).astype(j.int32)
                wings=state.wings.at[rank].add(j.array(WUNIT)[body])
                state=state._replace(wings=wings,wing_left=state.wing_left-1,
                    wing_min=rank+j.where(j.array(WUNIT)[body]==2,1,0))
                return state,ranks.at[i].set(rank)
            return jax.lax.cond(i<j.array(WINGS)[body],wing,lambda x:x,carry)
        _,ranks=jax.lax.fori_loop(0,5,add,(state,j.full(5,15,j.int32)))
        return env.encode_move(body,ranks)
    return jax.lax.cond(index>=N_BODY,bid,move,None)


def make_evaluator(model,games,memory_limit=88):
    @jax.jit
    def play(params,key):
        key,rk=jax.random.split(key)
        s=jax.vmap(env.reset)(jax.random.split(rk,games))
        seats=j.arange(games)%3
        total=j.zeros(games); finished=j.zeros(games,j.bool_)
        role=j.full(games,-1,j.int32); wins=j.zeros(games,j.bool_); invalid=j.int32(0)
        def condition(carry):
            return (carry[-1]<256)&~j.all(carry[3])
        def tick(carry):
            s,key,total,finished,role,wins,invalid,decisions=carry
            key,ak,sk=jax.random.split(key,3)
            obs=jax.vmap(env.observe)(s)
            logits,wings,_=jax.lax.cond(j.max(obs.hist_len)<=memory_limit,
                lambda _:model.apply({'params':params},obs,memory_length=memory_limit),
                lambda _:model.apply({'params':params},obs),None)
            learner,_,_=policy.sample(s,logits,wings,jax.random.split(ak,games))
            rival=jax.vmap(baseline)(s,jax.random.split(ak,games))
            a=j.where(s.turn==seats,learner,rival)
            ns,r,d,bad=jax.vmap(env.step)(s,a,jax.random.split(sk,games))
            score=j.take_along_axis(r,seats[:,None],axis=-1)[:,0]
            total+=j.where(~finished,score,0)
            wins=j.where(d&~finished,score>0,wins)
            role=j.where(d&~finished,ns.landlord==seats,role)
            invalid+=j.sum(bad&~finished)
            return ns,key,total,finished|d,role,wins,invalid,decisions+1
        result=jax.lax.while_loop(condition,tick,(s,key,total,finished,role,wins,invalid,j.int32(0)))
        return result[2:7]
    return play

_EVALUATORS={}
def evaluate(model,params,games,seed):
    cache=(model,games)
    if cache not in _EVALUATORS: _EVALUATORS[cache]=make_evaluator(model,games)
    scores,done,role,wins,invalid=jax.device_get(_EVALUATORS[cache](params,jax.random.PRNGKey(seed)))
    if not np.all(done) or invalid: raise RuntimeError('evaluation did not finish legally')
    return {'games':games,'mean_raw_score':float(scores.mean()),
            'score_stderr':float(scores.std(ddof=1)/np.sqrt(games)),
            'win_rate':float(wins.mean()),'landlord_games':int((role==1).sum()),
            'landlord_win_rate':float(wins[role==1].mean()) if (role==1).any() else None,
            'farmer_win_rate':float(wins[role==0].mean()) if (role==0).any() else None,
            'opponent':'fixed_shedding_v1','seat_balanced':True}
