"""Batched score evaluation against fixed baselines, balanced over all seats."""
import jax
import jax.numpy as j
import numpy as np
from . import env
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
    return j.argmax(j.where(legal,scores,-1e9)).astype(j.int32)


def make_evaluator(model,games):
    @jax.jit
    def play(params,key):
        key,rk=jax.random.split(key)
        s=jax.vmap(env.reset)(jax.random.split(rk,games))
        seats=j.arange(games)%3
        total=j.zeros(games); finished=j.zeros(games,j.bool_)
        role=j.full(games,-1,j.int32); wins=j.zeros(games,j.bool_); invalid=j.int32(0)
        def tick(carry,_):
            s,key,total,finished,role,wins,invalid=carry
            key,ak,sk=jax.random.split(key,3)
            obs=jax.vmap(env.observe)(s)
            logits,_=model.apply({'params':params},obs)
            learner=j.argmax(logits,axis=-1).astype(j.int32)
            rival=jax.vmap(baseline)(s,jax.random.split(ak,games))
            a=j.where(s.turn==seats,learner,rival)
            ns,r,d,bad=jax.vmap(env.step)(s,a,jax.random.split(sk,games))
            score=j.take_along_axis(r,seats[:,None],axis=-1)[:,0]
            total+=j.where(~finished,score,0)
            wins=j.where(d&~finished,score>0,wins)
            role=j.where(d&~finished,ns.landlord==seats,role)
            invalid+=j.sum(bad&~finished)
            return (ns,key,total,finished|d,role,wins,invalid),None
        result,_=jax.lax.scan(tick,(s,key,total,finished,role,wins,invalid),None,length=320)
        return result[2:]
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
