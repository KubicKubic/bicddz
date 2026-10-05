"""Batched V3 evaluation against a fixed legal shedding policy."""
from functools import partial
import numpy as np
import jax
import jax.numpy as j

from . import env_v2 as env
from . import candidates_v3 as moves


def _baseline(states,ids,mask):
    hands=np.asarray(jax.device_get(states.hands))
    turns=np.asarray(jax.device_get(states.turn))
    landlord=np.asarray(jax.device_get(states.landlord))
    last_seat=np.asarray(jax.device_get(states.last_seat))
    phase=np.asarray(jax.device_get(states.phase))
    choice=np.zeros(len(turns),np.int32)
    for i in range(len(turns)):
        if phase[i]==0:
            values=moves.BID_OF[ids[i]]
            choice[i]=int(np.flatnonzero(mask[i])[np.argmin(values[mask[i]])])
            continue
        c=moves.CARDS[ids[i]].sum(axis=-1)
        t=moves.TYPE_OF[ids[i]]
        rank=moves.RANK_OF[ids[i]]
        score=3*c-rank*.1-np.where(t>=13,12.,0.)
        score=np.where(c==hands[i,turns[i]].sum(),100.,score)
        same=(turns[i]!=landlord[i] and last_seat[i]!=landlord[i]
              and last_seat[i]>=0)
        score=np.where(ids[i]==0,90. if same else -10.,score)
        choice[i]=int(np.argmax(np.where(mask[i],score,-1e9)))
    return choice


def evaluate(model,params,games,seed,common_memory=88):
    @partial(jax.jit,static_argnames=('memory_length',))
    def forward(weights,obs,ids,mask,memory_length):
        return model.apply({'params':weights},obs,ids,mask,memory_length=memory_length)
    key=jax.random.PRNGKey(seed)
    key,deal_key=jax.random.split(key)
    states=env.batch_reset(jax.random.split(deal_key,games))
    seats=np.arange(games)%3
    score=np.zeros(games,np.float32);finished=np.zeros(games,np.bool_)
    role=np.full(games,-1,np.int32);wins=np.zeros(games,np.bool_)
    invalid=0
    for _ in range(192):
        if finished.all():break
        ids_np,mask_np,_=moves.batch_legal(states)
        obs=jax.vmap(env.observe)(states)
        memory_length=common_memory if int(j.max(states.hist_len))<=common_memory else env.HISTORY
        logits,_=forward(params,obs,j.asarray(ids_np),j.asarray(mask_np),memory_length)
        learner=np.asarray(jax.device_get(j.argmax(logits,axis=-1)))
        rival=_baseline(states,ids_np,mask_np)
        turns=np.asarray(jax.device_get(states.turn))
        selected=np.where(turns==seats,learner,rival)
        actions=j.asarray(moves.ACTION[ids_np[np.arange(games),selected]])
        key,step_key=jax.random.split(key)
        ns,reward,done,bad=env.batch_step(states,actions,
            jax.random.split(step_key,games))
        r=np.asarray(jax.device_get(reward))
        d=np.asarray(jax.device_get(done))
        b=np.asarray(jax.device_get(bad))
        active=~finished
        score=np.where(d&active,r[np.arange(games),seats],score)
        wins=np.where(d&active,score>0,wins)
        role=np.where(d&active,
            np.asarray(jax.device_get(ns.landlord))==seats,role)
        invalid+=int(np.sum(b&active))
        finished|=d
        states=ns
    if not finished.all() or invalid:
        raise RuntimeError(f'evaluation incomplete: finished={finished.sum()}, invalid={invalid}')
    return {'games':games,'mean_raw_score':float(score.mean()),
        'score_stderr':float(score.std(ddof=1)/np.sqrt(games)),
        'win_rate':float(wins.mean()),'landlord_games':int((role==1).sum()),
        'landlord_win_rate':float(wins[role==1].mean()) if (role==1).any() else None,
        'farmer_win_rate':float(wins[role==0].mean()) if (role==0).any() else None,
        'opponent':'fixed_shedding_v3','seat_balanced':True}
