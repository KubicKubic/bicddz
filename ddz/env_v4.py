"""Vectorized validation and one atomic commit for each complete move.

Public states, observations, scoring and event encoding are identical to V2.
Validation touches only the selected body and its 15 rank counts; no repeated
309-body legality scans or intermediate body/wing environment states.
"""
import jax
import jax.numpy as j
from . import env_v2 as base
from .env_v2 import State,Observation,HISTORY,EVENT_DIM,observe,reset,legal
from .actions import N_BODY,BID_OFFSET

encode_bid=base.encode_bid
encode_move=base.encode_move
decode_move=base.decode_move


def step(s,action,key):
    action=j.asarray(action,j.int32)
    def bid(_):
        value=-action-1
        must=(s.redeals>=3)&(s.bid_count==2)&(s.bid==0)
        valid=(s.phase==0)&(value>=0)&(value<=3)&((value>s.bid)|((value==0)&~must))
        valid=valid&~s.done&(s.pending<0)
        return jax.lax.cond(valid,
            lambda _:(*base._step(s,BID_OFFSET+value,key),j.array(False)),
            lambda _:(s,j.zeros(3),s.done,j.array(True)),None)

    def move(_):
        body,ranks=decode_move(action)
        count=base.WN[body];unit=base.WU[body]
        active=j.arange(5)<count
        wing=j.sum(jax.nn.one_hot(ranks,15,dtype=j.int32)*active[:,None],axis=0)*unit
        cards=base.C[body]+wing
        hand=s.hands[s.turn]
        leading=(s.last==0)|(s.last_seat==s.turn)
        stronger=((base.TY[body]==base.TY[s.last])&
                  (base.LE[body]==base.LE[s.last])&(base.RA[body]>base.RA[s.last]))
        stronger=stronger|((base.TY[body]==13)&(base.TY[s.last]!=13)&
                           (base.TY[s.last]!=14))|((base.TY[body]==14)&(base.TY[s.last]!=14))
        legal_body=j.where(body==0,~leading,leading|stronger)
        sorted_ok=j.all(j.where(j.arange(4)<count-1,
            ranks[1:]>=ranks[:-1]+j.where(unit==2,1,0),True))
        wing_ok=j.all(j.where(active,ranks<15,ranks==15))&sorted_ok
        wing_ok=wing_ok&j.all(j.where(base.C[body]>0,wing==0,True))
        wing_ok=wing_ok&j.where(unit==1,j.all(wing<=3)&~((wing[13]>0)&(wing[14]>0)),True)
        valid=(s.phase==1)&~s.done&(s.pending<0)&legal_body&wing_ok&j.all(hand>=cards)
        valid=valid&(action<=encode_move(N_BODY-1,j.full(5,15,j.int32)))
        def commit(_):
            ns,reward,done=jax.lax.cond(body==0,
                lambda _:base._step(s,j.int32(0),key),
                lambda _:base.finish_move(s,body,cards),None)
            return ns,reward,done,j.array(False)
        return jax.lax.cond(valid,commit,
            lambda _:(s,j.zeros(3),s.done,j.array(True)),None)
    return jax.lax.cond(action<0,bid,move,None)


batch_reset=jax.jit(jax.vmap(reset))
batch_step=jax.jit(jax.vmap(step))
batch_observe=jax.jit(jax.vmap(observe))
