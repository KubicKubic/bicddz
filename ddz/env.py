"""Pure JAX, rank-exact environment; no hidden cards enter observe().

A move with wings uses internal decisions by the SAME player. Only the complete
move becomes public. Rewards are the actual zero-sum API score.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from .actions import (COUNTS,TYPE,RANK,LENGTH,WUNIT,WINGS,N_BODY,N_ACTIONS,
                      BID_OFFSET,WING_OFFSET,TYPES)
C=j.array(COUNTS); TY=j.array(TYPE); RA=j.array(RANK); LE=j.array(LENGTH)
WU=j.array(WUNIT); WN=j.array(WINGS)
HISTORY=192
EVENT_DIM=22

class State(NamedTuple):
    hands: j.ndarray
    bottom: j.ndarray
    turn: j.ndarray
    phase: j.ndarray  # 0 bid, 1 play, 2 finished
    landlord: j.ndarray
    bid: j.ndarray
    bidder: j.ndarray
    bids: j.ndarray
    bid_count: j.ndarray
    redeals: j.ndarray
    last: j.ndarray
    last_seat: j.ndarray
    passes: j.ndarray
    bombs: j.ndarray
    played: j.ndarray
    plays: j.ndarray
    pending: j.ndarray  # -1 or body id
    wings: j.ndarray
    wing_left: j.ndarray
    wing_min: j.ndarray
    history: j.ndarray
    hist_len: j.ndarray
    done: j.ndarray


def deal(key):
    k1,k2=jax.random.split(key)
    d=jax.random.permutation(k1,54)
    ranks=j.where(d<52,d//4,d-39)
    h=j.sum(jax.nn.one_hot(ranks[:51].reshape(3,17),15,dtype=j.int32),axis=1)
    b=j.sum(jax.nn.one_hot(ranks[51:],15,dtype=j.int32),axis=0)
    return h,b,jax.random.randint(k2,(),0,3)

def reset(key):
    h,b,t=deal(key)
    z=j.int32(0)
    return State(h,b,t,z,j.int32(-1),z,j.int32(-1),j.full(3,-1,j.int32),z,z,
                 z,j.int32(-1),z,z,j.zeros((3,15),j.int32),j.zeros(3,j.int32),
                 j.int32(-1),j.zeros(15,j.int32),z,z,
                 j.zeros((HISTORY,EVENT_DIM),j.uint8),z,j.array(False))

def event(s,kind,seat,cards=None,body=0,value=0):
    cards=j.zeros(15,j.int32) if cards is None else cards
    e=j.concatenate((cards,j.array([kind,seat,TY[body],RA[body],LE[body],value,0]))).astype(j.uint8)
    return s._replace(history=s.history.at[s.hist_len].set(e),hist_len=s.hist_len+1)

def single_capacity(x):
    return j.sum(j.minimum(x[...,:13],3),axis=-1)+j.minimum(j.sum(x[...,13:],axis=-1),1)

def legal(s):
    h=s.hands[s.turn]
    remain=j.where(C==0,h[None,:],0)
    cap=j.where(WU==1,single_capacity(remain),j.sum(remain>=2,axis=-1))
    available=j.all(h>=C,axis=-1)&(cap>=WN)
    last=s.last
    stronger=((TY==TY[last])&(LE==LE[last])&(RA>RA[last]))
    stronger=stronger|((TY==13)&(TY[last]!=13)&(TY[last]!=14))|((TY==14)&(TY[last]!=14))
    leading=(last==0)|(s.last_seat==s.turn)
    body=available&(leading|stronger)
    body=body.at[0].set(~leading)
    must=(s.redeals>=3)&(s.bid_count==2)&(s.bid==0)
    bid=((j.arange(4)>s.bid)|((j.arange(4)==0)&~must))
    # Sorted wing selection with exact completion lookahead; never a dead end.
    p=j.maximum(s.pending,0)
    rem=j.where(C[p]==0,h-s.wings,0)
    unit=WU[p]
    caprank=j.where(unit==1,j.minimum(rem,3-s.wings), (rem>=2).astype(j.int32))
    caprank=j.where(j.arange(15)>=s.wing_min,caprank,0)
    caprank=caprank.at[13:].set(j.where(j.sum(s.wings[13:])>0,0,caprank[13:]))
    def possible(r):
        after=caprank.at[r].add(-1)
        after=j.where(j.arange(15)>=r+j.where(unit==2,1,0),after,0)
        after=after.at[13:].set(j.where(r>=13,0,after[13:]))
        capacity=j.where(unit==1,single_capacity(after),j.sum(after))
        return (caprank[r]>0)&(capacity>=s.wing_left-1)
    wing=jax.vmap(possible)(j.arange(15))
    out=j.concatenate((body,j.zeros(19,j.bool_)))
    out=j.where(s.phase==0,j.concatenate((j.zeros(N_BODY,j.bool_),bid,j.zeros(15,j.bool_))),out)
    out=j.where(s.pending>=0,j.concatenate((j.zeros(WING_OFFSET,j.bool_),wing)),out)
    return out&~s.done

def finish_move(s,body,cards):
    t=s.turn
    h=s.hands.at[t].add(-cards)
    plays=s.plays.at[t].add(1)
    bombs=s.bombs+((TY[body]==13)|(TY[body]==14)).astype(j.int32)
    done=j.sum(h[t])==0
    won=t==s.landlord
    spring=won&(j.sum(j.where(j.arange(3)!=s.landlord,plays,0))==0)
    anti=(~won)&(plays[s.landlord]==1)
    mult=1+bombs+spring.astype(j.int32)+anti.astype(j.int32)
    sign=j.where(won,1.,-1.)
    payoff=j.where(j.arange(3)==s.landlord,2.,-1.)*sign*s.bid*mult
    ns=event(s,3,t,cards,body)
    ns=ns._replace(hands=h,played=s.played.at[t].add(cards),plays=plays,bombs=bombs,
                   turn=(t+1)%3,last=body,last_seat=t,passes=j.int32(0),
                   pending=j.int32(-1),wings=j.zeros(15,j.int32),wing_left=j.int32(0),
                   wing_min=j.int32(0),done=done,phase=j.where(done,2,1))
    return ns,j.where(done,payoff,j.zeros(3)),done

def _step(s,a,key):
    def bidding(_):
        value=a-BID_OFFSET
        ns=event(s,1,s.turn,value=value)
        ns=ns._replace(bid=j.maximum(s.bid,value),bidder=j.where(value>0,s.turn,s.bidder),
                       bids=s.bids.at[s.turn].set(value),bid_count=s.bid_count+1,turn=(s.turn+1)%3)
        end=(value==3)|(ns.bid_count==3)
        def close(ns):
            def redeal(ns):
                h,b,t=deal(key)
                return event(ns,5,0)._replace(hands=h,bottom=b,turn=t,bid=j.int32(0),
                    bidder=j.int32(-1),bids=j.full(3,-1,j.int32),bid_count=j.int32(0),redeals=s.redeals+1)
            def landlord(ns):
                return event(ns,2,ns.bidder,ns.bottom,value=ns.bid)._replace(
                    hands=ns.hands.at[ns.bidder].add(ns.bottom),landlord=ns.bidder,
                    turn=ns.bidder,phase=j.int32(1))
            return jax.lax.cond(ns.bid==0,redeal,landlord,ns)
        ns=jax.lax.cond(end,close,lambda x:x,ns)
        return ns,j.zeros(3),j.array(False)
    def playing(_):
        def attach(_):
            r=a-WING_OFFSET
            p=s.pending
            wings=s.wings.at[r].add(WU[p])
            ns=s._replace(wings=wings,wing_left=s.wing_left-1,wing_min=r+j.where(WU[p]==2,1,0))
            return jax.lax.cond(ns.wing_left==0,lambda _:finish_move(ns,p,C[p]+wings),
                                lambda _:(ns,j.zeros(3),j.array(False)),None)
        def body(_):
            def passed(_):
                ns=event(s,4,s.turn)
                ns=ns._replace(turn=(s.turn+1)%3,passes=s.passes+1,
                    last=j.where(s.passes==1,0,s.last))
                return ns,j.zeros(3),j.array(False)
            def play(_):
                def start(_):
                    return s._replace(pending=a,wing_left=WN[a],wing_min=j.int32(0)),j.zeros(3),j.array(False)
                return jax.lax.cond(WN[a]>0,start,lambda _:finish_move(s,a,C[a]),None)
            return jax.lax.cond(a==0,passed,play,None)
        return jax.lax.cond(s.pending>=0,attach,body,None)
    return jax.lax.cond(s.phase==0,bidding,playing,None)

def step(s,a,key):
    """Invalid actions do not mutate state. Return state,reward,done,invalid."""
    valid=(a>=0)&(a<N_ACTIONS)&legal(s)[j.clip(a,0,N_ACTIONS-1)]
    ns,r,d=jax.lax.cond(valid,lambda _:_step(s,a,key),lambda _:(s,j.zeros(3),s.done),None)
    return ns,r,d,~valid

class Observation(NamedTuple):
    ranks: j.ndarray
    context: j.ndarray
    history: j.ndarray
    hist_len: j.ndarray
    legal: j.ndarray

def observe(s):
    t=s.turn
    order=(j.arange(3)+t)%3
    known=j.where(s.phase>0,s.bottom,0)
    played=s.played[order]
    unseen=j.array([4]*13+[1,1])-s.hands[t]-j.sum(played,axis=0)
    p=j.maximum(s.pending,0)
    ranks=j.stack((s.hands[t],known,played[0],played[1],played[2],unseen,
                   j.where(s.pending>=0,C[p],0),s.wings),axis=-1).astype(j.float32)/4
    role=j.where(s.landlord>=0,(s.landlord-t)%3,3)
    last_seat=j.where((s.last_seat>=0)&(s.last!=0),(s.last_seat-t)%3,3)
    ctx=j.concatenate((jax.nn.one_hot(s.phase,3),jax.nn.one_hot(role,4),
        j.sum(s.hands,axis=-1)[order]/20,s.bids[order]/3,
        j.array([s.bid/3,s.bid_count/3,s.redeals/3,s.bombs/10,s.passes/2,
                 s.wing_left/5,s.wing_min/15,(s.pending>=0).astype(j.float32)]),
        jax.nn.one_hot(TY[s.last],15),j.array([RA[s.last]/14,LE[s.last]/12]),
        jax.nn.one_hot(last_seat,4),s.plays[order]/20,
        jax.nn.one_hot(TY[p],15)))
    # Relative seats in history; masked padding has no effect.
    hist=s.history.at[:,16].set(((s.history[:,16].astype(j.int32)-t)%3).astype(j.uint8))
    return Observation(ranks,ctx,hist,s.hist_len,legal(s))

batch_reset=jax.jit(jax.vmap(reset))
batch_step=jax.jit(jax.vmap(step))
batch_observe=jax.jit(jax.vmap(observe))
