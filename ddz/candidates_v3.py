"""Exact complete-move catalogue and public-state legal-set enumeration.

The global table contains every physically possible complete interpretation
with at most 20 cards. Per-state bitset intersections select all and only
legal actions. Batches pad with the pass entry; padding is masked for policy
normalization. No candidate count cap or hint pruning is used.
"""
from itertools import combinations,combinations_with_replacement
import numpy as np

from .actions import (BODIES,COUNTS,TYPE,RANK,LENGTH,WUNIT,WINGS,N_BODY,
                      beats)

RANK_CAP=np.array([4]*13+[1,1],np.uint8)
POW16=np.array([1,16,256,4096,65536],np.int32)


def _catalogue():
    actions=[]; bodies=[]; cards=[]; body_cards=[]; wing_cards=[]
    for body_id,body in enumerate(BODIES):
        base=np.asarray(body.cards,np.uint8)
        n=body.wings
        if not n:
            options=[()]
        elif body.wing_unit==2:
            options=combinations((r for r in range(13) if base[r]==0),n)
        else:
            options=combinations_with_replacement((r for r in range(15) if base[r]==0),n)
        for ranks in options:
            wing=np.zeros(15,np.uint8)
            for r in ranks:wing[r]+=body.wing_unit
            if np.any(wing[:13]>3) or (wing[13] and wing[14]):continue
            total=base+wing
            if np.any(total>RANK_CAP) or int(total.sum())>20:continue
            sequence=np.full(5,15,np.int32);sequence[:n]=ranks
            actions.append(int(body_id+N_BODY*np.dot(sequence,POW16)))
            bodies.append(body_id);cards.append(total);body_cards.append(base);wing_cards.append(wing)
    # Bids live in the same candidate table, after all play actions.
    for bid in range(4):
        actions.append(-bid-1); bodies.append(0)
        cards.append(np.zeros(15,np.uint8));body_cards.append(np.zeros(15,np.uint8))
        wing_cards.append(np.zeros(15,np.uint8))
    return (np.asarray(actions,np.int32),np.asarray(bodies,np.int16),
            np.asarray(cards,np.uint8),np.asarray(body_cards,np.uint8),
            np.asarray(wing_cards,np.uint8))


ACTION,BODY,CARDS,BODY_CARDS,WING_CARDS=_catalogue()
N_PLAY=len(ACTION)-4
N_ALL=len(ACTION)
if BODY[0]!=0 or N_PLAY<1 or len(set(map(int,ACTION)))!=N_ALL:
    raise RuntimeError('complete-move catalogue identity failed')

TYPE_OF=np.asarray(TYPE[BODY],np.uint8)
RANK_OF=np.asarray(RANK[BODY],np.uint8)
LENGTH_OF=np.asarray(LENGTH[BODY],np.uint8)
WING_UNIT_OF=np.asarray(WUNIT[BODY],np.uint8)
WING_COUNT_OF=np.asarray(WINGS[BODY],np.uint8)
BID_OF=np.concatenate((np.zeros(N_PLAY,np.uint8),np.arange(4,dtype=np.uint8)))
IS_BID=np.arange(N_ALL)>=N_PLAY

# A 28k-ish candidate table becomes about 3.5 KB of bitset per threshold.
_packed_threshold=tuple(tuple(np.packbits(CARDS[:N_PLAY,r]<=k,bitorder='little')
                         for k in range(int(RANK_CAP[r])+1))
                         for r in range(15))
_body_follow=tuple(np.packbits(np.fromiter(
    (body==0 or beats(int(body),last) for body in BODY[:N_PLAY]),
    dtype=np.bool_,count=N_PLAY),bitorder='little') for last in range(N_BODY))
_body_lead=np.packbits(BODY[:N_PLAY]!=0,bitorder='little')


def legal_ids(hand,phase,bid=0,bid_count=0,redeals=0,last=0,last_seat=-1,turn=0,done=False):
    """All exact legal catalogue indices for a single decision state."""
    if done:return np.array([0],np.int32)
    if phase==0:
        must=redeals>=3 and bid_count==2 and bid==0
        values=[v for v in range(4) if v>bid or (v==0 and not must)]
        return np.asarray([N_PLAY+v for v in values],np.int32)
    mask=_body_lead.copy() if last==0 or last_seat==turn else _body_follow[int(last)].copy()
    h=np.asarray(hand)
    for r in range(15):
        mask&=_packed_threshold[r][int(h[r])]
    ids=np.flatnonzero(np.unpackbits(mask,bitorder='little')[:N_PLAY]).astype(np.int32)
    if not len(ids):raise RuntimeError('state has no legal complete move')
    return ids


def batch_legal(states):
    """Return (candidate ids, valid mask, counts), padded with pass actions.

    States may be a JAX batch; only small public/own-hand fields cross to CPU.
    The result width is the next power of two above the batch maximum; there
    is no hard cap. This limits JIT specializations without dropping actions.
    """
    import jax
    fields=jax.device_get((states.hands,states.turn,states.phase,states.bid,
        states.bid_count,states.redeals,states.last,states.last_seat,states.done))
    hands,turn,phase,bid,bid_count,redeals,last,last_seat,done=map(np.asarray,fields)
    lists=[legal_ids(hands[i,turn[i]],phase[i],bid[i],bid_count[i],redeals[i],
                     last[i],last_seat[i],turn[i],done[i])
           for i in range(len(turn))]
    counts=np.asarray([len(x) for x in lists],np.int32)
    width=1<<(int(counts.max())-1).bit_length()
    ids=np.zeros((len(lists),width),np.int32)
    mask=np.zeros((len(lists),width),np.bool_)
    for i,entry in enumerate(lists):
        ids[i,:len(entry)]=entry;mask[i,:len(entry)]=True
    return ids,mask,counts
