"""API adapter for the complete-move, current-deal-history policy."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization

from . import env_v2 as env
from .actions import (BODIES,COUNTS,TYPES,WUNIT,WINGS,N_BODY,
                      counts_of,physical_cards)
from .model_v4 import CompactMoveTransformer
from .api import Client,run,BODY_BY_PATTERN


def from_api(raw):
    """Rebuild current-deal play/pass history from public log on every turn."""
    seat=raw['seat']
    if seat is None or raw['phase']=='finished': raise ValueError('no player decision available')
    if raw['turn']!=seat: raise ValueError('not our turn')
    h=np.zeros((3,15),np.int32)
    for i,p in enumerate(raw['players']): h[i,0]=p['count']
    h[seat]=counts_of(raw['hand'])
    bottom=counts_of(raw.get('bottom') or [])
    hist=np.zeros((env.HISTORY,env.EVENT_DIM),np.uint8)
    played=np.zeros((3,15),np.int32); plays=np.zeros(3,np.int32)
    bids=np.full(3,-1,np.int32); bid_count=0; bidder=-1; bid=0
    landlord=-1; redeals=0; bombs=0; passes=0; last=0; last_seat=-1
    remaining=np.full(3,17,np.int32); n=0
    for e in raw.get('log',[]):
        kind=e['kind']; actor=e.get('seat',3)
        if kind=='bid':
            value=int(e['value']); bids[actor]=value; bid_count+=1
            if value>bid: bid=value; bidder=actor
            continue
        if kind=='landlord':
            landlord=actor; remaining[actor]+=3
            continue
        if kind=='redeal':
            bids[:]=-1; bid_count=0; bidder=-1; bid=0
            played[:]=0; plays[:]=0; passes=0; last=0; last_seat=-1
            remaining[:]=17; landlord=-1; bombs=0; n=0; hist[:]=0
            redeals+=1
            continue
        if kind not in ('play','pass'): continue
        cards=np.zeros(15,np.int32); body=0
        if kind=='play':
            cards=counts_of(e['cards'])
            body=BODY_BY_PATTERN[(e['pattern']['type'],e['pattern']['rank'],e['pattern']['len'])]
            played[actor]+=cards; remaining[actor]-=int(cards.sum()); plays[actor]+=1
            bombs+=int(BODIES[body].type in ('bomb','rocket'))
            passes=0; last=body; last_seat=actor
        else:
            passes+=1
            if passes==2: last=0
        if n>=env.HISTORY: raise ValueError('play history exceeds bound')
        b=BODIES[body]
        event=list(cards)+[3 if kind=='play' else 4,actor,TYPES.index(b.type),
            b.rank,b.length,0,int(e.get('auto',False))]
        event+=list(remaining)+[3 if landlord<0 else landlord,bid,bombs,passes,
            3 if last==0 else last_seat,TYPES.index(BODIES[last].type),
            BODIES[last].rank,BODIES[last].length,redeals,1,bid_count]
        event+=list(bids+1)+list(plays)+[(actor+1)%3]
        hist[n]=np.asarray(event,np.uint8); n+=1
    server_landlord=raw.get('landlord')
    if (None if landlord<0 else landlord)!=server_landlord:
        raise ValueError('landlord disagrees with public log')
    if bid!=raw['bid'] or redeals!=raw.get('redeals',0) or bombs!=raw.get('bombs',0):
        raise ValueError('public score state disagrees with log')
    if list(remaining)!=[p['count'] for p in raw['players']]:
        raise ValueError('remaining card counts disagree with public log')
    raw_last=raw.get('last')
    lid=0 if raw_last is None or raw.get('leading',False) else BODY_BY_PATTERN[
        (raw_last['pattern']['type'],raw_last['pattern']['rank'],raw_last['pattern']['len'])]
    if lid!=last and not (raw.get('leading',False) and last_seat==seat):
        raise ValueError('last move disagrees with public log')
    values=(h,bottom,seat,0 if raw['phase']=='bidding' else 1,
        landlord,bid,bidder,bids,bid_count,redeals,lid,last_seat,passes,bombs,
        played,plays,-1,np.zeros(15,np.int32),0,0,hist,n,False)
    state=env.State(*(j.asarray(v,j.uint8 if i==20 else j.bool_ if i==22 else j.int32)
                      for i,v in enumerate(values)))
    if raw['phase']=='bidding':
        must=bool((state.redeals>=3)&(state.bid_count==2)&(state.bid==0))
        if must!=bool(raw.get('must_bid',False)):
            raise ValueError('must_bid disagrees with public log')
    return state


class Policy:
    def __init__(self,config,checkpoint):
        cfg=json.loads(Path(config).read_text())
        self.model=CompactMoveTransformer(**cfg['model'])
        self.params=serialization.msgpack_restore(Path(checkpoint).read_bytes())
        self.forward=jax.jit(lambda s:self.model.apply({'params':self.params},
            jax.tree_util.tree_map(lambda x:x[None],env.observe(s))))
        self.forward(env.reset(jax.random.PRNGKey(0)))[0].block_until_ready()

    def decide(self,raw):
        state=from_api(raw)
        body_logits,wing_logits,_=self.forward(state)
        index=int(j.argmax(body_logits[0])); version=int(raw['version'])
        if int(state.phase)==0:
            return 'bid',{'version':version,'value':index-N_BODY}
        if index==0: return 'pass',{'version':version}
        body=index; wings=np.zeros(15,np.int32)
        state=state._replace(pending=j.int32(body),wing_left=j.int32(WINGS[body]),
                             wing_min=j.int32(0))
        for _ in range(WINGS[body]):
            mask=env.legal_wings(state)
            rank=int(j.argmax(j.where(mask,wing_logits[0,body],-1e9)))
            wings[rank]+=int(WUNIT[body])
            state=state._replace(wings=j.array(wings),wing_left=state.wing_left-1,
                                 wing_min=j.int32(rank+(WUNIT[body]==2)))
        return 'play',{'version':version,'cards':physical_cards(raw['hand'],COUNTS[body]+wings,
                       raw.get('bottom') or ()),
                       'choice':BODIES[body].choice}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    ap.add_argument('--token-env',default='DDZ_API_KEY')
    ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--mode',choices=['single','match'],default='single')
    ap.add_argument('--once',action='store_true')
    args=ap.parse_args()
    run(Client(args.base,os.environ[args.token_env]),
        Policy(args.config,args.checkpoint),args.mode,args.once)


if __name__=='__main__': main()
