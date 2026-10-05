"""Version-safe API adapter. Only public state and the player's own hand are used."""
import argparse
import json
import os
import time
from pathlib import Path
import urllib.request
import urllib.error
import http.client
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from . import env
from .actions import (BODIES,COUNTS,TYPES,WUNIT,WINGS,BID_OFFSET,WING_OFFSET,
                      counts_of,physical_cards)
from .model import MemoryTransformer

BODY_BY_PATTERN={(b.type,b.rank,b.length):i for i,b in enumerate(BODIES)}

class RetryDecision(Exception):
    """Refresh authoritative state without restarting a compiled policy."""

class ProtocolError(RetryDecision):
    """A truncated or incomplete response cannot be used for a decision."""

class APIError(RuntimeError):
    def __init__(self,code,message):
        self.code=code
        super().__init__(f'API HTTP {code}: {message}')

class TemporaryAPIError(RetryDecision):
    def __init__(self,code,delay=1):
        self.code=code; self.delay=delay
        super().__init__(f'Temporary API HTTP {code}')

TEMPORARY_HTTP={408,425,429,500,502,503,504,520,521,522,523,524}

def response_state(result,previous=None):
    """Check executor fields before accessing them or submitting an action."""
    state=result.get('state') if isinstance(result,dict) else None
    if not isinstance(state,dict): raise ProtocolError('Response has no state object')
    if state.get('unchanged'):
        if previous is None or state.get('version')!=previous.get('version'):
            raise ProtocolError('Unchanged response has no matching full state')
        return {**previous,**{k:v for k,v in state.items() if k!='unchanged'}}
    if state.get('phase')=='finished':
        result=state.get('result')
        if (not isinstance(result,dict) or not isinstance(result.get('deltas'),list) or
            len(result['deltas'])!=3 or any(type(v) is not int for v in result['deltas'])):
            raise ProtocolError('Finished state has no settlement')
        return state
    if state.get('phase') not in ('bidding','playing'):
        raise ProtocolError('Unknown or missing phase')
    players=state.get('players'); seat=state.get('seat'); turn=state.get('turn')
    if (not isinstance(players,list) or any(not isinstance(p,dict) for p in players) or type(seat) is not int or
        not 0<=seat<len(players) or type(turn) is not int or not 0<=turn<3 or
        not isinstance(players[seat],dict) or type(players[seat].get('auto')) is not bool or
        type(state.get('version')) is not int):
        raise ProtocolError('Incomplete active state')
    return state

def checked_request(client,path,payload=None):
    code,data=client.request(path,payload)
    if not isinstance(data,dict): raise ProtocolError('Response is not an object')
    if code in TEMPORARY_HTTP:
        raise TemporaryAPIError(code,data.get('retry_after',1))
    if code not in (200,409): raise APIError(code,str(data.get('error','request rejected')))
    return code,data

def body_id(pattern):
    return BODY_BY_PATTERN[(pattern['type'],pattern['rank'],pattern['len'])]

def from_api(raw):
    """Rebuild explicit memory on every request: safe on restart, 409 or redeal.

    Fake opponent rank arrays contain ONLY the public count. observe() must
    never use their rank contents. Completed-game revealed hands are ignored.
    """
    seat=raw['seat']
    if seat is None or raw['phase']=='finished': raise ValueError('no player decision available')
    if raw['turn']!=seat: raise ValueError('not our turn')
    h=np.zeros((3,15),np.int32)
    for i,p in enumerate(raw['players']): h[i,0]=p['count']
    h[seat]=counts_of(raw['hand'])
    bottom=counts_of(raw.get('bottom') or [])
    hist=np.zeros((env.HISTORY,env.EVENT_DIM),np.uint8)
    played=np.zeros((3,15),np.int32); plays=np.zeros(3,np.int32)
    bids=np.full(3,-1,np.int32); bid_count=0; last_bidder=-1; passes=0; n=0
    for e in raw.get('log',[]):
        kind=e['kind']; t=e.get('seat',0); cards=np.zeros(15,np.int32); body=0
        if kind=='bid':
            k=1; bids[t]=e['value']; bid_count+=1
            if e['value']>0: last_bidder=t
        elif kind=='landlord': k=2; cards=counts_of(e['cards'])
        elif kind=='play':
            k=3; cards=counts_of(e['cards']); body=body_id(e['pattern'])
            played[t]+=cards; plays[t]+=1; passes=0
        elif kind=='pass': k=4; passes+=1
        elif kind=='redeal':
            k=5; bids[:]=-1; bid_count=0; last_bidder=-1; played[:]=0; plays[:]=0; passes=0
        else: continue  # auto toggles handled by executor; finish has no decision
        b=BODIES[body]
        if n>=env.HISTORY: raise ValueError('public history exceeds proven game bound')
        hist[n]=np.array(list(cards)+[k,t,TYPES.index(b.type),b.rank,b.length,e.get('value',0),int(e.get('auto',False))])
        n+=1
    last=raw.get('last')
    lid=0 if last is None or raw.get('leading',False) else body_id(last['pattern'])
    landlord=raw.get('landlord')
    values=(h,bottom,seat,0 if raw['phase']=='bidding' else 1,
        -1 if landlord is None else landlord,raw['bid'],last_bidder,bids,bid_count,
        raw.get('redeals',0),lid,-1 if last is None else last['seat'],passes,
        raw.get('bombs',0),played,plays,-1,np.zeros(15,np.int32),0,0,hist,n,False)
    s=env.State(*(j.asarray(v,j.uint8 if i==20 else j.bool_ if i==22 else j.int32) for i,v in enumerate(values)))
    if raw['phase']=='bidding':
        expected=bool(raw.get('must_bid',False))
        actual=bool((s.redeals>=3)&(s.bid_count==2)&(s.bid==0))
        if expected!=actual: raise ValueError('must_bid disagrees with complete public log')
    return s

class Policy:
    def __init__(self,config,checkpoint):
        cfg=json.loads(Path(config).read_text())
        self.model=MemoryTransformer(**cfg['model'])
        self.params=serialization.msgpack_restore(Path(checkpoint).read_bytes())
        self.forward=jax.jit(lambda s:self.model.apply({'params':self.params},
             jax.tree_util.tree_map(lambda x:x[None],env.observe(s))))
        # Compile before entering a queue: first JIT must not spend the turn clock.
        self.forward(env.reset(jax.random.PRNGKey(0)))[0].block_until_ready()

    def decide(self,raw):
        s=from_api(raw); logits,_=self.forward(s); a=int(j.argmax(logits[0]))
        version=int(raw['version'])
        if s.phase==0: return 'bid',{'version':version,'value':a-BID_OFFSET}
        if a==0: return 'pass',{'version':version}
        body=a; wings=np.zeros(15,np.int32)
        s=s._replace(pending=j.int32(body),wing_left=j.int32(WINGS[body]),wing_min=j.int32(0))
        for _ in range(WINGS[body]):
            logits,_=self.forward(s); r=int(j.argmax(logits[0]))-WING_OFFSET
            wings[r]+=int(WUNIT[body])
            s=s._replace(wings=j.array(wings),wing_left=s.wing_left-1,
                         wing_min=j.int32(r+(WUNIT[body]==2)))
        return 'play',{'version':version,'cards':physical_cards(raw['hand'],COUNTS[body]+wings,
                       raw.get('bottom') or ()),
                       'choice':BODIES[body].choice}

class Client:
    def __init__(self,base,token,timeout=5):
        self.base=base.rstrip('/'); self.token=token; self.timeout=timeout
    def request(self,path,payload=None):
        req=urllib.request.Request(self.base+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json',
                     'Accept':'application/json','User-Agent':'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as r:
                try: data=json.load(r)
                except (ValueError,UnicodeError):
                    raise ProtocolError('Non-JSON success response; refresh state') from None
                if not isinstance(data,dict): raise ProtocolError('API response is not an object')
                return r.status,data
        except urllib.error.HTTPError as e:
            try:
                data=json.load(e)
            except (ValueError,UnicodeError):
                data={'error':'non-JSON response'}
            if not isinstance(data,dict): data={'error':'non-object response'}
            if 'error' in data:data['error']=str(data['error']).replace(self.token,'[REDACTED]')
            if e.code in TEMPORARY_HTTP:
                try: delay=float(e.headers.get('Retry-After',1))
                except (TypeError,ValueError): delay=1
                data['retry_after']=max(.1,min(5,delay))
                return e.code,data
            if e.code==409: return e.code,data
            # Never print authorization headers.
            message=str(data.get('error','request rejected')).replace(self.token,'[REDACTED]')
            raise APIError(e.code,message) from None

def run(client,policy,mode='single',once=False):
    game=None; state=None; last_finished=None
    def recovery(error):
        callback=getattr(client,'on_recovery',None)
        if callback: callback(error,game)
    while True:
        submitted=None
        try:
            if game is None:
                code,me=checked_request(client,'/me')  # heartbeat every second while queued
                if 'game' not in me: raise ProtocolError('Lobby response has no game field')
                game=me.get('game')
                if game is not None and game==last_finished:
                    game=None; time.sleep(1); continue
                if game is None:
                    if not me.get('queued'):
                        code,queued=checked_request(client,'/queue',{'mode':mode})
                        game=queued.get('game')
                    if game is None: time.sleep(1); continue
                state=None
            if state is None:
                code,result=checked_request(client,f'/games/{game}')
                state=response_state(result)
            if state['phase']=='finished':
                print(json.dumps({'game':game,'result':state.get('result')}),flush=True)
                if once: return
                last_finished=game; game=None; state=None; continue
            # Cancel auto immediately, including while another player acts.
            # Waiting for our turn races the server's 1.2-second auto move.
            if state['players'][state['seat']]['auto']:
                code,result=checked_request(client,f'/games/{game}/auto',
                    {'version':int(state['version']),'on':False})
                state=response_state(result)
                continue
            if state['turn']!=state['seat']:
                version=state['version']; time.sleep(.3)
                code,result=checked_request(client,f'/games/{game}?version={version}')
                state=response_state(result,state)
                continue
            endpoint,payload=policy.decide(state)
            submitted=(state,endpoint,payload)
            code,result=checked_request(client,f'/games/{game}/{endpoint}',payload)
            # 409 has authoritative state: recompute, never replay stale actions.
            fresh=response_state(result)
            if code==409 and fresh.get('version')==state['version']:
                callback=getattr(policy,'on_rejected',None)
                if callback: callback(state,endpoint,payload,result.get('error'))
                recovery(RetryDecision('Action rejected at unchanged version'))
                time.sleep(.1)
            state=fresh
        except APIError as error:
            if error.code==401: raise  # invalid credentials need operator attention
            recovery(error)
            if error.code==400 and submitted:
                old,endpoint,payload=submitted
                callback=getattr(policy,'on_rejected',None)
                if callback: callback(old,endpoint,payload,str(error))
            if error.code in (403,404): game=None
            state=None; time.sleep(1)
        except TemporaryAPIError as error:
            recovery(error); state=None; time.sleep(error.delay)
        except RetryDecision as error:
            recovery(error); state=None; time.sleep(.1)
        except (TimeoutError,OSError,urllib.error.URLError,http.client.HTTPException) as error:
            # POST may already have committed: GET authoritative state next.
            recovery(error)
            state=None; time.sleep(1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True,help='https://site/api/v1/doudizhu')
    ap.add_argument('--token-env',default='DDZ_API_KEY')
    ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--mode',choices=['single','match'],default='single'); ap.add_argument('--once',action='store_true')
    args=ap.parse_args()
    policy=Policy(args.config,args.checkpoint)
    run(Client(args.base,os.environ[args.token_env]),policy,args.mode,args.once)
if __name__=='__main__': main()
