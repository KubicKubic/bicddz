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
from .model_v5 import InteractionMoveTransformer
from .policy_v5 import greedy_one
from .api import Client,run,BODY_BY_PATTERN

def displayed_values(relative,seat,landlord,own_clock):
    """Other seats share the final team payoff; own-clock heads lack own targets."""
    if not own_clock:return np.roll(relative,seat).tolist()
    own=float(relative[0])
    if landlord is None:
        result=[None]*3;result[seat]=own;return result
    landlord_value=own if seat==landlord else -2*own
    return [landlord_value if i==landlord else -landlord_value/2 for i in range(3)]


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
    if bid!=raw['bid'] or bombs!=raw.get('bombs',0):
        raise ValueError('public score state disagrees with log')
    # The live server clears its consecutive-redeal counter once a landlord
    # is selected. Training retains the auction's count in vectorized context.
    # Keep that log-derived model input, and accept the verified live reset.
    allowed_redeals=(0,redeals) if raw['phase']=='playing' else (redeals,)
    if raw.get('redeals',0) not in allowed_redeals:
        raise ValueError('public redeal count disagrees with log')
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
        params=serialization.msgpack_restore(Path(checkpoint).read_bytes())
        self._initialize(cfg,params)

    def _initialize(self,cfg,params):
        self.config=cfg
        self.own_value_clock=cfg.get('ppo',{}).get('gae_clock','public')=='own'
        model_type=InteractionMoveTransformer
        if {'belief_head','team_value'}&set(cfg['model']):
            from .model_efficiency import EfficientMoveTransformer
            model_type=EfficientMoveTransformer
        self.model=model_type(**cfg['model'])
        self.params=jax.tree_util.tree_map(j.asarray,params)
        # Weights must be a JIT argument: closing over self.params would keep
        # using the first compiled checkpoint after a live weight replacement.
        self._forward=jax.jit(lambda params,s:self.model.apply({'params':params},
            jax.tree_util.tree_map(lambda x:x[None],env.observe(s))))
        self.choose=jax.jit(greedy_one)
        self.observe=jax.jit(env.observe)
        self.capture_observation=False
        self.last_inference=None
        # Compile both the network and complete-action decoder before queueing.
        initial=env.reset(jax.random.PRNGKey(0))
        logits,context,_=self.forward(initial)
        context=jax.tree_util.tree_map(lambda x:x[0],context)
        self.choose(initial,logits[0],context).block_until_ready()
        self.observe(initial).ranks.block_until_ready()
        self.last_bid_diagnostics=None

    def prepare_replacement(self,params,config):
        """Warm an explicitly supported architecture upgrade off the game thread."""
        from .upgrade_v6 import validate_growth
        if config.get('model_family')!='V6':
            raise ValueError('Unregistered model family upgrade')
        if config.get('ppo',{}).get('gae_clock','public')!=self.config.get('ppo',{}).get('gae_clock','public'):
            raise ValueError('Architecture upgrade changes value semantics')
        validate_growth(self.config['model'],config['model'])
        model=InteractionMoveTransformer(**config['model'])
        initial=env.reset(jax.random.PRNGKey(0))
        obs=jax.tree_util.tree_map(lambda x:x[None],env.observe(initial))
        expected=jax.eval_shape(lambda:model.init(jax.random.PRNGKey(1),obs,memory_length=4)['params'])
        before,structure=jax.tree_util.tree_flatten(expected)
        after,new_structure=jax.tree_util.tree_flatten(params)
        if structure!=new_structure:
            raise ValueError('Replacement parameter tree differs from configured model')
        if sum(x.size for x in after)>config.get('parameter_limit',8_100_000):
            raise ValueError('Replacement parameter cap exceeded')
        for target,value in zip(before,after):
            if target.shape!=value.shape or target.dtype!=value.dtype or not np.all(np.isfinite(value)):
                raise ValueError('Replacement parameter shape/dtype/finite check failed')
        candidate=type(self).__new__(type(self))
        candidate._initialize(config,params)
        candidate.params=candidate.prepare_params(candidate.params,config)
        candidate.capture_observation=self.capture_observation
        return candidate

    def forward(self,state):
        return self._forward(self.params,state)

    def prepare_params(self,params,config):
        """Validate and warm new weights without changing the serving policy."""
        if (config.get('model')!=self.config.get('model') or
            config.get('ppo',{}).get('gae_clock','public')!=
            self.config.get('ppo',{}).get('gae_clock','public')):
            raise ValueError('Checkpoint model or value semantics differ from serving policy')
        old,structure=jax.tree_util.tree_flatten(self.params)
        new,new_structure=jax.tree_util.tree_flatten(params)
        if structure!=new_structure:
            raise ValueError('Checkpoint parameter tree differs from serving policy')
        for before,after in zip(old,new):
            if before.shape!=after.shape or before.dtype!=after.dtype:
                raise ValueError('Checkpoint parameter shape/dtype differs from serving policy')
            if not np.all(np.isfinite(after)):
                raise ValueError('Non-finite checkpoint parameter')
        params=jax.tree_util.tree_map(j.asarray,params)
        initial=env.reset(jax.random.PRNGKey(0))
        for state in (initial,initial._replace(phase=j.int32(1),landlord=j.int32(0))):
            output=self._forward(params,state)
            if not all(np.all(np.isfinite(np.asarray(leaf)))
                       for leaf in jax.tree_util.tree_leaves(output)):
                raise ValueError('Non-finite checkpoint inference')
        return params

    def decide(self,raw):
        self.last_inference=None
        self.last_bid_diagnostics=None
        state=from_api(raw)
        logits,context,value=self.forward(state)
        relative_value=np.asarray(value[0],np.float64)
        if not np.all(np.isfinite(relative_value)) or not np.all(np.isfinite(np.asarray(logits))):
            raise ValueError('Non-finite model output')
        context=jax.tree_util.tree_map(lambda x:x[0],context)
        action=int(self.choose(state,logits[0],context));version=int(raw['version'])
        absolute_value=np.roll(relative_value,int(raw['seat']))
        self.last_inference={'game':raw.get('id'),'version':version,'seat':raw['seat'],
            'value_relative':relative_value.tolist(),'value_by_seat':absolute_value.tolist(),
            'own_value':float(relative_value[0]),
            'display_value_by_seat':displayed_values(relative_value,int(raw['seat']),
                raw.get('landlord'),self.own_value_clock),
            'value_display_source':'own critic + exact team payoff ratio' if self.own_value_clock else 'three supervised critic heads',
            'value_training_clock':'own' if self.own_value_clock else 'public',
            'value_semantics':'Predicted final raw score delta, not win probability',
            'history_events':int(state.hist_len),'selected_action':action}
        if self.capture_observation:
            obs=jax.device_get(self.observe(state))
            self.last_inference['observation']={'ranks':obs.ranks.tolist(),
                'context':obs.context.tolist(),'history':obs.history[:int(obs.hist_len)].tolist(),
                'hist_len':int(obs.hist_len),'legal_indices':np.flatnonzero(obs.legal).tolist(),
                'relative_seats':[int((raw['seat']+i)%3) for i in range(3)]}
        if action<0:
            scores=np.asarray(logits[0,N_BODY:N_BODY+4],np.float64)
            probabilities=np.exp(scores-scores.max())
            probabilities/=probabilities.sum()
            self.last_bid_diagnostics={'bid_probabilities_0_1_2_3':probabilities.tolist(),
                'highest_bid':int(raw['bid']),'must_bid':bool(raw.get('must_bid',False)),
                'selected_bid':-action-1,'own_hand':raw['hand']}
            return 'bid',{'version':version,'value':-action-1}
        body,ranks=env.decode_move(j.int32(action));body=int(body)
        if body==0:return 'pass',{'version':version}
        wings=np.zeros(15,np.int32)
        for rank in np.asarray(ranks)[:WINGS[body]]:wings[rank]+=WUNIT[body]
        return 'play',{'version':version,'cards':physical_cards(raw['hand'],COUNTS[body]+wings,
            raw.get('bottom') or ()), 'choice':BODIES[body].choice}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    credentials=ap.add_mutually_exclusive_group()
    credentials.add_argument('--token-env',default='DDZ_API_KEY')
    credentials.add_argument('--token-file',type=Path,
                             help='Read a Bearer key from a private file')
    ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--mode',choices=['single','match'],default='single')
    ap.add_argument('--once',action='store_true')
    args=ap.parse_args()
    token=(args.token_file.read_text().strip() if args.token_file else os.environ[args.token_env])
    if not token: ap.error('API key is empty')
    run(Client(args.base,token),
        Policy(args.config,args.checkpoint),args.mode,args.once)


if __name__=='__main__': main()
