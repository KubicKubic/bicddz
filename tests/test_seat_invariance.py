"""Absolute seat labels must not change a fixed relative DDZ position."""
import copy
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from ddz import env_v2 as env
from ddz.actions import BODIES,N_BODY
from ddz.api_v5 import from_api,displayed_values

def rotate_api(raw,shift):
    raw=copy.deepcopy(raw)
    rotate=lambda seat:(seat+shift)%3 if type(seat) is int and 0<=seat<3 else seat
    for field in ('seat','turn','landlord'):raw[field]=rotate(raw.get(field))
    for field in ('players','table','reserve','hands'):
        if isinstance(raw.get(field),list) and len(raw[field])==3:
            raw[field]=raw[field][-shift:]+raw[field][:-shift] if shift else raw[field]
    if raw.get('last'):raw['last']['seat']=rotate(raw['last']['seat'])
    for entry in raw.get('log',()):
        if 'seat' in entry:entry['seat']=rotate(entry['seat'])
    return raw

def rotate_state(state,shift):
    rotate=lambda seat:j.where(seat>=0,(seat+shift)%3,seat).astype(j.int32)
    history=np.asarray(state.history).copy();n=int(state.hist_len)
    for column in (env.EVENT_ACTOR,env.EVENT_LANDLORD,env.EVENT_LAST_ACTOR,env.EVENT_NEXT_ACTOR):
        history[:n,column]=np.where(history[:n,column]==3,3,(history[:n,column]+shift)%3)
    for column in (env.EVENT_REMAIN,env.EVENT_BIDS,env.EVENT_PLAYS):
        history[:n,column]=np.roll(history[:n,column],shift,axis=-1)
    return state._replace(hands=j.roll(state.hands,shift,axis=0),
        played=j.roll(state.played,shift,axis=0),plays=j.roll(state.plays,shift),
        bids=j.roll(state.bids,shift),turn=rotate(state.turn),landlord=rotate(state.landlord),
        bidder=rotate(state.bidder),last_seat=rotate(state.last_seat),history=j.asarray(history))

def positions():
    """Legal auction/play/pass transitions across all roles and history lengths."""
    step=jax.jit(env.step);legal=jax.jit(env.legal)
    out=[]
    for seed in (71,72):
        key=jax.random.PRNGKey(seed);state=env.reset(key)
        out.append(state)
        forced=state._replace(redeals=j.int32(3),bid_count=j.int32(2),
            bids=j.zeros(3,j.int32).at[state.turn].set(-1))
        out.append(forced)
        state,_,_,bad=step(state,env.encode_bid(3),key);assert not bool(bad)
        for i in range(48):
            out.append(state)
            mask=np.asarray(legal(state))[:N_BODY]
            body=0 if mask[0] else next(i for i in np.flatnonzero(mask) if BODIES[i].type=='single')
            state,_,done,bad=step(state,env.encode_move(body,j.full(5,15,j.int32)),key)
            assert not bool(bad) and not bool(done)
    return out

def assert_observations_equal(left,right):
    for name in left._fields:
        np.testing.assert_array_equal(np.asarray(getattr(left,name)),np.asarray(getattr(right,name)))

def test_all_visible_features_are_invariant_to_absolute_seat_numbers():
    observe=jax.jit(env.observe)
    states=positions()
    assert {int((s.turn-s.landlord)%3) for s in states if s.phase==1}=={0,1,2}
    assert max(int(s.hist_len) for s in states)>=45
    for state in states:
        base=observe(state)
        for shift in (1,2):assert_observations_equal(base,observe(rotate_state(state,shift)))

def test_api_rotation_preserves_redeal_history_and_forced_bid_context():
    path=Path(__file__).parent/'fixtures/qoj_redeal_landlord_reset_26894.json'
    raw=json.loads(path.read_text());base=env.observe(from_api(raw))
    for shift in (1,2):assert_observations_equal(base,env.observe(from_api(rotate_api(raw,shift))))
    raw.update(phase='bidding',bottom=None,landlord=None,bid=0,redeals=3,
               must_bid=True,leading=True,last=None)
    raw['players']=[{'count':17} for _ in range(3)];raw['log']=[]
    for _ in range(3):
        raw['log'] += [{'kind':'bid','seat':seat,'value':0} for seat in range(3)]
        raw['log'].append({'kind':'redeal'})
    raw['log'] += [{'kind':'bid','seat':seat,'value':0} for seat in (1,2)]
    base=env.observe(from_api(raw))
    for shift in (1,2):assert_observations_equal(base,env.observe(from_api(rotate_api(raw,shift))))


def test_hidden_opponent_ranks_and_unrevealed_bottom_cannot_enter_any_input_tensor():
    observe=jax.jit(env.observe)
    for state in positions()[::7]:
        changed=state.hands
        for seat in range(3):
            if seat!=int(state.turn):changed=changed.at[seat].set(j.roll(changed[seat],seat+1))
        altered=state._replace(hands=changed)
        if state.phase==0:altered=altered._replace(bottom=j.roll(state.bottom,4))
        assert_observations_equal(observe(state),observe(altered))
    path=Path(__file__).parent/'fixtures/qoj_redeal_landlord_reset_26894.json'
    raw=json.loads(path.read_text());before=observe(from_api(raw))
    raw['hands']=[[53]*20]*3
    raw['fairness']={'deals':[{'deck':list(reversed(range(54)))}]}
    raw['chat']=['hidden text']
    for player in raw['players']:player['username']='different identity'
    assert_observations_equal(before,observe(from_api(raw)))


def test_other_displayed_scores_follow_team_payoffs_without_opponent_hands():
    for own in (-12.,3.,0.):
        for seat in range(3):
            for landlord in range(3):
                value=displayed_values(np.array([own,100.,-100.-own]),seat,landlord,True)
                assert value[seat]==own
                farmers=[i for i in range(3) if i!=landlord]
                assert value[farmers[0]]==value[farmers[1]]==-value[landlord]/2
                assert sum(value)==0
            bidding=displayed_values(np.array([own,123.,-123.-own]),seat,None,True)
            assert bidding[seat]==own and sum(x is None for x in bidding)==2
    np.testing.assert_array_equal(displayed_values(np.array([1.,2.,-3.]),2,1,False),[2.,-3.,1.])
