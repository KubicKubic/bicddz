"""Parity with pinned publisher observations and ResNet weights."""
import sys
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import pytest
import torch
from ddz import env_v4 as env
from ddz.compare_efficiency_douzero import ROLES, make_move_encoder
from ddz.compare_douzero import values_from_counts, counts_from_values
from ddz.douzero_jax import convert_weights, resnet_values, public_observation, cards_array, cards_array_numpy, padded_candidates
from ddz.douzero_resnet_adapter import load_best_models, make_observation
REPO=Path('/tmp/ddz_douzero_resnet_2_0')
OFFICIAL=Path('/tmp/ddz_douzero_official')
WEIGHTS=Path('runs/douzero_resnet_2_0_reference/weights/BEST')
pytestmark=pytest.mark.skipif(not REPO.exists(),reason='Pinned publisher fixture unavailable')
sys.path.insert(0,str(OFFICIAL))

def test_publisher_network_translation():
    torch.set_num_threads(2)
    models,_=load_best_models(WEIGHTS,REPO)
    rng=np.random.default_rng(42)
    z=rng.integers(-1,2,(16,40,54)).astype(np.float32)
    x=rng.normal(size=(16,15)).astype(np.float32)
    for role in ROLES:
        with torch.inference_mode():
            reference=models[role](torch.from_numpy(z),torch.from_numpy(x),return_value=True)['values'][:,0].numpy()
        actual=np.asarray(resnet_values(convert_weights(models[role]),j.asarray(z),j.asarray(x)))
        np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=3e-5)

def test_visible_inputs_passes_bottom_history_and_legal_order():
    from douzero.env import game as gm, move_detector
    from douzero.env.env import DummyAgent
    get_obs=make_observation(REPO)
    encode=make_move_encoder(move_detector)
    step=jax.jit(env.step); observe=jax.jit(public_observation)
    rng=np.random.default_rng(41)
    counts={'events':0,'passes':0,'history_over_32':0,'bottom_removed':0}
    for seed in range(4):
        s=env.reset(jax.random.PRNGKey(seed+100))
        s,_,_,bad=step(s,env.encode_bid(3),jax.random.PRNGKey(0))
        assert not bad
        ll=int(s.landlord)
        game=gm.GameEnv({role:DummyAgent(role) for role in ROLES})
        game.card_play_init({**{role:values_from_counts(np.asarray(s.hands[(ll+r)%3]))
            for r,role in enumerate(ROLES)},'three_landlord_cards':values_from_counts(np.asarray(s.bottom))})
        while not game.game_over:
            role=game.acting_player_position; legal=game.game_infoset.legal_actions
            light=SimpleNamespace(acting_player_position=role,
                info_sets={role:SimpleNamespace(player_hand_cards=values_from_counts(np.asarray(s.hands[int(s.turn)])))},
                card_play_action_seq=game.card_play_action_seq[-2:])
            assert gm.GameEnv.get_legal_card_play_actions(light)==legal
            z,x=map(np.asarray,observe(s)); original=get_obs(game.game_infoset)
            ac=cards_array_numpy([counts_from_values(a) for a in legal])
            actual=np.concatenate((ac[:,None],np.broadcast_to(z,(len(legal),39,54))),axis=1)
            np.testing.assert_array_equal(actual,original['z_batch'])
            np.testing.assert_array_equal(np.broadcast_to(x,(len(legal),15)),original['x_batch'])
            opp=[(ll+r)%3 for r in range(3) if (ll+r)%3 != int(s.turn)]
            h=np.asarray(s.hands).copy()
            r0=np.flatnonzero(h[opp[0]])[0]; r1=np.flatnonzero(h[opp[1]])[-1]
            h[opp[0],r0]-=1; h[opp[1],r1]-=1; h[opp[0],r1]+=1; h[opp[1],r0]+=1
            np.testing.assert_array_equal(np.asarray(observe(s._replace(hands=j.asarray(h)))[0]),z)
            action=legal[int(rng.integers(len(legal)))]
            game.players[role].set_action(action); game.step()
            s,_,done,bad=step(s,encode(tuple(action)),jax.random.PRNGKey(0))
            assert not bad and bool(done)==game.game_over
            counts['events']+=1; counts['passes']+=not action
            counts['history_over_32']+=int(s.hist_len)>32
            counts['bottom_removed']+=len(game.three_landlord_cards)<3
    assert counts['passes']>0 and counts['history_over_32']>0 and counts['bottom_removed']>0

def test_padding_preserves_all_candidates():
    ids,counts=padded_candidates(list(range(131)),np.ones((131,15),np.int32))
    assert ids.shape==(256,) and counts.shape==(256,15)
    np.testing.assert_array_equal(ids[:131],np.arange(131))
    np.testing.assert_array_equal(counts[:131],1)

def test_training_rules_exclude_rocket_attachments_only():
    from douzero.env import move_detector
    from ddz.compare_efficiency_douzero_jax import make_qoj_filter
    select=make_qoj_filter(make_move_encoder(move_detector))
    moves=[[7,7,7,7,20,30],[7,7,7,7,3,4],[20,30],[]]
    assert select(moves)==moves[1:]

def test_complete_catalogue_preserves_ambiguous_planes_and_all_interpretations():
    from ddz import candidates_v3 as cat
    from ddz.compare_efficiency_douzero_jax import make_catalogue_order
    from douzero.env import move_detector
    action=[3,3,3,6,6,6,7,7,7,8,8,8,9,9,9,10]
    assert move_detector.get_move_type(action)['type']==15
    hand=counts_from_values(action)
    ids=cat.legal_ids(hand,1,last=0,last_seat=-1,turn=0)
    order=make_catalogue_order(make_move_encoder(move_detector))(ids,[action])
    assert set(order.tolist())==set(ids.tolist())
    assert len(order)==len(ids)
    selected=[k for k in order if np.array_equal(cat.CARDS[k],hand)]
    assert selected
    s=env.reset(jax.random.PRNGKey(41))
    s=s._replace(hands=s.hands.at[0].set(hand),phase=j.int32(1),turn=j.int32(0),landlord=j.int32(0),bid=j.int32(3))
    step=jax.jit(env.step)
    for k in selected:
        ns,_,done,bad=step(s,int(cat.ACTION[k]),jax.random.PRNGKey(0))
        assert not bad and done and np.all(np.asarray(ns.hands[0])==0)
    # A reply must match the explicit plane body rather than re-detecting the
    # ambiguous card multiset as a different shape.
    body=int(cat.BODY[selected[0]])
    follow=cat.legal_ids(hand,1,last=body,last_seat=1,turn=0)
    assert all(int(cat.BODY[k])!=body for k in follow)

def test_cached_legal_queries_preserve_order_under_passes_and_seat_rotation():
    from ddz import candidates_v3 as cat
    from ddz.compare_efficiency_douzero_jax import make_catalogue_order,make_cached_legals
    from ddz.actions import BODIES,COUNTS
    from douzero.env import game as gm,move_detector
    ordered=make_catalogue_order(make_move_encoder(move_detector))
    cached=make_cached_legals(gm,ordered)
    for seed in range(4):
        hands=np.asarray(env.reset(jax.random.PRNGKey(seed+41)).hands)
        for turn in range(3):
            hand=hands[turn]
            for last_body,sequence in [(0,[]),(1,[[3]]),(1,[[3],[]]),(0,[[],[]])]:
                last_seat=(turn+1)%3
                ids=cat.legal_ids(hand,1,last=last_body,last_seat=last_seat,turn=turn)
                light=SimpleNamespace(acting_player_position='landlord',
                    info_sets={'landlord':SimpleNamespace(player_hand_cards=values_from_counts(hand))},
                    card_play_action_seq=sequence)
                expected=ordered(ids,gm.GameEnv.get_legal_card_play_actions(light))
                actual=cached(hand,last_body,last_seat,turn,sequence)
                np.testing.assert_array_equal(actual,expected)
                # The same visible hand/rival query rotates to another seat.
                np.testing.assert_array_equal(cached(hand,last_body,(last_seat+1)%3,(turn+1)%3,sequence),expected)
