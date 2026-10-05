"""Complete-move and current-deal public history invariants."""
import numpy as np
import jax
import jax.numpy as jnp

from ddz import env_v2
from ddz.actions import BODIES, N_BODY, WING_OFFSET
from ddz.model_v2 import FullAttentionMoveTransformer
from ddz.api_v2 import from_api


def test_preplay_is_vectorized_and_roles_are_explicit():
    key=jax.random.PRNGKey(12)
    state=env_v2.reset(key)
    for value in (1,2,3):
        state,_,_,bad=env_v2.step(state,env_v2.encode_bid(value),key)
        assert not bool(bad)
    assert int(state.phase)==1
    assert int(state.hist_len)==0
    assert tuple(map(int,state.bids))!=( -1,-1,-1)
    for delta in range(3):
        turn=(int(state.landlord)+delta)%3
        obs=env_v2.observe(state._replace(turn=jnp.int32(turn)))
        assert int(jnp.argmax(obs.context[7:11]))==delta


def test_pass_is_a_full_event_with_actor_and_public_snapshot():
    state=env_v2.reset(jax.random.PRNGKey(13))._replace(
        phase=jnp.int32(1),landlord=jnp.int32(0),turn=jnp.int32(1),
        bid=jnp.int32(1),last=jnp.int32(1),last_seat=jnp.int32(0))
    expected=np.asarray(state.hands).sum(axis=1)
    state,_,_,bad=env_v2.step(state,env_v2.encode_move(0,jnp.full(5,15)),jax.random.PRNGKey(14))
    assert not bool(bad)
    assert int(state.hist_len)==1
    event=np.asarray(state.history[0])
    assert event[15]==4 and event[16]==1 and event[42]==2
    assert np.all(event[:15]==0)
    np.testing.assert_array_equal(event[22:25],expected)
    obs=np.asarray(env_v2.observe(state).history[0])
    assert obs[16]==2  # actor 1 relative to current player 2
    assert obs[25]==1  # landlord 0 relative to current player 2


def test_wing_move_is_atomic_and_records_full_cards():
    key=jax.random.PRNGKey(15)
    state=env_v2.reset(key)
    hand=jnp.array([3,1]+[0]*13,jnp.int32)
    state=state._replace(hands=state.hands.at[0].set(hand),turn=jnp.int32(0),
        phase=jnp.int32(1),landlord=jnp.int32(0),bid=jnp.int32(1))
    body=next(i for i,b in enumerate(BODIES) if b.type=='trio1' and b.rank==0)
    move=env_v2.encode_move(body,jnp.array([1,15,15,15,15]))
    atomic,reward,done,bad=env_v2.step(state,move,key)
    assert not bool(bad)
    assert int(atomic.hist_len)==1
    np.testing.assert_array_equal(np.asarray(atomic.history[0,:15]),np.asarray(hand))
    intermediate,_,_,bad=env_v2.micro_step(state,jnp.int32(body),key)
    assert not bool(bad) and int(intermediate.hist_len)==0
    manual,manual_reward,manual_done,bad=env_v2.micro_step(intermediate,jnp.int32(WING_OFFSET+1),key)
    assert not bool(bad)
    np.testing.assert_array_equal(np.asarray(atomic.hands),np.asarray(manual.hands))
    np.testing.assert_array_equal(np.asarray(reward),np.asarray(manual_reward))
    assert bool(done)==bool(manual_done)


def test_opponent_hidden_ranks_do_not_enter_observation():
    state=env_v2.reset(jax.random.PRNGKey(16))._replace(
        phase=jnp.int32(1),landlord=jnp.int32(0),turn=jnp.int32(0))
    before=env_v2.observe(state)
    opponents=state.hands.at[1].set(state.hands[2]).at[2].set(state.hands[1])
    after=env_v2.observe(state._replace(hands=opponents))
    for x,y in zip(before,after):
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))


def test_full_history_attention_respects_order_and_padding():
    state=env_v2.reset(jax.random.PRNGKey(17))._replace(
        phase=jnp.int32(1),landlord=jnp.int32(0),turn=jnp.int32(0))
    first=jnp.array([1]+[0]*14,jnp.int32)
    second=jnp.array([0,1]+[0]*13,jnp.int32)
    state=env_v2.event(state,3,1,first,1)
    state=env_v2.event(state,3,2,second,2)
    obs=env_v2.observe(state)
    batch=lambda x:jax.tree_util.tree_map(lambda a:a[None],x)
    model=FullAttentionMoveTransformer(width=48,layers=1,heads=3,ff=96,
        memory_ff=96,bf16=False)
    params=model.init(jax.random.PRNGKey(18),batch(obs))
    baseline=model.apply(params,batch(obs))
    padded=state._replace(history=state.history.at[5,0].set(3))
    padded_output=model.apply(params,batch(env_v2.observe(padded)))
    for x,y in zip(baseline,padded_output):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-6)
    reversed_history=state.history.at[0].set(state.history[1]).at[1].set(state.history[0])
    reversed_output=model.apply(params,batch(env_v2.observe(state._replace(history=reversed_history))))
    assert np.max(np.abs(np.asarray(baseline[0]-reversed_output[0])))>1e-7


def test_api_reconstructs_training_features_from_public_log():
    key=jax.random.PRNGKey(19)
    state=env_v2.reset(key); log=[]
    for value in (1,2,3):
        actor=int(state.turn)
        state,_,_,bad=env_v2.step(state,env_v2.encode_bid(value),key)
        assert not bool(bad)
        log.append({'kind':'bid','seat':actor,'value':value})
    log.append({'kind':'landlord','seat':int(state.landlord),
                'cards':[(4*r+i if r<13 else r+39) for r,n in enumerate(np.asarray(state.bottom))
                         for i in range(int(n))]})
    body=next(int(i) for i in np.flatnonzero(np.asarray(env_v2.legal(state)[:N_BODY]))
              if i and not BODIES[i].wings)
    actor=int(state.turn)
    state,_,_,bad=env_v2.step(state,env_v2.encode_move(body,jnp.full(5,15)),key)
    assert not bool(bad)
    pools=[[4*r+i for i in range(4)] if r<13 else [r+39] for r in range(15)]
    cards=[]
    for rank,count in enumerate(BODIES[body].cards):
        for _ in range(count): cards.append(pools[rank].pop(0))
    log.append({'kind':'play','seat':actor,'cards':cards,'pattern':{
        'type':BODIES[body].type,'rank':BODIES[body].rank,'len':BODIES[body].length}})
    hand=[]
    for rank,count in enumerate(np.asarray(state.hands[state.turn])):
        hand+=pools[rank][:int(count)]
    raw={'seat':int(state.turn),'turn':int(state.turn),'phase':'playing',
         'players':[{'count':int(x.sum())} for x in np.asarray(state.hands)],
         'hand':hand,'bottom':log[3]['cards'],'landlord':int(state.landlord),
         'bid':int(state.bid),'bombs':int(state.bombs),'redeals':int(state.redeals),
         'leading':bool(state.last==0 or state.last_seat==state.turn),
         'last':{'seat':actor,'pattern':log[-1]['pattern']},'log':log}
    parsed=from_api(raw)
    for x,y in zip(env_v2.observe(state),env_v2.observe(parsed)):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-6)
    passer=int(state.turn)
    state,_,_,bad=env_v2.step(state,env_v2.encode_move(0,jnp.full(5,15)),key)
    assert not bool(bad)
    raw['log'].append({'kind':'pass','seat':passer})
    raw['seat']=raw['turn']=int(state.turn)
    raw['players']=[{'count':int(x.sum())} for x in np.asarray(state.hands)]
    raw['hand']=[pools[rank][i] for rank,count in enumerate(np.asarray(state.hands[state.turn]))
                 for i in range(int(count))]
    raw['leading']=bool(state.last==0 or state.last_seat==state.turn)
    parsed=from_api(raw)
    for x,y in zip(env_v2.observe(state),env_v2.observe(parsed)):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-6)
