import itertools
import numpy as np
import pytest
import jax
import jax.numpy as j
from ddz import env
from ddz.actions import *
from ddz.reference import classify,score

LEGAL=jax.jit(env.legal); STEP=jax.jit(env.step)

def state(hand=None):
    s=env.reset(jax.random.PRNGKey(1))._replace(phase=j.int32(1),landlord=j.int32(0),
          turn=j.int32(0),bid=j.int32(2))
    if hand is not None: s=s._replace(hands=s.hands.at[0].set(j.array(hand,j.int32)))
    return s

def ident(t,r,n=1):
    return next(i for i,b in enumerate(BODIES) if (b.type,b.rank,b.length)==(t,r,n))

def act(s,a): return STEP(s,j.int32(a),jax.random.PRNGKey(3))

def test_deal_and_hidden_information():
    s=env.reset(jax.random.PRNGKey(41))
    np.testing.assert_array_equal(s.hands.sum(0)+s.bottom,[4]*13+[1,1])
    np.testing.assert_array_equal(s.hands.sum(1),[17]*3)
    s=s._replace(turn=j.int32(0))
    alt=s._replace(hands=s.hands.at[1:].set(j.roll(s.hands[1:],1,axis=1)),bottom=j.roll(s.bottom,1))
    for a,b in zip(env.observe(s),env.observe(alt)): np.testing.assert_array_equal(a,b)

def test_bids_redeal_and_force():
    s=env.reset(jax.random.PRNGKey(1))
    for deal in range(3):
        for k in range(3):
            s,r,d,bad=act(s,BID_OFFSET); assert not bad and not d and not np.any(r)
        assert int(s.redeals)==deal+1 and s.phase==0
    s,_,_,_=act(s,BID_OFFSET); s,_,_,_=act(s,BID_OFFSET)
    assert not LEGAL(s)[BID_OFFSET]
    ns,r,d,bad=act(s,BID_OFFSET); assert bad
    np.testing.assert_array_equal(ns.hands,s.hands)
    t=int(s.turn); s,_,_,_=act(s,BID_OFFSET+1)
    assert s.landlord==t and s.turn==t and s.hands[t].sum()==20
    assert s.phase==1 and s.bid==1

@pytest.mark.parametrize('value',[1,2,3])
def test_highest_bid(value):
    s=env.reset(jax.random.PRNGKey(4)); t=int(s.turn)
    s,_,_,_=act(s,BID_OFFSET+value)
    if value<3:
        assert not LEGAL(s)[BID_OFFSET+value]
        s,_,_,_=act(s,BID_OFFSET); s,_,_,_=act(s,BID_OFFSET)
    assert s.landlord==t and s.bid==value and s.turn==t

def test_pass_reset_and_bombs():
    s=state([1]*12+[0,0,0]); s,_,_,bad=act(s,0); assert bad
    s,_,_,_=act(s,ident('single',0)); assert not LEGAL(s)[ident('single',0)]
    s,_,_,_=act(s,0); s,_,_,_=act(s,0)
    assert s.turn==0 and s.last==0 and not LEGAL(s)[0]

def test_exact_scores_and_no_intermediate_reward():
    for landlord in range(3):
        for winner in range(3):
            for bombs in [0,1,3]:
                for plays in [[0,0,0],[1,2,2],[2,1,0]]:
                    s=state()._replace(landlord=j.int32(landlord),turn=j.int32(winner),
                        bombs=j.int32(bombs),plays=j.array(plays,j.int32))
                    h=j.zeros(15,j.int32).at[0].set(1)
                    s=s._replace(hands=s.hands.at[winner].set(h))
                    ns,r,d,bad=act(s,ident('single',0))
                    pp=plays.copy(); pp[winner]+=1
                    assert d and not bad
                    np.testing.assert_array_equal(r,score(landlord,winner,2,bombs,pp))
                    assert r.sum()==0
    # Two bombs mean x3; spring adds one, NOT doubling.
    s=state([4]+[0]*14)._replace(bombs=j.int32(1))
    ns,r,d,bad=act(s,ident('bomb',0))
    np.testing.assert_array_equal(r,[16,-8,-8])

def test_ambiguous_and_illegal_patterns():
    c=np.array([3]*4+[0]*11)
    expected={('plane',3,4),('plane1',3,3),('plane1',2,3)}
    assert classify(c)==expected
    assert {(BODIES[i].type,BODIES[i].rank,BODIES[i].length) for i in interpretations(c)}==expected
    for c in [[4,4]+[0]*13,[4]+[0]*12+[1,1],[3,3,4]+[0]*12]: assert not classify(c)
    assert ('four2',0,1) in classify([4,2]+[0]*13)

def test_all_bodies_and_random_interpretations_against_oracle():
    rng=np.random.default_rng(5)
    seen=set()
    for i,b in enumerate(BODIES[1:],1):
        for _ in range(3):
            c=COUNTS[i].copy()
            eligible=[r for r in range(13) if c[r]==0]
            if b.wings:
                for r in rng.choice(eligible,b.wings,replace=False): c[r]+=b.wing_unit
            expected=classify(c)
            got={(BODIES[k].type,BODIES[k].rank,BODIES[k].length) for k in interpretations(c)}
            assert got==expected
            assert (b.type,b.rank,b.length) in got
            seen.add(b.type)
    assert seen==set(TYPES)-{'pass'}
    for _ in range(300):
        c=np.zeros(15,np.int32)
        for r in rng.choice(15,rng.integers(1,8),replace=False): c[r]=rng.integers(1,5) if r<13 else 1
        if c.sum()>20: continue
        got={(BODIES[k].type,BODIES[k].rank,BODIES[k].length) for k in interpretations(c)}
        assert got==classify(c)

def test_wings_complete_no_deadends_and_no_hidden_publication():
    # Exhaustively compare all subsets of adversarial small hands to reachable
    # factorized actions. Includes repeated singles, pair wings and both jokers.
    for hand in [[4,3,3,2]+[0]*9+[1,1],[3,3,3,3]+[0]*11,[4,4,2,2]+[0]*11]:
        s=state(hand); got=set()
        def walk(ss,body):
            mask=np.asarray(LEGAL(ss))
            assert mask.any()
            for a in np.flatnonzero(mask):
                ns,r,d,bad=act(ss,int(a)); assert not bad
                if ns.pending>=0:
                    assert ns.hist_len==s.hist_len
                    walk(ns,body)
                else:
                    c=np.asarray(ss.hands[0]-ns.hands[0])
                    got.add((tuple(c),(BODIES[body].type,BODIES[body].rank,BODIES[body].length)))
        for a in np.flatnonzero(np.asarray(LEGAL(s))):
            ns,r,d,bad=act(s,int(a)); assert not bad
            if ns.pending>=0: walk(ns,int(a))
            else: got.add((tuple(np.asarray(s.hands[0]-ns.hands[0])),(BODIES[a].type,BODIES[a].rank,BODIES[a].length)))
        expected=set()
        for subset in itertools.product(*(range(n+1) for n in hand)):
            for pat in classify(subset): expected.add((subset,pat))
        assert got==expected

def test_random_batched_games_invariants():
    batch=256
    @jax.jit
    def run(key):
        key,r=jax.random.split(key)
        s=jax.vmap(env.reset)(jax.random.split(r,batch))
        done=j.zeros(batch,j.bool_); pay=j.zeros((batch,3)); errors=j.zeros(5,j.int32)
        def tick(carry,_):
            s,key,done,pay,errors=carry
            key,a,k=jax.random.split(key,3)
            mask=jax.vmap(env.legal)(s)
            action=jax.random.categorical(a,j.where(mask,0.,-1e9)).astype(j.int32)
            ns,reward,d,bad=jax.vmap(env.step)(s,action,jax.random.split(k,batch))
            # Inventory includes unrevealed bottom during bidding only.
            inventory=ns.hands.sum(1)+ns.played.sum(1)+j.where((ns.phase==0)[:,None],ns.bottom,0)
            errors+=j.array([j.sum(bad&~done),j.sum(j.any(inventory!=j.array([4]*13+[1,1]),axis=-1)),
                j.sum(j.any(ns.hands<0,axis=(1,2))),j.sum(j.any(reward!=0,axis=-1)&~d),
                j.sum(ns.hist_len>=env.HISTORY)])
            pay+=j.where(done[:,None],0,reward)
            return (ns,key,done|d,pay,errors),None
        result,_=jax.lax.scan(tick,(s,key,done,pay,errors),None,length=320)
        return result
    s,key,done,pay,errors=run(jax.random.PRNGKey(42))
    assert np.all(done); np.testing.assert_array_equal(errors,0)
    np.testing.assert_array_equal(pay.sum(-1),0)
    for n in range(batch):
        ss=jax.tree_util.tree_map(lambda x:np.asarray(x[n]),s)
        winner=(int(ss.turn)+2)%3
        np.testing.assert_array_equal(pay[n],score(int(ss.landlord),winner,int(ss.bid),int(ss.bombs),ss.plays))
        # Independently validate every publicly played interpretation.
        for e in ss.history[:int(ss.hist_len)]:
            if e[15]==3: assert (TYPES[e[17]],int(e[18]),int(e[19])) in classify(e[:15])

def test_beating_rules_and_rejected_actions_are_noops():
    for a,b,expected in [('bomb','single',True),('rocket','bomb',True),
                         ('bomb','rocket',False),('pair','single',False)]:
        aa=ident(a,14 if a=='rocket' else 3); bb=ident(b,14 if b=='rocket' else 2)
        assert beats(aa,bb)==expected
    s=state()
    for a in [-1,N_ACTIONS,10000]:
        ns,r,d,bad=act(s,a); assert bad and not np.any(r)
        for x,y in zip(s,ns): np.testing.assert_array_equal(x,y)
