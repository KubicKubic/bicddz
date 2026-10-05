"""Full candidate enumeration and padded cross-attention invariants."""
from itertools import product
import numpy as np
import jax
import jax.numpy as j

from ddz import candidates_v3 as candidates,env_v2,reference
from ddz.actions import BODIES,COUNTS,N_BODY
from ddz.model_v3 import CandidateSetTransformer


BODY_BY_PATTERN={(b.type,b.rank,b.length):i for i,b in enumerate(BODIES)}


def oracle_moves(hand,last=None):
    expected=set()
    for values in product(*(range(int(c)+1) for c in hand)):
        cards=np.asarray(values,np.int32)
        if not cards.any():continue
        for pattern in reference.classify(cards):
            if last is not None and not reference.beats(pattern,last):continue
            body=BODY_BY_PATTERN[pattern]
            wing=cards-COUNTS[body]
            if np.any(wing<0):continue
            unit=BODIES[body].wing_unit
            ranks=[] if not unit else [r for r,n in enumerate(wing)
                                   for _ in range(int(n)//unit)]
            if len(ranks)!=BODIES[body].wings:continue
            ranks+= [15]*(5-len(ranks))
            expected.add(int(body+N_BODY*sum(r*16**i for i,r in enumerate(ranks))))
    if last is not None:
        expected.add(int(candidates.ACTION[0]))
    return expected


def test_catalogue_is_exact_against_independent_small_hand_oracle():
    rng=np.random.default_rng(21)
    for _ in range(18):
        ids=rng.choice(54,size=9,replace=False)
        hand=np.bincount(np.where(ids<52,ids//4,ids-39),minlength=15)
        actual=set(map(int,candidates.ACTION[candidates.legal_ids(hand,1)]))
        assert actual==oracle_moves(hand)
        last=('single',int(rng.integers(0,13)),1)
        last_id=BODY_BY_PATTERN[last]
        actual=set(map(int,candidates.ACTION[candidates.legal_ids(
            hand,1,last=last_id,last_seat=1,turn=0)]))
        assert actual==oracle_moves(hand,last)


def test_dense_hand_has_all_candidates_without_truncation():
    hand=np.array([1,1,1,1,1,3,3,3,3,1,1,1,0,0,0],np.int32)
    ids=candidates.legal_ids(hand,1)
    assert len(ids)>=500
    assert len(np.unique(candidates.ACTION[ids]))==len(ids)


def test_follow_rules_match_oracle_for_bombs_rocket_and_sequence_lengths():
    hand=np.array([4,1,1,1,1,1,2,0,0,0,0,0,0,1,1],np.int32)
    patterns=[('single',7,1),('pair',5,1),('straight',4,5),
              ('bomb',4,1),('rocket',14,1)]
    for pattern in patterns:
        last=BODY_BY_PATTERN[pattern]
        ids=candidates.legal_ids(hand,1,last=last,last_seat=1,turn=0)
        actual=set(map(int,candidates.ACTION[ids]))
        assert actual==oracle_moves(hand,pattern)


def test_pass_padding_does_not_change_valid_scores_or_value():
    state=env_v2.reset(jax.random.PRNGKey(31))
    obs=jax.tree_util.tree_map(lambda x:x[None],env_v2.observe(state))
    ids=j.array([[candidates.N_PLAY,candidates.N_PLAY+1]],j.int32)
    mask=j.array([[True,True]])
    model=CandidateSetTransformer(width=48,layers=1,heads=3,ff=96,
        memory_ff=96,candidate_layers=1,candidate_ff=96,bf16=False)
    params=model.init(jax.random.PRNGKey(32),obs,ids,mask)
    logits,value=model.apply(params,obs,ids,mask)
    padded=j.pad(ids,((0,0),(0,3)))
    valid=j.pad(mask,((0,0),(0,3)))
    padded_logits,padded_value=model.apply(params,obs,padded,valid)
    np.testing.assert_allclose(np.asarray(logits),np.asarray(padded_logits[:,:2]),atol=1e-6)
    np.testing.assert_allclose(np.asarray(value),np.asarray(padded_value),atol=1e-6)
    assert np.all(np.asarray(padded_logits[:,2:])<-1e8)


def test_v3_api_prefers_revealed_bottom_card_id(monkeypatch):
    from ddz import api_v3
    state=env_v2.reset(jax.random.PRNGKey(54))
    monkeypatch.setattr(api_v3,'from_api',lambda raw:state)
    single=next(i for i in range(candidates.N_PLAY)
                if candidates.TYPE_OF[i]==1 and candidates.RANK_OF[i]==1)
    policy=object.__new__(api_v3.Policy)
    policy._score=lambda _:single
    kind,payload=policy.decide({'version':7,'hand':[4,5,6,7,12],
                                'bottom':[7]})
    assert kind=='play' and payload['cards']==[7]
