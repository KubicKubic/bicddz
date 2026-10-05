"""One model forward, one complete move, one PPO log probability."""
import jax
import jax.numpy as j
from . import env_v2 as env
from .actions import N_BODY, WINGS, WUNIT

WN=j.asarray(WINGS)
WU=j.asarray(WUNIT)


def _entropy(logits):
    lp=jax.nn.log_softmax(logits)
    return -j.sum(j.exp(lp)*lp)


def _pending(state,body):
    return state._replace(pending=body,wing_left=WN[body],wing_min=j.int32(0),
                          wings=j.zeros(15,j.int32))


def _advance(state,body,rank):
    wings=state.wings.at[rank].add(WU[body])
    return state._replace(wings=wings,wing_left=state.wing_left-1,
                          wing_min=rank+j.where(WU[body]==2,1,0))


def sample_one(state,body_logits,wing_logits,key):
    """Sample an atomic packed action and its joint log probability."""
    index=jax.random.categorical(key,body_logits).astype(j.int32)
    body_lp=jax.nn.log_softmax(body_logits)[index]
    body_ent=_entropy(body_logits)
    def bid(_):
        return env.encode_bid(index-N_BODY),body_lp,body_ent
    def move(_):
        body=index
        logits=wing_logits[body]
        state0=_pending(state,body)
        def choose(i,carry):
            state,ranks,logp,ent=carry
            def wing(carry):
                state,ranks,logp,ent=carry
                mask=env.legal_wings(state)
                masked=j.where(mask,logits,-1e9)
                rank=jax.random.categorical(jax.random.fold_in(key,i+1),masked).astype(j.int32)
                lp=jax.nn.log_softmax(masked)[rank]
                return (_advance(state,body,rank),ranks.at[i].set(rank),
                        logp+lp,ent+_entropy(masked))
            return jax.lax.cond(i<WN[body],wing,lambda x:x,carry)
        _,ranks,logp,ent=jax.lax.fori_loop(0,5,choose,
            (state0,j.full(5,15,j.int32),body_lp,body_ent))
        return env.encode_move(body,ranks),logp,ent
    return jax.lax.cond(index>=N_BODY,bid,move,None)


def score_one(state,body_logits,wing_logits,action):
    """Joint log probability and sampled conditional entropy for PPO."""
    index=j.where(action<0,-action-1+N_BODY,action%N_BODY)
    body_lp=jax.nn.log_softmax(body_logits)[index]
    body_ent=_entropy(body_logits)
    def bid(_):
        return body_lp,body_ent
    def move(_):
        body,ranks=env.decode_move(action)
        logits=wing_logits[body]
        state0=_pending(state,body)
        def choose(i,carry):
            state,logp,ent=carry
            def wing(carry):
                state,logp,ent=carry
                mask=env.legal_wings(state)
                masked=j.where(mask,logits,-1e9)
                rank=ranks[i]
                lp=jax.nn.log_softmax(masked)[rank]
                return _advance(state,body,rank),logp+lp,ent+_entropy(masked)
            return jax.lax.cond(i<WN[body],wing,lambda x:x,carry)
        _,logp,ent=jax.lax.fori_loop(0,5,choose,(state0,body_lp,body_ent))
        return logp,ent
    return jax.lax.cond(action<0,bid,move,None)


sample=jax.vmap(sample_one)
score=jax.vmap(score_one)
