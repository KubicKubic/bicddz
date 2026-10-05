"""One complete action and joint logp, with learned conditional wing scores."""
from typing import NamedTuple
import math
import numbers
import jax
import jax.numpy as j
from . import env_v2 as env
from .actions import N_BODY, COUNTS, WINGS, WUNIT
from .policy_v2 import _entropy, _pending, _advance

WN=j.asarray(WINGS)
WU=j.asarray(WUNIT)
C=j.asarray(COUNTS)


def validate_random_action_prob(value):
    """Static config probability; missing fields default to zero at callers."""
    if (isinstance(value,bool) or not isinstance(value,numbers.Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('ppo.random_action_prob must be a finite number in [0, 1]')
    return float(value)


class WingContext(NamedTuple):
    base: j.ndarray
    ranks: j.ndarray
    state: j.ndarray
    bodies: j.ndarray
    remaining_kernel: j.ndarray
    output: j.ndarray


def _selected_wing_scores(state,context,body):
    """Score all next ranks against the hypothetical hand after that choice.

    Earlier selections change learned preferences, as well as the exact legal
    mask. Everything is computed from public context and our own hand.
    """
    remaining=(state.hands[state.turn]-C[body]-state.wings).astype(j.float32)
    selected=state.wings.astype(j.float32)@context.ranks/5
    body_pool=C[body].astype(j.float32)@context.ranks/20
    post_pool=(remaining@context.ranks)[None]/20-WU[body]*context.ranks/20
    post_counts=(remaining@context.remaining_kernel)[None]/4-WU[body]*context.remaining_kernel/4
    query=context.state+context.bodies+selected+body_pool
    hidden=jax.nn.gelu(query[None]+context.ranks+post_pool+post_counts)
    delta=hidden@context.output/(context.ranks.shape[-1]**.5)
    return context.base+delta


def select_body(context,body):
    # Gather once before the conditional scan. Reverse-mode then scatters the
    # combined gradient once, without a 309-body adjoint on every wing choice.
    return context._replace(base=context.base[body],bodies=context.bodies[body])


def wing_scores(state,context,body):
    return _selected_wing_scores(state,select_body(context,body),body)


def _decide(state,logits,context,key,action=None,greedy=False,uniform=False):
    if uniform:
        logits=j.where(env.legal(state)[:N_BODY+4],0.,-1e9)
    index=(j.argmax(logits) if greedy else jax.random.categorical(key,logits)).astype(j.int32)
    if action is not None:
        index=j.where(action<0,-action-1+N_BODY,action%N_BODY)
    logp=jax.nn.log_softmax(logits)[index]
    entropy=_entropy(logits)
    def bid(_):
        return env.encode_bid(index-N_BODY),logp,entropy
    def move(_):
        body=index
        selected_context=select_body(context,body)
        specified=env.decode_move(action)[1] if action is not None else j.full(5,15,j.int32)
        def choose(i,carry):
            current,ranks,lp,ent=carry
            def wing(carry):
                current,ranks,lp,ent=carry
                scores=j.zeros(15) if uniform else _selected_wing_scores(current,selected_context,body)
                masked=j.where(env.legal_wings(current),scores,-1e9)
                rank=(j.argmax(masked) if greedy else jax.random.categorical(
                    jax.random.fold_in(key,i+1),masked)).astype(j.int32)
                if action is not None: rank=specified[i]
                return (_advance(current,body,rank),ranks.at[i].set(rank),
                    lp+jax.nn.log_softmax(masked)[rank],ent+_entropy(masked))
            return jax.lax.cond(i<WN[body],wing,lambda x:x,carry)
        _,ranks,lp,ent=jax.lax.fori_loop(0,5,choose,
            (_pending(state,body),j.full(5,15,j.int32),logp,entropy))
        return env.encode_move(body,ranks),lp,ent
    return jax.lax.cond(index>=N_BODY,bid,move,None)


def _mixture_logp(policy_logp,random_logp,prob):
    if prob == 1:
        return random_logp
    return j.logaddexp(j.log1p(-prob)+policy_logp,j.log(prob)+random_logp)


def sample_one(state,logits,context,key,random_action_prob=0.):
    """One branch draw per complete action, never per individual wing.

    The random branch is uniform at each legal tree node, NOT uniform over
    complete leaves. Both branches have exact, normalized joint probabilities.
    Zero preserves the original RNG stream, results and computation graph.
    """
    if random_action_prob == 0:
        return _decide(state,logits,context,key)
    branch_key,action_key=jax.random.split(key)
    if random_action_prob == 1:
        action=_decide(state,logits,context,action_key,uniform=True)[0]
    else:
        action=jax.lax.cond(jax.random.bernoulli(branch_key,float(random_action_prob)),
            lambda _:_decide(state,logits,context,action_key,uniform=True)[0],
            lambda _:_decide(state,logits,context,action_key)[0],None)
    logp,entropy=score_one(state,logits,context,action,random_action_prob)
    return action,logp,entropy


def score_one(state,logits,context,action,random_action_prob=0.):
    key=jax.random.PRNGKey(0)
    if random_action_prob == 0:
        return _decide(state,logits,context,key,action=action)[1:]
    random_lp=_decide(state,logits,context,key,action=action,uniform=True)[1]
    if random_action_prob == 1:
        logp=random_lp
    else:
        policy_lp=_decide(state,logits,context,key,action=action)[1]
        logp=_mixture_logp(policy_lp,random_lp,random_action_prob)
    # For mixed policies this is sampled surprisal, not conditional entropy.
    # PPO applies importance weighting to obtain the mixture entropy gradient.
    return logp,-logp


def greedy_one(state,logits,context):
    return _decide(state,logits,context,jax.random.PRNGKey(0),greedy=True)[0]


def sample(states,logits,context,keys,random_action_prob=0.):
    return jax.vmap(lambda s,l,c,k:sample_one(s,l,c,k,random_action_prob))(
        states,logits,context,keys)


def score(states,logits,context,actions,random_action_prob=0.):
    return jax.vmap(lambda s,l,c,a:score_one(s,l,c,a,random_action_prob))(
        states,logits,context,actions)
greedy=jax.vmap(greedy_one)
