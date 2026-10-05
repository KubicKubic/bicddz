"""Role mirrored raw-score matches for complete-move V2 checkpoints."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import jax.numpy as j

from . import env_v2 as env
from .actions import N_BODY,WINGS,WUNIT

WN=j.array(WINGS); WU=j.array(WUNIT)


def greedy_one(state,body_logits,wing_logits):
    index=j.argmax(body_logits).astype(j.int32)
    def bid(_): return env.encode_bid(index-N_BODY)
    def move(_):
        body=index
        state0=state._replace(pending=body,wing_left=WN[body],
            wing_min=j.int32(0),wings=j.zeros(15,j.int32))
        def choose(i,carry):
            state,ranks=carry
            def wing(carry):
                state,ranks=carry
                rank=j.argmax(j.where(env.legal_wings(state),wing_logits[body],-1e9)).astype(j.int32)
                wings=state.wings.at[rank].add(WU[body])
                return (state._replace(wings=wings,wing_left=state.wing_left-1,
                    wing_min=rank+j.where(WU[body]==2,1,0)),ranks.at[i].set(rank))
            return jax.lax.cond(i<WN[body],wing,lambda x:x,carry)
        _,ranks=jax.lax.fori_loop(0,5,choose,(state0,j.full(5,15,j.int32)))
        return env.encode_move(body,ranks)
    return jax.lax.cond(index>=N_BODY,bid,move,None)


greedy=jax.vmap(greedy_one)


def make_score_evaluator(model,chunk_deals):
    """Return candidate terminal team payoff for each role mirrored deal."""
    batch=2*chunk_deals
    a_is_landlord=j.concatenate((j.ones(chunk_deals,j.bool_),
                                  j.zeros(chunk_deals,j.bool_)))
    def action(params,states,obs):
        variables={'params':params}
        logits,wings,_=jax.lax.cond(j.max(obs.hist_len)<=96,
            lambda _:model.apply(variables,obs,memory_length=96),
            lambda _:model.apply(variables,obs),None)
        return greedy(states,logits,wings)
    @jax.jit
    def evaluate(params_a,params_b,seed,deal_offset):
        key=jax.random.PRNGKey(seed)
        key,deal_key,bid_key=jax.random.split(key,3)
        states=jax.vmap(env.reset)(jax.random.split(deal_key,chunk_deals))
        states=states._replace(turn=(j.arange(chunk_deals,dtype=j.int32)+deal_offset)%3)
        states,_,_,bad=jax.vmap(env.step)(states,j.full((chunk_deals,),env.encode_bid(3)),
                                         jax.random.split(bid_key,chunk_deals))
        states=jax.tree_util.tree_map(lambda x:j.concatenate((x,x),axis=0),states)
        finished=j.zeros(batch,j.bool_); score=j.zeros(batch,j.float32)
        invalid=j.concatenate((bad,bad)).astype(j.int32)
        def condition(carry):
            return j.any(~carry[2])&(carry[5]<192)
        def body(carry):
            states,key,finished,score,invalid,steps=carry
            key,step_key=jax.random.split(key)
            obs=jax.vmap(env.observe)(states)
            action_a=action(params_a,states,obs)
            action_b=action(params_b,states,obs)
            a_turn=j.where(a_is_landlord,states.turn==states.landlord,
                           states.turn!=states.landlord)
            actions=j.where(a_turn,action_a,action_b)
            ns,reward,done,bad=jax.vmap(env.step)(states,actions,
                jax.random.split(step_key,batch))
            landlord_payoff=j.take_along_axis(reward,states.landlord[:,None],axis=1)[:,0]
            payoff=j.where(a_is_landlord,landlord_payoff,-landlord_payoff)
            score=j.where(done&~finished,payoff,score)
            return ns,key,finished|done,score,invalid+(bad&~finished).astype(j.int32),steps+1
        result=jax.lax.while_loop(condition,body,
            (states,key,finished,score,invalid,j.int32(0)))
        return result[3],result[2],result[4],result[5]
    return evaluate
