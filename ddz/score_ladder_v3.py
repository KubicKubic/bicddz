"""Role-mirrored expected-score matches for exact candidate-set policies."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from functools import partial
import numpy as np
import jax
import jax.numpy as j

from . import env_v2 as env
from . import candidates_v3 as moves


def make_score_evaluator(model, chunk_deals, memory_limit=88):
    """Return candidate scores with each deal played in both team roles."""
    games=2*chunk_deals
    candidate_landlord=np.arange(games)<chunk_deals

    @partial(jax.jit, static_argnames=('memory_length',))
    def forward(params, obs, ids, mask, memory_length):
        return model.apply({'params':params},obs,ids,mask,
                           memory_length=memory_length)[0]

    def evaluate(params_a,params_b,seed,deal_offset):
        key=jax.random.PRNGKey(seed)
        key,deal_key,bid_key=jax.random.split(key,3)
        states=env.batch_reset(jax.random.split(deal_key,chunk_deals))
        states=states._replace(turn=(j.arange(chunk_deals,dtype=j.int32)+deal_offset)%3)
        states,_,_,bad=env.batch_step(states,j.full((chunk_deals,),env.encode_bid(3)),
                                      jax.random.split(bid_key,chunk_deals))
        states=jax.tree_util.tree_map(lambda x:j.concatenate((x,x),axis=0),states)
        finished=np.zeros(games,np.bool_)
        scores=np.zeros(games,np.float32)
        invalid=np.concatenate((np.asarray(bad),np.asarray(bad))).astype(np.int32)
        for decision in range(192):
            if finished.all():break
            ids_np,mask_np,_=moves.batch_legal(states)
            ids=j.asarray(ids_np);mask=j.asarray(mask_np)
            obs=jax.vmap(env.observe)(states)
            memory=memory_limit if int(j.max(states.hist_len))<=memory_limit else env.HISTORY
            logits_a=forward(params_a,obs,ids,mask,memory)
            logits_b=forward(params_b,obs,ids,mask,memory)
            actions_a=np.asarray(j.argmax(logits_a,axis=-1))
            actions_b=np.asarray(j.argmax(logits_b,axis=-1))
            turns=np.asarray(states.turn);landlords=np.asarray(states.landlord)
            owns_a=np.where(candidate_landlord,turns==landlords,turns!=landlords)
            chosen=np.where(owns_a,actions_a,actions_b)
            actions=j.asarray(moves.ACTION[ids_np[np.arange(games),chosen]])
            key,step_key=jax.random.split(key)
            ns,reward,done,bad=env.batch_step(states,actions,
                                               jax.random.split(step_key,games))
            terminal=np.asarray(done)&~finished
            payoff=np.asarray(j.take_along_axis(reward,states.landlord[:,None],axis=1)[:,0])
            scores=np.where(terminal,np.where(candidate_landlord,payoff,-payoff),scores)
            invalid+=np.asarray(bad&j.asarray(~finished),np.int32)
            finished|=np.asarray(done)
            states=ns
        return scores,finished,invalid,decision+1
    return evaluate
