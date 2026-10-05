"""Three-seat PPO. Raw terminal scores only; gamma=lambda=1 by default.

Vector returns stay in absolute seat order, including across same-seat internal
wing decisions and bidding. A two-player alternating sign GAE is WRONG here.
The actor uses its own component; farmers naturally share the same payoff.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
import optax
from . import env
from .model import absolute_value
from .env import HISTORY

class Transition(NamedTuple):
    obs: env.Observation
    turn: j.ndarray
    action: j.ndarray
    logp: j.ndarray
    value: j.ndarray
    reward: j.ndarray
    done: j.ndarray


def gae(values,rewards,dones,last,gamma=1.,lam=1.):
    def backward(carry,x):
        adv,vnext=carry
        v,r,d=x
        keep=(~d)[...,None]
        delta=r+gamma*keep*vnext-v
        adv=delta+gamma*lam*keep*adv
        return (adv,v),adv
    _,a=jax.lax.scan(backward,(j.zeros_like(last),last),(values,rewards,dones),reverse=True)
    return a,a+values


def explained_variance(predictions, targets):
    """Variance explained by rollout-time values; zero for constant targets."""
    target_variance=j.var(targets)
    residual_variance=j.var(targets-predictions)
    return j.where(target_variance>1e-8,
                   1.0-residual_variance/j.maximum(target_variance,1e-8),0.0)


def make_rollout(model,horizon):
    def rollout(params,states,key):
        b=states.turn.shape[0]
        def tick(carry,_):
            states,key=carry
            key,ak,sk,rk=jax.random.split(key,4)
            obs=jax.vmap(env.observe)(states)
            logits,v=model.apply({'params':params},obs)
            v=absolute_value(v,states.turn)
            action=jax.random.categorical(ak,logits).astype(j.int32)
            logp=j.take_along_axis(jax.nn.log_softmax(logits),action[:,None],axis=-1)[:,0]
            ns,reward,done,invalid=jax.vmap(env.step)(states,action,jax.random.split(sk,b))
            fresh=jax.vmap(env.reset)(jax.random.split(rk,b))
            next_states=jax.tree_util.tree_map(lambda a,z:j.where(done.reshape((b,)+(1,)*(a.ndim-1)),z,a),ns,fresh)
            tr=Transition(obs,states.turn,action,logp,v,reward,done)
            stats=j.array([j.sum(done),j.sum(invalid),j.sum(done&(states.turn==states.landlord)),
                           j.sum(j.max(j.abs(reward),axis=-1)),j.max(ns.hist_len)])
            return (next_states,key),(tr,stats)
        (states,key),(tr,stats)=jax.lax.scan(tick,(states,key),None,length=horizon)
        _,v=model.apply({'params':params},jax.vmap(env.observe)(states))
        last=absolute_value(v,states.turn)
        return states,key,tr,last,stats
    return jax.jit(rollout)


def make_update(model,cfg,train_memory_limit=HISTORY):
    def apply_train_model(params,obs):
        variables={'params':params}
        if train_memory_limit==HISTORY:
            return model.apply(variables,obs)
        return jax.lax.cond(j.max(obs.hist_len)<=train_memory_limit,
            lambda _:model.apply(variables,obs,memory_length=train_memory_limit),
            lambda _:model.apply(variables,obs),operand=None)

    def loss(params,tr,adv,target,entropy_coef,vf_coef):
        logits,values=apply_train_model(params,tr.obs)
        values=absolute_value(values,tr.turn)
        lp=jax.nn.log_softmax(logits)
        logp=j.take_along_axis(lp,tr.action[:,None],axis=-1)[:,0]
        ratio=j.exp(logp-tr.logp)
        policy=-j.mean(j.minimum(ratio*adv,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*adv))
        # No reward transform, clipping, shaping, auxiliary task or value clipping.
        vl=j.mean(j.square(values-target))
        ent=-j.mean(j.sum(j.where(tr.obs.legal,j.exp(lp)*lp,0),axis=-1))
        kl=j.mean((ratio-1)-(logp-tr.logp))
        total=policy+vf_coef*vl-entropy_coef*ent
        return total,{'loss':total,'policy_loss':policy,'value_loss':vl,'entropy':ent,
                      'kl':kl,'clip_fraction':j.mean(j.abs(ratio-1)>cfg['clip'])}
    grad=jax.value_and_grad(loss,has_aux=True)
    def update(ts,tr,last,key,entropy_coef,vf_coef):
        a,tgt=gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        value_ev=explained_variance(tr.value,tgt)
        acting_value=j.take_along_axis(tr.value,tr.turn[...,None],axis=-1)[...,0]
        acting_target=j.take_along_axis(tgt,tr.turn[...,None],axis=-1)[...,0]
        acting_value_ev=explained_variance(acting_value,acting_target)
        actor=j.take_along_axis(a,tr.turn[...,None],axis=-1)[...,0]
        actor=(actor-actor.mean())/(actor.std()+1e-8)
        n=actor.size
        flat=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:]),tr)
        actor=actor.reshape(n); tgt=tgt.reshape(n,3)
        mb=cfg['minibatch']
        def epoch(carry,ek):
            ts,stop=carry
            idx=jax.random.permutation(ek,n).reshape(-1,mb)
            def minibatch(carry,ix):
                ts,stop=carry
                b=jax.tree_util.tree_map(lambda x:x[ix],flat)
                (_,m),g=grad(ts.params,b,actor[ix],tgt[ix],entropy_coef,vf_coef)
                norm=optax.global_norm(g)
                finite=j.isfinite(norm)&j.isfinite(m['loss'])
                apply=(~stop)&finite
                ts=jax.lax.cond(apply,lambda x:x.apply_gradients(grads=g),lambda x:x,ts)
                stop=stop|(m['kl']>cfg['target_kl'])|~finite
                return (ts,stop),dict(m,grad_norm=norm,applied=apply.astype(j.float32),nonfinite=(~finite).astype(j.float32))
            (ts,stop),m=jax.lax.scan(minibatch,(ts,stop),idx)
            return (ts,stop),jax.tree_util.tree_map(j.mean,m)
        (ts,_),m=jax.lax.scan(epoch,(ts,j.array(False)),jax.random.split(key,cfg['epochs']))
        m=jax.tree_util.tree_map(j.mean,m)
        return ts,dict(m,value_explained_variance=value_ev,
                       acting_value_explained_variance=acting_value_ev)
    return jax.jit(update)


def make_gradient_diagnostic(model,cfg):
    """Separate unweighted policy/value gradients, both full and shared trunk."""
    @jax.jit
    def diagnose(params,tr,last,key):
        a,t=gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        adv=j.take_along_axis(a,tr.turn[...,None],axis=-1)[...,0]
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        n=adv.size
        ix=jax.random.permutation(key,n)[:cfg['minibatch']]
        b=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:])[ix],tr)
        adv=adv.reshape(n)[ix]; target=t.reshape(n,3)[ix]
        def objectives(p):
            logits,v=model.apply({'params':p},b.obs)
            v=absolute_value(v,b.turn)
            lp=j.take_along_axis(jax.nn.log_softmax(logits),b.action[:,None],axis=-1)[:,0]
            ratio=j.exp(lp-b.logp)
            pl=-j.mean(j.minimum(ratio*adv,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*adv))
            vl=j.mean(j.square(v-target))
            return j.stack((pl,vl))
        _,pullback=jax.vjp(objectives,params)
        pg=pullback(j.array([1.,0.]))[0]; vg=pullback(j.array([0.,1.]))[0]
        shared=lambda g:{k:v for k,v in g.items() if not k.startswith(('actor','critic'))}
        return {'policy_grad_l2':optax.global_norm(pg),'value_grad_l2':optax.global_norm(vg),
                'policy_shared_grad_l2':optax.global_norm(shared(pg)),
                'value_shared_grad_l2':optax.global_norm(shared(vg))}
    return diagnose
