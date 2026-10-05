"""Device-resident PPO over compressed exact complete moves."""
from typing import NamedTuple
import jax
import jax.numpy as j
import optax
from . import env_v4 as env
from . import policy_v5 as policy
from .model import absolute_value
from .env_v2 import HISTORY

class Transition(NamedTuple):
    state: env.State
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


def policy_terms(logp,old_logp,adv,entropy,cfg):
    """PPO on actual behavior q, including a differentiable entropy estimate.

    Mixed-policy entropy is E_qold[-(q/qold) log q]. Its gradient includes
    the importance weight; averaging -log q alone has zero expected gradient
    at q=qold. The disabled path retains the original conditional entropy.
    """
    ratio=j.exp(logp-old_logp)
    pl=-j.mean(j.minimum(ratio*adv,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*adv))
    ent=-j.mean(ratio*logp) if cfg.get('random_action_prob',0.)>0 else j.mean(entropy)
    return {'policy_loss':pl,'entropy':ent,
            'kl':j.mean((ratio-1)-(logp-old_logp)),
            'clip_fraction':j.mean(j.abs(ratio-1)>cfg['clip'])}


def make_rollout(model,horizon,memory_limit=HISTORY,random_action_prob=0.):
    random_action_prob=policy.validate_random_action_prob(random_action_prob)
    def apply_model(params,obs):
        variables={'params':params}
        if memory_limit==HISTORY:
            return model.apply(variables,obs)
        return jax.lax.cond(j.max(obs.hist_len)<=memory_limit,
            lambda _:model.apply(variables,obs,memory_length=memory_limit),
            lambda _:model.apply(variables,obs),operand=None)
    def rollout(params,states,key):
        b=states.turn.shape[0]
        def tick(carry,_):
            states,key=carry
            key,ak,sk,rk=jax.random.split(key,4)
            obs=jax.vmap(env.observe)(states)
            logits,wings,v=apply_model(params,obs)
            v=absolute_value(v,states.turn)
            action,logp,_=policy.sample(states,logits,wings,jax.random.split(ak,b),random_action_prob)
            ns,reward,done,invalid=jax.vmap(env.step)(states,action,jax.random.split(sk,b))
            fresh=jax.vmap(env.reset)(jax.random.split(rk,b))
            next_states=jax.tree_util.tree_map(lambda a,z:j.where(done.reshape((b,)+(1,)*(a.ndim-1)),z,a),ns,fresh)
            tr=Transition(states,states.turn,action,logp,v,reward,done)
            stats=j.array([j.sum(done),j.sum(invalid),j.sum(done&(states.turn==states.landlord)),
                           j.sum(j.max(j.abs(reward),axis=-1)),j.maximum(j.max(ns.hist_len),j.max(states.hist_len))])
            return (next_states,key),(tr,stats)
        (states,key),(tr,stats)=jax.lax.scan(tick,(states,key),None,length=horizon)
        _,_,v=apply_model(params,jax.vmap(env.observe)(states))
        last=absolute_value(v,states.turn)
        return states,key,tr,last,stats
    return jax.jit(rollout)


def make_update(model,cfg,train_memory_limit=HISTORY,fixed_memory_length=False):
    random_action_prob=policy.validate_random_action_prob(cfg.get('random_action_prob',0.))
    def apply_train_model(params,obs):
        variables={'params':params}
        if fixed_memory_length or train_memory_limit==HISTORY:
            return model.apply(variables,obs,memory_length=train_memory_limit)
        return jax.lax.cond(j.max(obs.hist_len)<=train_memory_limit,
            lambda _:model.apply(variables,obs,memory_length=train_memory_limit),
            lambda _:model.apply(variables,obs),operand=None)

    def loss(params,tr,adv,target,entropy_coef,vf_coef):
        obs=jax.vmap(env.observe)(tr.state)
        logits,wings,values=apply_train_model(params,obs)
        values=absolute_value(values,tr.turn)
        logp,entropy=policy.score(tr.state,logits,wings,tr.action,random_action_prob)
        terms=policy_terms(logp,tr.logp,adv,entropy,cfg)
        # No reward transform, clipping, shaping, auxiliary task or value clipping.
        vl=j.mean(j.square(values-target))
        total=terms['policy_loss']+vf_coef*vl-entropy_coef*terms['entropy']
        return total,dict(terms,loss=total,value_loss=vl)
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


def make_gradient_diagnostic(model,cfg,memory_limit=HISTORY):
    """Separate unweighted policy/value gradients, both full and shared trunk."""
    random_action_prob=policy.validate_random_action_prob(cfg.get('random_action_prob',0.))
    @jax.jit
    def diagnose(params,tr,last,key):
        a,t=gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        adv=j.take_along_axis(a,tr.turn[...,None],axis=-1)[...,0]
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        n=adv.size
        ix=jax.random.permutation(key,n)[:cfg.get('vf_diagnostic_batch',256)]
        b=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:])[ix],tr)
        adv=adv.reshape(n)[ix]; target=t.reshape(n,3)[ix]
        def objectives(p):
            obs=jax.vmap(env.observe)(b.state)
            logits,wings,v=jax.lax.cond(j.max(obs.hist_len)<=memory_limit,
                lambda _:model.apply({'params':p},obs,memory_length=memory_limit),
                lambda _:model.apply({'params':p},obs),None)
            v=absolute_value(v,b.turn)
            lp,_=policy.score(b.state,logits,wings,b.action,random_action_prob)
            ratio=j.exp(lp-b.logp)
            pl=-j.mean(j.minimum(ratio*adv,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*adv))
            vl=j.mean(j.square(v-target))
            return j.stack((pl,vl))
        _,pullback=jax.vjp(objectives,params)
        pg=pullback(j.array([1.,0.]))[0]; vg=pullback(j.array([0.,1.]))[0]
        # Shared gradients include the encoder and action features read by
        # the new critic. Dedicated actor/critic/wing outputs are excluded.
        shared=lambda g:{k:v for k,v in g.items() if not k.startswith(
            ('actor','critic','wing','body_wing_factors','prefix'))}
        return {'policy_grad_l2':optax.global_norm(pg),'value_grad_l2':optax.global_norm(vg),
                'policy_shared_grad_l2':optax.global_norm(shared(pg)),
                'value_shared_grad_l2':optax.global_norm(shared(vg))}
    return diagnose
