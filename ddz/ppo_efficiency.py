"""Controlled sample-efficiency contrasts over the unchanged complete action space."""
from typing import NamedTuple
import jax
import jax.numpy as j
import optax
from . import env_v4 as env, policy_v5 as policy
from .model import absolute_value
from .env_v2 import HISTORY
from .ppo_v5 import gae, policy_terms
from .policy_v2 import _pending, _advance
from .actions import N_BODY, WINGS
from .env_v2 import legal_wings


class Transition(NamedTuple):
    state: env.State
    turn: j.ndarray
    action: j.ndarray
    logp: j.ndarray
    value: j.ndarray
    reward: j.ndarray
    done: j.ndarray
    learner: j.ndarray


class ArenaState(NamedTuple):
    games: env.State
    pool_id: j.ndarray
    focus: j.ndarray
    complement: j.ndarray
    use_pool: j.ndarray


def weighted_mean(x, mask, axis_name=None):
    numerator=j.sum(j.where(mask,x,0));denominator=j.sum(mask)
    if axis_name is not None:
        numerator=jax.lax.psum(numerator,axis_name)
        denominator=jax.lax.psum(denominator,axis_name)
    return numerator/j.maximum(denominator,1)


def loss_mean(x,mask,axis_name=None):
    """Local contribution whose averaged gradient is the global masked loss."""
    if axis_name is None:return weighted_mean(x,mask)
    count=jax.lax.psum(j.sum(mask),axis_name)
    ranks=jax.lax.psum(j.float32(1),axis_name)
    return j.sum(j.where(mask,x,0))*ranks/j.maximum(count,1)


def weighted_ev(value, target, mask, axis_name=None):
    def variance(x):
        mean = weighted_mean(x, mask,axis_name)
        return weighted_mean(j.square(x - mean), mask,axis_name)
    var = variance(target)
    return j.where(var > 1e-8, 1 - variance(target - value) / j.maximum(var, 1e-8), 0.)


def own_seat_gae(values, rewards, dones, turns, gamma=1., lam=.95):
    """Trace between the SAME seat's actual observations.

    Terminal rewards are accumulated between decisions; gamma counts public
    transitions, lambda counts own decisions. Unresolved final decisions are
    excluded, rather than bootstrapping from another player's private view.
    The last own observation is still used as a bootstrap for the preceding
    own decision, with its unobserved advantage set to zero.
    """
    b = values.shape[1]
    initial = (j.zeros((b,3)), j.zeros((b,3)), j.zeros((b,3)),
               j.ones((b,3)), j.zeros((b,3),j.bool_))
    def back(carry, row):
        next_v, next_a, accumulated, discount, known = carry
        v, reward, done, turn = row
        terminal = done[:,None]
        next_v = j.where(terminal, 0, next_v)
        next_a = j.where(terminal, 0, next_a)
        accumulated = reward + gamma * j.where(terminal, 0, accumulated)
        discount = gamma * j.where(terminal, 0, discount)
        known = known | terminal
        seat = jax.nn.one_hot(turn,3,dtype=j.bool_)
        valid = j.sum(known & seat,axis=-1).astype(j.bool_)
        delta = accumulated + discount * next_v - v
        a = j.sum(j.where(seat, delta + discount * lam * next_a, 0),axis=-1)
        a = j.where(valid, a, 0)
        target = a + j.sum(j.where(seat,v,0),axis=-1)
        next_v = j.where(seat,v,next_v)
        next_a = j.where(seat,a[:,None],next_a)
        accumulated = j.where(seat,0,accumulated)
        discount = j.where(seat,1,discount)
        known = known | seat
        return (next_v,next_a,accumulated,discount,known),(a,target,valid)
    _, result = jax.lax.scan(back, initial, (values,rewards,dones,turns),reverse=True)
    return result


def belief_targets(states):
    """Privileged labels only, never passed to model.apply or env.observe."""
    other = (states.turn[:,None] + j.array([1,2])[None]) % 3
    return j.take_along_axis(states.hands,other[:,:,None],axis=1)


def forced_one(state,action):
    forced=j.sum(env.legal(state)[:N_BODY+4])==1
    def move(_):
        body,ranks=env.decode_move(action)
        def wing(i,carry):
            pending,forced=carry
            def attach(_):
                return _advance(pending,body,ranks[i]),forced&(j.sum(legal_wings(pending))==1)
            return jax.lax.cond(i<j.asarray(WINGS)[body],attach,lambda _:carry,None)
        return jax.lax.fori_loop(0,5,wing,(_pending(state,body),forced))[1]
    return jax.lax.cond(action<0,lambda _:forced,move,None)


def prepare_targets(tr, last, cfg, axis_name=None):
    if cfg.get('gae_clock','public') == 'own':
        actor, own_target, valid = own_seat_gae(tr.value,tr.reward,tr.done,tr.turn,cfg['gamma'],cfg['lambda'])
        value_mask = jax.nn.one_hot(tr.turn,3,dtype=j.bool_) & valid[...,None]
        target = jax.nn.one_hot(tr.turn,3) * own_target[...,None]
    else:
        all_adv, target = gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        actor = j.take_along_axis(all_adv,tr.turn[...,None],axis=-1)[...,0]
        valid = j.ones_like(tr.done)
        value_mask = j.ones_like(target,j.bool_)
    actor_mask = valid & tr.learner
    if cfg.get('optional_actor_only',False):
        shape=tr.action.shape
        states=jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),tr.state)
        forced=jax.vmap(forced_one)(states,tr.action.reshape(-1)).reshape(shape)
        actor_mask = actor_mask & ~forced
    mean = weighted_mean(actor,actor_mask,axis_name)
    std = j.sqrt(weighted_mean(j.square(actor-mean),actor_mask,axis_name))
    actor = j.where(actor_mask,(actor-mean)/(std+1e-8),0)
    return actor,target,value_mask,actor_mask,valid


def make_rollout(model,horizon,memory_limit=88,pool_probability=0.,random_action_prob=0.,pool_bucket=256):
    prob = policy.validate_random_action_prob(random_action_prob)
    if not 0 <= pool_probability <= 1:raise ValueError('invalid pool probability')
    def forward(params, states):
        obs = jax.vmap(env.observe)(states)
        return jax.lax.cond(j.max(states.hist_len)<=memory_limit,
            lambda _:model.apply({'params':params},obs,memory_length=memory_limit),
            lambda _:model.apply({'params':params},obs),None)
    @jax.jit
    def rollout(params, arena, key, pool_params):
        b = arena.games.turn.shape[0]
        pool_size = jax.tree_util.tree_leaves(pool_params)[0].shape[0]
        def tick(carry,_):
            arena,key = carry;states=arena.games
            key,ak,sk,rk,pk,fk,ck,uk = jax.random.split(key,8)
            logits,wings,v=forward(params,states)
            action,lp,_=policy.sample(states,logits,wings,jax.random.split(ak,b),prob)
            learner=j.ones(b,j.bool_)
            if pool_probability > 0:
                learner=~arena.use_pool | j.where(arena.complement,states.turn!=arena.focus,states.turn==arena.focus)
                rival_actions=j.zeros(b,j.int32)
                # Compile one body shared by all frozen models; never copy
                # per-example parameters or duplicate three model graphs.
                def rival(i,rival_actions):
                    def choose(_):
                        p=jax.tree_util.tree_map(lambda x:x[i],pool_params)
                        def subset(size):
                            ids=j.nonzero(use,size=size,fill_value=b)[0]
                            small=jax.tree_util.tree_map(lambda x:x[j.minimum(ids,b-1)],states)
                            l,c,_=forward(p,small)
                            selected=policy.greedy(small,l,c)
                            return j.zeros(b,j.int32).at[ids].set(selected,mode='drop')
                        cap=min(b,pool_bucket)
                        return jax.lax.cond(j.sum(use)<=cap,lambda _:subset(cap),lambda _:subset(b),None)
                    use=(arena.pool_id==i)&~learner
                    selected=jax.lax.cond(j.any(use),choose,lambda _:j.zeros(b,j.int32),None)
                    return j.where(use,selected,rival_actions)
                rival_actions=jax.lax.fori_loop(0,pool_size,rival,rival_actions)
                action=j.where(learner,action,rival_actions)
            ns,reward,done,invalid=env.batch_step(states,action,jax.random.split(sk,b))
            fresh=env.batch_reset(jax.random.split(rk,b))
            games=jax.tree_util.tree_map(lambda x,y:j.where(done.reshape((b,)+(1,)*(x.ndim-1)),y,x),ns,fresh)
            next_arena=ArenaState(games,
                j.where(done,jax.random.randint(pk,(b,),0,pool_size),arena.pool_id),
                j.where(done,jax.random.randint(fk,(b,),0,3),arena.focus),
                j.where(done,jax.random.bernoulli(ck,.5,(b,)),arena.complement),
                j.where(done,jax.random.bernoulli(uk,float(pool_probability),(b,)),arena.use_pool))
            tr=Transition(states,states.turn,action,lp,absolute_value(v,states.turn),reward,done,learner)
            stats=j.array([j.sum(done),j.sum(invalid),j.sum(learner),j.maximum(j.max(ns.hist_len),j.max(states.hist_len))])
            return (next_arena,key),(tr,stats)
        (arena,key),(tr,stats)=jax.lax.scan(tick,(arena,key),None,length=horizon)
        _,_,v=forward(params,arena.games)
        return arena,key,tr,absolute_value(v,arena.games.turn),stats
    return rollout


def make_update(model,cfg,memory_length=88,axis_name=None):
    prob=policy.validate_random_action_prob(cfg.get('random_action_prob',0.))
    metric_names=('loss','policy_loss','value_loss','entropy','kl','clip_fraction',
                  'belief_loss','belief_accuracy','grad_norm','nonfinite','applied','evaluated')
    zero={k:j.float32(0) for k in metric_names}
    def loss(params,tr,adv,target,value_mask,actor_mask,entropy_coef,vf_coef):
        obs=jax.vmap(env.observe)(tr.state)
        logits,context,values,belief=model.apply({'params':params},obs,
            memory_length=memory_length,return_aux=True)
        values=absolute_value(values,tr.turn)
        lp,ent=policy.score(tr.state,logits,context,tr.action,prob)
        # Opponent actions have no learner logp: mask BEFORE forming ratios.
        delta=j.where(actor_mask,lp-tr.logp,0)
        ratio=j.exp(delta)
        pl=-loss_mean(j.minimum(ratio*adv,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*adv),actor_mask,axis_name)
        entropy=loss_mean(-ratio*lp if prob>0 else ent,actor_mask,axis_name)
        vl=loss_mean(j.square(values-target),value_mask,axis_name)
        bl=j.float32(0);accuracy=j.float32(0)
        if model.belief_head:
            labels=belief_targets(tr.state)
            bl=j.mean(optax.softmax_cross_entropy_with_integer_labels(belief,labels))
            accuracy=j.mean(j.argmax(belief,axis=-1)==labels)
        total=pl+vf_coef*vl-entropy_coef*entropy+cfg.get('belief_coef',0.)*bl
        return total,{'loss':total,'policy_loss':pl,'value_loss':vl,'entropy':entropy,
            'kl':loss_mean((ratio-1)-delta,actor_mask,axis_name),
            'clip_fraction':loss_mean(j.abs(ratio-1)>cfg['clip'],actor_mask,axis_name),
            'belief_loss':bl,'belief_accuracy':accuracy}
    grad=jax.value_and_grad(loss,has_aux=True)
    @jax.jit
    def update(ts,tr,last,key,entropy_coef,vf_coef,learning_rate):
        actor,target,value_mask,actor_mask,valid=prepare_targets(tr,last,cfg,axis_name)
        n=actor.size
        flat=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:]),tr)
        actor=actor.reshape(n);target=target.reshape(n,3)
        value_mask=value_mask.reshape(n,3);actor_mask=actor_mask.reshape(n)
        indices=jax.vmap(lambda k:jax.random.permutation(k,n))(jax.random.split(key,cfg['epochs']))
        indices=indices.reshape(-1,cfg['minibatch'])
        def step(carry,ix):
            ts,stop=carry
            def evaluate(_):
                batch=jax.tree_util.tree_map(lambda x:x[ix],flat)
                (_,m),g=grad(ts.params,batch,actor[ix],target[ix],value_mask[ix],actor_mask[ix],entropy_coef,vf_coef)
                if axis_name is not None:
                    g=jax.lax.pmean(g,axis_name)
                    m=jax.lax.pmean(m,axis_name)
                norm=optax.global_norm(g)
                finite=j.isfinite(norm)&j.isfinite(m['loss'])
                exceeded=m['kl']>cfg['target_kl']
                apply=finite&~exceeded
                def commit(ts):
                    updates,opt_state=ts.tx.update(g,ts.opt_state,ts.params)
                    updates=jax.tree_util.tree_map(lambda x:x*learning_rate,updates)
                    return ts.replace(step=ts.step+1,params=optax.apply_updates(ts.params,updates),opt_state=opt_state)
                updated=jax.lax.cond(apply,commit,lambda x:x,ts)
                return (updated,exceeded|~finite),dict(m,grad_norm=norm,
                    nonfinite=(~finite).astype(j.float32),applied=apply.astype(j.float32),evaluated=j.float32(1))
            return jax.lax.cond(stop,lambda _:((ts,stop),zero),evaluate,None)
        (ts,_),m=jax.lax.scan(step,(ts,j.array(False)),indices)
        evaluated=j.sum(m['evaluated']);planned=len(indices)
        result={k:j.sum(v)/j.maximum(evaluated,1) for k,v in m.items() if k not in ('applied','evaluated','nonfinite')}
        result.update(applied=j.sum(m['applied'])/planned,evaluated_fraction=evaluated/planned,
            kl_max=j.max(m['kl']),kl_early_stop=j.any(m['kl']>cfg['target_kl']).astype(j.float32),
            nonfinite=j.sum(m['nonfinite']),evaluated_minibatches=evaluated,
            applied_minibatches=j.sum(m['applied']),
            actor_eligible_fraction=j.mean(actor_mask),value_eligible_fraction=j.mean(value_mask),
            gae_resolved_fraction=j.mean(valid))
        result['value_explained_variance']=weighted_ev(tr.value,target.reshape(tr.value.shape),value_mask.reshape(tr.value.shape),axis_name)
        acting_v=j.take_along_axis(tr.value,tr.turn[...,None],axis=-1)[...,0]
        acting_t=j.take_along_axis(target.reshape(tr.value.shape),tr.turn[...,None],axis=-1)[...,0]
        result['acting_value_explained_variance']=weighted_ev(acting_v,acting_t,valid,axis_name)
        if axis_name is not None:result=jax.lax.pmean(result,axis_name)
        return ts,result
    return update


def make_diagnostic(model,cfg,memory_length=88,axis_name=None):
    """Adaptive vf uses the same valid actor/value masks as the update."""
    @jax.jit
    def diagnose(params,tr,last,key):
        actor,target,vm,am,_=prepare_targets(tr,last,cfg,axis_name);n=actor.size
        ix=jax.random.permutation(key,n)[:cfg.get('vf_diagnostic_batch',256)]
        flat=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:])[ix],tr)
        actor=actor.reshape(n)[ix];target=target.reshape(n,3)[ix]
        vm=vm.reshape(n,3)[ix];am=am.reshape(n)[ix]
        def objectives(p):
            l,c,v=model.apply({'params':p},jax.vmap(env.observe)(flat.state),memory_length=memory_length)
            lp,_=policy.score(flat.state,l,c,flat.action,cfg.get('random_action_prob',0.))
            delta=j.where(am,lp-flat.logp,0);ratio=j.exp(delta)
            pl=-loss_mean(j.minimum(ratio*actor,j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*actor),am,axis_name)
            vl=loss_mean(j.square(absolute_value(v,flat.turn)-target),vm,axis_name)
            return j.stack((pl,vl))
        _,back=jax.vjp(objectives,params)
        policy_g=back(j.array([1.,0.]))[0];value_g=back(j.array([0.,1.]))[0]
        if axis_name is not None:
            policy_g=jax.lax.pmean(policy_g,axis_name);value_g=jax.lax.pmean(value_g,axis_name)
        return {'policy_grad_l2':optax.global_norm(policy_g),'value_grad_l2':optax.global_norm(value_g)}
    return diagnose
