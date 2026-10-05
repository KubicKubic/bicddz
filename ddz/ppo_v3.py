"""PPO over the exact variable-size set of complete legal moves."""
from functools import partial
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as j
import optax

from . import env_v2 as env
from . import candidates_v3 as moves
from .model import absolute_value
from .ppo_v2 import gae,explained_variance


class Transition(NamedTuple):
    state: env.State
    candidates: j.ndarray
    candidate_mask: j.ndarray
    turn: j.ndarray
    action_index: j.ndarray
    logp: j.ndarray
    value: j.ndarray
    reward: j.ndarray
    done: j.ndarray


def make_rollout(model,horizon,common_memory=88,forward_group_size=None):
    @partial(jax.jit,static_argnames=('memory_length',))
    def forward(params,obs,ids,mask,memory_length):
        return model.apply({'params':params},obs,ids,mask,memory_length=memory_length)

    def rollout(params,states,key):
        rows=[]; stats=[]; batch=states.turn.shape[0]
        action_codes=j.asarray(moves.ACTION)
        for _ in range(horizon):
            ids_np,mask_np,counts=moves.batch_legal(states)
            ids=j.asarray(ids_np);mask=j.asarray(mask_np)
            obs=jax.vmap(env.observe)(states)
            if forward_group_size is None:
                memory_length=(common_memory if int(j.max(states.hist_len))<=common_memory
                               else env.HISTORY)
                logits,value=forward(params,obs,ids,mask,memory_length)
                value=absolute_value(value,states.turn)
                key,ak=jax.random.split(key)
                index=jax.random.categorical(ak,logits).astype(j.int32)
                lp=j.take_along_axis(jax.nn.log_softmax(logits),index[:,None],axis=-1)[:,0]
                choice=j.take_along_axis(ids,index[:,None],axis=-1)[:,0]
            else:
                lengths=np.asarray(jax.device_get(states.hist_len))
                order=np.argsort(counts,kind='stable')
                indexes=[];logps=[];values=[];choices=[]
                for start in range(0,batch,forward_group_size):
                    sample_ix=order[start:start+forward_group_size]
                    width=1<<(int(counts[sample_ix].max())-1).bit_length()
                    subobs=jax.tree_util.tree_map(lambda x:x[sample_ix],obs)
                    subids=j.asarray(ids_np[sample_ix,:width])
                    submask=j.asarray(mask_np[sample_ix,:width])
                    memory_length=(common_memory if int(lengths[sample_ix].max())<=common_memory
                                   else env.HISTORY)
                    logits,value=forward(params,subobs,subids,submask,memory_length)
                    value=absolute_value(value,states.turn[sample_ix])
                    key,ak=jax.random.split(key)
                    index=jax.random.categorical(ak,logits).astype(j.int32)
                    lp=j.take_along_axis(jax.nn.log_softmax(logits),index[:,None],axis=-1)[:,0]
                    choice=j.take_along_axis(subids,index[:,None],axis=-1)[:,0]
                    indexes.append(index);logps.append(lp)
                    values.append(value);choices.append(choice)
                inverse=j.asarray(np.argsort(order))
                index=j.concatenate(indexes)[inverse]
                lp=j.concatenate(logps)[inverse]
                value=j.concatenate(values)[inverse]
                choice=j.concatenate(choices)[inverse]
            key,sk,rk=jax.random.split(key,3)
            actions=action_codes[choice]
            ns,reward,done,bad=env.batch_step(states,actions,jax.random.split(sk,batch))
            fresh=env.batch_reset(jax.random.split(rk,batch))
            next_states=jax.tree_util.tree_map(
                lambda a,z:j.where(done.reshape((batch,)+(1,)*(a.ndim-1)),z,a),ns,fresh)
            rows.append(Transition(states,ids,mask,states.turn,index,lp,value,reward,done))
            stats.append(j.array([j.sum(done),j.sum(bad),
                j.sum(done&(states.turn==states.landlord)),
                j.sum(j.max(j.abs(reward),axis=-1)),j.max(ns.hist_len),
                j.max(j.sum(mask,axis=-1))]))
            states=next_states
        max_width=max(int(row.candidates.shape[1]) for row in rows)
        def pad(row):
            gap=max_width-row.candidates.shape[1]
            return row._replace(
                candidates=j.pad(row.candidates,((0,0),(0,gap))),
                candidate_mask=j.pad(row.candidate_mask,((0,0),(0,gap))))
        transitions=jax.tree_util.tree_map(lambda *xs:j.stack(xs),*(pad(row) for row in rows))
        ids_np,mask_np,_=moves.batch_legal(states)
        obs=jax.vmap(env.observe)(states)
        memory_length=common_memory if int(j.max(states.hist_len))<=common_memory else env.HISTORY
        _,last=forward(params,obs,j.asarray(ids_np),j.asarray(mask_np),memory_length)
        last=absolute_value(last,states.turn)
        return states,key,transitions,last,j.stack(stats)
    return rollout


def make_update(model,cfg,common_memory=88):
    max_tokens=cfg.get('candidate_tokens_per_minibatch',65536)
    max_mb=cfg.get('max_minibatch',2048)
    @partial(jax.jit,static_argnames=('memory_length',))
    def train_minibatch(ts,states,ids,mask,turn,action_index,old_logp,
                        advantage,target,sample_weight,entropy_coef,vf_coef,memory_length):
        def loss(params):
            weight=sample_weight/j.maximum(j.sum(sample_weight),1.)
            weighted=lambda x:j.sum(x*weight)
            obs=jax.vmap(env.observe)(states)
            logits,values=model.apply({'params':params},obs,ids,mask,
                                      memory_length=memory_length)
            values=absolute_value(values,turn)
            lp=jax.nn.log_softmax(logits)
            logp=j.take_along_axis(lp,action_index[:,None],axis=-1)[:,0]
            ratio=j.exp(logp-old_logp)
            policy=-weighted(j.minimum(ratio*advantage,
                j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*advantage))
            value_loss=weighted(j.mean(j.square(values-target),axis=-1))
            entropy=-weighted(j.sum(j.where(mask,j.exp(lp)*lp,0),axis=-1))
            kl=weighted((ratio-1)-(logp-old_logp))
            total=policy+vf_coef*value_loss-entropy_coef*entropy
            return total,{'loss':total,'policy_loss':policy,'value_loss':value_loss,
                'entropy':entropy,'kl':kl,
                'clip_fraction':weighted((j.abs(ratio-1)>cfg['clip']).astype(j.float32))}
        (_,metrics),grads=jax.value_and_grad(loss,has_aux=True)(ts.params)
        norm=optax.global_norm(grads)
        finite=j.isfinite(norm)&j.isfinite(metrics['loss'])
        ts=jax.lax.cond(finite,lambda x:x.apply_gradients(grads=grads),lambda x:x,ts)
        return ts,dict(metrics,grad_norm=norm,
                       applied=finite.astype(j.float32),
                       nonfinite=(~finite).astype(j.float32))

    def update(ts,tr,last,key,entropy_coef,vf_coef):
        a,target=gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        value_ev=explained_variance(tr.value,target)
        acting_value=j.take_along_axis(tr.value,tr.turn[...,None],axis=-1)[...,0]
        acting_target=j.take_along_axis(target,tr.turn[...,None],axis=-1)[...,0]
        acting_ev=explained_variance(acting_value,acting_target)
        actor=j.take_along_axis(a,tr.turn[...,None],axis=-1)[...,0]
        actor=(actor-actor.mean())/(actor.std()+1e-8)
        n=actor.size
        flat=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:]),tr)
        actor=actor.reshape(n);target=target.reshape(n,3)
        counts=np.asarray(jax.device_get(j.sum(flat.candidate_mask,axis=-1)))
        lengths=np.asarray(jax.device_get(flat.state.hist_len))
        buckets=1<<np.ceil(np.log2(np.maximum(counts,1))).astype(np.int32)
        memory=np.where(lengths<=common_memory,common_memory,env.HISTORY)
        seed=int(jax.random.randint(key,(),0,2**31-1))
        rng=np.random.default_rng(seed)
        sums={};weight=0;stopped=False;max_minibatch_kl=0.;last_epoch_kl=0.
        group_keys=list(set(zip(buckets,memory)))
        for _ in range(int(cfg.get('epochs',1))):
            epoch_kl_sum=0.;epoch_weight=0
            groups=group_keys.copy()
            rng.shuffle(groups)
            for bucket,mem in groups:
                ix=np.flatnonzero((buckets==bucket)&(memory==mem))
                rng.shuffle(ix)
                # Keep one training shape per candidate bucket. Small groups pad
                # their final minibatch instead of compiling a new graph whenever
                # the number of examples changes between PPO updates.
                mb=max(1,min(max_mb,max_tokens//int(bucket),512))
                for offset in range(0,len(ix),mb):
                    selected=ix[offset:offset+mb]
                    effective=len(selected)
                    if effective<mb:
                        selected=np.pad(selected,(0,mb-effective),constant_values=selected[0])
                    sample_weight=j.asarray(np.arange(mb)<effective,j.float32)
                    batch=jax.tree_util.tree_map(lambda x:x[selected],flat)
                    ts,metrics=train_minibatch(ts,batch.state,
                        batch.candidates[:,:bucket],batch.candidate_mask[:,:bucket],
                        batch.turn,batch.action_index,batch.logp,
                        actor[selected],target[selected],sample_weight,entropy_coef,vf_coef,
                        memory_length=int(mem))
                    metrics=jax.device_get(metrics)
                    for name,value in metrics.items():
                        sums[name]=sums.get(name,0.)+float(value)*effective
                    weight+=effective
                    epoch_kl_sum+=float(metrics['kl'])*effective
                    epoch_weight+=effective
                    max_minibatch_kl=max(max_minibatch_kl,float(metrics['kl']))
                    if not np.isfinite(float(metrics['loss'])):
                        stopped=True;break
                if stopped:break
            if stopped:break
            last_epoch_kl=epoch_kl_sum/max(1,epoch_weight)
            if last_epoch_kl>cfg['target_kl']:
                stopped=True;break
        result={name:value/max(1,weight) for name,value in sums.items()}
        result['value_explained_variance']=float(value_ev)
        result['acting_value_explained_variance']=float(acting_ev)
        result['processed_decisions']=int(weight)
        result['effective_epochs']=float(weight/n)
        result['requested_epochs']=int(cfg.get('epochs',1))
        result['last_epoch_kl']=float(last_epoch_kl)
        result['max_minibatch_kl']=float(max_minibatch_kl)
        result['early_stop']=float(stopped)
        return ts,result
    return update


def make_gradient_diagnostic(model,cfg,common_memory=88):
    """Measure separate PPO policy/value gradients on a small rollout sample."""
    sample_size=int(cfg.get('vf_diagnostic_batch',64))
    if sample_size<1:
        raise ValueError('vf_diagnostic_batch must be positive')

    @partial(jax.jit,static_argnames=('memory_length',))
    def gradient_norms(params,states,ids,mask,turn,action_index,old_logp,
                       advantage,target,memory_length):
        def objectives(p):
            obs=jax.vmap(env.observe)(states)
            logits,values=model.apply({'params':p},obs,ids,mask,
                                      memory_length=memory_length)
            values=absolute_value(values,turn)
            lp=jax.nn.log_softmax(logits)
            selected=j.take_along_axis(lp,action_index[:,None],axis=-1)[:,0]
            ratio=j.exp(selected-old_logp)
            policy=-j.mean(j.minimum(ratio*advantage,
                j.clip(ratio,1-cfg['clip'],1+cfg['clip'])*advantage))
            value=j.mean(j.mean(j.square(values-target),axis=-1))
            return j.stack((policy,value))

        _,pullback=jax.vjp(objectives,params)
        policy_grad=pullback(j.array([1.,0.]))[0]
        value_grad=pullback(j.array([0.,1.]))[0]
        policy_head=('candidate_cross','candidate_ff','candidate_out_norm',
                     'candidate_score')
        shared=lambda grads:{name:value for name,value in grads.items()
            if not name.startswith(policy_head+('critic',))}
        return {'policy_grad_l2':optax.global_norm(policy_grad),
            'value_grad_l2':optax.global_norm(value_grad),
            'policy_shared_grad_l2':optax.global_norm(shared(policy_grad)),
            'value_shared_grad_l2':optax.global_norm(shared(value_grad))}

    def diagnose(params,tr,last,key):
        advantage,target=gae(tr.value,tr.reward,tr.done,last,cfg['gamma'],cfg['lambda'])
        actor=j.take_along_axis(advantage,tr.turn[...,None],axis=-1)[...,0]
        actor=(actor-actor.mean())/(actor.std()+1e-8)
        n=actor.size
        flat=jax.tree_util.tree_map(lambda x:x.reshape((n,)+x.shape[2:]),tr)
        rng=np.random.default_rng(int(jax.random.randint(key,(),0,2**31-1)))
        selected=rng.choice(n,min(sample_size,n),replace=False)
        counts=np.asarray(jax.device_get(j.sum(flat.candidate_mask[selected],axis=-1)))
        width=1<<(int(counts.max())-1).bit_length()
        memory=(common_memory if int(j.max(flat.state.hist_len[selected]))<=common_memory
                else env.HISTORY)
        result=gradient_norms(params,
            jax.tree_util.tree_map(lambda x:x[selected],flat.state),
            flat.candidates[selected,:width],flat.candidate_mask[selected,:width],
            flat.turn[selected],flat.action_index[selected],flat.logp[selected],
            actor.reshape(n)[selected],target.reshape(n,3)[selected],
            memory_length=memory)
        return {name:float(value) for name,value in jax.device_get(result).items()}

    return diagnose
