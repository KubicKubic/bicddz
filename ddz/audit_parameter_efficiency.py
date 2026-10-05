"""Read-only checkpoint morphology and small CPU functional diagnostics.

These measurements identify compression/scaling candidates. They do not
establish playing strength or replace retrained, paired model comparisons.
"""
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from flax.traverse_util import flatten_dict
from . import env_v2 as env
from .actions import N_BODY,WINGS
from .model_v5 import InteractionMoveTransformer
from .policy_v5 import greedy_one,wing_scores
from .policy_v2 import _pending


def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))


def spectrum(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim<2:return {}
    # DenseGeneral Q/K/V: [input,head,head_dim]; O: [head,head_dim,out].
    matrix=x.reshape(x.shape[0],-1) if x.ndim==3 and x.shape[0]>16 else x.reshape(-1,x.shape[-1])
    singular=np.linalg.svd(matrix,compute_uv=False)
    energy=singular**2;total=float(energy.sum())
    if total==0:return {'rank90':0,'rank99':0,'stable_rank':0,'participation_rank':0}
    cumulative=np.cumsum(energy)/total
    return {'matrix_shape':list(matrix.shape),'rank90':int(np.searchsorted(cumulative,.9)+1),
        'rank99':int(np.searchsorted(cumulative,.99)+1),
        'stable_rank':float(total/energy[0]),
        'participation_rank':float(total**2/np.square(energy).sum()),
        'top1_energy_fraction':float(energy[0]/total)}


def category(name):
    if re.fullmatch(r'ff_(in|out|norm)\d+',name):return 'state_ffn'
    if re.fullmatch(r'(self|cross)(_norm)?\d+',name):return 'state_attention'
    if name.startswith('memory_self'):return 'history_attention'
    if name.startswith('memory_ff'):return 'history_ffn'
    if name.startswith(('memory_','event_','null_memory')):return 'history_inputs'
    if name.startswith(('action_','actor_candidate','prefix_','wing_','body_wing')):return 'action_interaction'
    if name.startswith(('actor','critic')):return 'actor_critic_mlp'
    return 'state_inputs'


def estimate_budget(state_width,history_width,state_ff,history_ff,
                    action_width=48,interaction_width=64,action_hidden=96,wing_rank=16):
    """Exact V5-schema allocation; independent widths are prospective designs."""
    d,m=state_width,history_width
    a,i,h,r=action_width,interaction_width,action_hidden,wing_rank
    p,b=65,N_BODY  # Current prefix descriptors, and complete-action body count.
    groups={
        'state_inputs':91*d,
        'history_inputs':292*m,
        'history_attention':len(history_ff)*(4*m*m+6*m),
        'history_ffn':sum((2*m+1)*f+3*m for f in history_ff),
        'state_attention':len(state_ff)*(6*d*d+2*m*d+12*d),
        'state_ffn':sum((2*d+1)*f+3*d for f in state_ff),
        'actor_critic_mlp':6*d*d+4*d+(b+7)*d+(b+7)+3*h+3,
        'action_interaction':(p+1)*a+(d+1)*a+(a+1)*d+(p+1)*r+
            (d+1)*15*r+b*r+(d+1)*i+(i+1)*i+15*i+i+(p+1)*i+
            (i+1)*i+3*(d+1)*i+(i+16)*h+h+1,
    }
    return {'parameters':sum(groups.values()),'groups':groups}


def adam_state(x):
    if isinstance(x,dict):
        if 'mu' in x and 'nu' in x:return x
        for child in x.values():
            found=adam_state(child)
            if found is not None:return found
    return None


def select_states(saved,n):
    rng=np.random.default_rng(41);state=saved['env']
    roles=np.where(state['landlord']<0,3,(state['turn']-state['landlord'])%3)
    history=state['hist_len'];bins=np.digitize(history,[1,25,56,89])
    selected=[];strata=[]
    groups=[(r,b,np.flatnonzero((roles==r)&(bins==b)&~state['done']))
            for r in range(4) for b in range(5)]
    groups=[(r,b,ix) for r,b,ix in groups if len(ix)]
    budget=max(1,n//len(groups))
    for role,b,ix in groups:
        chosen=rng.choice(ix,min(budget,len(ix)),replace=False)
        selected.extend(chosen.tolist());strata.append({'role':role,'history_bin':b,
            'available':len(ix),'sampled':len(chosen)})
    remaining=np.setdiff1d(np.flatnonzero(~state['done']),selected)
    if len(selected)<n:selected.extend(rng.choice(remaining,n-len(selected),replace=False).tolist())
    selected=np.array(selected[:n],np.int32)
    return env.State(*(j.asarray(state[field][selected]) for field in env.State._fields)),selected,strata


def analyse(run,origin,out,n=96):
    out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    raw=(run/'latest.msgpack').read_bytes()
    full_sha=hashlib.sha256(raw).hexdigest();saved=serialization.msgpack_restore(raw);del raw
    cfg=saved['config'];params=saved['train']['params'];ema=saved['ema']['params']
    initial=serialization.msgpack_restore(origin.read_bytes())
    flat=flatten_dict(params);start=flatten_dict(initial);average=flatten_dict(ema)
    if set(flat)!=set(start) or any(flat[p].shape!=start[p].shape for p in flat):
        raise ValueError('Morphology origin must have exactly the current V5 schema')
    adam=adam_state(saved['train']['opt_state'])
    mu=flatten_dict(adam['mu']);nu=flatten_dict(adam['nu'])
    states,indices,strata=select_states(saved,n)
    (out/'diagnostic_states.msgpack').write_bytes(serialization.msgpack_serialize(
        {field:np.asarray(getattr(states,field)) for field in env.State._fields}))
    iteration=saved['iteration'];total=sum(x.size for x in flat.values())
    summary={'state':'in_progress','iteration':iteration,
        'global_step':cfg.get('global_source_iteration',0)+iteration,'parameters':total,
        'source_run':str(run),'full_checkpoint_sha256':full_sha,'origin':str(origin),
        'origin_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),
        'policy':str(run/f'policy_{iteration:07d}.msgpack'),
        'ema_policy':str(run/'ema'/f'policy_{iteration:07d}.msgpack'),
        'ema_rollout_updates':saved['ema']['updates'],
        'diagnostics':{'positions':n,'sampling':'role/history strata plus random fill',
            'strata':strata,'indices':indices.tolist(),'history_lengths':np.asarray(states.hist_len).tolist(),
            'roles':np.where(np.asarray(states.landlord)<0,3,
                (np.asarray(states.turn)-np.asarray(states.landlord))%3).tolist(),
            'backend':'CPU FP32 standard attention reference; not a match evaluation'},
        'config':cfg}
    rows=[];modules={};categories={}
    for path,weight in flat.items():
        name='.'.join(path);key=path[0];count=weight.size
        delta=np.asarray(weight)-np.asarray(start[path]);mom=np.asarray(mu[path]);second=np.asarray(nu[path])
        row={'name':name,'parameters':count,'shape':list(weight.shape),'weight_rms':norm(weight)/np.sqrt(count),
            'initial_rms':norm(start[path])/np.sqrt(count),'delta_rms':norm(delta)/np.sqrt(count),
            'delta_to_current':norm(delta)/max(norm(weight),1e-30),
            'ema_gap_to_current':norm(weight-average[path])/max(norm(weight),1e-30),
            'gradient_second_moment_rms':float(np.sqrt(np.mean(second,dtype=np.float64))),
            'adam_direction_rms':float(np.sqrt(np.mean(np.square(mom/(np.sqrt(second)+1e-8)),dtype=np.float64))),
            'zero_second_moment_fraction':float(np.mean(second==0)),**spectrum(weight)}
        rows.append(row)
        group=modules.setdefault(key,{'parameters':0,'weight2':0.,'delta2':0.,'gradient2':0.,'initial2':0.,'ema_gap2':0.})
        for field,value in [('parameters',count),('weight2',norm(weight)**2),('delta2',norm(delta)**2),
                            ('gradient2',float(second.sum(dtype=np.float64))),('initial2',norm(start[path])**2),
                            ('ema_gap2',norm(weight-average[path])**2)]:group[field]+=value
        categories[category(key)]=categories.get(category(key),0)+count
    for group in modules.values():
        group.update(delta_to_current=np.sqrt(group['delta2']/max(group['weight2'],1e-30)),
            gradient_rms=np.sqrt(group['gradient2']/group['parameters']),
            ema_gap_to_current=np.sqrt(group['ema_gap2']/max(group['weight2'],1e-30)))
    summary['modules']=modules;summary['parameter_categories']=categories
    current=cfg['model']
    predicted=estimate_budget(current['width'],current['width'],
        [current['ff']]*current['layers'],[current['memory_ff']]*current['memory_layers'],
        current['action_width'],current['interaction_width'],current['action_hidden'],current['wing_rank'])
    if predicted['groups']!=categories:raise ValueError('Budget formula disagrees with actual checkpoint')
    plans=[]
    for name,d,m,ff,mff in [('control',192,192,[1536]*4,[512]*3),
        ('compact',192,192,[1024]*4,[512]*3),('state_224',224,192,[1024]*4,[512]*3),
        ('state_256',256,192,[1024]*4,[512]*3),
        ('anisotropic',192,192,[1024,1024,1024,512],[512,256,256]),
        ('slim_memory',192,128,[1024]*4,[256]*3)]:
        plans.append({'name':name,'state_width':d,'history_width':m,'state_ff':ff,
            'history_ff':mff,**estimate_budget(d,m,ff,mff),
            'status':'analytic budget; not a trained model'})
    summary['scaling_budgets']=plans
    summary['matrix_spectra']=rows
    head_stats={}
    for name in params:
        if not re.fullmatch(r'(memory_self|self|cross)\d+',name):continue
        p=params[name];q=p['query']['kernel'];k=p['key']['kernel'];v=p['value']['kernel'];o=p['out']['kernel']
        qk=np.stack([(q[:,h]@k[:,h].T).reshape(-1) for h in range(q.shape[1])])
        vo=np.stack([(v[:,h]@o[h]).reshape(-1) for h in range(q.shape[1])])
        def max_cos(matrix):
            unit=matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-30)
            cosine=unit@unit.T;np.fill_diagonal(cosine,0)
            return float(np.max(np.abs(cosine)))
        head_stats[name]={'max_abs_qk_head_cosine':max_cos(qk),'max_abs_vo_head_cosine':max_cos(vo),
            'qk_head_norms':np.linalg.norm(qk,axis=1).tolist(),'vo_head_norms':np.linalg.norm(vo,axis=1).tolist()}
    summary['attention_heads']=head_stats
    del saved,initial,ema,adam,mu,nu,flat,start,average
    (out/'weights.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'phase':'weight_morphology_done','parameters':total,'global_step':summary['global_step'],
                      'categories':categories,'seconds':time.monotonic()-begin}),flush=True)
    model=InteractionMoveTransformer(**{**cfg['model'],'bf16':False,'attention_backend':'standard'})
    obs=jax.jit(jax.vmap(env.observe))(states)
    device=jax.tree_util.tree_map(j.asarray,params)
    forward=jax.jit(lambda p:model.apply({'params':p},obs))
    capture=lambda module,method:method=='__call__' and module.name is not None and (
        re.fullmatch(r'(memory_)?ff_(in|out)\d+',module.name) is not None)
    baseline,intermediates=jax.jit(lambda p:model.apply({'params':p},obs,
        capture_intermediates=capture,mutable=['intermediates']))(device)
    valid=np.concatenate((np.ones((n,1),bool),np.arange(env.HISTORY)[None]<np.asarray(states.hist_len)[:,None]),axis=1)
    ffn=[]
    for prefix,layers in [('',cfg['model']['layers']),('memory_',cfg['model']['memory_layers'])]:
        for i in range(layers):
            incoming=np.asarray(intermediates['intermediates'][f'{prefix}ff_in{i}']['__call__'][0])
            hidden=np.asarray(jax.nn.gelu(j.asarray(incoming)))
            if prefix:hidden=hidden[valid]
            else:hidden=hidden.reshape(-1,hidden.shape[-1])
            output=params[f'{prefix}ff_out{i}']['kernel'];rms=np.sqrt(np.mean(hidden**2,axis=0))
            score=rms*np.linalg.norm(output,axis=1)
            energy=np.square(score);cumulative=np.cumsum(np.sort(energy)[::-1])/max(float(energy.sum()),1e-30)
            old_width=384 if prefix else 640
            learned_tail=output[old_width:]
            ffn.append({'layer':f'{prefix}ff{i}','neurons':len(score),
                'score_p90_neurons':int(np.searchsorted(cumulative,.9)+1),
                'score_p99_neurons':int(np.searchsorted(cumulative,.99)+1),
                'tail_neurons':len(learned_tail),'tail_output_energy_fraction':norm(learned_tail)**2/max(norm(output)**2,1e-30),
                'tail_activation_weight_energy_fraction':float(energy[old_width:].sum()/max(energy.sum(),1e-30)),
                'small_score_fraction_1pct_of_max':float(np.mean(score<score.max()*.01)),
                'activation_spectrum':spectrum(hidden[::max(1,int(np.ceil(len(hidden)/512)))]-hidden.mean(axis=0)),
                'score_quantiles':np.quantile(score,[0,.1,.5,.9,1]).tolist()})
    summary['ffn_activations']=ffn
    del intermediates
    base_logits,base_context,base_value=baseline
    base_logp=np.asarray(jax.nn.log_softmax(base_logits));base_prob=np.exp(base_logp)
    choose=jax.jit(jax.vmap(greedy_one));base_action=np.asarray(choose(states,base_logits,base_context))
    # Exact first-wing conditional KL, averaged over all baseline-probability
    # weighted legal bodies with wings. Later conditional choices are not tested.
    def wing_logp(state,context):
        def one(body):
            current=_pending(state,body);mask=env.legal_wings(current)
            scores=wing_scores(current,context,body)
            return jax.nn.log_softmax(j.where(mask,scores,-1e9))
        return jax.vmap(one)(j.arange(N_BODY,dtype=j.int32))
    wings=jax.jit(jax.vmap(wing_logp))
    base_wlp=np.asarray(wings(states,base_context));base_wp=np.exp(base_wlp)
    wing_mass=base_prob[:,:N_BODY]*(np.asarray(WINGS)>0)[None]
    def evaluate(candidate,label,count):
        logits,context,value=forward(candidate)
        logp=np.asarray(jax.nn.log_softmax(logits))
        body_kl=np.sum(base_prob*(base_logp-logp),axis=-1)
        wlp=np.asarray(wings(states,context))
        conditional=np.sum(base_wp*(base_wlp-wlp),axis=-1)
        action=np.asarray(choose(states,logits,context))
        own_delta=np.asarray(value)[:,0]-np.asarray(base_value)[:,0]
        return {'name':label,'parameters':count,'body_kl_mean':float(body_kl.mean()),
            'body_kl_per_100k_parameters':float(body_kl.mean()/max(count,1)*1e5),
            'body_greedy_change':float(np.mean(np.argmax(logp,axis=-1)!=np.argmax(base_logp,axis=-1))),
            'complete_greedy_change':float(np.mean(action!=base_action)),
            'first_wing_conditional_kl':float(np.sum(conditional*wing_mass)/max(float(wing_mass.sum()),1e-30)),
            'own_value_delta_rms':float(np.sqrt(np.mean(own_delta**2))),
            'own_value_delta_to_baseline_std':float(np.sqrt(np.mean(own_delta**2))/max(float(np.std(np.asarray(base_value)[:,0])),1e-30))}
    ablations=[]
    targets=[]
    for prefix,layers in [('',cfg['model']['layers']),('memory_',cfg['model']['memory_layers'])]:
        for i in range(layers):targets.append((f'{prefix}ff{i}',[f'{prefix}ff_out{i}'],False))
    for name in head_stats:targets.append((name,[name],True))
    for name in ['actor_candidate_output','prefix_context','prefix_query','prefix_wing','wing_output',
                 'wing_state','body_wing_factors','critic_action']:
        targets.append((name,[name],False))
    for label,names,is_attention in targets:
        candidate=dict(device)
        for name in names:
            if is_attention:
                candidate[name]={**device[name],'out':jax.tree_util.tree_map(j.zeros_like,device[name]['out'])}
            else:candidate[name]=jax.tree_util.tree_map(j.zeros_like,device[name])
        count=modules[label]['parameters'] if label in modules else sum(
            modules[name]['parameters'] for name in (label.replace('ff','ff_in'),label.replace('ff','ff_out')))
        # A gate's own parameter count understates the cost of the branch it
        # disables. Rank its whole action scorer rather than the 97-param gate.
        if label=='actor_candidate_output':count=sum(modules[name]['parameters'] for name in
            ('action_body_embed','action_body_queries','action_rank_keys','action_rank_values',
             'action_state','action_fusion','actor_candidate_output'))
        result=evaluate(candidate,label,count);ablations.append(result)
        print(json.dumps({'phase':'functional_ablation',**result}),flush=True)
    summary['ablations']=ablations
    # Direct FFN pruning sensitivity: fixed architecture, masked neurons. This
    # predicts candidate damage, not throughput of a physically smaller model.
    pruning=[]
    for width in [w for w in (1024,768,512) if w<cfg['model']['ff']]:
        candidate=dict(device)
        for i in range(cfg['model']['layers']):
            weight=params[f'ff_out{i}']['kernel']
            keep=np.argsort(np.linalg.norm(weight,axis=1))[-width:]
            mask=np.zeros(weight.shape[0],np.float32);mask[keep]=1
            candidate[f'ff_out{i}']={**device[f'ff_out{i}'],
                'kernel':device[f'ff_out{i}']['kernel']*j.asarray(mask)[:,None]}
        result=evaluate(candidate,f'state_ffn_neuron_norm_prune_{width}',
            (cfg['model']['ff']-width)*(2*cfg['model']['width']+1)*cfg['model']['layers'])
        result['remaining_parameters']=total-result['parameters'];pruning.append(result)
        print(json.dumps({'phase':'pruning_probe',**result}),flush=True)
    summary['pruning_probes']=pruning
    summary.update(state='complete',seconds=time.monotonic()-begin,time=time.time())
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns=['name','parameters','weight_rms','initial_rms','delta_rms','delta_to_current',
             'gradient_second_moment_rms','zero_second_moment_fraction','rank90','rank99','stable_rank']
    with (out/'matrices.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    import importlib.util
    if importlib.util.find_spec('matplotlib') is not None:
        from .render_parameter_efficiency import render
        render(out)
    else:
        print('Plot separately with a matplotlib environment: python -m ddz.render_parameter_efficiency '+str(out),flush=True)
    print(json.dumps({'state':'complete','out':str(out),'seconds':summary['seconds']}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--origin',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--positions',type=int,default=96)
    args=parser.parse_args()
    analyse(args.run.resolve(),args.origin.resolve(),args.out.resolve(),args.positions)


if __name__=='__main__':main()
