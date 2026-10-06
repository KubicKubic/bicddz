"""Behavior-policy probabilities, exploration legality and exact continuation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import serialization

from ddz import env_v4 as env, policy_v5 as policy, policy_v2 as prior
from ddz.actions import BODIES, N_BODY, WINGS
from ddz.env_v2 import legal_wings
from ddz.ppo_v5 import make_rollout, make_update, make_gradient_diagnostic, policy_terms


def context():
    def values(shape,offset):
        return j.sin(j.arange(np.prod(shape),dtype=j.float32).reshape(shape)*.13+offset)
    return policy.WingContext(values((N_BODY,15),0),values((15,4),1),
        values((4,),2),values((N_BODY,4),3),values((15,4),4),values((4,),5))


def position(hand=None):
    state=env.reset(jax.random.PRNGKey(601))
    hand=hand or [3,3,3,3,2,1,1,1,0,0,0,0,0,0,0]
    return state._replace(hands=state.hands.at[0].set(j.array(hand)),turn=j.int32(0),
        phase=j.int32(1),landlord=j.int32(0),bid=j.int32(3))


def legal_actions(state):
    from ddz import candidates_v3 as catalogue
    ids=catalogue.legal_ids(np.asarray(state.hands[state.turn]),int(state.phase),
        bid=int(state.bid),bid_count=int(state.bid_count),redeals=int(state.redeals),
        last=int(state.last),last_seat=int(state.last_seat),turn=int(state.turn))
    return j.asarray(catalogue.ACTION[ids])


def logits_for(state):
    return j.where(env.legal(state)[:N_BODY+4],j.sin(j.arange(N_BODY+4)*.23)*3,-1e9)


def random_probability(state,action):
    """Independent reference: multiply reciprocals of legal node counts."""
    probability=1/int(env.legal(state)[:N_BODY+4].sum())
    if int(action)<0:return probability
    body,ranks=env.decode_move(action)
    current=prior._pending(state,body)
    for rank in np.asarray(ranks)[:WINGS[int(body)]]:
        mask=np.asarray(legal_wings(current))
        assert mask[rank]
        probability/=int(mask.sum())
        current=prior._advance(current,body,j.int32(rank))
    return probability


@pytest.mark.parametrize('bad',[-.01,1.01,float('nan'),float('inf'),True,False,'0.01',None])
def test_invalid_probability_rejected_before_compilation(bad):
    for call in (lambda:policy.validate_random_action_prob(bad),
                 lambda:make_rollout(None,1,random_action_prob=bad),
                 lambda:make_update(None,{'random_action_prob':bad}),
                 lambda:make_gradient_diagnostic(None,{'random_action_prob':bad})):
        with pytest.raises(ValueError,match='random_action_prob'):call()


@pytest.mark.parametrize('kind',['lead','follow','rocket','bid','forced_bid'])
def test_complete_mixture_normalizes_and_matches_independent_random_distribution(kind):
    state=position()
    if kind=='follow':
        body=next(i for i,b in enumerate(BODIES) if b.type=='trio1' and b.rank==0)
        state=state._replace(last=j.int32(body),last_seat=j.int32(1))
    elif kind=='rocket':
        body=next(i for i,b in enumerate(BODIES) if b.type=='rocket')
        state=state._replace(last=j.int32(body),last_seat=j.int32(1))
    elif kind in ('bid','forced_bid'):
        state=state._replace(phase=j.int32(0),bid=j.int32(0),
            bid_count=j.int32(2),redeals=j.int32(3 if kind=='forced_bid' else 0))
    actions=legal_actions(state);logits=logits_for(state);ctx=context()
    evaluate=lambda p:jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,ctx,a,p)))(actions)
    model_lp,_=evaluate(0.)
    random_lp,_=evaluate(1.)
    expected_random=np.array([random_probability(state,a) for a in actions])
    np.testing.assert_allclose(j.exp(random_lp),expected_random,rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(expected_random.sum(),1,rtol=2e-6)
    for p in (.01,.02,.3,.9):
        lp,surprisal=evaluate(p)
        expected=(1-p)*np.exp(model_lp)+p*expected_random
        np.testing.assert_allclose(j.exp(lp),expected,rtol=3e-6,atol=1e-8)
        np.testing.assert_allclose(j.exp(lp).sum(),1,rtol=3e-6)
        np.testing.assert_array_equal(surprisal,-lp)


def test_disabled_path_is_exact_and_pure_random_ignores_policy():
    state=position();ctx=context();logits=logits_for(state)
    keys=jax.random.split(jax.random.PRNGKey(602),128)
    old=jax.jit(jax.vmap(lambda k:policy._decide(state,logits,ctx,k)))(keys)
    default=jax.jit(jax.vmap(lambda k:policy.sample_one(state,logits,ctx,k)))(keys)
    zero=jax.jit(jax.vmap(lambda k:policy.sample_one(state,logits,ctx,k,0.)))(keys)
    for xs in (default,zero):
        for a,b in zip(old,xs):np.testing.assert_array_equal(a,b)
    for action in old[0][:8]:
        expected=policy._decide(state,logits,ctx,jax.random.PRNGKey(0),action=action)[1:]
        for a,b in zip(expected,policy.score_one(state,logits,ctx,action,0.)):
            np.testing.assert_array_equal(a,b)
    sample=lambda l,c:jax.jit(jax.vmap(lambda k:policy.sample_one(state,l,c,k,1.)))(keys)
    a=sample(logits,ctx)
    b=sample(j.arange(N_BODY+4,dtype=j.float32)*100,
        jax.tree_util.tree_map(lambda x:x*123,ctx))
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_sampling_matches_whole_action_mixture_not_per_node_mixing():
    state=position([3,1,1]+[0]*12);ctx=context()
    body=next(i for i,b in enumerate(BODIES) if b.type=='trio1' and b.rank==0)
    logits=j.where(env.legal(state)[:N_BODY+4],-10.,-1e9).at[body].set(3.)
    ctx=ctx._replace(base=j.zeros((N_BODY,15)).at[body,1].set(3.),output=j.zeros(4))
    actions=legal_actions(state);p=.3
    lp,_=jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,ctx,a,p)))(actions)
    keys=jax.random.split(jax.random.PRNGKey(603),16384)
    samples=jax.jit(jax.vmap(lambda k:policy.sample_one(state,logits,ctx,k,p)))(keys)
    scored=jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,ctx,a,p)))(samples[0])
    np.testing.assert_allclose(samples[1],scored[0],atol=2e-6)
    assert np.isin(samples[0],actions).all()
    frequencies=np.array([(np.asarray(samples[0])==int(a)).mean() for a in actions])
    np.testing.assert_allclose(frequencies,np.exp(lp),atol=.012,rtol=0)
    # The dominant trio1 + rank1 leaf separates one branch per action from
    # independently mixing the body and wing decisions.
    action=env.encode_move(j.int32(body),j.array([1,15,15,15,15]))
    pi_body=float(jax.nn.softmax(logits)[body])
    per_node=((1-p)*pi_body+p/int(env.legal(state)[:313].sum()))*((1-p)*float(jax.nn.softmax(j.array([3.,0.]))[0])+p/2)
    observed=frequencies[np.flatnonzero(np.asarray(actions)==int(action))[0]]
    assert abs(observed-per_node)>.05  # More than four times the sampling tolerance.


def test_mixture_gradient_and_importance_weighted_entropy_match_exact_expectations():
    state=position([3,1,1]+[0]*12);ctx=context();actions=legal_actions(state)
    base=logits_for(state);direction=j.sin(j.arange(313)*.5);p=.3
    def log_probs(theta):
        return jax.vmap(lambda a:policy.score_one(state,base+theta*direction,ctx,a,p)[0])(actions)
    old=jax.lax.stop_gradient(log_probs(0.));weights=j.exp(old)
    advantages=j.cos(j.arange(len(actions))*.7)
    def surrogate(theta):
        lp=log_probs(theta)
        terms=jax.vmap(lambda l,o,a:policy_terms(l[None],o[None],a[None],(-l)[None],
            {'clip':.2,'random_action_prob':p}))(lp,old,advantages)
        return j.sum(weights*terms['policy_loss']),j.sum(weights*terms['entropy'])
    exact_policy=lambda t:-j.sum(j.exp(log_probs(t))*advantages)
    exact_entropy=lambda t:-j.sum(j.exp(log_probs(t))*log_probs(t))
    for i,exact in enumerate((exact_policy,exact_entropy)):
        actual=jax.grad(lambda t:surrogate(t)[i])(0.)
        np.testing.assert_allclose(actual,jax.grad(exact)(0.),rtol=2e-5,atol=2e-6)
    # p=1 removes actor dependence entirely; value learning can still proceed.
    action=actions[0]
    assert float(jax.grad(lambda t:policy.score_one(state,base+t*direction,ctx,action,1.)[0])(0.))==0
    # Disabled entropy remains exactly the previous conditional-entropy term.
    terms=policy_terms(old,old,advantages,j.arange(len(actions),dtype=j.float32),{'clip':.2})
    assert float(terms['kl'])==0 and float(terms['clip_fraction'])==0
    np.testing.assert_array_equal(terms['entropy'],j.arange(len(actions),dtype=j.float32).mean())


@pytest.mark.parametrize('p',[.02,.03,1.])
def test_exploration_finishes_real_games_without_illegal_actions(p):
    ctx=context();states=env.batch_reset(jax.random.split(jax.random.PRNGKey(604),24))
    @jax.jit
    def play(states,key):
        def tick(carry):
            states,key,steps,bad=carry;key,ak,sk=jax.random.split(key,3)
            def one(s,k,step_key):
                def act(_):
                    action,_,_=policy.sample_one(s,logits_for(s),ctx,k,p)
                    ns,_,_,invalid=env.step(s,action,step_key)
                    return ns,invalid
                return jax.lax.cond(s.done,lambda _:(s,j.array(False)),act,None)
            states,invalid=jax.vmap(one)(states,jax.random.split(ak,24),jax.random.split(sk,24))
            return states,key,steps+1,bad+j.sum(invalid)
        return jax.lax.while_loop(lambda c:(~j.all(c[0].done))&(c[2]<256),tick,
            (states,key,j.int32(0),j.int32(0)))
    result=play(states,jax.random.PRNGKey(605))
    assert np.all(result[0].done) and int(result[3])==0
    assert int(result[0].hist_len.max())<=env.HISTORY
    assert set(np.asarray(result[0].landlord))=={0,1,2}


def small_config(p=.3):
    return {'seed':41,'envs':2,'horizon':4,'updates':3,'save_every':1,
        'eval_every':0,'eval_games':3,'memory_limit':8,
        'model':dict(width=16,layers=1,heads=2,ff=32,memory_layers=1,memory_ff=32,
            bf16=False,wing_rank=2,action_width=8,interaction_width=8,action_hidden=12),
        'ppo':dict(gamma=1.,**{'lambda':.95},clip=.2,lr=1e-4,lr_min=1e-5,warmup=1,
            minibatch=8,epochs=1,target_kl=.03,max_grad_norm=1.,weight_decay=1e-4,
            value_coef=.03,entropy_start=.002,entropy_end=.002,entropy_updates=100,
            vf_balance_every=100,vf_diagnostic_batch=8,random_action_prob=p)}


def test_fork_rejects_weights_and_accepts_legacy_disabled_config(tmp_path):
    from ddz.train_v5 import load_v5_fork
    cfg=small_config(.01)
    path=tmp_path/'policy.msgpack'
    path.write_bytes(serialization.msgpack_serialize({'actor_candidate_output':{}}))
    with pytest.raises(ValueError,match='full V5 training checkpoint'):
        load_v5_fork(path,cfg)
    old=copy.deepcopy(cfg);del old['ppo']['random_action_prob']
    payload={'train':{'params':{'actor_candidate_output':{}},'step':17},
        'env':{},'key':np.array([1,2],np.uint32),'iteration':1,'config':old,'runtime':{}}
    path.write_bytes(serialization.msgpack_serialize(payload))
    obj,proof=load_v5_fork(path,cfg)
    assert proof['previous_random_action_prob']==0 and proof['random_action_prob']==.01
    assert obj['train']['step']==17 and obj['config']==old
    payload['iteration']=cfg['updates']
    path.write_bytes(serialization.msgpack_serialize(payload))
    with pytest.raises(ValueError,match='reached'):load_v5_fork(path,cfg)


def test_resident_ppo_uses_mixture_logp_diagnostics_and_exact_checkpoint_resume(tmp_path):
    from ddz.train_v5 import create,checkpoint,load_v5_fork
    cfg=small_config();model,ts=create(cfg)
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(606),2));key=jax.random.PRNGKey(607)
    roll=make_rollout(model,4,8,cfg['ppo']['random_action_prob'])
    states,key,tr,last,stats=roll(ts.params,states,key)
    assert int(stats[:,1].sum())==0
    flat=jax.tree_util.tree_map(lambda x:x.reshape((8,)+x.shape[2:]),tr)
    logits,ctx,_=model.apply({'params':ts.params},jax.vmap(env.observe)(flat.state),memory_length=8)
    lp,_=policy.score(flat.state,logits,ctx,flat.action,.3)
    np.testing.assert_allclose(lp,flat.logp,atol=2e-6)
    trained,metrics=make_update(model,cfg['ppo'],8,fixed_memory_length=True)(
        ts,tr,last,key,j.float32(.002),j.float32(.03))
    assert int(trained.step)>0 and float(metrics['nonfinite'])==0 and float(metrics['applied'])>0
    assert all(np.isfinite(float(v)) for v in metrics.values())
    norms=make_gradient_diagnostic(model,cfg['ppo'],8)(trained.params,tr,last,key)
    assert all(np.isfinite(float(v)) for v in norms.values()) and float(norms['policy_grad_l2'])>0
    checkpoint(tmp_path,trained,states,key,1,cfg,{'vf_coef':.03})
    obj=serialization.msgpack_restore((tmp_path/'latest.msgpack').read_bytes())
    _,empty=create(obj['config'])
    restored=serialization.from_state_dict(empty,obj['train'])
    recovered=serialization.from_state_dict(states,obj['env'])
    a=roll(trained.params,states,key);b=roll(restored.params,recovered,j.asarray(obj['key']))
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x,y)
    fork_cfg=copy.deepcopy(cfg);fork_cfg['ppo']['random_action_prob']=.01
    fork,proof=load_v5_fork(tmp_path/'latest.msgpack',fork_cfg)
    assert proof['random_action_prob']==.01 and proof['previous_random_action_prob']==.3
    assert fork['train']['step']==obj['train']['step'] and len(proof['sha256'])==64
    for field in ('envs','seed','horizon'):
        invalid=copy.deepcopy(fork_cfg);invalid[field]+=1
        with pytest.raises(ValueError,match='only'):load_v5_fork(tmp_path/'latest.msgpack',invalid)
    invalid=copy.deepcopy(fork_cfg);invalid['ppo']['lr']*=2
    with pytest.raises(ValueError,match='only'):load_v5_fork(tmp_path/'latest.msgpack',invalid)


def test_training_cli_new_fork_and_resume(tmp_path):
    cfg=small_config(.01);config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    source=tmp_path/'source';branch=tmp_path/'branch'
    def run(*args):
        result=subprocess.run([sys.executable,'-m','ddz.train_v5',*args],
            cwd=Path(__file__).resolve().parents[1],env={**os.environ,'JAX_PLATFORMS':'cpu'},
            capture_output=True,text=True,timeout=240)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    run('--config',str(config),'--out',str(source),'--stop-after','1')
    original=serialization.msgpack_restore((source/'latest.msgpack').read_bytes())
    cfg['ppo']['random_action_prob']=.03;config.write_text(json.dumps(cfg))
    run('--config',str(config),'--out',str(branch),'--fork-from-v5',str(source/'latest.msgpack'),
        '--stop-after','2')
    branched=serialization.msgpack_restore((branch/'latest.msgpack').read_bytes())
    assert branched['iteration']==2 and branched['runtime']['exploration']['random_action_prob']==.03
    assert branched['runtime']['fork_from_v5']['iteration']==1
    np.testing.assert_array_equal(serialization.msgpack_restore((source/'latest.msgpack').read_bytes())['key'],original['key'])
    run('--config',str(config),'--out',str(branch),'--resume','--stop-after','3')
    resumed=serialization.msgpack_restore((branch/'latest.msgpack').read_bytes())
    assert resumed['iteration']==3 and resumed['config']['ppo']['random_action_prob']==.03
    rows=[json.loads(l) for l in (branch/'metrics.jsonl').read_text().splitlines()]
    assert [r['iteration'] for r in rows]==[2,3]
    assert all(r['random_action_prob']==.03 and r['invalid_actions']==r['nonfinite']==0 for r in rows)
