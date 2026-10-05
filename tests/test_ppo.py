import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from ddz import env
from ddz.ppo import gae,explained_variance,make_rollout,make_update,make_gradient_diagnostic
from ddz.train import create,checkpoint,learning_rate_schedule,resume_config_migration

CFG=json.loads((Path(__file__).parents[1]/'configs/a100.json').read_text())


def test_recorded_checkpoint_cadence_and_lambda_migrations():
    saved=json.loads(json.dumps(CFG)); saved['save_every']=25
    assert resume_config_migration(saved,CFG)=={'field':'save_every','from':25,'to':100}
    assert resume_config_migration(CFG,CFG) is None
    saved['ppo']['lr']*=2
    with np.testing.assert_raises(ValueError):
        resume_config_migration(saved,CFG)
    saved=json.loads(json.dumps(CFG)); saved['ppo']['lambda']=1.0
    assert resume_config_migration(saved,CFG)=={'field':'ppo.lambda','from':1.0,'to':.95}
    assert saved['ppo']['lambda']==1.0
    saved['save_every']=25
    with np.testing.assert_raises(ValueError):
        resume_config_migration(saved,CFG)


def test_resized_minibatch_keeps_learning_rate_continuous():
    cfg=json.loads(json.dumps(CFG))
    base=learning_rate_schedule(cfg)
    anchor=711835
    tuned=learning_rate_schedule(cfg,dict(schedule_real_anchor_step=anchor,
        schedule_virtual_anchor_step=float(anchor),schedule_step_scale=2.0))
    np.testing.assert_allclose(tuned(anchor),base(anchor),rtol=1e-6)
    np.testing.assert_allclose(tuned(anchor+64),base(anchor+128),rtol=1e-6)

def test_three_seat_returns_terminal_and_truncation():
    v=j.arange(5*2*3,dtype=j.float32).reshape(5,2,3)/30
    r=j.zeros_like(v).at[2,0].set(j.array([12,-6,-6])).at[4,0].set(j.array([-2,1,1]))
    done=j.zeros((5,2),j.bool_).at[2,0].set(True).at[4,0].set(True)
    last=j.array([[1.,2.,3.],[4.,-2.,-2.]])
    a,t=gae(v,r,done,last)
    np.testing.assert_allclose(t[:3,0],np.tile([12,-6,-6],(3,1)),atol=1e-6)
    np.testing.assert_allclose(t[3:,0],np.tile([-2,1,1],(2,1)),atol=1e-6)
    np.testing.assert_allclose(t[:,1],np.tile([4,-2,-2],(5,1)),atol=1e-6)
    np.testing.assert_allclose(t,a+v)

def test_value_explained_variance():
    target=j.array([-2.,-1.,1.,2.])
    assert np.isclose(explained_variance(target,target),1.0)
    assert np.isclose(explained_variance(j.zeros_like(target),target),0.0)
    assert np.isclose(explained_variance(j.ones_like(target),j.ones_like(target)),0.0)

def test_model_memory_mask_zero_sum_and_size():
    cfg=json.loads(json.dumps(CFG)); cfg['model']['bf16']=False
    model,ts=create(cfg)
    assert 1_900_000<sum(x.size for x in jax.tree_util.tree_leaves(ts.params))<2_100_000
    s=env.batch_reset(jax.random.split(jax.random.PRNGKey(2),2)); obs=env.batch_observe(s)
    f=jax.jit(lambda o:model.apply({'params':ts.params},o))
    logits,v=f(obs); assert np.isfinite(logits).all() and np.isfinite(v).all()
    short_logits,short_v=model.apply({'params':ts.params},obs,memory_length=32)
    np.testing.assert_allclose(short_logits,logits,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(short_v,v,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(v.sum(-1),0,atol=1e-6)
    other=obs._replace(history=obs.history.at[:,100,0].set(4))
    l2,v2=f(other); np.testing.assert_array_equal(logits,l2); np.testing.assert_array_equal(v,v2)
    # Distinct public history with identical current cards changes predictions.
    other=obs._replace(hist_len=j.ones(2,j.int32),history=obs.history.at[:,0,15].set(1).at[:,0,20].set(3))
    l2,v2=f(other); assert np.max(np.abs(np.asarray(v2-v)))>1e-5
    short_l2,short_v2=model.apply({'params':ts.params},other,memory_length=32)
    np.testing.assert_allclose(short_l2,l2,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(short_v2,v2,rtol=1e-5,atol=1e-5)
    ordered=obs._replace(hist_len=j.full((2,),2,j.int32),
        history=obs.history.at[:,0,15].set(1).at[:,1,15].set(2))
    reversed_events=ordered._replace(history=ordered.history.at[:,0,15].set(2).at[:,1,15].set(1))
    ordered_logits,ordered_values=f(ordered)
    reversed_logits,reversed_values=f(reversed_events)
    assert np.max(np.abs(np.asarray(ordered_logits-reversed_logits)))>1e-7
    assert np.max(np.abs(np.asarray(ordered_values-reversed_values)))>1e-7


def test_ppo_update_gradients_and_exact_checkpoint_resume(tmp_path):
    cfg=json.loads(json.dumps(CFG)); cfg.update(envs=4,horizon=16,updates=2)
    cfg['ppo'].update(minibatch=16,epochs=1,warmup=1)
    # Small model for this functional test; the real 2M model is GPU-smoke tested.
    cfg['model'].update(width=32,layers=1,heads=2,ff=64,bf16=False)
    model,ts=create(cfg); old=ts.params
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(2),4)); key=jax.random.PRNGKey(7)
    roll=make_rollout(model,16)
    states,key,tr,last,stats=roll(ts.params,states,key)
    assert stats[:,1].sum()==0
    update=make_update(model,cfg['ppo'])
    before=ts
    ts,m=update(ts,tr,last,key,j.float32(.02),j.float32(.5))
    short_ts,short_m=make_update(model,cfg['ppo'],train_memory_limit=8)(
        before,tr,last,key,j.float32(.02),j.float32(.5))
    for a,b in zip(jax.tree_util.tree_leaves(ts.params),jax.tree_util.tree_leaves(short_ts.params)):
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)
    for k in m:
        np.testing.assert_allclose(m[k],short_m[k],rtol=1e-5,atol=1e-5)
    assert np.isfinite(m['value_explained_variance'])
    assert np.isfinite(m['acting_value_explained_variance'])
    assert m['nonfinite']==0 and m['applied']>0
    assert any(np.any(np.asarray(a)!=np.asarray(b)) for a,b in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(ts.params)))
    norms=make_gradient_diagnostic(model,cfg['ppo'])(ts.params,tr,last,key)
    assert all(np.isfinite(x) and x>0 for x in norms.values())
    runtime={'vf_coef':.2,'stable_updates':20}
    checkpoint(tmp_path,ts,states,key,1,cfg,runtime)
    saved=serialization.msgpack_restore((tmp_path/'latest.msgpack').read_bytes())
    restored=serialization.from_state_dict(ts,saved['train']); es=serialization.from_state_dict(states,saved['env'])
    out1=roll(ts.params,states,key); out2=roll(restored.params,es,j.array(saved['key']))
    for a,b in zip(jax.tree_util.tree_leaves(out1),jax.tree_util.tree_leaves(out2)): np.testing.assert_array_equal(a,b)
    assert saved['runtime']==runtime

def test_dense_embedding_matches_gather_forward_and_gradient():
    from ddz.model import DenseEmbedding
    # Include repeated padding: this is the bottleneck in large history batches.
    indices=j.array([[0,0,0,1],[2,0,0,1]],j.int32)
    module=DenseEmbedding(3,8,j.float32)
    table=j.arange(24,dtype=j.float32).reshape(3,8)/24
    params={'embedding':table}
    f=lambda p:module.apply({'params':p},indices)
    reference=lambda p:p['embedding'][indices]
    np.testing.assert_array_equal(f(params),reference(params))
    g1=jax.grad(lambda p:j.square(f(p)).sum())(params)
    g2=jax.grad(lambda p:j.square(reference(p)).sum())(params)
    np.testing.assert_allclose(g1['embedding'],g2['embedding'],atol=1e-6)
