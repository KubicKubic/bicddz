"""Function-preserving growth, live new capacity, Adam ages and PPO resume."""
import copy
import numpy as np
import jax
import jax.numpy as j
import pytest
from flax import serialization
from flax.traverse_util import flatten_dict
from ddz import env_v4 as env, policy_v5 as policy
from ddz.model_v5 import InteractionMoveTransformer
from ddz.model_efficiency import EfficientMoveTransformer
from ddz.upgrade_v6 import grow_parameters, grow_births, inherited_births, revised_config
from tests.test_v5_exploration import small_config


def configs(bf16=False):
    old=small_config()
    old['model'].update(layers=2,memory_layers=2,bf16=bf16)
    new=copy.deepcopy(old)
    # Keep the source head size, expand its bank, and insert a block early.
    d=old['model']['width'];heads=old['model']['heads'];hd=d//heads
    new['model'].update(layers=3,memory_layers=3,inherited_layers=2,
        inherited_memory_layers=2,attention_width=d+hd,attention_heads=heads+1,
        memory_attention_width=d+2*hd,memory_attention_heads=heads+2,new_layer_ff=48)
    new.update(model_family='V6',parameter_limit=8_100_000)
    return old,new


def observations():
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(810),4))
    states=states._replace(phase=j.ones(4,j.int32),landlord=j.arange(4,dtype=j.int32)%3)
    obs=jax.vmap(env.observe)(states)
    history=obs.history.at[:,:,0].set((j.arange(env.HISTORY)[None,:]%5).astype(j.uint8))
    history=history.at[:,:,15].set(j.uint8(4)).at[:,:,16].set((j.arange(env.HISTORY)[None,:]%3).astype(j.uint8))
    return states,obs._replace(history=history,hist_len=j.array([0,2,89,env.HISTORY]))


def test_preset_budget_and_execution_order():
    old=copy.deepcopy(small_config())
    old['model']=dict(width=192,heads=6,layers=4,memory_layers=3,ff=1536,
        memory_ff=512,wing_rank=16,action_width=48,interaction_width=64,
        action_hidden=96,bf16=True,attention_backend='cudnn')
    cfg=revised_config(old)
    model=EfficientMoveTransformer(**cfg['model'])
    _,obs=observations()
    params=jax.eval_shape(lambda:model.init(jax.random.PRNGKey(811),obs,memory_length=4)['params'])
    count=sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert 7_900_000<count<8_100_000
    assert model.block_order(6,4)==(0,4,1,5,2,3)
    assert model.block_order(4,3)==(0,3,1,2)
    assert cfg['ppo']==old['ppo'] and cfg['envs']==old['envs']
    with pytest.raises(ValueError):
        InteractionMoveTransformer(attention_width=321,attention_heads=10).attention_options()


@pytest.mark.parametrize('bf16',[False,True])
def test_expanded_heads_depth_preserve_all_outputs_actions_and_history(bf16):
    a,b=configs(bf16)
    old=EfficientMoveTransformer(**a['model']);new=EfficientMoveTransformer(**b['model'])
    states,obs=observations()
    source=old.init(jax.random.PRNGKey(812),obs)['params']
    template=new.init(jax.random.PRNGKey(813),obs)['params']
    params=grow_parameters(template,source,a['model'],b['model'])
    x=old.apply({'params':source},obs);y=new.apply({'params':params},obs)
    for xx,yy in zip(jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)):
        np.testing.assert_allclose(xx,yy,atol=2e-5 if bf16 else 3e-6,rtol=2e-5)
    choose=jax.jit(jax.vmap(policy.greedy_one))
    np.testing.assert_array_equal(choose(states,x[0],x[1]),choose(states,y[0],y[1]))
    # Arbitrary invisible padding must remain invisible, even at full capacity.
    valid=j.arange(env.HISTORY)[None,:]<obs.hist_len[:,None]
    changed=obs._replace(history=j.where(valid[:,:,None],obs.history,0))
    z=new.apply({'params':params},changed)
    for yy,zz in zip(jax.tree_util.tree_leaves(y),jax.tree_util.tree_leaves(z)):
        np.testing.assert_array_equal(yy,zz)
    reversed_history=obs.history.at[1,:2].set(obs.history[1,:2][::-1])
    z=new.apply({'params':params},obs._replace(history=reversed_history))
    assert np.max(np.abs(np.asarray(z[2][1]-y[2][1])))>1e-7
    invalid=copy.deepcopy(b['model']);invalid['attention_heads']=2
    with pytest.raises(ValueError,match='head dimension'):
        grow_parameters(template,source,a['model'],invalid)


def test_added_attention_outputs_receive_gradients_and_old_coordinates_stay_exact():
    a,b=configs();_,obs=observations();obs=jax.tree_util.tree_map(lambda x:x[:1],obs)
    obs=obs._replace(hist_len=j.array([2]))
    old=EfficientMoveTransformer(**a['model']);new=EfficientMoveTransformer(**b['model'])
    source=old.init(jax.random.PRNGKey(814),obs,memory_length=4)['params']
    params=grow_parameters(new.init(jax.random.PRNGKey(815),obs,memory_length=4)['params'],
        source,a['model'],b['model'])
    def loss(p):
        logits,_,values=new.apply({'params':p},obs,memory_length=4)
        return -jax.nn.log_softmax(logits)[0,j.argmax(logits[0])]+j.square(values-1).sum()
    gradients=jax.grad(loss)(params)
    for name in ('cross2','self2','memory_self2'):
        assert np.linalg.norm(np.asarray(gradients[name]['out']['kernel']))>0,name
    for name in ('cross_extra0','self_extra0','memory_self_extra0'):
        assert np.linalg.norm(np.asarray(gradients[name]['out']['kernel']))>0,name
    for key,value in flatten_dict(source).items():
        grown=flatten_dict(params)[key]
        np.testing.assert_array_equal(grown[tuple(slice(0,n) for n in value.shape)],value)


def test_grown_births_match_fresh_adam_and_preserve_three_existing_ages():
    import optax
    from ddz.optimizer_efficiency import birth_corrected_adam
    source={'w':j.ones((2,2))}
    params={'w':j.ones((3,2)),'added':j.ones(2)}
    prior={'w':np.array([[0,100],[500,0]],np.int32)}
    births=grow_births(params,source,prior,1000)
    np.testing.assert_array_equal(births['w'],[[0,100],[500,0],[1000,1000]])
    assert births['added']==1000
    from ddz.upgrade_v5 import grow_optimizer
    base=optax.scale_by_adam()
    state=base.init(params)._replace(count=j.int32(1000),
        mu={'w':j.array([[.2,.3],[.4,.5],[0.,0.]]),'added':j.zeros(2)},
        nu={'w':j.array([[.1,.2],[.3,.4],[0.,0.]]),'added':j.zeros(2)})
    fresh=base.init(params);tx=birth_corrected_adam(births)
    for i in range(3):
        grads=jax.tree_util.tree_map(lambda x:j.sin(x+i)+.1,params)
        u,state=tx.update(grads,state);v,fresh=base.update(grads,fresh)
        np.testing.assert_allclose(u['w'][2],v['w'][2],atol=1e-7)
        np.testing.assert_allclose(u['added'],v['added'],atol=1e-7)
    assert inherited_births(source,{'coordinate_births':prior}) is prior


def test_grown_production_ppo_update_ema_and_resume(tmp_path):
    from ddz.train_v5 import create as base_create
    from ddz.train_efficiency import create
    from ddz.ppo_efficiency import ArenaState,make_rollout,make_update
    from ddz.ema_weights import update_weights
    a,b=configs();_,old=base_create(a)
    old=old.apply_gradients(grads=jax.tree_util.tree_map(lambda x:j.ones_like(x)*.01,old.params))
    source={'config':a,'train':serialization.to_state_dict(old),'runtime':{}}
    model,ts=create(b,source)
    assert int(ts.step)==1 and int(ts.opt_state[1][0].count)==1
    assert np.all(np.asarray(ts.opt_state[1][0].mu['cross2']['out']['kernel'])==0)
    # Old optimizer moments, including widened projections, are inherited.
    for k,v in flatten_dict(old.opt_state[1][0].mu).items():
        actual=flatten_dict(ts.opt_state[1][0].mu)[k]
        np.testing.assert_array_equal(actual[tuple(slice(0,n) for n in v.shape)],v)
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(816),a['envs']))
    n=a['envs'];arena=ArenaState(states,j.zeros(n,j.int32),j.zeros(n,j.int32),j.zeros(n,j.bool_),j.zeros(n,j.bool_))
    pool=jax.tree_util.tree_map(lambda x:x[None],ts.params)
    arena,key,tr,last,stats=make_rollout(model,a['horizon'],8)(ts.params,arena,jax.random.PRNGKey(817),pool)
    assert int(stats[:,1].sum())==0
    updated,metrics=make_update(model,b['ppo'],8)(ts,tr,last,key,j.float32(.002),j.float32(.03),j.float32(1e-5))
    assert float(metrics['nonfinite'])==0 and float(metrics['applied'])>0
    assert all(np.isfinite(float(x)) for x in metrics.values())
    ema=update_weights(ts.params,updated.params,.999)
    for before,after,mixed in zip(jax.tree_util.tree_leaves(ts.params),jax.tree_util.tree_leaves(updated.params),jax.tree_util.tree_leaves(ema)):
        np.testing.assert_allclose(mixed,.999*before+.001*after,rtol=1e-6,atol=1e-7)
    births=grow_births(ts.params,old.params,inherited_births(old.params,{}),1)
    payload={'config':b,'train':serialization.to_state_dict(updated),'runtime':{'coordinate_births':births}}
    payload=serialization.msgpack_restore(serialization.msgpack_serialize(payload))
    _,restored=create(b,payload)
    for x,y in zip(jax.tree_util.tree_leaves(updated),jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(x,y)
