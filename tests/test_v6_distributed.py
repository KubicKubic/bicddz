"""Eight CPU engineering ranks exercise the grown model's real PPO and Adam."""
import jax
import jax.numpy as j
import numpy as np
from flax import serialization
from ddz import env_v4 as env
from ddz.train_v5 import create as create_base
from ddz.train_efficiency import create
from ddz.ppo_efficiency import ArenaState,make_rollout,make_update
from ddz.ema_weights import update_weights
from tests.test_v6_scaling import configs


def test_grown_model_global_update_replica_agreement_and_resume():
    assert jax.local_device_count()==8
    old,cfg=configs();old['ppo'].update(random_action_prob=0.,gae_clock='own')
    cfg['ppo'].update(random_action_prob=0.,gae_clock='own')
    _,base=create_base(old)
    base=base.apply_gradients(grads=jax.tree_util.tree_map(lambda x:j.ones_like(x)*.01,base.params))
    model,state=create(cfg,{'config':old,'train':serialization.to_state_dict(base),'runtime':{}})
    devices=jax.local_devices();ts=jax.device_put_replicated(state,devices)
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(913),16))
    states=jax.tree_util.tree_map(lambda x:x.reshape((8,2)+x.shape[1:]),states)
    arena=ArenaState(states,j.zeros((8,2),j.int32),j.zeros((8,2),j.int32),
                     j.zeros((8,2),j.bool_),j.zeros((8,2),j.bool_))
    keys=jax.random.split(jax.random.PRNGKey(914),8)
    pool=jax.tree_util.tree_map(lambda x:x[None],state.params)
    collect=jax.pmap(make_rollout(model,4,8),axis_name='replicas',in_axes=(0,0,0,None))
    _,keys,tr,last,stats=collect(ts.params,arena,keys,pool)
    assert int(np.asarray(stats)[:,:,1].sum())==0
    options={**cfg['ppo'],'minibatch':8}
    update=jax.pmap(make_update(model,options,8,'replicas'),axis_name='replicas',
                    in_axes=(0,0,0,0,None,None,None))
    updated,metrics=update(ts,tr,last,keys,j.float32(.002),j.float32(.03),j.float32(1e-5))
    assert np.all(np.asarray(metrics['nonfinite'])==0) and np.all(np.asarray(metrics['applied'])==1)
    for value in jax.tree_util.tree_leaves(updated):
        np.testing.assert_array_equal(value,j.broadcast_to(value[0],value.shape))
    combined=jax.tree_util.tree_map(lambda x:j.transpose(x,(1,0,2)+tuple(range(3,x.ndim))).reshape(
        (x.shape[1],-1)+x.shape[3:]),tr)
    one_last=last.reshape((-1,3))
    reference,expected=make_update(model,{**options,'minibatch':64},8)(state,combined,one_last,keys[0],
                                j.float32(.002),j.float32(.03),j.float32(1e-5))
    for a,b in zip(jax.tree_util.tree_leaves(updated.params),jax.tree_util.tree_leaves(reference.params)):
        np.testing.assert_allclose(a[0],b,rtol=2e-5,atol=3e-6)
    for name in ('policy_loss','value_loss','kl','grad_norm'):
        np.testing.assert_allclose(metrics[name],expected[name],rtol=2e-4,atol=1e-5)
    one=jax.tree_util.tree_map(lambda x:x[0],updated)
    from ddz.upgrade_v6 import grow_births,inherited_births
    ages=grow_births(one.params,base.params,inherited_births(base.params,{}),int(base.step))
    checkpoint=serialization.msgpack_restore(serialization.msgpack_serialize(jax.device_get({
        'config':cfg,'train':serialization.to_state_dict(one),'runtime':{'coordinate_births':ages}})))
    _,restored=create(cfg,checkpoint)
    for a,b in zip(jax.tree_util.tree_leaves(one),jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a,b)
    ema=jax.pmap(lambda a,b:update_weights(a,b,.999))(ts.params,updated.params)
    for value in jax.tree_util.tree_leaves(ema):
        np.testing.assert_array_equal(value,j.broadcast_to(value[0],value.shape))
