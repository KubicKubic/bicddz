"""Eight replica PPO equals one global batch, including unequal valid masks."""
import jax
import jax.numpy as j
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from ddz.ppo_efficiency import make_update,make_diagnostic,loss_mean
from tests.test_efficiency import toy_rollout
from tests.test_v5_exploration import small_config


def test_global_masked_gradient_with_empty_replica():
    assert jax.local_device_count()==8
    x=j.arange(32,dtype=j.float32).reshape(8,4)
    mask=x%3==0;mask=mask.at[0].set(False)
    def objective(w,x,mask):return loss_mean(w*x,mask,'replicas')
    g=jax.pmap(jax.grad(objective),axis_name='replicas',in_axes=(None,0,0))(j.float32(2),x,mask)
    np.testing.assert_allclose(j.mean(g),j.sum(j.where(mask,x,0))/j.sum(mask),rtol=1e-6)


@pytest.mark.parametrize('clock',['public','own'])
@pytest.mark.parametrize('keep_fraction',[1.,.5])
def test_eight_rank_ppo_and_vf_diagnostic_match_global_batch(clock,keep_fraction):
    model,params,_,result=toy_rollout();_,key,tr,last,_=result
    local_n=tr.action.size
    shards=[]
    for rank in range(8):
        reward=j.sin(j.arange(tr.reward.size).reshape(tr.reward.shape)+rank*.7)
        learner=(j.arange(tr.action.size).reshape(tr.action.shape)+rank)%3!=0
        if rank==0:learner=j.zeros_like(learner)
        shards.append(tr._replace(reward=reward,learner=learner,done=tr.done.at[-1].set(True)))
    sharded=jax.tree_util.tree_map(lambda *xs:j.stack(xs),*shards)
    combined=jax.tree_util.tree_map(lambda *xs:j.concatenate(xs,axis=1),*shards)
    keys=jax.random.split(key,8);lasts=j.stack([last]*8)
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    replicated=jax.device_put_replicated(ts,jax.local_devices())
    cfg={**small_config()['ppo'],'epochs':1,'minibatch':local_n,'gae_clock':clock,
         'vf_diagnostic_batch':local_n,'adv_keep_fraction':keep_fraction}
    update=jax.pmap(make_update(model,cfg,8,'replicas'),axis_name='replicas',
                    in_axes=(0,0,0,0,None,None,None))
    updated,m=update(replicated,sharded,lasts,keys,j.float32(.002),j.float32(.03),j.float32(.001))
    global_cfg={**cfg,'minibatch':8*local_n,'vf_diagnostic_batch':8*local_n}
    reference,metrics=make_update(model,global_cfg,8)(ts,combined,j.concatenate([last]*8),key,
                                      j.float32(.002),j.float32(.03),j.float32(.001))
    np.testing.assert_allclose(updated.params['w'],reference.params['w'],rtol=1e-6,atol=1e-6)
    for name in ('policy_loss','value_loss','kl','acting_value_explained_variance',
                 'actor_eligible_fraction','value_explained_variance','grad_norm'):
        np.testing.assert_allclose(m[name],metrics[name],rtol=2e-5,atol=2e-5,err_msg=name)
    diagnose=jax.pmap(make_diagnostic(model,cfg,8,'replicas'),axis_name='replicas',in_axes=(0,0,0,0))
    norms=diagnose(replicated.params,sharded,lasts,keys)
    expected=make_diagnostic(model,global_cfg,8)(params,combined,j.concatenate([last]*8),key)
    for name in expected:np.testing.assert_allclose(norms[name],expected[name],rtol=2e-5,atol=2e-5)
    # One bad replica must prevent every replica from updating, without divergence.
    broken=sharded._replace(reward=sharded.reward.at[3].set(j.nan))
    stopped,bad=update(replicated,broken,lasts,keys,j.float32(.002),j.float32(.03),j.float32(.001))
    np.testing.assert_array_equal(stopped.step,j.zeros(8))
    np.testing.assert_array_equal(bad['applied_minibatches'],j.zeros(8))
    np.testing.assert_array_equal(stopped.params['w'],j.full(8,params['w']))
