import jax
import jax.numpy as j
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from ddz.advantage_sampling import keep_largest_advantages,validate_keep_fraction
from ddz.ppo_efficiency import make_update,make_diagnostic,prepare_targets
from tests.test_efficiency import ToyModel,toy_rollout
from tests.test_v5_exploration import small_config


@pytest.mark.parametrize('fraction',[.25,.5,.75,1.])
def test_absolute_raw_advantage_matches_sort_with_invalid_and_tied_rows(fraction):
    raw=j.array([-8.,2.,9.,-5.,5.,1.,0.,0.,100.,-99.])
    valid=j.array([True]*8+[False]*2)
    mask=jax.jit(lambda a,v:keep_largest_advantages(a,v,fraction))(raw,valid)
    indices=np.flatnonzero(valid)
    ordered=indices[np.argsort(-np.abs(np.asarray(raw)[indices]),kind='stable')]
    expected=np.zeros(10,bool);expected[ordered[:int(len(indices)*fraction)]]=True
    np.testing.assert_array_equal(mask,expected)


def test_empty_odd_and_all_zero_rollouts():
    for raw,valid in ((j.zeros(5),j.ones(5,bool)),
                      (j.ones(7),j.zeros(7,bool)),
                      (j.array([1.,-2.,3.]),j.ones(3,bool))):
        mask=jax.jit(lambda a,v:keep_largest_advantages(a,v,.5))(raw,valid)
        assert int(mask.sum())==int(valid.sum())//2
        assert not bool((mask & ~valid).any())
    np.testing.assert_array_equal(keep_largest_advantages(j.zeros(5),j.ones(5,bool),.5),
                                  [True,True,False,False,False])


@pytest.mark.parametrize('fraction',[0.,-1.,1.1,float('nan'),float('inf')])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError,match='adv_keep_fraction'):validate_keep_fraction(fraction)


def test_crop_really_skips_half_the_gradient_work_and_keeps_full_ev():
    _,params,_,result=toy_rollout();_,key,tr,last,_=result
    tr=tr._replace(done=tr.done.at[-1].set(True),
        reward=j.sin(j.arange(tr.reward.size).reshape(tr.reward.shape)*.7))
    cfg={**small_config()['ppo'],'epochs':1,'minibatch':4,'gae_clock':'own','target_kl':1.}
    calls=[];model=ToyModel(calls)
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    full,full_m=make_update(model,cfg,8)(ts,tr,last,key,j.float32(.002),j.float32(.03),j.float32(.0001))
    jax.block_until_ready(full_m);jax.effects_barrier();full_calls=len(calls);calls.clear()
    crop_cfg={**cfg,'adv_keep_fraction':.5}
    cropped,m=make_update(model,crop_cfg,8)(ts,tr,last,key,j.float32(.002),j.float32(.03),j.float32(.0001))
    jax.block_until_ready(m);jax.effects_barrier()
    assert len(calls)==full_calls//2==int(m['applied_minibatches'])==2
    assert int(cropped.step)==2 and int(full.step)==4
    assert float(m['retained_fraction'])==.5
    assert int(m['retained_train_samples'])==int(m['applied_train_samples'])==8
    for metric in ('value_explained_variance','acting_value_explained_variance'):
        np.testing.assert_array_equal(m[metric],full_m[metric])
    assert float(m['nonfinite'])==0
    # Same complete-trajectory GAE/normalization, then raw (not centered) rank.
    *_,raw=prepare_targets(tr,last,crop_cfg,return_raw=True)
    assert raw.shape==tr.action.shape


def test_zero_eligible_samples_do_not_update_or_compile_a_fake_batch():
    model,params,_,result=toy_rollout();_,key,tr,last,_=result
    tr=tr._replace(learner=j.zeros_like(tr.learner))
    cfg={**small_config()['ppo'],'adv_keep_fraction':.5,'minibatch':4,'gae_clock':'own'}
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    updated,m=make_update(model,cfg,8)(ts,tr,last,key,j.float32(.002),j.float32(.03),j.float32(.001))
    assert int(updated.step)==int(m['evaluated_minibatches'])==int(m['retained_train_samples'])==0
    assert all(np.isfinite(float(x)) for x in m.values())
    np.testing.assert_array_equal(updated.params['w'],params['w'])
    diagnostic=make_diagnostic(model,cfg,8)(params,tr,last,key)
    assert float(diagnostic['policy_grad_l2'])==float(diagnostic['value_grad_l2'])==0


def test_explicit_keep_one_matches_default_optimizer_and_metrics():
    model,params,_,result=toy_rollout();_,key,tr,last,_=result
    cfg={**small_config()['ppo'],'minibatch':4}
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    def update(options):
        return make_update(model,options,8)(ts,tr,last,key,j.float32(.002),j.float32(.03),j.float32(.001))
    a,am=update(cfg);b,bm=update({**cfg,'adv_keep_fraction':1.})
    for x,y in zip(jax.tree_util.tree_leaves((a,am)),jax.tree_util.tree_leaves((b,bm))):
        np.testing.assert_array_equal(x,y)


@pytest.mark.skipif(jax.local_device_count()!=8,reason='Run with eight CPU devices for collective engineering checks')
def test_eight_rank_cutoff_is_global_even_with_an_empty_rank_and_concentrated_extremes():
    raw=j.arange(64,dtype=j.float32).reshape(8,8)-17
    raw=raw.at[7].set(j.array([90.,-90.,90.,-90.,90.,-90.,90.,-90.]))
    valid=j.ones((8,8),bool).at[0].set(False).at[4,2].set(False)
    mask=jax.pmap(lambda a,v:keep_largest_advantages(a,v,.5,'replicas'),axis_name='replicas')(raw,valid)
    reference=keep_largest_advantages(raw,valid,.5)
    np.testing.assert_array_equal(mask,reference)
    assert int(mask.sum())==int(valid.sum())//2
    assert not bool(mask[0].any()) and bool(mask[7].all())
