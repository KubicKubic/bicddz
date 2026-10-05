import copy
from pathlib import Path
import numpy as np
import pytest
import jax.numpy as j
import optax
from flax import serialization
from flax.training.train_state import TrainState
from ddz.ema_weights import update_weights,validate_decay
from ddz.scale_rollout_campaign import revised_config,expand_checkpoint,choose,equal
from ddz.optimizer_efficiency import sampled_lr
from ddz.train_v2 import checkpoint


def source():
    cfg={'envs':16,'per_gpu_envs':2,'horizon':64,'updates':100000,
         'global_source_iteration':26554,'continuation_updates':73446,
         'ppo':{'lr':1e-4,'lr_min':1e-5,'minibatch':16,'epochs':1,'warmup':20,'gamma':1.,'lambda':.95}}
    return {'config':cfg,'iteration':39000,'train':{'params':{'w':np.array([3.,-2.],np.float32)},
            'opt_state':{'mu':np.array([.5,-.1],np.float32),'count':np.int32(9000)},'step':np.int32(9000)},
        'env':{'turn':np.arange(16,dtype=np.int32)%3,'history':np.arange(96,dtype=np.int16).reshape(16,6)},
        'key':np.arange(16,dtype=np.uint32).reshape(8,2),
        'runtime':{'arena':{'focus':np.arange(16,dtype=np.int32)%3,'use_pool':np.zeros(16,bool)},
                   'distributed':{'nranks':8,'per_gpu_envs':2},'vf_coef':.05,'stable_updates':17,'source_sha256':'fixed'}}


@pytest.mark.parametrize('factor',[2,4])
def test_scale_preserves_each_rank_weights_adam_rng_history_and_matches_4x_budget(factor):
    saved=source();original=copy.deepcopy(saved);cfg=revised_config(saved['config'],factor)
    count=cfg['envs']-saved['config']['envs']
    fresh={'turn':np.full(count,2,np.int32),'history':np.full((count,6),-123,np.int16)}
    result=expand_checkpoint(saved,cfg,fresh,39000)
    restored=serialization.msgpack_restore(serialization.msgpack_serialize(result))
    assert equal(saved,original)
    for name in ('train','key','iteration'):assert equal(restored[name],saved[name])
    for name,value in saved['env'].items():
        new=restored['env'][name].reshape((8,2*factor)+value.shape[1:])
        np.testing.assert_array_equal(new[:,:2],value.reshape((8,2)+value.shape[1:]))
    assert cfg['envs']*cfg['horizon']==4*saved['config']['envs']*saved['config']['horizon']
    assert cfg['updates']==200000 and cfg['continuation_updates']==173446
    assert restored['runtime']['vf_coef']==.05 and restored['runtime']['stable_updates']==17
    assert restored['ema']['updates']==0 and equal(restored['ema']['params'],saved['train']['params'])
    changed=copy.deepcopy(cfg);changed['ppo']['gamma']=.99
    with pytest.raises(RuntimeError,match='authorized'):expand_checkpoint(saved,changed,fresh,39000)
    schedule=sampled_lr(cfg,3000000)
    for it in (0,39000,100000,173446):np.testing.assert_allclose(schedule(it),1e-5,rtol=1e-6)


def test_ema_formula_checkpoint_and_resume_are_continuous(tmp_path):
    previous={'w':j.array([1.,-1.],j.float32)};current={'w':j.array([3.,5.],j.float32)}
    ema=update_weights(previous,current)
    np.testing.assert_allclose(ema['w'],[1.002,-.994],rtol=1e-6)
    ts=TrainState.create(apply_fn=lambda *args:None,params=current,tx=optax.sgd(.1))
    checkpoint(tmp_path,ts,{'turn':np.array([1],np.int32)},np.array([1,2],np.uint32),7,{}, {},
               ema={'params':ema,'decay':.999,'updates':1})
    saved=serialization.msgpack_restore((tmp_path/'latest.msgpack').read_bytes())
    assert saved['ema']['decay']==.999 and saved['ema']['updates']==1
    np.testing.assert_array_equal(saved['ema']['params']['w'],ema['w'])
    snapshot=serialization.msgpack_restore((tmp_path/'ema/policy_0000007.msgpack').read_bytes())
    np.testing.assert_array_equal(snapshot['w'],ema['w'])
    resumed=update_weights(saved['ema']['params'],{'w':j.array([-4.,2.],j.float32)})
    direct=update_weights(ema,{'w':j.array([-4.,2.],j.float32)})
    np.testing.assert_array_equal(resumed['w'],direct['w'])
    assert [p.name for p in tmp_path.glob('policy_*.msgpack')]==['policy_0000007.msgpack']
    for bad in (-.1,1.,float('nan')):
        with pytest.raises(ValueError):validate_decay(bad)


def test_choose_honors_health_and_prefers_longer_horizon_when_speed_is_close():
    a={'factor':2,'passed':True,'eligible_decisions_per_second':100.}
    b={'factor':4,'passed':True,'eligible_decisions_per_second':102.}
    assert choose([a,b])==a
    b['eligible_decisions_per_second']=110.
    assert choose([a,b])==b
    b['passed']=False
    assert choose([a,b])==a
    with pytest.raises(RuntimeError,match='No rollout'):choose([{'passed':False}])
