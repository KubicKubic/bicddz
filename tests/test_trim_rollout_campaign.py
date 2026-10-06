import copy
import numpy as np
import pytest
from ddz.trim_rollout_campaign import revised_config,migrate,verify_trim_rows
from ddz.switch_distributed_gae import equal


def checkpoint():
    return {'iteration':51000,'config':{'ppo':{'lr':1e-5},'ema_decay':.999,'model_family':'V6'},
        'train':{'step':np.int32(21),'params':{'w':np.arange(7,dtype=np.float32)},
                 'opt_state':{'mu':np.arange(7,dtype=np.float32),'births':np.zeros(7,np.int32)}},
        'env':{'history':np.arange(24,dtype=np.int32).reshape(8,3)},
        'key':np.arange(16,dtype=np.uint32).reshape(8,2),
        'runtime':{'vf_coef':.047,'arena':{'pool_id':np.zeros(8,np.int32)},
                   'new_coordinate_warmup_steps':1024,'source_sha256':'original'},
        'ema':{'params':{'w':np.arange(7,dtype=np.float32)},'updates':11008,'decay':.999}}


def test_only_fraction_changes_and_full_state_is_retained_without_mutating_source():
    saved=checkpoint();original=copy.deepcopy(saved)
    cfg=revised_config(saved['config']);result=migrate(saved,cfg,51000)
    assert result['config']['ppo']['adv_keep_fraction']==.5
    assert equal(saved,original)
    for field in ('train','env','key','iteration','ema'):assert equal(saved[field],result[field])
    assert result['runtime']['vf_coef']==saved['runtime']['vf_coef']
    assert result['runtime']['new_coordinate_warmup_steps']==1024
    result['runtime']['config_migrations'].clear()
    assert 'config_migrations' not in saved['runtime']


def test_protocol_checkpoint_and_ema_identity_fail_closed():
    saved=checkpoint();cfg=revised_config(saved['config'])
    changed=copy.deepcopy(cfg);changed['ppo']['lr']=1e-4
    with pytest.raises(ValueError,match='Only'):migrate(saved,changed,51000)
    with pytest.raises(ValueError,match='Complete'):migrate(saved,cfg,51001)
    broken=copy.deepcopy(saved);broken['ema']['decay']=.99
    with pytest.raises(ValueError,match='EMA'):migrate(broken,cfg,51000)
    with pytest.raises(ValueError,match='already'):revised_config(cfg)


def test_acceptance_checks_exact_global_half_and_actual_selected_updates():
    good={'available_train_samples':101,'retained_train_samples':50,'applied_train_samples':50,
          'adv_keep_fraction':.5,'loss':.3}
    verify_trim_rows([good])
    for change in ({'retained_train_samples':51},{'applied_train_samples':49},{'loss':float('nan')},
                   {'adv_keep_fraction':1.}):
        with pytest.raises(RuntimeError):verify_trim_rows([{**good,**change}])
