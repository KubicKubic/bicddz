import copy
from pathlib import Path
import numpy as np
import pytest
from flax import serialization
from ddz.entropy_campaign import revised_config,migrate,verify_entropy_rows,TRANSITION
from ddz.scale_rollout_campaign import read,write
from ddz.switch_distributed_gae import equal
from ddz.local_a100_controller import follow_architecture
from scripts.prepare_exploration_resume import owned_source,checkpoint_ready
from scripts.prepare_local_v6 import reviewed_entropy_handoff
from tests.test_trim_rollout_campaign import checkpoint
from tests.test_exploration_campaign import exploration_setup


def source_checkpoint():
    saved=checkpoint()
    saved['config'].update(global_source_iteration=26554)
    saved['config']['ppo'].update(random_action_prob=.02,entropy_end=.002,
        entropy_start=.02,entropy_updates=20000,adv_keep_fraction=.5)
    saved['runtime']['config_migrations']=[{'field':'previous','iteration':50000}]
    return saved


def test_checkpoint_roundtrip_changes_only_sampling_and_entropy_and_keeps_all_retained_state():
    saved=source_checkpoint();before=copy.deepcopy(saved);cfg=revised_config(saved['config'])
    result=migrate(saved,cfg,51000)
    restored=serialization.msgpack_restore(serialization.msgpack_serialize(result))
    assert equal(saved,before)
    for field in ('train','env','key','iteration','ema'):assert equal(saved[field],restored[field])
    for field in saved['runtime']:
        if field!='config_migrations':assert equal(saved['runtime'][field],restored['runtime'][field])
    assert restored['runtime']['config_migrations'][:1]==saved['runtime']['config_migrations']
    p=cfg['ppo'];assert p['random_action_prob']==0 and p['adv_keep_fraction']==.5
    effective=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*min(1,(26554+51000)/p['entropy_updates'])
    assert effective==pytest.approx(.01)
    changed=copy.deepcopy(cfg);changed['ppo']['random_action_prob']=.02;changed['ppo']['entropy_end']=.002
    assert changed==saved['config']


@pytest.mark.parametrize('fault',['lr','trimming','clock','rng','ema','schedule'])
def test_protocol_or_partial_checkpoint_changes_are_rejected(fault):
    saved=source_checkpoint();cfg=revised_config(saved['config']);at=51000
    if fault=='lr':cfg['ppo']['lr']=1e-4
    if fault=='trimming':cfg['ppo']['adv_keep_fraction']=1.
    if fault=='clock':at+=1
    if fault=='rng':saved['key']=saved['key'][:1]
    if fault=='ema':saved['ema']['decay']=.99
    if fault=='schedule':saved['config']['global_source_iteration']=100
    with pytest.raises(ValueError):migrate(saved,cfg,at)


def test_actual_rollout_acceptance_requires_zero_random_probability_and_raised_entropy():
    verify_entropy_rows([{'random_action_prob':0.,'entropy_coef':.01}])
    for row in ({'random_action_prob':.02,'entropy_coef':.01},
                {'random_action_prob':0.,'entropy_coef':.002},{}):
        with pytest.raises(RuntimeError):verify_entropy_rows([row])


def entropy_setup(tmp_path):
    cfg,regular,root,campaign,run,proof=exploration_setup(tmp_path)
    source=Path(regular['run_dir']);original=read(run/'config.json')
    write(source/'config.json',original)
    write(run/'config.json',revised_config(original))
    write(campaign/'production/config.json',revised_config(original))
    request=read(campaign/'REQUEST.json');request['transition']=TRANSITION;write(campaign/'REQUEST.json',request)
    evidence=read(proof);evidence.update(random_action_prob=0.,entropy_coefficient=.01)
    write(proof,evidence)
    return cfg,regular,root,campaign,run,proof


def test_local_evaluator_follows_only_registered_entropy_configuration(tmp_path):
    cfg,regular,root,campaign,run,proof=entropy_setup(tmp_path)
    result=follow_architecture(cfg,regular,root)
    assert result['run_dir']==str(run) and result['protocol']['interval_updates']==500
    assert read(root/'architecture_handoff.json')['transition']==TRANSITION


@pytest.mark.parametrize('fault',['probability','entropy','retention','configuration'])
def test_bad_entropy_handoff_is_not_promoted(tmp_path,fault):
    cfg,regular,root,campaign,run,proof=entropy_setup(tmp_path)
    if fault in ('probability','entropy'):
        evidence=read(proof);evidence['random_action_prob' if fault=='probability' else 'entropy_coefficient']=.02
        write(proof,evidence)
    if fault=='retention':write(campaign/'migration_receipt.json',{'iteration':51000})
    if fault=='configuration':
        target=read(run/'config.json');target['ppo']['entropy_end']=.02;write(run/'config.json',target)
    with pytest.raises(RuntimeError):follow_architecture(cfg,regular,root)


def test_checkpoint_wait_and_nested_exploration_ownership(tmp_path):
    assert not checkpoint_ready({'iteration':50708},50800)
    assert checkpoint_ready({'iteration':50800},50800)
    assert checkpoint_ready({'iteration':50900},50800)
    old=tmp_path/'old'
    active={'id':'task','command':'python -m ddz.exploration_campaign --root '+str(old)}
    owned_source(active,old,{'state':'running','queue_id':'task'})
    with pytest.raises(RuntimeError):owned_source(active,old,{'state':'running','queue_id':'unrelated'})


def test_old_controller_registry_failure_is_reviewable_only_after_successful_evaluation(tmp_path):
    cfg,regular,root,campaign,run,proof=entropy_setup(tmp_path)
    owner={'root':str(root)};source=Path(regular['run_dir'])
    write(root/'architecture_handoff.json',{'new_run':str(source),'initial_step':50708})
    status={'state':'failed_waiting_review','error':"RuntimeError('Unregistered sample-selection handoff')"}
    with pytest.raises(RuntimeError):reviewed_entropy_handoff(owner,status,campaign)
    output=root/'step_0050708';(output/'BEST').mkdir(parents=True)
    write(output/'BEST/summary.json',{'completed':True})
    assert reviewed_entropy_handoff(owner,status,campaign)
    assert not reviewed_entropy_handoff(owner,{**status,'error':'evaluation failed'},campaign)
    write(output/'FAILED.json',{'exit_code':1})
    with pytest.raises(RuntimeError):reviewed_entropy_handoff(owner,status,campaign)
