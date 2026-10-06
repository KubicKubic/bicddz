"""Preserved resume state and proof-gated single-controller architecture handoff."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from ddz.upgrade_v6 import revised_config
from ddz.scale_architecture_v6 import migrate_checkpoint
from ddz.local_a100_controller import follow_architecture
from ddz.scale_rollout_campaign import write


def production_config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/cluster8_env2x_batch2x_v1.json').read_text())


def saved_checkpoint():
    cfg=production_config()
    return {'config':cfg,'iteration':50600,'key':np.zeros((8,2),np.uint32),
        'env':{'turn':np.arange(8,dtype=np.int32)},
        'train':{'step':1000,'params':{'cross0':{'out':{'kernel':np.ones((6,32,192),np.float32)}}},
                 'opt_state':{'0':{'count':1000,'mu':{'w':np.ones((2,2),np.float32)},
                                   'nu':{'w':np.ones((2,2),np.float32)*2}}}},
        'runtime':{'arena':{'pool_id':np.zeros(8,np.int32)},'vf_coef':.07,'stable_updates':18},
        'ema':{'params':{'cross0':{'out':{'kernel':np.ones((6,32,192),np.float32)*.5}}},
               'decay':.999,'updates':11600}}


def test_migration_keeps_environment_rng_ema_and_adam_ages_and_rejects_corruption():
    saved=saved_checkpoint();cfg=revised_config(saved['config']);train=copy.deepcopy(saved['train'])
    train['params']['cross0']['out']['kernel']=np.pad(train['params']['cross0']['out']['kernel'],((0,4),(0,0),(0,0)))
    for key in ('mu','nu'):train['opt_state']['0'][key]['w']=np.pad(train['opt_state']['0'][key]['w'],((0,1),(0,0)))
    result=migrate_checkpoint(saved,cfg,train,'hash','/frozen/source.msgpack')
    assert result['key'] is saved['key'] and result['env'] is saved['env']
    assert result['train']['step']==1000 and result['ema']['updates']==11600
    np.testing.assert_array_equal(result['ema']['params']['cross0']['out']['kernel'][:6],.5)
    assert not result['ema']['params']['cross0']['out']['kernel'][6:].any()
    birth=result['runtime']['coordinate_births']['cross0']['out']['kernel']
    assert not birth[:6].any() and np.all(birth[6:]==1000)
    changed=copy.deepcopy(train);changed['opt_state']['0']['mu']['w'][0,0]=9
    with pytest.raises(ValueError,match='Adam'):migrate_checkpoint(saved,cfg,changed,'hash','path')
    changed=copy.deepcopy(cfg);changed['ppo']['lr']=1e-4
    with pytest.raises(ValueError,match='architecture-only'):migrate_checkpoint(saved,changed,train,'hash','path')


def setup(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';controller=tmp_path/'controller'
    for root in (old,new,controller):root.mkdir()
    cfg=production_config();target=revised_config(cfg)
    write(old/'config.json',cfg);write(new/'config.json',target)
    write(new/'latest.json',{'iteration':50608,'policy':'policy_0050608.msgpack'})
    proof=tmp_path/'proof.json'
    write(proof,{'passed':True,'nranks':8,'NCCL_evidence':['NCCL INFO nranks 8'],'parameters':8014192})
    pointer=tmp_path/'current.json'
    write(pointer,{'run':str(new),'nranks':8,'proof':str(proof)})
    options={'run_dir':str(old),'initial_steps':[39300],'protocol':{'interval_updates':500}}
    return {'follow_production_pointer':str(pointer)},options,controller,new,proof


def test_local_controller_follows_only_verified_v6_and_preserves_initial_checkpoint_on_restart(tmp_path):
    cfg,old,root,new,proof=setup(tmp_path)
    result=follow_architecture(cfg,old,root)
    assert result['run_dir']==str(new) and result['initial_steps']==[50608]
    assert old['initial_steps']==[39300]
    write(new/'latest.json',{'iteration':51000})
    assert follow_architecture(cfg,old,root)['initial_steps']==[50608]
    assert follow_architecture(cfg,result,root) is result
    assert follow_architecture({},old,root) is old


def test_eight_rank_source_rng_can_initialize_resume_template_without_consuming_saved_streams():
    import jax
    from ddz.train_distributed_v5 import source_template_key
    keys=np.asarray(jax.random.split(jax.random.PRNGKey(41),8))
    before=keys.copy()
    template=source_template_key(keys,True)
    np.testing.assert_array_equal(template,keys[0])
    assert jax.random.split(template,16).shape==(16,2)
    np.testing.assert_array_equal(keys,before)
    np.testing.assert_array_equal(source_template_key(keys[0],False),keys[0])
    with pytest.raises(ValueError):source_template_key(keys,False)


@pytest.mark.parametrize('fault',['ranks','nccl','passed','parameters','hyperparameters'])
def test_missing_proof_or_protocol_change_never_hands_off_local_gpu(tmp_path,fault):
    cfg,old,root,new,proof=setup(tmp_path)
    data=json.loads(proof.read_text())
    if fault=='ranks':data['nranks']=1
    if fault=='nccl':data['NCCL_evidence']=[]
    if fault=='passed':data['passed']=False
    if fault=='parameters':data['parameters']=100
    if fault=='hyperparameters':
        changed=json.loads((new/'config.json').read_text());changed['ppo']['lambda']=.5
        write(new/'config.json',changed)
    write(proof,data)
    with pytest.raises(RuntimeError):follow_architecture(cfg,old,root)
    assert not (root/'architecture_handoff.json').exists()
