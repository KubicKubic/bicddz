import json
from pathlib import Path
import pytest
from ddz.local_a100_controller import follow_architecture
from ddz.scale_rollout_campaign import write
from ddz.trim_rollout_campaign import revised_config


def setup(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    old=tmp_path/'old';new=tmp_path/'new';controller=tmp_path/'controller'
    old_run=old/'production/training';new_run=new/'production/training'
    for directory in (old_run,new_run,controller):directory.mkdir(parents=True)
    original=json.loads((repo/'configs/cluster8_attention_v6_8m.json').read_text())
    write(old_run/'config.json',original);write(new_run/'config.json',revised_config(original))
    write(new_run/'latest.json',{'iteration':51008})
    write(new_run/'status.json',{'parameters':8014192,'nranks':8})
    write(new/'REQUEST.json',{'old_root':str(old),'at_iteration':51000})
    write(new/'migration_receipt.json',{'weights_adam_env_rng_vf_ema_retained':True,'iteration':51000})
    proof=new/'engineering_READY.json'
    write(proof,{'passed':True,'nranks':8,'NCCL_evidence':['NCCL INFO nranks 8'],'adv_keep_fraction':.5})
    pointer=tmp_path/'current.json';write(pointer,{'run':str(new_run),'nranks':8,'proof':str(proof)})
    regular={'run_dir':str(old_run),'protocol':{'interval_updates':500},'initial_steps':[50008]}
    return {'follow_production_pointer':str(pointer)},regular,controller,new,new_run,proof


def test_registered_trim_follows_legacy_acceptance_schema_without_repeating_initial_point(tmp_path):
    cfg,regular,root,campaign,run,proof=setup(tmp_path)
    result=follow_architecture(cfg,regular,root)
    assert result['run_dir']==str(run) and result['initial_steps']==[51008]
    assert regular['initial_steps']==[50008]
    write(run/'latest.json',{'iteration':52000})
    assert follow_architecture(cfg,regular,root)['initial_steps']==[51008]
    assert follow_architecture(cfg,result,root) is result


@pytest.mark.parametrize('fault',['ranks','nccl','parameters','retention','registration','hyperparameters'])
def test_trim_proof_protocol_or_retention_mismatch_never_promotes(tmp_path,fault):
    cfg,regular,root,campaign,run,proof=setup(tmp_path)
    if fault in ('ranks','nccl'):
        data=json.loads(proof.read_text())
        data['nranks' if fault=='ranks' else 'NCCL_evidence']=1 if fault=='ranks' else []
        write(proof,data)
    if fault=='parameters':write(run/'status.json',{'parameters':100,'nranks':8})
    if fault=='retention':write(campaign/'migration_receipt.json',{'iteration':51000})
    if fault=='registration':write(campaign/'REQUEST.json',{'old_root':str(campaign),'at_iteration':51000})
    if fault=='hyperparameters':
        data=json.loads((run/'config.json').read_text());data['ppo']['lr']=1e-4
        write(run/'config.json',data)
    with pytest.raises(RuntimeError):follow_architecture(cfg,regular,root)
    assert not (root/'architecture_handoff.json').exists()
