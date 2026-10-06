"""Immediate handoff must prove source ownership, retained state and process exit."""
import copy
import hashlib
import os
from pathlib import Path
import pytest
from ddz.scale_architecture_v6 import verify_source
from ddz.scale_rollout_campaign import write,sha


def source(tmp_path):
    run=tmp_path/'old/production/training';run.mkdir(parents=True)
    queue=tmp_path/'queue';(queue/'interrupted').mkdir(parents=True)
    (run/'latest.msgpack').write_bytes(b'retained full state')
    write(run/'status.json',{'state':'running','iteration':105,'nranks':8})
    log=tmp_path/'source.log';log.write_text('nranks=8;\nNCCL INFO nranks 8\n')
    proof=tmp_path/'proof.json'
    write(proof,{'passed':True,'nranks':8,'NCCL_evidence':['NCCL INFO nranks 8'],
        'queue_id':'source','accepted_checkpoint_iteration':100,'source_log':str(log),'source_log_sha256':sha(log)})
    write(queue/'interrupted/source.json',{'id':'source','status':'interrupted','task_pid':99999998})
    pause=tmp_path/'pause.json'
    write(pause,{'interrupted_queue_id':'source','retained_checkpoint_iteration':100,
        'retained_checkpoint_sha256':sha(run/'latest.msgpack'),'source_status_sha256':sha(run/'status.json'),
        'remote_training_pid':99999999,'source_health_proof':str(proof)})
    request={'at_iteration':100,'active_source_queue_id':'source','queue':str(queue),'source_pause_receipt':str(pause)}
    return request,run,proof,pause,log


def test_paused_source_accepts_retained_checkpoint_without_faking_completed_status(tmp_path):
    request,run,*_=source(tmp_path)
    assert verify_source(request,run)['accepted_checkpoint_iteration']==100


@pytest.mark.parametrize('fault',['weights','status','log','owner','nccl','alive','iteration'])
def test_paused_source_rejects_changed_or_unowned_state(tmp_path,fault):
    import json
    request,run,proof,pause,log=source(tmp_path)
    if fault=='weights':(run/'latest.msgpack').write_bytes(b'changed')
    if fault=='status':write(run/'status.json',{'state':'running','iteration':106,'nranks':8})
    if fault=='log':log.write_text('changed')
    if fault in ('owner','nccl'):
        data=json.loads(proof.read_text())
        data['queue_id' if fault=='owner' else 'NCCL_evidence']='other' if fault=='owner' else []
        write(proof,data)
    if fault in ('alive','iteration'):
        data=json.loads(pause.read_text())
        data['remote_training_pid' if fault=='alive' else 'retained_checkpoint_iteration']=os.getpid() if fault=='alive' else 99
        write(pause,data)
    with pytest.raises(RuntimeError):verify_source(request,run)


def test_normal_boundary_still_requires_completed_matching_eight_rank_proof(tmp_path):
    request,run,proof,*_=source(tmp_path)
    del request['source_pause_receipt']
    write(run/'status.json',{'state':'completed','iteration':100,'nranks':8})
    import json
    write(run.parent/'queue_job_status.json',json.loads(proof.read_text()))
    assert verify_source(request,run)['passed']
    write(run/'status.json',{'state':'stopped','iteration':100,'nranks':8})
    with pytest.raises(RuntimeError):verify_source(request,run)


def test_interrupted_wrapper_can_exit_before_its_owned_trainer(monkeypatch,tmp_path):
    from ddz.scale_architecture_v6 import wait_for_source_exit
    process=tmp_path/'42/cmdline';process.parent.mkdir();process.write_bytes(b'owned trainer')
    monkeypatch.setattr('ddz.scale_architecture_v6.time.sleep',lambda _:process.write_bytes(b''))
    wait_for_source_exit([42],10,proc_root=tmp_path)
    process.write_bytes(b'owned trainer')
    with pytest.raises(RuntimeError):wait_for_source_exit([42],0,proc_root=tmp_path)
