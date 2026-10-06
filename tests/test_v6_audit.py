"""Failure cases found while auditing the frozen 8M migration path."""
import copy
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import serialization
from ddz import env_v4 as env
from ddz.model_v5 import fused_full_attention
from ddz.train_distributed_v5 import restore_arena
from ddz.upgrade_v6 import grow_births,inherited_births,validate_growth
from ddz.scale_architecture_v6 import expected_followup
from ddz.qoj_checkpoint_watch import model_signature
from tests.test_qoj_checkpoint_watch import setup_watch


@pytest.mark.parametrize('memory',[False,True])
@pytest.mark.parametrize('length',[15,88,192])
def test_fused_lengths_depend_on_query_semantics_not_coincident_shapes(monkeypatch,memory,length):
    received={}
    def attention(q,k,v,**kwargs):
        received.update(kwargs,qshape=q.shape,kshape=k.shape)
        return j.zeros_like(q)
    monkeypatch.setattr(jax.nn,'dot_product_attention',attention)
    qlength=length+1 if memory else 16
    q=j.zeros((2,qlength,4,32),j.bfloat16);k=j.zeros((2,length+1,4,32),j.bfloat16)
    valid=j.arange(length+1)[None,:]<j.array([1,3])[:,None]
    output=fused_full_attention(q,k,k,valid[:,None,None,:],query_is_memory=memory)
    np.testing.assert_array_equal(received['key_value_seq_lengths'],[1,3])
    np.testing.assert_array_equal(received['query_seq_lengths'],[1,3] if memory else [16,16])
    assert received['qshape'][1]%2==received['kshape'][1]%2==0
    assert output.shape==q.shape


def checkpoint(n=16):
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(991),n))
    return {'env':serialization.to_state_dict(states),'key':np.asarray(jax.random.split(jax.random.PRNGKey(992),8)),
        'runtime':{'arena':{'pool_id':np.zeros(n,np.int32),'focus':np.zeros(n,np.int32),
                          'complement':np.zeros(n,np.bool_),'use_pool':np.zeros(n,np.bool_)}}}


def test_eight_rank_resume_restores_all_fields_without_dealing(monkeypatch):
    saved=checkpoint();before=copy.deepcopy(saved)
    monkeypatch.setattr(env,'batch_reset',lambda *_:pytest.fail('Resume must not redeal'))
    arena,key=restore_arena(saved,{'envs':16,'per_gpu_envs':2})
    np.testing.assert_array_equal(key,before['key'])
    for name in env.State._fields:
        np.testing.assert_array_equal(np.asarray(getattr(arena.games,name)).reshape(before['env'][name].shape),before['env'][name])


@pytest.mark.parametrize('fault',['key_dtype','missing_field','history_dtype','history_length','turn','arena_dtype'])
def test_resume_rejects_corrupt_state_before_device_sharding(fault):
    saved=checkpoint()
    if fault=='key_dtype':saved['key']=saved['key'].astype(np.int32)
    if fault=='missing_field':del saved['env']['bottom']
    if fault=='history_dtype':saved['env']['history']=saved['env']['history'].astype(np.int32)
    if fault=='history_length':saved['env']['hist_len']=np.full(16,193,np.int32)
    if fault=='turn':saved['env']['turn']=np.full(16,3,np.int32)
    if fault=='arena_dtype':saved['runtime']['arena']['use_pool']=np.zeros(16,np.int32)
    with pytest.raises(ValueError):restore_arena(saved,{'envs':16,'per_gpu_envs':2})


@pytest.mark.parametrize('birth',[np.zeros(3,np.int32),-1,11,1.5])
def test_invalid_coordinate_ages_cannot_silently_broadcast(birth):
    params={'w':np.zeros((2,2),np.float32)}
    with pytest.raises(ValueError):grow_births(params,params,{'w':birth},10)
    if np.ndim(birth)!=0 or birth!=11:
        with pytest.raises(ValueError):inherited_births(params,{'coordinate_births':{'w':birth}})


def test_growth_does_not_enable_an_untrained_value_or_belief_branch():
    from tests.test_v6_scaling import configs
    a,b=configs();b['model']['team_value']=True
    with pytest.raises(ValueError,match='team_value'):validate_growth(a['model'],b['model'])


def test_duplicate_followups_are_detected_before_any_cancellation(tmp_path):
    queue=tmp_path/'queue';old=tmp_path/'old'
    (queue/'pending').mkdir(parents=True);(queue/'interrupted').mkdir()
    command=f'python -m ddz.cluster_distributed_job --root {old} --phase production --steps 51600 --resume'
    for task in ('a','b'):
        record={'id':task,'status':'pending','command':command,
            'command_sha256':hashlib.sha256(command.encode()).hexdigest()}
        (queue/'pending'/f'{task}.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='Exactly one'):expected_followup(queue,old,51600)
    assert len(list((queue/'pending').glob('*.json')))==2 and not list((queue/'interrupted').iterdir())
    (queue/'pending/b.json').unlink()
    assert expected_followup(queue,old,51600)=='a'
    with pytest.raises(RuntimeError,match='boundary'):expected_followup(queue,old,52600)


@pytest.mark.parametrize('claimed',[False,True])
def test_reviewed_supersession_retains_receipt_and_never_cancels_a_claim(tmp_path,claimed):
    from scripts.prepare_v6 import supersede_pending
    queue=tmp_path/'queue';old=tmp_path/'old';old.mkdir()
    for state in ('pending','running','interrupted'):(queue/state).mkdir(parents=True)
    command=f'python -m ddz.scale_architecture_v6 --root {old}'
    record={'id':'v3','status':'pending','command':command,
        'command_sha256':hashlib.sha256(command.encode()).hexdigest()}
    pending=queue/'pending/v3.json';pending.write_text(json.dumps(record))
    if claimed:
        (queue/'running/v3.json').write_text('{}')
        with pytest.raises(RuntimeError):supersede_pending(queue,'v3',tmp_path/'new')
        assert pending.exists() and not (old/'SUPERSEDED.json').exists()
    else:
        supersede_pending(queue,'v3',tmp_path/'new')
        assert not pending.exists() and (old/'SUPERSEDED.json').exists()
        receipt=json.loads((queue/'interrupted/v3.json').read_text())
        assert receipt['command']==command and receipt['status']=='interrupted'


def test_reporting_failure_does_not_turn_a_committed_model_into_a_fallback(tmp_path):
    watch,model,run,cfg,root,pointer,events,_=setup_watch(tmp_path)
    def fail(_):raise OSError('dashboard is unavailable')
    watch.on_activate=fail
    assert watch.poll_once() and watch.decide({})==2
    assert json.loads((root/'active_model.json').read_text())['global_step']==107
    assert any(e['event']=='model_activation_reporting_failed' for e in events)


def test_user_offline_marker_blocks_the_qoj_start_script(tmp_path):
    import os,subprocess
    root=tmp_path/'offline';root.mkdir();(root/'OFFLINE.json').write_text('{}')
    script=Path(__file__).resolve().parents[1]/'scripts/qoj_match.sh'
    result=subprocess.run(['bash',str(script),'start'],env={**os.environ,
        'DDZ_MATCH_ROOT':str(root),'DDZ_QOJ_ENABLE_ONLINE':'0'},capture_output=True,text=True)
    assert result.returncode==77 and 'disabled by the user' in result.stdout
    assert not (root/'status.json').exists()


def test_background_preparation_cannot_apply_an_old_tree_to_a_new_architecture(tmp_path):
    watch,model,run,cfg,root,pointer,events,_=setup_watch(tmp_path)
    model.config=cfg
    assert watch.poll_once()
    model.config={**cfg,'model':{'width':2}}
    assert not watch.commit_ready()
    assert model.decide({})==1 and not (root/'active_model.json').exists()
    assert events[-1]['event']=='model_update_rejected'


def test_new_attention_interiors_learn_after_the_zero_output_gate_opens():
    from tests.test_v6_scaling import configs,observations
    from ddz.model_efficiency import EfficientMoveTransformer
    from ddz.upgrade_v6 import grow_parameters
    a,b=configs();_,obs=observations()
    obs=jax.tree_util.tree_map(lambda x:x[1:2],obs)
    old=EfficientMoveTransformer(**a['model']);new=EfficientMoveTransformer(**b['model'])
    source=old.init(jax.random.PRNGKey(997),obs,memory_length=4)['params']
    grown=grow_parameters(new.init(jax.random.PRNGKey(998),obs,memory_length=4)['params'],
                          source,a['model'],b['model'])
    def loss(p):
        logits,_,value=new.apply({'params':p},obs,memory_length=4)
        return -jax.nn.log_softmax(logits)[0,j.argmax(logits[0])]+j.square(value[0,0]-1.)
    derivative=jax.jit(jax.grad(loss))
    grad=derivative(grown)
    moved=jax.tree_util.tree_map(lambda p,g:p-.001*g,grown,grad)
    second=derivative(moved)
    for name in ('self2','cross2','memory_self2','self_extra0','cross_extra0','memory_self_extra0'):
        assert np.linalg.norm(np.asarray(second[name]['value']['kernel']))>0,name
        assert np.linalg.norm(np.asarray(second[name]['query']['kernel']))>0,name
