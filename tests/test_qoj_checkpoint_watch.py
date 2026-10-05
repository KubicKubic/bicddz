import copy
import json
import threading
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import serialization
from ddz import env_v2 as env
from ddz.actions import N_BODY
from ddz.api_v5 import Policy
from ddz.model_v5 import InteractionMoveTransformer
from ddz.qoj_checkpoint_watch import (CheckpointWatcher,atomic_json,discover,
                                     resume_paths)


@pytest.fixture(scope='module')
def policy_files(tmp_path_factory):
    root=tmp_path_factory.mktemp('serving-model')
    cfg={'model':dict(width=32,layers=2,heads=2,ff=64,memory_ff=64,
        memory_layers=2,bf16=False,wing_rank=4,action_width=16,
        interaction_width=16,action_hidden=24),'ppo':{'gae_clock':'own'},
        'global_source_iteration':100}
    model=InteractionMoveTransformer(**cfg['model'])
    state=env.reset(jax.random.PRNGKey(4))
    params=model.init(jax.random.PRNGKey(5),
        jax.tree_util.tree_map(lambda x:x[None],env.observe(state)))['params']
    (root/'config.json').write_text(json.dumps(cfg))
    (root/'policy.msgpack').write_bytes(serialization.to_bytes(params))
    return root,cfg,params,state


def publish(run,cfg,params,step):
    run.mkdir(exist_ok=True)
    (run/'config.json').write_text(json.dumps(cfg))
    name=f'policy_{step:07d}.msgpack'
    (run/name).write_bytes(serialization.to_bytes(params))
    atomic_json(run/'latest.json',{'iteration':step,'policy':name})


def test_real_jit_replacement_changes_bid_and_value_without_recompilation(policy_files):
    root,cfg,params,state=policy_files
    model=Policy(root/'config.json',root/'policy.msgpack')
    original=model.forward(state)
    changed=copy.deepcopy(params)
    chosen=(int(j.argmax(original[0][0,N_BODY:]))+1)%4
    changed['actor']['bias']=j.zeros_like(changed['actor']['bias']).at[N_BODY+chosen].set(1000)
    changed['critic']['bias']=changed['critic']['bias']+j.array([9.,0.,0.])
    compiled=model._forward._cache_size()
    prepared=model.prepare_params(changed,cfg)
    np.testing.assert_array_equal(model.forward(state)[0],original[0])
    model.params=prepared
    logits,context,value=model.forward(state)
    assert int(model.choose(state,logits[0],jax.tree_util.tree_map(lambda x:x[0],context)))==-chosen-1
    # V5 projects values to zero sum; changing only our head moves all three.
    np.testing.assert_allclose(value,original[2]+j.array([6.,-3.,-3.]),atol=2e-5)
    assert model._forward._cache_size()==compiled
    model.params=params


@pytest.mark.parametrize('fault',['nan','shape','dtype','tree','architecture','clock'])
def test_incompatible_or_bad_weights_do_not_change_serving_model(policy_files,fault):
    root,cfg,params,state=policy_files
    model=Policy(root/'config.json',root/'policy.msgpack')
    candidate=copy.deepcopy(params);config=copy.deepcopy(cfg)
    if fault=='nan':candidate['critic']['bias']=j.full_like(candidate['critic']['bias'],j.nan)
    if fault=='shape':candidate['critic']['bias']=j.zeros(4)
    if fault=='dtype':candidate['critic']['bias']=j.zeros(3,j.int32)
    if fault=='tree':del candidate['critic']
    if fault=='architecture':config['model']['layers']+=1
    if fault=='clock':config['ppo']['gae_clock']='public'
    before=model.forward(state)
    with pytest.raises(ValueError):model.prepare_params(candidate,config)
    after=model.forward(state)
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[2],after[2])


class Serving:
    def __init__(self):self.params={'w':np.array([1.],np.float32)}
    def prepare_params(self,params,config):
        if not np.all(np.isfinite(params['w'])):raise ValueError('bad weights')
        return params
    def decide(self,raw):return float(self.params['w'][0])


def setup_watch(tmp_path):
    cfg={'model':{'width':1},'global_source_iteration':100,'ppo':{'gae_clock':'own'}}
    run=tmp_path/'training';root=tmp_path/'bot';root.mkdir()
    publish(run,cfg,{'w':np.array([2.],np.float32)},7)
    pointer=tmp_path/'current.json';atomic_json(pointer,{'run':str(run)})
    events=[];activations=[];model=Serving()
    watch=CheckpointWatcher(model,root,pointer,{'global_step':106},
        activations.append,events.append,poll_seconds=.05)
    return watch,model,run,cfg,root,pointer,events,activations


def test_complete_marker_controls_activation_and_next_decision_recovers_on_restart(tmp_path):
    watch,model,run,cfg,root,pointer,events,activations=setup_watch(tmp_path)
    assert watch.poll_once() and model.decide({})==1
    assert watch.decide({})==2
    assert activations[-1]['global_step']==107
    assert watch.poll_once() is False  # Does not load the same snapshot twice.
    # A raw policy written ahead of latest.json is not a completed publication.
    (run/'policy_0000008.msgpack').write_bytes(serialization.to_bytes({'w':np.array([3.],np.float32)}))
    assert watch.poll_once() is False
    atomic_json(run/'latest.json',{'iteration':8,'policy':'policy_0000008.msgpack'})
    assert watch.poll_once() and watch.decide({})==3
    frozen=tmp_path/'frozen';frozen.mkdir()
    (frozen/'config.json').write_text(json.dumps(cfg))
    (frozen/'policy.msgpack').write_bytes(serialization.to_bytes({'w':np.array([1.],np.float32)}))
    deployment={'model_dir':str(frozen),'global_step':106}
    config_path,weights,resumed=resume_paths(root,deployment,events.append)
    assert config_path==run/'config.json' and weights==run/'policy_0000008.msgpack'
    assert resumed['global_step']==108
    weights.write_bytes(b'corrupted after activation')
    assert resume_paths(root,deployment,events.append)[2] is None
    assert events[-1]['event']=='model_resume_rejected'


def test_background_loading_does_not_block_decisions_and_corruption_keeps_previous(tmp_path):
    watch,model,run,cfg,root,pointer,events,activations=setup_watch(tmp_path)
    entered=threading.Event();release=threading.Event();prepared=threading.Event()
    validate=model.prepare_params
    def slow(params,cfg):
        entered.set();assert release.wait(5)
        return validate(params,cfg)
    model.prepare_params=slow
    watch.event=lambda record:(events.append(record),prepared.set() if record['event']=='model_prepared' else None)
    watch.start()
    try:
        assert entered.wait(5)
        # Loading is still waiting; game decisions continue using previous weights.
        assert not release.is_set() and watch.decide({})==1
        release.set();assert prepared.wait(5)
        assert watch.decide({})==2
    finally:release.set();watch.close()
    publish(run,cfg,{'w':np.array([np.nan],np.float32)},8)
    with pytest.raises(ValueError,match='bad weights'):watch.poll_once()
    assert watch.decide({})==2 and watch.active['global_step']==107


def test_pointer_migration_uses_global_step_and_never_rolls_back(tmp_path):
    watch,model,run,cfg,root,pointer,events,activations=setup_watch(tmp_path)
    watch.poll_once();watch.commit_ready()
    replacement=tmp_path/'new-training'
    publish(replacement,{**cfg,'global_source_iteration':105},{'w':np.array([4.],np.float32)},3)
    atomic_json(pointer,{'run':str(replacement)})
    assert watch.poll_once() and watch.decide({})==4
    atomic_json(pointer,{'run':str(run)})
    assert watch.poll_once() is False and watch.decide({})==4


@pytest.mark.parametrize('marker',[{'iteration':7,'policy':'../../key'},
    {'iteration':True,'policy':'policy_0000001.msgpack'},
    {'iteration':-1,'policy':'policy_-000001.msgpack'}])
def test_invalid_publication_cannot_select_a_model(tmp_path,marker):
    watch,model,run,cfg,root,pointer,events,activations=setup_watch(tmp_path)
    atomic_json(run/'latest.json',marker)
    with pytest.raises(ValueError):discover(pointer)
    assert watch.decide({})==1


def test_deployed_worker_records_actual_model_after_hot_update(tmp_path,monkeypatch):
    import runpy
    watch,serving,run,cfg,root,pointer,events,activations=setup_watch(tmp_path)
    frozen=tmp_path/'frozen';frozen.mkdir()
    (frozen/'config.json').write_text(json.dumps(cfg))
    (frozen/'policy.msgpack').write_bytes(serialization.to_bytes(serving.params))
    (root/'key').write_text('FAKE_KEY_MUST_NOT_APPEAR')
    deployment={'token_file':str(root/'key'),'base':'http://test','username':'Fortune',
        'tmux_session':'test','source_checkpoint':str(frozen/'policy.msgpack'),
        'model_dir':str(frozen),'global_step':106,'relative_step':6,
        'checkpoint_sha256':'original','checkpoint_watch':{'pointer':str(pointer)}}
    atomic_json(root/'deployment.json',deployment)
    class Client:
        def __init__(self,*args):pass
        def request(self,path,payload=None):
            if path=='/info':return 200,{'match_rounds':9}
            if path=='/me':return 200,{'username':'Fortune','game':None,'score':0}
            raise AssertionError(path)
    class Model(Serving):
        def __init__(self,*args):super().__init__();self.last_bid_diagnostics=None
        def decide(self,raw):
            self.last_inference={'display_value_by_seat':[1.,None,None],
                                 'value_by_seat':[1.,-1.,0.]}
            return 'bid',{'version':raw['version'],'value':int(self.params['w'][0])}
    def play(client,guard,**kwargs):
        loader=guard.model
        loader.close()  # Run the two publications synchronously for this test.
        loader.poll_once();loader.commit_ready()
        publish(run,cfg,{'w':np.array([3.],np.float32)},8)
        assert loader.poll_once()
        kind,payload=guard.decide({'id':1,'version':1,'remaining':10000})
        assert kind=='bid' and payload['value']==3
    monkeypatch.setenv('DDZ_MATCH_ROOT',str(root))
    monkeypatch.delenv('DDZ_DEPLOYMENT_FILE',raising=False)
    monkeypatch.setattr('ddz.api.Client',Client)
    monkeypatch.setattr('ddz.api_v5.Policy',Model)
    monkeypatch.setattr('ddz.api.run',play)
    worker=Path(__file__).resolve().parents[1]/'deploy/qoj/worker.py'
    state=runpy.run_path(str(worker))
    try:
        status=json.loads((root/'status.json').read_text())
        decision=json.loads((root/'last_decision.json').read_text())
        assert status['global_step']==decision['global_step']==108
        assert status['checkpoint_sha256']==decision['checkpoint_sha256']
        assert json.loads((root/'deployment.json').read_text())['global_step']==108
        assert status['checkpoint_watch_enabled']
        assert 'FAKE_KEY_MUST_NOT_APPEAR' not in (root/'events.jsonl').read_text()
    finally:state['lock'].close()
