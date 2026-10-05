import json
import pytest
from ddz.local_douzero_watch import pending_steps,merge_history


def test_initial_latest_snapshot_then_complete_500_step_cadence(tmp_path):
    run=tmp_path/'run';out=tmp_path/'out';run.mkdir();out.mkdir()
    for step in (39000,39300,39400,39500,40000):
        (run/f'policy_{step:07d}.msgpack').touch()
    assert pending_steps(run,39500,out,[39300])==[39300,39500,40000]
    done=out/'step_0039300/BEST';done.mkdir(parents=True);(done/'summary.json').touch()
    assert pending_steps(run,39500,out,[39300])==[39500,40000]
    # EMA policies live in a separate directory and never enter the raw queue.
    (run/'ema').mkdir();(run/'ema/policy_0039500.msgpack').touch()
    assert pending_steps(run,40000,out)==[40000]


def test_mixed_sample_curve_preserves_intervals_and_never_pools_versions(tmp_path):
    protocol={'seed':911001,'equal_role_score':'equal roles /2 landlord','bidding':'fixed bid3','interval_updates':500}
    old={'step':39000,'checkpoint_sha256':'same','deals':1024,'low':-.08,'high':.09}
    earlier={'step':38500,'checkpoint_sha256':'earlier','deals':1024,'low':-.09,'high':.1}
    path=tmp_path/'history.json';path.write_text(json.dumps({'protocol':protocol,'rows':[earlier,old]}))
    cfg={'history_ratings':str(path),'protocol':protocol}
    precise={'step':39000,'checkpoint_sha256':'same','deals':65536,'low':.01,'high':.03}
    latest={'step':39300,'checkpoint_sha256':'new','deals':65536,'low':.02,'high':.04}
    merged=merge_history(cfg,[precise,latest])
    assert merged==[earlier,precise,latest]
    assert json.loads(path.read_text())['rows']==[earlier,old]
    wrong={**precise,'checkpoint_sha256':'different'}
    with pytest.raises(RuntimeError,match='identity'):merge_history(cfg,[wrong])
    changed={'history_ratings':str(path),'protocol':{**protocol,'seed':42}}
    with pytest.raises(RuntimeError,match='protocol'):merge_history(changed,[latest])
