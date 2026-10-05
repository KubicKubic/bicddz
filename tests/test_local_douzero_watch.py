import json
import subprocess
import sys
import numpy as np
from ddz.local_douzero_watch import pending_steps, stop_child


def test_pending_snapshots_keep_every_500_step_in_order(tmp_path):
    run = tmp_path/'run'
    output = tmp_path/'out'
    run.mkdir()
    output.mkdir()
    for step in (0,500,1000,1500,2000,2100,2500,3000):
        (run/f'policy_{step:07d}.msgpack').touch()
    assert pending_steps(run,2000,output) == [2000,2500,3000]
    complete = output/'step_0002000/BEST'
    complete.mkdir(parents=True)
    (complete/'summary.json').write_text('{}')
    assert pending_steps(run,2000,output) == [2500,3000]
    assert pending_steps(run,3500,output) == []


def test_stop_only_owned_idle_process_before_evaluation():
    owned = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    other = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    try:
        stop_child(owned)
        assert owned.poll() is not None
        assert other.poll() is None
        stop_child(owned)
        stop_child(None)
    finally:
        stop_child(other)


def test_score_equal_role_weights_and_whole_deal_bootstrap():
    from ddz.compare_efficiency_douzero import bootstrap_summary
    # Each role has normalized outcomes [+1,-1] in both control legs.
    raw = np.tile(np.array([1.,-1.]),(3,2,1))
    raw[0] *= 2
    wins = raw > 0
    result = bootstrap_summary([{'focus_raw_scores':raw.tolist(),
                                 'candidate_team_wins':wins.tolist()}],41)
    assert result['games'] == 12
    assert result['equal_role_expected_score']['mean'] == 0
    assert result['equal_role_team_win_rate']['mean'] == .5
    assert result['equal_role_expected_score']['ci95'] == [-1.,1.]
    for role in result['roles'].values():
        assert role['expected_score']['mean'] == 0
