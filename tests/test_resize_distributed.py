import copy
import hashlib
import json
import numpy as np
import pytest
from flax import serialization
from ddz.resize_distributed import revised_config, expand_checkpoint, verify_source
from ddz.switch_distributed_gae import equal
from test_rollout_scale_ema import source


def saved_state():
    saved = source()
    saved['config'].update(per_gpu_minibatch=2, ema_decay=.999)
    saved['ema'] = {'params': {'w': np.array([2., -1.], np.float32)}, 'decay': .999, 'updates': 1234}
    return saved


def test_resize_retains_adam_rng_ema_and_each_ranks_original_envs_after_serialization():
    saved = saved_state()
    original = copy.deepcopy(saved)
    cfg = revised_config(saved['config'])
    fresh = {k: np.full_like(v, 2) for k, v in saved['env'].items()}
    result = serialization.msgpack_restore(serialization.msgpack_serialize(
        expand_checkpoint(saved, cfg, fresh, saved['iteration'])))
    assert equal(saved, original)
    for key in ('train', 'key', 'ema', 'iteration'):
        assert equal(result[key], saved[key])
    assert cfg['horizon'] == saved['config']['horizon']
    assert cfg['updates'] == saved['config']['updates']
    assert cfg['ppo']['minibatch'] == 2 * saved['config']['ppo']['minibatch']
    for key, value in saved['env'].items():
        shaped = result['env'][key].reshape((8, 4) + value.shape[1:])
        np.testing.assert_array_equal(shaped[:, :2], value.reshape((8, 2) + value.shape[1:]))
        np.testing.assert_array_equal(shaped[:, 2:], fresh[key].reshape((8, 2) + value.shape[1:]))
    for key, value in saved['runtime']['arena'].items():
        shaped = result['runtime']['arena'][key].reshape(8, 4)
        np.testing.assert_array_equal(shaped[:, :2], value.reshape(8, 2))
        assert not shaped[:, 2:].any()
    assert result['runtime']['distributed']['per_gpu_minibatch'] == 4
    assert result['runtime']['vf_coef'] == saved['runtime']['vf_coef']


@pytest.mark.parametrize('fault', ['horizon', 'lr', 'ema', 'boundary', 'dtype'])
def test_resize_rejects_unapproved_config_changes_and_incomplete_states(fault):
    saved = saved_state()
    cfg = revised_config(saved['config'])
    fresh = {k: np.zeros_like(v) for k, v in saved['env'].items()}
    at = saved['iteration']
    if fault == 'horizon': cfg['horizon'] *= 2
    if fault == 'lr': cfg['ppo']['lr'] *= .1
    if fault == 'ema': saved.pop('ema')
    if fault == 'boundary': at += 1
    if fault == 'dtype': fresh['history'] = fresh['history'].astype(np.int32)
    with pytest.raises(RuntimeError):
        expand_checkpoint(saved, cfg, fresh, at)


def test_reviewed_immediate_pause_requires_exact_checkpoint_and_nccl_proof(tmp_path):
    run = tmp_path / 'production/training'
    run.mkdir(parents=True)
    (run / 'latest.msgpack').write_bytes(b'complete retained state')
    (run / 'status.json').write_text(json.dumps({'state': 'stopped', 'iteration': 123, 'nranks': 8}))
    queue = tmp_path / 'queue'
    (queue / 'interrupted').mkdir(parents=True)
    (queue / 'interrupted/task.json').write_text(json.dumps({'status': 'interrupted'}))
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'passed': True, 'nranks': 8, 'NCCL_evidence': ['real log evidence']}))
    receipt = tmp_path / 'pause.json'
    pause = {'interrupted_queue_id': 'task', 'retained_checkpoint_iteration': 123,
             'remote_training_pid': 2147483647, 'source_health_proof': str(proof),
             'retained_checkpoint_sha256': hashlib.sha256((run / 'latest.msgpack').read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(pause))
    request = {'queue': str(queue), 'source_pause_receipt': str(receipt)}
    assert verify_source(request, run, 123)
    with pytest.raises(RuntimeError, match='identity'): verify_source(request, run, 124)
    (run / 'status.json').write_text(json.dumps({'state': 'running', 'iteration': 123, 'nranks': 8}))
    with pytest.raises(RuntimeError, match='identity'): verify_source(request, run, 123)
    pause['reviewed_stale_running_status'] = True
    receipt.write_text(json.dumps(pause))
    assert verify_source(request, run, 123)
    (run / 'latest.msgpack').write_bytes(b'changed checkpoint')
    with pytest.raises(RuntimeError, match='identity'): verify_source(request, run, 123)
