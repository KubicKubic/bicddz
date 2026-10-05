import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from flax import serialization
from ddz.export_checkpoint import export_bundle, verify_bundle
from ddz.qoj_deployment import prepare, verify_frozen


def training(tmp_path):
    run = tmp_path / 'training'
    run.mkdir()
    (run / 'ema').mkdir()
    raw = {'w': np.array([1., 2.], np.float32)}
    ema = {'w': np.array([.5, 1.5], np.float32)}
    saved = {'iteration': 7, 'config': {'ema_decay': .999, 'global_source_iteration': 10},
             'train': {'params': raw, 'step': 100},
             'ema': {'params': ema, 'decay': .999, 'updates': 3},
             'env': {}, 'runtime': {}, 'key': np.array([1, 2], np.uint32)}
    for name, data in [('latest.msgpack', saved), ('policy_0000007.msgpack', raw),
                       ('ema/policy_0000007.msgpack', ema)]:
        (run / name).write_bytes(serialization.msgpack_serialize(data))
    return run


def test_bundle_pins_matching_raw_ema_full_state_and_detects_corruption(tmp_path):
    run = training(tmp_path)
    out = tmp_path / 'models'
    manifest = export_bundle(run, out)
    assert manifest['global_step'] == 17 and manifest['ema_rollout_updates'] == 3
    assert manifest['parameters'] == 2 and verify_bundle(out) == manifest
    assert (out / 'latest_checkpoint.msgpack').read_bytes() == (run / 'latest.msgpack').read_bytes()
    with pytest.raises(FileExistsError): export_bundle(run, out)
    (out / 'policy.msgpack').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'): verify_bundle(out)


@pytest.mark.parametrize('fault', ['mismatch', 'missing_ema'])
def test_bundle_refuses_incomplete_or_mixed_checkpoint_publication(tmp_path, fault):
    run = training(tmp_path)
    if fault == 'mismatch':
        (run / 'policy_0000007.msgpack').write_bytes(serialization.msgpack_serialize(
            {'w': np.array([99., 2.], np.float32)}))
    else:
        (run / 'ema/policy_0000007.msgpack').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        export_bundle(run, tmp_path / 'models')
    assert not (tmp_path / 'models').exists()


def test_deployment_freezes_worker_code_weights_without_copying_credentials(tmp_path):
    repo = tmp_path / 'relocated clone'
    (repo / 'ddz').mkdir(parents=True)
    (repo / 'ddz/__init__.py').write_text('')
    (repo / 'ddz/qoj_deployment.py').write_text('# frozen code\n')
    (repo / 'deploy/qoj').mkdir(parents=True)
    (repo / 'deploy/qoj/worker.py').write_text('# worker\n')
    models = repo / 'models'
    export_bundle(training(tmp_path), models)
    token = tmp_path / 'private-key'
    token.write_text('TEST_SECRET_DO_NOT_COPY')
    root = repo / 'runs/match'
    cfg = prepare(repo, root, models, token, 'user', 'http://localhost', 'test', '/usr/bin/python3')
    frozen = Path(cfg['active_code']).parent
    verify_frozen(frozen)
    assert Path(cfg['model_dir']).is_relative_to(root)
    assert cfg['token_file'] == str(token)
    assert 'TEST_SECRET_DO_NOT_COPY' not in ''.join(p.read_text(errors='ignore') for p in root.rglob('*') if p.is_file())
    assert prepare(repo, root, models, token, 'user', 'http://localhost', 'test', '/usr/bin/python3') == cfg
    (repo / 'ddz/qoj_deployment.py').write_text('# later edits\n')
    verify_frozen(frozen)  # Running release stays independent of source edits.
    (frozen / 'worker.py').write_text('# corrupted\n')
    with pytest.raises(ValueError, match='Frozen'): verify_frozen(frozen)
