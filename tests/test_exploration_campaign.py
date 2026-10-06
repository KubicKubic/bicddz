import copy
from pathlib import Path
import pytest
from flax import serialization
from ddz.exploration_campaign import revised_config, migrate, TRANSITION
from ddz.switch_distributed_gae import equal
from ddz.scale_rollout_campaign import read, write
from ddz.local_a100_controller import follow_architecture
from scripts.prepare_exploration_resume import owned_source, supersede_trim
from tests.test_trim_rollout_campaign import checkpoint
from tests.test_local_advantage_handoff import setup


def test_serialized_resume_preserves_all_weights_adam_ages_environment_rng_vf_and_rollout_ema():
    saved = checkpoint()
    saved['runtime']['coordinate_births'] = {'w': saved['train']['opt_state']['births']}
    before = copy.deepcopy(saved)
    cfg = revised_config(saved['config']); result = migrate(saved, cfg, saved['iteration'])
    restored = serialization.msgpack_restore(serialization.msgpack_serialize(result))
    assert equal(saved, before)
    for field in ('train', 'env', 'key', 'iteration', 'ema'):
        assert equal(saved[field], restored[field])
    for name in saved['runtime']:
        assert equal(saved['runtime'][name], restored['runtime'][name])
    assert restored['config']['ppo']['random_action_prob'] == .02
    assert restored['config']['ppo']['adv_keep_fraction'] == .5
    migrations = restored['runtime']['config_migrations']
    assert [m['field'] for m in migrations] == ['ppo.random_action_prob', 'ppo.adv_keep_fraction']


@pytest.mark.parametrize('fault', ['lr', 'gamma', 'architecture', 'clock', 'ema', 'ranks'])
def test_unregistered_changes_or_partial_checkpoint_are_rejected(fault):
    saved = checkpoint(); cfg = revised_config(saved['config']); at = saved['iteration']
    if fault == 'lr': cfg['ppo']['lr'] = .0001
    if fault == 'gamma': cfg['ppo']['gamma'] = .99
    if fault == 'architecture': cfg['model'] = {'width': 300}
    if fault == 'clock': at += 1
    if fault == 'ema': saved['ema']['decay'] = .99
    if fault == 'ranks': saved['key'] = saved['key'][:1]
    with pytest.raises(ValueError): migrate(saved, cfg, at)


def exploration_setup(tmp_path):
    cfg, regular, root, campaign, run, proof = setup(tmp_path)
    original = read(Path(regular['run_dir']) / 'config.json')
    write(run / 'config.json', revised_config(original))
    registration = read(campaign / 'REQUEST.json'); registration['transition'] = TRANSITION
    write(campaign / 'REQUEST.json', registration)
    evidence = read(proof); evidence.update(parameters=8014192, random_action_prob=.02)
    write(proof, evidence)
    return cfg, regular, root, campaign, run, proof


def test_controller_follows_registered_exploration_and_crop_and_retains_500_step_protocol(tmp_path):
    cfg, regular, root, campaign, run, proof = exploration_setup(tmp_path)
    result = follow_architecture(cfg, regular, root)
    assert result['run_dir'] == str(run) and result['initial_steps'] == [51008]
    assert result['protocol']['interval_updates'] == 500
    assert read(root / 'architecture_handoff.json')['transition'] == TRANSITION
    assert follow_architecture(cfg, result, root) is result


@pytest.mark.parametrize('fault', ['proof_probability', 'configuration_probability', 'unregistered'])
def test_controller_rejects_unproved_probability_or_unregistered_handoff(tmp_path, fault):
    cfg, regular, root, campaign, run, proof = exploration_setup(tmp_path)
    if fault == 'proof_probability':
        data = read(proof); data.pop('random_action_prob'); write(proof, data)
    if fault == 'configuration_probability':
        data = read(run / 'config.json'); data['ppo']['random_action_prob'] = .03
        write(run / 'config.json', data)
    if fault == 'unregistered':
        data = read(campaign / 'REQUEST.json'); data.pop('transition')
        write(campaign / 'REQUEST.json', data)
    with pytest.raises(RuntimeError): follow_architecture(cfg, regular, root)


def test_nested_architecture_source_is_owned_but_other_task_is_never_interrupted(tmp_path):
    old = tmp_path / 'source'
    active = {'id': 'active', 'command': 'python -m ddz.scale_architecture_v6 --root ' + str(old)}
    production = {'queue_id': 'active', 'state': 'running'}
    owned_source(active, old, production)
    with pytest.raises(RuntimeError): owned_source(active, tmp_path / 'other', production)
    with pytest.raises(RuntimeError): owned_source(active, old, {**production, 'queue_id': 'other'})


def test_exact_pending_crop_is_archived_without_rewriting_frozen_dependencies(tmp_path):
    import hashlib
    queue = tmp_path / 'queue'; previous = tmp_path / 'old'; root = tmp_path / 'new'
    previous.mkdir(); frozen = previous / 'REQUEST.json'; write(frozen, {'immutable': True})
    command = 'python -m ddz.trim_rollout_campaign --root ' + str(previous)
    record = {'id': 'trim', 'status': 'pending', 'command': command,
              'command_sha256': hashlib.sha256(command.encode()).hexdigest()}
    (queue / 'pending').mkdir(parents=True); (queue / 'interrupted').mkdir()
    write(queue / 'pending/trim.json', record)
    original = frozen.read_bytes()
    supersede_trim(queue, record, previous, root)
    assert frozen.read_bytes() == original and not (queue / 'pending/trim.json').exists()
    assert read(queue / 'interrupted/trim.json')['replacement_root'] == str(root)
    assert read(previous / 'SUPERSEDED.json')['never_started']
