"""Restore model sampling and raise its original entropy coefficient."""
import copy
import time
import numpy as np
from .switch_distributed_gae import equal


TRANSITION = 'model_entropy_01_without_random_actions'


def revised_config(old):
    if (old.get('model_family') != 'V6'
            or old['ppo'].get('random_action_prob') != .02
            or old['ppo'].get('entropy_end') != .002
            or old['ppo'].get('adv_keep_fraction') != .5):
        raise ValueError('Registered V6 p=.02, entropy=.002 and top50 source required')
    if old['global_source_iteration'] <= old['ppo']['entropy_updates']:
        raise ValueError('The registered continuation requires the completed entropy schedule')
    cfg = copy.deepcopy(old)
    cfg['ppo'].update(random_action_prob=0., entropy_end=.01)
    return cfg


def migrate(saved, cfg, at):
    if cfg != revised_config(saved['config']):
        raise ValueError('Only random_action_prob and entropy_end may change')
    if (saved['iteration'] != at or np.shape(saved['key']) != (8, 2)
            or 'ema' not in saved or 'runtime' not in saved):
        raise ValueError('Complete eight-rank checkpoint with EMA and runtime required')
    if saved['ema']['decay'] != cfg.get('ema_decay', .999):
        raise ValueError('EMA decay differs')
    result = {**saved, 'config': copy.deepcopy(cfg),
              'runtime': copy.deepcopy(saved['runtime'])}
    for field in ('random_action_prob', 'entropy_end'):
        result['runtime'].setdefault('config_migrations', []).append({
            'field': 'ppo.' + field, 'from': saved['config']['ppo'][field],
            'to': cfg['ppo'][field], 'iteration': at,
            'reason': 'user requested original sampling and stronger model entropy',
            'time': time.time()})
    for name in ('train', 'env', 'key', 'iteration', 'ema'):
        if not equal(saved[name], result[name]):
            raise ValueError('Retained state differs: ' + name)
    for name in saved['runtime']:
        if name != 'config_migrations' and not equal(saved['runtime'][name], result['runtime'][name]):
            raise ValueError('Retained runtime differs: ' + name)
    return result


def verify_entropy_rows(rows):
    if any(row.get('random_action_prob') != 0.
           or abs(row.get('entropy_coef', 0.) - .01) > 1e-12 for row in rows):
        raise RuntimeError('Fresh rollout/update must use p=0 and entropy coefficient .01')


if __name__ == '__main__':
    from .trim_rollout_campaign import main
    try:
        main()
    except Exception as error:
        import sys
        from pathlib import Path
        from .scale_rollout_campaign import write
        root = Path(sys.argv[sys.argv.index('--root') + 1])
        write(root / 'FAILED.json', {'error': str(error), 'time': time.time()})
        raise
