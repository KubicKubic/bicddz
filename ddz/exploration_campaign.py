"""Registered V6 exploration change, retaining the full checkpoint state."""
import copy
import time
import numpy as np
from .policy_v5 import validate_random_action_prob
from .switch_distributed_gae import equal


TRANSITION = 'random_action_02_and_advantage_trim'


def revised_config(old):
    if old.get('model_family') != 'V6':
        raise ValueError('The registered exploration handoff requires V6')
    previous = validate_random_action_prob(old['ppo'].get('random_action_prob', 0.))
    if previous >= .02 or old['ppo'].get('adv_keep_fraction', 1.) not in (1., .5):
        raise ValueError('Inspect an already changed exploration or trimming configuration')
    cfg = copy.deepcopy(old)
    cfg['ppo'].update(random_action_prob=.02, adv_keep_fraction=.5)
    return cfg


def migrate(saved, cfg, at):
    if cfg != revised_config(saved['config']):
        raise ValueError('Only registered exploration and advantage selection may change')
    if (saved['iteration'] != at or np.shape(saved['key']) != (8, 2)
            or 'ema' not in saved or 'runtime' not in saved):
        raise ValueError('Complete eight-rank checkpoint with EMA and runtime required')
    if saved['ema']['decay'] != cfg.get('ema_decay', .999):
        raise ValueError('EMA decay differs')
    result = {**saved, 'config': copy.deepcopy(cfg),
              'runtime': copy.deepcopy(saved['runtime'])}
    for field, default in (('random_action_prob', 0.), ('adv_keep_fraction', 1.)):
        previous = saved['config']['ppo'].get(field, default)
        following = cfg['ppo'][field]
        if previous != following:
            result['runtime'].setdefault('config_migrations', []).append({
                'field': 'ppo.' + field, 'from': previous, 'to': following,
                'iteration': at, 'reason': 'explicit user instruction', 'time': time.time()})
    for name in ('train', 'env', 'key', 'iteration', 'ema'):
        if not equal(saved[name], result[name]):
            raise ValueError('Retained state differs: ' + name)
    for name in saved['runtime']:
        if name != 'config_migrations' and not equal(saved['runtime'][name], result['runtime'][name]):
            raise ValueError('Retained runtime differs: ' + name)
    return result


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
