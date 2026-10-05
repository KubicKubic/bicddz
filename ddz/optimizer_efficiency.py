"""Retain Adam ages across V4 -> V5 -> auxiliary-head migrations."""
import numpy as np
import jax
import jax.numpy as j
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
from .optimizer_v5 import new_coordinate_mask
from optax.tree_utils import tree_bias_correction


def coordinate_births(params, source_params, origin, fork_step):
    legacy = new_coordinate_mask(params, origin['source_shapes']) if origin else None
    flat_legacy = flatten_dict(legacy) if legacy else {}
    old = flatten_dict(source_params)
    births = {}
    for name, value in flatten_dict(params).items():
        if name not in old:
            births[name] = int(fork_step)
        else:
            if np.shape(old[name]) != value.shape:
                raise ValueError(f'unsupported shape change {name}')
            mask = flat_legacy.get(name, False)
            births[name] = (int(origin['source_step']) if mask else 0) if isinstance(mask, bool) else j.where(mask, origin['source_step'], 0)
    return unflatten_dict(births)


def birth_corrected_adam(births):
    base = optax.scale_by_adam()
    def update(grads, state, params=None):
        updates, new = base.update(grads, state, params)
        def correct(u, mu, nu, birth):
            if isinstance(birth, int) and birth == 0:
                return u
            age = j.maximum(new.count - j.asarray(birth, j.int32), 1)
            mu_hat = tree_bias_correction(mu, .9, age)
            nu_hat = tree_bias_correction(nu, .999, age)
            corrected = mu_hat / (j.sqrt(nu_hat) + 1e-8)
            return corrected if isinstance(birth, int) else j.where(birth > 0, corrected, u)
        return jax.tree_util.tree_map(correct, updates, new.mu, new.nu, births), new
    return optax.GradientTransformation(base.init, update)


def sampled_lr(cfg, source_step):
    """Schedule progresses in fresh-data updates, independently of PPO epochs.

    At the fork, this equals the exact inherited V5 LR. Applied optimizer
    steps advance by the legacy minibatches per newly collected rollout,
    regardless of epochs, KL stopping, or the long-history minibatch size.
    """
    from .train_v2 import learning_rate_schedule
    import copy
    schedule_cfg = copy.deepcopy(cfg)
    schedule_cfg['ppo']['epochs'] = 1
    base = learning_rate_schedule(schedule_cfg)
    steps = cfg['envs'] * cfg['horizon'] // cfg['ppo']['minibatch']
    return lambda fresh_updates: base(source_step + fresh_updates * steps)
