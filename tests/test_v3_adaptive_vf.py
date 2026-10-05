import numpy as np
import jax
import jax.numpy as j

from ddz import candidates_v3, env_v2
from ddz.model_v3 import CandidateSetTransformer
from ddz.ppo_v3 import Transition, make_gradient_diagnostic
from ddz.train_v3 import balanced_vf_coefficient


def test_vf_balance_is_bounded_and_ignores_unusable_gradient_ratios():
    cfg = {'vf_coef_min': .001, 'vf_coef_max': 10.}
    assert balanced_vf_coefficient(1., 1., 100., cfg) == .5
    assert balanced_vf_coefficient(1., 100., 1., cfg) == 2.
    assert balanced_vf_coefficient(.1, 0., 1., cfg) == .1
    assert balanced_vf_coefficient(.1, np.nan, 1., cfg) == .1


def test_candidate_policy_and_value_gradient_diagnostic_is_finite():
    model = CandidateSetTransformer(width=48, layers=1, heads=3, ff=96,
        memory_ff=96, candidate_layers=1, candidate_ff=96, bf16=False)
    states = env_v2.batch_reset(jax.random.split(jax.random.PRNGKey(11), 4))
    ids, mask, _ = candidates_v3.batch_legal(states)
    ids, mask = j.asarray(ids), j.asarray(mask)
    obs = jax.vmap(env_v2.observe)(states)
    params = model.init(jax.random.PRNGKey(12), obs, ids, mask)['params']
    logits, value = model.apply({'params': params}, obs, ids, mask)
    index = j.argmax(logits, axis=-1)
    logp = j.take_along_axis(jax.nn.log_softmax(logits), index[:, None], axis=1)[:, 0]
    add_time = lambda x: x[None]
    tr = Transition(jax.tree_util.tree_map(add_time, states), ids[None], mask[None],
        states.turn[None], index[None], logp[None], value[None],
        j.array([[[1., 0., -1.], [0., -1., 1.], [1., -1., 0.], [-1., 1., 0.]]]),
        j.zeros((1, 4), j.bool_))
    diagnose = make_gradient_diagnostic(model,
        {'gamma': 1., 'lambda': .95, 'clip': .2, 'vf_diagnostic_batch': 4})
    norms = diagnose(params, tr, j.zeros((4, 3)), jax.random.PRNGKey(13))
    assert all(np.isfinite(v) and v >= 0 for v in norms.values())
    assert norms['policy_grad_l2'] > 0
    assert norms['value_grad_l2'] > 0
