"""Fresh-coordinate warmup preserves mature updates and resumes its Adam clock."""
import jax
import jax.numpy as j
import numpy as np
import optax
import pytest
from flax import serialization
from ddz.optimizer_efficiency import birth_corrected_adam,V6_NEW_COORDINATE_WARMUP_STEPS


def test_new_coordinate_ramp_and_mature_updates_are_exact_across_cold_restore():
    assert V6_NEW_COORDINATE_WARMUP_STEPS==1024
    params={'w':j.ones(4)};births={'w':j.array([0,400,10000,10000],j.int32)}
    base=optax.scale_by_adam();state=base.init(params)._replace(count=j.int32(10000),
        mu={'w':j.array([.2,-.3,0,0])},nu={'w':j.array([.1,.2,0,0])})
    original=birth_corrected_adam(births);warm=birth_corrected_adam(births,32)
    raw_state=state
    baseline=jax.jit(lambda g,s:original.update(g,s))
    step=jax.jit(lambda g,s:warm.update(g,s))
    for age in range(1,35):
        g={'w':j.array([.3,-.2,.7,-.4])}
        expected,raw_state=baseline(g,raw_state)
        actual,state=step(g,state)
        np.testing.assert_array_equal(actual['w'][:2],expected['w'][:2])
        np.testing.assert_allclose(actual['w'][2:],expected['w'][2:]*min(age/32,1),rtol=1e-6)
        if age==17:
            restored=serialization.from_bytes(base.init(params),serialization.to_bytes(state))
            np.testing.assert_array_equal(restored.count,state.count)
            for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a,b)
            state=restored
            # Reconstruct the transform as a cold process would.
            warm=birth_corrected_adam(births,32);step=jax.jit(lambda g,s:warm.update(g,s))


@pytest.mark.parametrize('steps',[-1,1.5,None])
def test_invalid_new_coordinate_warmup_cannot_change_optimizer_silently(steps):
    with pytest.raises(ValueError):birth_corrected_adam({'w':0},steps)
