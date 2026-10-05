"""Inference-only JAX translation of the pinned publisher ResNet and public inputs.

Weights, FP32 arithmetic, evaluation batch normalization and candidate ordering
are preserved. No retraining or candidate truncation is performed.
"""
import jax
import jax.numpy as j
import numpy as np


def convert_weights(model):
    return {name: j.asarray(value.detach().cpu().numpy())
            for name, value in model.state_dict().items()
            if not name.endswith('num_batches_tracked')}


@jax.jit
def resnet_values(p, z, x):
    def conv(a, name, stride=1, padding=1):
        w = p[name+'.weight'].transpose(2, 1, 0)
        return jax.lax.conv_general_dilated(a, w, (stride,),
            ((padding, padding),), dimension_numbers=('NWC', 'WIO', 'NWC'),
            precision=jax.lax.Precision.HIGHEST)
    def bn(a, name):
        return ((a-p[name+'.running_mean']) *
                jax.lax.rsqrt(p[name+'.running_var']+1e-5) *
                p[name+'.weight'] + p[name+'.bias'])
    a = jax.nn.relu(bn(conv(z.transpose(0, 2, 1), 'conv1', 2), 'bn1'))
    for layer in range(1, 4):
        for block in range(2):
            name = f'layer{layer}.{block}'
            stride = 2 if block == 0 else 1
            skip = a
            b = jax.nn.relu(bn(conv(a, name+'.conv1', stride), name+'.bn1'))
            b = bn(conv(b, name+'.conv2'), name+'.bn2')
            if block == 0:
                skip = bn(conv(skip, name+'.shortcut.0', 2, 0), name+'.shortcut.1')
            a = jax.nn.relu(b+skip)
    a = a.transpose(0, 2, 1).reshape((a.shape[0], -1))
    a = j.concatenate((x, x, x, x, a), axis=-1)
    for layer in range(1, 5):
        name = f'linear{layer}'
        a = j.matmul(a, p[name+'.weight'].T,
                     precision=jax.lax.Precision.HIGHEST)+p[name+'.bias']
        a = jax.nn.leaky_relu(a, negative_slope=.01)
    return a[:, 0]


def cards_array(counts):
    counts = j.asarray(counts)
    ordinary = (counts[..., :13, None] > j.arange(4)).reshape(counts.shape[:-1]+(52,))
    return j.concatenate((ordinary, counts[..., 13:] > 0), axis=-1).astype(j.float32)


def cards_array_numpy(counts):
    counts = np.asarray(counts)
    ordinary = (counts[..., :13, None] > np.arange(4)).reshape(counts.shape[:-1]+(52,))
    return np.concatenate((ordinary, counts[..., 13:] > 0), axis=-1).astype(np.float32)


def public_observation(s):
    """Exact publisher 39x54 state rows and 15 scalars, using visible data only."""
    ll = s.landlord
    order = j.array([ll, (ll+2)%3, (ll+1)%3])
    own = s.hands[s.turn]
    unseen = j.array([4]*13+[1, 1])-s.played.sum(axis=0)-own
    left = s.hands.sum(axis=-1)
    counts = j.concatenate((jax.nn.one_hot(left[order[0]]-1, 20),
                            jax.nn.one_hot(left[order[1]]-1, 17),
                            jax.nn.one_hot(left[order[2]]-1, 17)))
    bottom = j.maximum(s.bottom-s.played[ll], 0)
    present = j.minimum(s.hist_len, 32)
    rows = j.arange(32)
    valid = rows >= 32-present
    index = j.clip(s.hist_len-1-(rows-(32-present)), 0, s.history.shape[0]-1)
    history = j.where(valid[:, None], cards_array(s.history[index, :15]), -1.)
    z = j.concatenate((counts[None], cards_array(own)[None],
        cards_array(unseen)[None], cards_array(bottom)[None],
        cards_array(s.played[order]), history), axis=0)
    landlord = j.array([1, .5, 1, 1, 1, 1, 1, 5, -4, 1, 1, 1, 1, 1, 1])
    farmer = j.array([1, .2, 1, 1, 3.5, 1, 1, 5, 4, 1.035, 1, .15, 1, 2.5, 1.2])
    return z, j.where(s.turn == ll, landlord, farmer).astype(j.float32)


batch_public_observation = jax.jit(jax.vmap(public_observation))


@jax.jit
def candidate_values(p, state_z, state_x, game_ids, action_counts):
    z = j.concatenate((cards_array(action_counts)[:, None], state_z[game_ids]), axis=1)
    return resnet_values(p, z, state_x[game_ids])


def padded_candidates(game_ids, counts, minimum=128):
    """Finite compile shapes; padding never removes a legal action."""
    n = len(game_ids)
    size = max(minimum, 1 << (n-1).bit_length())
    ids = np.zeros(size, np.int32)
    actions = np.zeros((size, 15), np.int32)
    ids[:n] = game_ids
    actions[:n] = counts
    return ids, actions
