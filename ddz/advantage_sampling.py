"""Exact global absolute-advantage trimming without gathering observations."""
import math
import jax
import jax.numpy as j


def validate_keep_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('adv_keep_fraction must be in (0, 1]')
    return value


def keep_largest_advantages(advantage, eligible, fraction, axis_name=None):
    """Keep floor(fraction * global eligible count), including both signs.

    Positive float32 magnitudes have the same order as their uint32 bits.
    Four 256-bin histogram reductions find the exact cutoff; ties use global
    rank followed by local row order. Only small histograms cross devices.
    GAE and normalization must be computed on the original complete rollout.
    """
    fraction = validate_keep_fraction(fraction)
    if fraction == 1:
        return eligible
    shape = advantage.shape
    bits = jax.lax.bitcast_convert_type(j.abs(advantage.reshape(-1).astype(j.float32)), j.uint32)
    eligible = eligible.reshape(-1)
    def total(x):
        return x if axis_name is None else jax.lax.psum(x, axis_name)
    count = j.floor(total(j.sum(eligible)) * fraction).astype(j.int32)
    remaining = j.maximum(count, 1)
    candidates = eligible
    cutoff = j.uint32(0)
    for shift in (24, 16, 8, 0):
        digit = ((bits >> shift) & 255).astype(j.int32)
        histogram = total(j.bincount(digit, weights=candidates.astype(j.int32), length=256))
        bucket = 255 - j.argmax(j.cumsum(histogram[::-1]) >= remaining)
        remaining -= j.sum(j.where(j.arange(256) > bucket, histogram, 0))
        candidates &= digit == bucket
        cutoff |= bucket.astype(j.uint32) << shift
    higher = eligible & (bits > cutoff)
    ties = eligible & (bits == cutoff)
    offset = j.int32(0)
    if axis_name is not None:
        counts = jax.lax.all_gather(j.sum(ties), axis_name)
        offset = j.sum(j.where(j.arange(counts.size) < jax.lax.axis_index(axis_name), counts, 0))
    tie_order = offset + j.cumsum(ties.astype(j.int32))
    kept = (higher | (ties & (tie_order <= count - total(j.sum(higher))))) & (count > 0)
    return kept.reshape(shape)


def shuffled_selected_indices(key, selected, epochs):
    """Pack shuffled selected rows first so unused suffix minibatches can skip."""
    selected = selected.reshape(-1)
    def shuffle(k):
        scores = jax.random.uniform(k, selected.shape)
        return j.argsort(j.where(selected, scores, j.inf), stable=True)
    return jax.vmap(shuffle)(jax.random.split(key, epochs))
