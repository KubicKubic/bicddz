"""Model weight EMA, advanced once per rollout, independently of optimizer steps."""
import jax
import jax.numpy as j


def validate_decay(decay):
    decay=float(decay)
    if not 0<=decay<1:raise ValueError('EMA decay must be in [0, 1)')
    return decay


def update_weights(previous, current, decay=0.999):
    decay=validate_decay(decay)
    def blend(old,new):
        # Keep the stored parameter dtype; accumulate float weights in FP32.
        result=j.asarray(old,j.float32)*decay+j.asarray(new,j.float32)*(1-decay)
        return result.astype(old.dtype)
    return jax.tree_util.tree_map(blend,previous,current)
