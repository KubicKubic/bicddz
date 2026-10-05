"""Retain mature Adam state while new V5 coordinates use their own age."""
import numpy as np
import jax
import jax.numpy as j
import optax
from optax.tree_utils import tree_bias_correction
from flax.traverse_util import flatten_dict,unflatten_dict


def new_coordinate_mask(params,source_shapes):
    masks={}
    for name,value in flatten_dict(params).items():
        old=source_shapes.get('.'.join(name))
        if old is None:masks[name]=True
        elif tuple(old)==value.shape:masks[name]=False
        else:
            if len(old)!=value.ndim or any(a>b for a,b in zip(old,value.shape)):
                raise ValueError(f'invalid source coordinate shape: {name}')
            mask=np.ones(value.shape,np.bool_)
            mask[tuple(slice(0,n) for n in old)]=False
            masks[name]=j.asarray(mask)
    return unflatten_dict(masks)


def age_corrected_adam(mask,source_step):
    """Same ScaleByAdamState schema; only new-coordinate bias correction differs.

    Old coordinates use Optax verbatim. New/expanded coordinates have zero
    moments at migration, so their bias corrections use count-source_step.
    """
    base=optax.scale_by_adam()
    def update(grads,state,params=None):
        updates,new=base.update(grads,state,params)
        age=j.maximum(new.count-j.int32(source_step),j.int32(1))
        mu=tree_bias_correction(new.mu,.9,age)
        nu=tree_bias_correction(new.nu,.999,age)
        def correct(u,m,v,is_new):
            if isinstance(is_new,bool) and not is_new:return u
            fresh=m/(j.sqrt(v)+1e-8)
            return fresh if isinstance(is_new,bool) else j.where(is_new,fresh,u)
        updates=jax.tree_util.tree_map(correct,updates,mu,nu,mask)
        return updates,new
    return optax.GradientTransformation(base.init,update)
