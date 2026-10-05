"""Grow V4 into V5 while retaining policy, optimizer and environment state."""
import re
from pathlib import Path
import numpy as np
import jax.numpy as j
from flax import serialization
from flax.traverse_util import flatten_dict,unflatten_dict


def warm_start_v4(params,path):
    source=flatten_dict(serialization.msgpack_restore(Path(path).read_bytes()))
    target=flatten_dict(params)
    copied=[]
    for name,old in source.items():
        if name not in target: raise ValueError(f'missing source parameter {name}')
        old=np.asarray(old);new=np.asarray(target[name]).copy()
        if old.shape==new.shape:
            target[name]=j.asarray(old)
        elif name[0].startswith(('ff_in','ff_out','memory_ff_in','memory_ff_out')) and all(
                a<=b for a,b in zip(old.shape,new.shape)) and old.ndim==new.ndim:
            if 'ff_out' in name[0]: new[:]=0
            new[tuple(slice(0,n) for n in old.shape)]=old
            target[name]=j.asarray(new)
        else:
            raise ValueError(f'unsupported shape change {name}: {old.shape} -> {new.shape}')
        copied.append('.'.join(name))
    # Added attention/FF blocks start as identities. Output projections learn
    # first, then their interior weights; no frozen or permanently dead branch.
    for name,value in list(target.items()):
        if name in source: continue
        attention=(re.fullmatch(r'(cross|self|memory_self)\d+',name[0])
                   and len(name)>1 and name[1]=='out')
        ff=re.fullmatch(r'(ff_out|memory_ff_out)\d+',name[0])
        if attention or ff: target[name]=j.zeros_like(value)
    return unflatten_dict(target),copied


def grow_optimizer(source,target,path=()):
    """Copy Adam moments/counts; new rows and parameters have zero moments."""
    if isinstance(target,dict):
        if not isinstance(source,dict): raise ValueError(f'optimizer schema changed: {path}')
        if not set(source)<=set(target): raise ValueError(f'optimizer keys missing: {path}')
        return {k:grow_optimizer(source[k],v,path+(k,)) if k in source else v
                for k,v in target.items()}
    old=np.asarray(source);new=np.asarray(target).copy()
    if old.shape==new.shape: return old
    if old.ndim!=new.ndim or not all(a<=b for a,b in zip(old.shape,new.shape)):
        raise ValueError(f'optimizer shape mismatch: {path}')
    new[:]=0;new[tuple(slice(0,n) for n in old.shape)]=old
    return new


def upgrade_state(train_state,source_payload,source_policy):
    params,copied=warm_start_v4(train_state.params,source_policy)
    state=serialization.to_state_dict(train_state.replace(params=params))
    state['opt_state']=grow_optimizer(source_payload['train']['opt_state'],state['opt_state'])
    state['step']=source_payload['train']['step']
    return serialization.from_state_dict(train_state,state),copied
