"""Grow useful attention banks and insert blocks, preserving the V5 function.

Each old attention head retains its original 32 coordinates and temperature.
Additional banks have random Q/K/V and zero output projections. Added blocks have
zero attention/FFN outputs. Residual widths and all existing FFNs stay intact.
"""
import copy
import re
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


def revised_config(old):
    cfg = copy.deepcopy(old)
    m = cfg['model']
    if (m['width'], m['heads'], m['layers'], m['memory_layers'], m['ff'], m['memory_ff']) != (
            192, 6, 4, 3, 1536, 512):
        raise ValueError('V6 preset requires the audited production V5 architecture')
    if m.get('attention_width', 0) or m.get('inherited_layers', 0):
        raise ValueError('Source already has an attention growth migration')
    m.update(attention_width=320, attention_heads=10,
             memory_attention_width=256, memory_attention_heads=8,
             layers=6, memory_layers=4, inherited_layers=4,
             inherited_memory_layers=3, new_layer_ff=768)
    cfg['parameter_limit'] = 8_100_000
    cfg['model_family'] = 'V6'
    return cfg


def validate_growth(source, target):
    """Do not accept a shape-compatible migration that changes head temperature."""
    fixed = ('width', 'heads', 'ff', 'memory_ff', 'wing_rank', 'action_width',
             'interaction_width', 'action_hidden', 'bf16', 'attention_backend')
    for key in fixed:
        if source.get(key) != target.get(key):
            raise ValueError('Unsupported architecture change: ' + key)
    from .model_v5 import InteractionMoveTransformer
    a, b = InteractionMoveTransformer(**source), InteractionMoveTransformer(**target)
    for memory in (False, True):
        old, new = a.attention_options(memory), b.attention_options(memory)
        if (new['qkv_features'] < old['qkv_features'] or
                old['qkv_features']//old['num_heads'] != new['qkv_features']//new['num_heads']):
            raise ValueError('Attention migration must retain each inherited head dimension')
    if (target['inherited_layers'] != source['layers'] or
            target['inherited_memory_layers'] != source['memory_layers']):
        raise ValueError('Inherited block counts differ from source')
    for count, inherited in ((b.layers, b.inherited_layers), (b.memory_layers, b.inherited_memory_layers)):
        order = b.block_order(count, inherited)
        if sorted(order) != list(range(count)):
            raise ValueError('Invalid block execution order')


def grow_parameters(template, source, source_model, target_model):
    validate_growth(source_model, target_model)
    old = flatten_dict(source)
    target = flatten_dict(template)
    for name, value in old.items():
        if name not in target:
            raise ValueError('Missing inherited parameter: ' + '.'.join(name))
        value = np.asarray(value)
        new = np.asarray(target[name]).copy()
        if value.shape == new.shape:
            target[name] = value.copy()
            continue
        if not re.fullmatch(r'(self|cross|memory_self)\d+', name[0]):
            raise ValueError('Only attention projection banks may widen')
        if value.ndim != new.ndim or any(a>b for a,b in zip(value.shape,new.shape)):
            raise ValueError('Invalid attention growth: ' + '.'.join(name))
        if name[1] == 'out':
            new[:] = 0
        new[tuple(slice(0,n) for n in value.shape)] = value
        target[name] = new
    for name, value in list(target.items()):
        if name in old:
            continue
        attention = re.fullmatch(r'(self|cross|memory_self)(_extra)?\d+', name[0])
        output = attention and name[1] == 'out'
        if output or re.fullmatch(r'(ff_out|memory_ff_out)\d+', name[0]):
            target[name] = np.zeros_like(value)
    return unflatten_dict(target)


def inherited_births(params, origin):
    """Recover all previous coordinate ages; a saved map takes precedence."""
    from .optimizer_efficiency import coordinate_births
    if origin.get('coordinate_births') is not None:
        result = origin['coordinate_births']
        if set(flatten_dict(result)) != set(flatten_dict(params)):
            raise ValueError('Saved coordinate birth map differs from parameters')
        return result
    return coordinate_births(params, params, origin.get('optimizer_origin'), 0)


def grow_births(template, source, old_births, fork_step):
    old, births = flatten_dict(source), flatten_dict(old_births)
    result = {}
    for name, value in flatten_dict(template).items():
        if name not in old:
            result[name] = int(fork_step)
            continue
        prior = np.asarray(births[name], np.int32)
        if np.shape(old[name]) == value.shape:
            result[name] = int(prior) if prior.ndim == 0 else prior.copy()
        else:
            new = np.full(value.shape, fork_step, np.int32)
            new[tuple(slice(0,n) for n in np.shape(old[name]))] = prior
            result[name] = new
    return unflatten_dict(result)
