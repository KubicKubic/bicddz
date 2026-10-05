"""Load publisher-released best models with their exact ResNet observation code."""
import copy
import importlib.util
from pathlib import Path
import torch
from .compare_douzero import file_hash


def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_best_models(weights_dir,repo):
    source=Path(repo)/'Douzero_Resnet/douzero/dmc/models_res.py'
    definitions=module_from_file('edwardpooh_resnet_models',source)
    models={};hashes={}
    for role in ('landlord','landlord_down','landlord_up'):
        path=Path(weights_dir)/f'{role}.ckpt'
        model=definitions.model_dict_resnet[role]()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.eval();models[role]=model;hashes[role]=file_hash(path)
    return models,hashes


def make_observation(repo):
    source=Path(repo)/'Douzero_Resnet/douzero/env/env_res.py'
    definitions=module_from_file('edwardpooh_resnet_observation',source)
    roles=('landlord','landlord_down','landlord_up')
    def observe(infoset):
        # Original DouZero stores just cards, while this publisher stores
        # (actor, cards). Play starts with landlord and rotates deterministically.
        # All other public observation fields use the publisher's code verbatim.
        info=copy.copy(infoset)
        info.card_play_action_seq=[(roles[i%3],cards)
                                  for i,cards in enumerate(infoset.card_play_action_seq)]
        return definitions._get_obs_resnet(info,info.player_position)
    return observe
