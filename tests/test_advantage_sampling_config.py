import copy
import pytest
from ddz.train_efficiency import validate_config
from tests.test_v5_exploration import small_config


def test_local_efficiency_cli_accepts_trim_and_rejects_invalid_fraction():
    base=small_config();cfg=copy.deepcopy(base);cfg['ppo']['adv_keep_fraction']=.5
    validate_config(cfg,base)
    cfg['ppo']['adv_keep_fraction']=0
    with pytest.raises(ValueError,match='adv_keep_fraction'):validate_config(cfg,base)


def test_allowing_trim_does_not_allow_unmatched_ppo_protocol_changes():
    base=small_config();cfg=copy.deepcopy(base)
    cfg['ppo'].update(adv_keep_fraction=.5,gamma=.98)
    with pytest.raises(ValueError,match='unmatched PPO option gamma'):validate_config(cfg,base)
