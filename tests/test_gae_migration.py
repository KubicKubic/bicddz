"""Only the GAE setting changes; serialization retains all eight-rank state."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from flax import serialization
from ddz.switch_distributed_gae import migrate,equal,cancel_old_followups


def test_checkpoint_migration_preserves_weights_adam_all_environments_and_rng():
    old={'config':{'envs':8192,'ppo':{'gamma':1.,'lambda':.95,'minibatch':16384}},
         'train':{'step':np.int32(2500000),'params':{'w':np.arange(17,dtype=np.float32)},
                  'opt_state':{'mu':np.linspace(-1,1,17,dtype=np.float32),'count':np.int32(2500000)}},
         'env':{'turn':np.arange(8192,dtype=np.int32)%3,'history':np.arange(64,dtype=np.int32).reshape(8,8)},
         'key':np.arange(16,dtype=np.uint32).reshape(8,2),'iteration':1000,
         'runtime':{'source_sha256':'fixed','vf_coef':.037,'stable_updates':800,
                    'arena':{'pool_id':np.zeros(8192,dtype=np.int32)}}}
    cfg=copy.deepcopy(old['config']);cfg['ppo']['gae_clock']='own'
    new=migrate(old,cfg,'fixed',1000)
    restored=serialization.msgpack_restore(serialization.msgpack_serialize(new))
    for key in ('train','env','key','iteration'):assert equal(old[key],restored[key])
    assert equal(old['runtime']['arena'],restored['runtime']['arena'])
    assert restored['runtime']['vf_coef']==.037 and restored['runtime']['stable_updates']==0
    assert restored['config']['ppo']['gae_clock']=='own' and 'gae_clock' not in old['config']['ppo']
    changed=copy.deepcopy(cfg);changed['ppo']['gamma']=.99
    with pytest.raises(RuntimeError,match='Only'):migrate(old,changed,'fixed',1000)
    with pytest.raises(RuntimeError,match='identity'):migrate(old,cfg,'tampered',1000)


def test_cancel_only_old_ddz_followups_keep_unrelated_queue_items(tmp_path):
    queue=tmp_path/'queue';(queue/'pending').mkdir(parents=True);(queue/'interrupted').mkdir()
    old=tmp_path/'old';commands=[
        f'cd /code && python -u -m ddz.cluster_distributed_job --root {old} --phase production --steps 2000 --resume',
        f'python -m ddz.cluster_distributed_job --root {tmp_path / "other"} --phase production --steps 1000',
        'python /Q/finance_runner.py config.json']
    for i,cmd in enumerate(commands):
        (queue/'pending'/f'{i}.json').write_text(json.dumps({'id':str(i),'status':'pending',
             'command':cmd,'command_sha256':hashlib.sha256(cmd.encode()).hexdigest()}))
    assert cancel_old_followups(queue,old)==['0']
    assert sorted(p.name for p in (queue/'pending').glob('*.json'))==['1.json','2.json']
    receipt=json.loads((queue/'interrupted'/'0.json').read_text())
    assert receipt['status']=='interrupted' and receipt['command']==commands[0]
