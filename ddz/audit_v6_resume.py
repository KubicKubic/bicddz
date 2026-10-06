"""Rehearse the real complete checkpoint's 8M migration and cold restore on CPU."""
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import jax
from flax import serialization
from .scale_rollout_campaign import read,write,sha
from .switch_distributed_gae import equal
from .upgrade_v6 import revised_config
from .scale_architecture_v6 import migrate_checkpoint
from .train_efficiency import create
from .train_distributed_v5 import restore_arena,merge_env
from .train_v2 import atomic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if jax.default_backend()!='cpu':raise RuntimeError('Full resume rehearsal is CPU engineering only')
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();raw=(args.run/'latest.msgpack').read_bytes()
    source=serialization.msgpack_restore(raw);cfg=revised_config(source['config'])
    if source['config']!=read(args.run/'config.json'):raise RuntimeError('Current source config differs')
    source_hash=hashlib.sha256(raw).hexdigest();atomic(args.out/'source.msgpack',raw);del raw
    model,state=create(cfg,source)
    migrated=jax.device_get(migrate_checkpoint(source,cfg,serialization.to_state_dict(state),source_hash,args.out/'source.msgpack'))
    for tree in (migrated['train'],migrated['ema']):
        if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree)):
            raise RuntimeError('Full state contains nonfinite weights/moments')
    atomic(args.out/'migrated.msgpack',serialization.msgpack_serialize(migrated))
    restored=serialization.msgpack_restore((args.out/'migrated.msgpack').read_bytes())
    if not equal(migrated,restored):raise RuntimeError('Complete checkpoint serialization is not exact')
    _,cold=create(cfg,restored)
    for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(cold)):
        np.testing.assert_array_equal(a,b)
    arena,key=restore_arena(restored,cfg)
    if not equal(serialization.to_state_dict(merge_env(arena.games)),restored['env']):
        raise RuntimeError('Environment rank partition is not exact')
    np.testing.assert_array_equal(key,source['key'])
    write(args.out/'config.json',cfg)
    result={'passed':True,'scope':'Real full checkpoint, CPU migration/serialization/cold restore; not eight-GPU evidence',
        'source_iteration':source['iteration'],'global_iteration':cfg['global_source_iteration']+source['iteration'],
        'source_sha256':source_hash,'migrated_sha256':sha(args.out/'migrated.msgpack'),
        'parameters':sum(x.size for x in jax.tree_util.tree_leaves(state.params)),
        'optimizer_step':int(state.step),'ema_updates':source['ema']['updates'],'ema_decay':source['ema']['decay'],
        'envs':cfg['envs'],'rng_shape':list(key.shape),'inherited_weights_moments_and_ages_exact':True,
        'all_environment_fields_and_arena_restored':True,'cold_restore_exact':True,
        'seconds':time.monotonic()-start,'time':time.time()}
    write(args.out/'RESUME_VERIFICATION.json',result)
    print(__import__('json').dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
