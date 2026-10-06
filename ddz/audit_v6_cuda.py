"""Local CUDA engineering rehearsal; remote eight-rank acceptance is separate."""
import argparse
import time
from pathlib import Path
import jax
from flax import serialization
from .scale_rollout_campaign import write
from .scale_architecture_v6 import function_probe
from .model_efficiency import EfficientMoveTransformer
from .v6_preflight import training_signatures,attention_mask_parity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True)
    args=ap.parse_args();root=args.audit
    if jax.default_backend()!='gpu' or jax.local_device_count()!=1:
        raise RuntimeError('Requires one local engineering GPU')
    if 'A100' not in jax.local_devices()[0].device_kind:raise RuntimeError('Requires local A100')
    saved=serialization.msgpack_restore((root/'source.msgpack').read_bytes())
    migrated=serialization.msgpack_restore((root/'migrated.msgpack').read_bytes())
    probe=function_probe(saved,migrated,precision_modes=('bf16',))
    jax.clear_caches()
    model=EfficientMoveTransformer(**migrated['config']['model'])
    signatures=training_signatures(model,jax.tree_util.tree_map(jax.numpy.asarray,migrated['train']['params']),migrated['config'])
    masks=attention_mask_parity()
    result={'passed':True,'scope':'Single local A100 CUDA engineering; not NCCL/eight-rank or strength evidence',
            'source_iteration':saved['iteration'],'function_probe':probe,
            'training_signatures':signatures,'attention_mask_parity':masks,'time':time.time()}
    write(root/'CUDA_VERIFICATION.json',result)
    print(__import__('json').dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
