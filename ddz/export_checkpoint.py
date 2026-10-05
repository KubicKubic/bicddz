"""Export one consistent raw/EMA/resume checkpoint bundle for Git LFS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import jax
from flax import serialization
from .switch_distributed_gae import equal


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def export_bundle(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Export to a new directory; existing releases are immutable')
    # The trainer atomically replaces this file. A single read pins one inode.
    raw = (run / 'latest.msgpack').read_bytes()
    saved = serialization.msgpack_restore(raw)
    iteration = saved['iteration']
    cfg = saved['config']
    if 'ema' not in saved or saved['ema']['decay'] != cfg.get('ema_decay', .999):
        raise ValueError('A complete, matching EMA checkpoint is required')
    raw_policy = run / f'policy_{iteration:07d}.msgpack'
    ema_policy = run / 'ema' / raw_policy.name
    # A checkpoint being published is incomplete until both immutable policies exist.
    policies = [(raw_policy, saved['train']['params']), (ema_policy, saved['ema']['params'])]
    for path, params in policies:
        if not equal(serialization.msgpack_restore(path.read_bytes()), params):
            raise ValueError('Policy snapshot does not match full checkpoint: ' + path.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.checkpoint-export-', dir=out.parent))
    try:
        (temporary / 'latest_checkpoint.msgpack').write_bytes(raw)
        shutil.copyfile(raw_policy, temporary / 'policy.msgpack')
        shutil.copyfile(ema_policy, temporary / 'ema_policy.msgpack')
        (temporary / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        manifest = {
            'family': 'V5', 'policy_kind': 'raw', 'relative_step': iteration,
            'global_step': cfg.get('global_source_iteration', 0) + iteration,
            'parameters': sum(a.size for a in jax.tree_util.tree_leaves(saved['train']['params'])),
            'ema_decay': saved['ema']['decay'], 'ema_rollout_updates': saved['ema']['updates'],
            'optimizer_step': int(saved['train']['step']),
            'training_config': 'config.json', 'raw_policy': 'policy.msgpack',
            'ema_policy': 'ema_policy.msgpack', 'resume_checkpoint': 'latest_checkpoint.msgpack',
            'source_run': str(run),
            'files_sha256': {p.name: sha(p) for p in temporary.iterdir()},
        }
        (temporary / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(temporary, out)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'release.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = Path(name)
        if path.name != name or sha(bundle / name) != expected:
            raise ValueError('Release file checksum differs: ' + name)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(export_bundle(args.run, args.out), indent=2))


if __name__ == '__main__':
    main()
