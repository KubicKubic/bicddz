"""Prepare frozen, portable CPU deployments; account credentials stay external."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from .export_checkpoint import sha, verify_bundle


def prepare(repo, root, models, token_file, username, base, session, python):
    repo, root, models = (Path(p).resolve() for p in (repo, root, models))
    release = verify_bundle(models)
    sources = sorted(p for p in (repo / 'ddz').rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    worker = repo / 'deploy/qoj/worker.py'
    source_hash = hashlib.sha256((''.join(str(p.relative_to(repo)) + sha(p) for p in sources)
                                  + sha(worker)).encode()).hexdigest()
    name = f"v5-{release['global_step']}-{source_hash[:10]}"
    frozen = root / 'releases' / name
    config = {
        'username': username, 'base': base, 'mode': 'match', 'tmux_session': session,
        'token_file': str(Path(token_file).expanduser().resolve()),
        'source_checkpoint': str(models / 'policy.msgpack'),
        'relative_step': release['relative_step'], 'global_step': release['global_step'],
        'checkpoint_sha256': release['files_sha256']['policy.msgpack'],
        'backend': 'cpu', 'automatic_requeue': True, 'once': False,
        'active_client_version': 8, 'model_dir': str(frozen / 'models'),
        'active_code': str(frozen / 'code'), 'active_worker': str(frozen / 'worker.py'),
        'active_launcher': str(root / 'launch_current.sh'), 'release': name,
    }
    if frozen.exists():
        existing = json.loads((frozen / 'deployment.json').read_text())
        if existing != config:
            raise ValueError('Existing frozen deployment settings differ')
        verify_frozen(frozen)
    else:
        frozen.mkdir(parents=True)
        shutil.copytree(repo / 'ddz', frozen / 'code/ddz',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(worker, frozen / 'worker.py')
        (frozen / 'models').mkdir()
        for filename in ('policy.msgpack', 'config.json', 'release.json'):
            shutil.copyfile(models / filename, frozen / 'models' / filename)
        (frozen / 'deployment.json').write_text(json.dumps(config, indent=2) + '\n')
        binding = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*') if p.is_file()}
        (frozen / 'files_sha256.json').write_text(json.dumps(binding, indent=2) + '\n')
    # A running supervisor retains its existing release-specific environment.
    # Replacing this entrypoint affects only the next deliberate restart.
    quote = shlex.quote
    launcher = '\n'.join([
        '#!/usr/bin/env bash', 'set -euo pipefail',
        f'export DDZ_MATCH_ROOT={quote(str(root))}',
        'if [[ -f "$DDZ_MATCH_ROOT/network.env" ]]; then source "$DDZ_MATCH_ROOT/network.env"; fi',
        f'export DDZ_DEPLOYMENT_FILE={quote(str(frozen / "deployment.json"))}',
        f'export PYTHONPATH={quote(str(frozen / "code"))}',
        "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1",
        f'{quote(str(python))} -m ddz.qoj_deployment verify-frozen --release {quote(str(frozen))}',
        f'exec {quote(str(python))} -u -m ddz.api_supervisor --root "$DDZ_MATCH_ROOT" --worker {quote(str(frozen / "worker.py"))}',
        '',
    ])
    temporary = root / 'launch_current.sh.tmp'
    temporary.write_text(launcher)
    temporary.chmod(0o755)
    os.replace(temporary, root / 'launch_current.sh')
    (root / 'prepared_deployment.json').write_text(json.dumps(config, indent=2) + '\n')
    return config


def verify_frozen(frozen):
    frozen = Path(frozen)
    for name, expected in json.loads((frozen / 'files_sha256.json').read_text()).items():
        path = frozen / name
        if path.resolve().is_relative_to(frozen.resolve()) is False or sha(path) != expected:
            raise ValueError('Frozen deployment changed: ' + name)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--repo', type=Path, required=True)
    prepare_parser.add_argument('--root', type=Path, required=True)
    prepare_parser.add_argument('--models', type=Path, required=True)
    prepare_parser.add_argument('--token-file', type=Path, required=True)
    prepare_parser.add_argument('--username', required=True)
    prepare_parser.add_argument('--base', default='https://qoj.ac/api/v1/doudizhu')
    prepare_parser.add_argument('--session', default='ddz_qoj_match')
    verify_parser = sub.add_parser('verify-frozen')
    verify_parser.add_argument('--release', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'verify-frozen':
        verify_frozen(args.release)
        print('Frozen deployment verified')
    else:
        config = prepare(args.repo, args.root, args.models, args.token_file, args.username,
                         args.base, args.session, sys.executable)
        print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
