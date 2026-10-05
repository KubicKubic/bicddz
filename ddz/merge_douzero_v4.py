"""Validate complete disjoint DouZero shards and publish whole-deal statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare_douzero_v4 import bootstrap_summary, file_hash
from .elo_ladder import atomic_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    root=args.output
    launch=json.loads((root/'launch.json').read_text())
    if file_hash(Path(launch['checkpoint']))!=launch['sha256']:
        raise RuntimeError('frozen checkpoint changed')
    engine_root=Path('/tmp/ddz_douzero_official')
    engine_manifest=json.loads((root/'official_engine_manifest.json').read_text())
    for name,digest in engine_manifest.items():
        if file_hash(engine_root/name)!=digest:
            raise RuntimeError(f'official engine file changed: {name}')
    comparison={}
    for baseline in launch['baselines']:
        chunks=[]
        identity=None
        for shard in range(launch['shards_per_baseline']):
            folder=root/f'{baseline}_shard{shard}'
            status=json.loads((folder/'status.json').read_text())
            if status['state']!='complete':
                raise RuntimeError(f'incomplete shard: {folder}')
            for path in sorted((folder/baseline).glob('chunk_*.json')):
                chunk=json.loads(path.read_text())
                identity=identity or chunk['identity']
                if chunk['identity']!=identity or chunk['invalid_actions']!=0:
                    raise RuntimeError(f'chunk identity or validity mismatch: {path}')
                n=chunk['deals']
                for key in ('focus_raw_scores','candidate_team_wins'):
                    arr=np.asarray(chunk[key])
                    if arr.shape!=(3,2,n) or not np.isfinite(arr).all():
                        raise RuntimeError(f'bad chunk data: {path} {key}')
                if not np.isin(chunk['candidate_team_wins'],[0,1]).all():
                    raise RuntimeError(f'invalid winner data: {path}')
                if chunk['deal_offset']//identity['chunk_deals']%launch['shards_per_baseline']!=shard:
                    raise RuntimeError(f'wrong shard assignment: {path}')
                chunks.append(chunk)
        chunks.sort(key=lambda c:c['deal_offset'])
        offsets=[c['deal_offset'] for c in chunks]
        if offsets!=list(range(0,launch['deals_per_baseline'],identity['chunk_deals'])):
            raise RuntimeError('missing or overlapping deals')
        if identity['checkpoint_sha256']!=launch['sha256'] or identity['seed']!=launch['seed']:
            raise RuntimeError('launch and result identity mismatch')
        for name,digest in identity['source_sha256'].items():
            if file_hash(Path(__file__).with_name(name))!=digest:
                raise RuntimeError(f'evaluation dependency changed: {name}')
        if 'resnet_source_sha256' in identity:
            reference=Path('runs/douzero_resnet_2_0_reference')
            for name,digest in identity['resnet_source_sha256'].items():
                if file_hash(reference/'source'/name)!=digest:raise RuntimeError(f'ResNet source changed: {name}')
            for role,digest in identity['weight_sha256'].items():
                if file_hash(reference/'weights'/'BEST'/f'{role}.ckpt')!=digest:raise RuntimeError(f'ResNet weight changed: {role}')
        summary=bootstrap_summary(chunks,identity['seed'])
        if summary['games']!=launch['games_per_baseline']:
            raise RuntimeError('wrong total game count')
        comparison[baseline]={'identity':identity,'summary':summary,
            'invalid_actions':0,'completed_chunks':len(chunks)}
    result={'checkpoint_step':identity['checkpoint_step'],
        'checkpoint_sha256':launch['sha256'],'baselines':comparison,
        'official_engine_source_sha256':engine_manifest,
        'merge_source_sha256':file_hash(Path(__file__))}
    atomic_json(root/'comparison.json',result)
    atomic_json(root/'status.json',{'state':'complete',
        'checkpoint_step':identity['checkpoint_step'],
        'games':sum(v['summary']['games'] for v in comparison.values())})
    print(json.dumps({b:v['summary'] for b,v in comparison.items()},indent=2))


if __name__=='__main__': main()
