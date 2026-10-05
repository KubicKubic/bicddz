"""Read-only aggregation of disjoint large-validation blocks."""
import json
from pathlib import Path


def completed_blocks(cfg):
    root = Path(cfg['root'])/'blocks'
    blocks = []
    for path in sorted(root.glob('deal_*/BEST/summary.json')):
        result = json.loads(path.read_text())
        identity = result['identity']
        if ('accepted_protocols' in cfg and identity['protocol'] not in cfg['accepted_protocols']):
            raise RuntimeError('Unaccepted evaluation protocol: '+str(path))
        if ('weight_sha256' in cfg and identity.get('weight_sha256') != cfg['weight_sha256']):
            raise RuntimeError('Publisher weight identity changed: '+str(path))
        offset = int(path.parents[1].name.split('_')[1])
        if (identity['checkpoint_sha256'] != cfg['checkpoint_sha256']
                or identity['seed'] != cfg['seed']
                or identity.get('global_deal_offset',0) != offset
                or identity['resnet_commit'] != cfg['best_commit']
                or result['summary']['deals'] != cfg['block_deals']
                or identity.get('hardware',{}).get('policy_devices') != 1
                or result['invalid_actions'] != 0):
            raise RuntimeError('Large-validation block identity mismatch: '+str(path))
        if offset != len(blocks)*cfg['block_deals']:
            raise RuntimeError('Large-validation deal gap or duplicate')
        blocks.append(path)
    return blocks


def precision_level(stage):
    # All planned looks share total alpha .05, rather than repeatedly testing
    # unadjusted 95% intervals until one happens to look favorable.
    return 1-.05/(2**stage)


def aggregate(cfg, paths, stage):
    from .compare_efficiency_douzero import bootstrap_summary
    chunks = []
    for path in paths:
        data = json.loads(path.read_text())
        identity = data['identity']
        chunk_deals = identity['chunk_deals']
        block_chunks = sorted(path.parent.glob('chunk_*.json'))
        if len(block_chunks)*chunk_deals != cfg['block_deals']:
            raise RuntimeError('Incomplete deal block: '+str(path))
        for index, chunk_path in enumerate(block_chunks):
            chunk = json.loads(chunk_path.read_text())
            if (chunk['identity'] != identity or chunk['deal_offset'] != index*chunk_deals
                    or chunk['deals'] != chunk_deals
                    or chunk['invalid_actions'] != 0):
                raise RuntimeError('Chunk identity or contiguous-deal audit failed')
            chunks.append(chunk)
    confidence = precision_level(stage)
    summary = bootstrap_summary(chunks,cfg['bootstrap_seed'],confidence=confidence,
                                rounds=cfg['bootstrap_rounds'],batch=32)
    if summary['deals'] != len(paths)*cfg['block_deals']:
        raise RuntimeError('Aggregated independent-deal count differs from audited blocks')
    interval = summary['equal_role_expected_score']['ci95']
    width = interval[1]-interval[0]
    complete = width <= cfg['target_interval_width']
    def labels(value):
        if isinstance(value,dict):
            return {('interval' if k=='ci95' else k):labels(v) for k,v in value.items()}
        return value
    return {'state':'complete' if complete else 'extend','step':cfg['step'],
        'global_step':cfg['global_step'],'checkpoint_sha256':cfg['checkpoint_sha256'],
        'deals':summary['deals'],'games':summary['games'],'stage':stage,
        'confidence_level_at_this_look':confidence,'overall_coverage_target':.95,
        'method':'whole-deal paired percentile bootstrap; geometric alpha spending across precision looks',
        'interval_width':width,'target_interval_width':cfg['target_interval_width'],
        'expected_score':{'mean':summary['equal_role_expected_score']['mean'],'interval':interval},
        'summary':labels(summary),'fresh_seed':cfg['seed'],'bidding':'fixed bid 3',
        'invalid_actions':0,'block_files':[str(p) for p in paths]}
