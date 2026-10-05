import json
import numpy as np
import pytest
from ddz.compare_efficiency_douzero import bootstrap_summary,deal_seed_values
from ddz.douzero_precision import completed_blocks,precision_level


def test_streamed_bootstrap_matches_full_draws_for_scores_wins_and_roles():
    rng = np.random.default_rng(71)
    raw = rng.integers(-8,9,size=(3,2,37)).astype(float)
    wins = (raw>0).astype(float)
    draws = np.random.default_rng(42).integers(37,size=(600,37))
    level = .975
    bounds = [(1-level)*50,(1+level)*50]
    result = bootstrap_summary([{'focus_raw_scores':raw.tolist(),
                                  'candidate_team_wins':wins.tolist()}],41,
                               confidence=level,rounds=600,batch=17)
    normalized = raw/np.array([2.,1.,1.])[:,None,None]
    def check(metric,values):
        assert metric['mean'] == pytest.approx(values.mean())
        np.testing.assert_allclose(metric['ci95'],np.percentile(values[draws].mean(axis=1),bounds),atol=1e-14)
    check(result['equal_role_expected_score'],normalized.mean(axis=(0,1)))
    check(result['equal_role_team_win_rate'],wins.mean(axis=(0,1)))
    for i,role in enumerate(('landlord','landlord_down','landlord_up')):
        r = result['roles'][role]
        check(r['expected_score'],normalized[i].mean(axis=0))
        check(r['candidate_alone_team_win_rate'],wins[i,0])
        check(r['candidate_on_other_two_team_win_rate'],wins[i,1])
    check(result['team_mirror']['win_rate'],wins[0].mean(axis=0))


def test_global_deal_seeds_are_partition_invariant_fresh_and_int32_safe():
    whole = deal_seed_values(1220041,0,4096)
    blocks = np.concatenate([deal_seed_values(1220041,start,1024) for start in range(0,4096,1024)])
    np.testing.assert_array_equal(whole,blocks)
    assert len(np.unique(whole)) == 4096
    assert not set(whole.tolist()) & set(deal_seed_values(911001,0,1024).tolist())
    large = deal_seed_values(1220041,262144,1024)
    assert (large>=0).all() and (large<2**31-1).all()
    assert len(np.unique(large)) == 1024
    np.testing.assert_array_equal(deal_seed_values(911001,0,16),911001+np.arange(16)*9973)


def test_block_audit_rejects_gaps_and_changed_checkpoint(tmp_path):
    cfg = {'root':str(tmp_path),'checkpoint_sha256':'frozen','seed':1220041,
           'best_commit':'pinned','block_deals':1024}
    def put(offset,sha='frozen'):
        path = tmp_path/f'blocks/deal_{offset:07d}/BEST/summary.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        data = {'identity':{'checkpoint_sha256':sha,'seed':1220041,'global_deal_offset':offset,
                           'resnet_commit':'pinned','hardware':{'policy_devices':1}},
                'summary':{'deals':1024},'invalid_actions':0}
        path.write_text(json.dumps(data))
        return path
    put(0)
    assert len(completed_blocks(cfg))==1
    path = put(2048)
    with pytest.raises(RuntimeError,match='gap'):
        completed_blocks(cfg)
    path.unlink()
    put(1024,sha='changed')
    with pytest.raises(RuntimeError,match='identity'):
        completed_blocks(cfg)


def test_planned_precision_looks_spend_at_most_five_percent_alpha():
    assert precision_level(1)==.975
    assert precision_level(2)==.9875
    assert sum(1-precision_level(stage) for stage in range(1,25)) < .05


def test_aggregation_preserves_completed_blocks_with_different_parallelism(tmp_path):
    from ddz.douzero_precision import aggregate
    cfg={'root':str(tmp_path),'checkpoint_sha256':'frozen','seed':41,'best_commit':'pinned',
         'block_deals':4,'chunk_deals':2,'bootstrap_seed':41,'bootstrap_rounds':100,
         'target_interval_width':.025,'step':1,'global_step':2}
    paths=[]
    for offset,chunk_size in [(0,1),(4,2)]:
        folder=tmp_path/f'blocks/deal_{offset:07d}/BEST';folder.mkdir(parents=True)
        identity={'checkpoint_sha256':'frozen','seed':41,'global_deal_offset':offset,
                  'resnet_commit':'pinned','hardware':{'policy_devices':1},'chunk_deals':chunk_size}
        path=folder/'summary.json';path.write_text(json.dumps({'identity':identity,'summary':{'deals':4},'invalid_actions':0}))
        for start in range(0,4,chunk_size):
            (folder/f'chunk_{start:05d}.json').write_text(json.dumps({'identity':identity,'deal_offset':start,'deals':chunk_size,
                'focus_raw_scores':np.ones((3,2,chunk_size)).tolist(),
                'candidate_team_wins':np.ones((3,2,chunk_size)).tolist(),'invalid_actions':0}))
        paths.append(path)
    result=aggregate(cfg,completed_blocks(cfg),1)
    assert result['deals']==8 and result['games']==48
    assert result['expected_score']['mean']==pytest.approx(5/6)
    chunk=paths[1].parent/'chunk_00002.json'
    data=json.loads(chunk.read_text());data['deals']=1;chunk.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='Chunk identity'):
        aggregate(cfg,paths,1)


def test_block_audit_pins_protocol_and_publisher_weights(tmp_path):
    cfg={'root':str(tmp_path),'checkpoint_sha256':'frozen','seed':41,'best_commit':'pinned',
         'block_deals':4,'accepted_protocols':['accepted'],'weight_sha256':{'landlord':'pinned'}}
    path=tmp_path/'blocks/deal_0000000/BEST/summary.json';path.parent.mkdir(parents=True)
    data={'identity':{'checkpoint_sha256':'frozen','seed':41,'resnet_commit':'pinned',
        'hardware':{'policy_devices':1},'protocol':'other','weight_sha256':{'landlord':'pinned'}},
        'summary':{'deals':4},'invalid_actions':0}
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='protocol'):completed_blocks(cfg)
    data['identity']['protocol']='accepted';data['identity']['weight_sha256']={'landlord':'wrong'}
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='weight'):completed_blocks(cfg)
