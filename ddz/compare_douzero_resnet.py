"""Frozen policy vs EdwardPooh released best ResNet weights, common deals and six role legs.

Both engines validate every move. No action replacement, hidden-card input,
history truncation, or training-process modification is allowed.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as j
import numpy as np
import torch
from flax import serialization

from . import env_v4 as env
from .actions import BODIES, COUNTS, N_BODY
from .compare_douzero import (TYPE_NAMES, VALUE_RANKS, check_hands,
    counts_from_values, file_hash, load_douzero_models, values_from_counts)
from .compare_v3_v2_all_roles import greedy_v2_one
from .elo_ladder import atomic_json
from .model_v4 import CompactMoveTransformer

ROLES = ('landlord', 'landlord_down', 'landlord_up')
PROTOCOL = 'edwardpooh_resnet2_best_fixed_bid3_six_role_legs_v1'
BODY_MAP = {(b.type, b.rank-b.length+1, b.length): i
            for i,b in enumerate(BODIES)}


def make_move_encoder(detector):
    @lru_cache(maxsize=65536)
    def encode(action):
        detected = detector.get_move_type(list(action))
        if detected['type'] >= len(TYPE_NAMES):
            raise RuntimeError(f'official detector rejected {action}')
        name = TYPE_NAMES[detected['type']]
        start = VALUE_RANKS[detected['rank']] if 'rank' in detected else 0
        length = detected.get('len', 1)
        if name == 'rocket': start = 14
        body = BODY_MAP[(name, start, length)]
        b = BODIES[body]
        remaining = counts_from_values(action) - COUNTS[body]
        if np.any(remaining < 0):
            raise RuntimeError(f'body mismatch for {action}')
        ranks = [r for r,n in enumerate(remaining)
                 for _ in range(int(n)//max(b.wing_unit,1))]
        if len(ranks) != b.wings or (b.wing_unit == 2 and np.any(remaining % 2)):
            raise RuntimeError(f'wing mismatch for {action}')
        ranks += [15]*(5-len(ranks))
        # The exact integer format used by env_v2.encode_move.
        packed = sum(r*16**i for i,r in enumerate(ranks))*N_BODY + body
        return packed
    return encode


def values_from_packed(action):
    body = action % N_BODY
    code = action // N_BODY
    counts = COUNTS[body].copy()
    for i in range(BODIES[body].wings):
        rank = (code // 16**i) % 16
        counts[rank] += BODIES[body].wing_unit
    return values_from_counts(counts)


def bootstrap_summary(chunks, seed):
    raw = np.concatenate([np.asarray(c['focus_raw_scores']) for c in chunks],axis=2)
    wins = np.concatenate([np.asarray(c['candidate_team_wins']) for c in chunks],axis=2)
    # Complementary leg reward is minus the focus seat's reward. Landlord /2
    # gives equal stake to each of the three roles.
    normalized = raw / np.asarray([2.,1.,1.])[:,None,None]
    paired = normalized.mean(axis=1)
    rng = np.random.default_rng(seed+1)
    indices = rng.integers(raw.shape[2],size=(5000,raw.shape[2]))
    def metric(values):
        sampled = values[indices].mean(axis=1)
        return {'mean':float(values.mean()),
                'ci95':np.percentile(sampled,[2.5,97.5]).tolist()}
    return {'deals':raw.shape[2], 'games':int(raw.shape[2]*6),
        'equal_role_expected_score':metric(paired.mean(axis=0)),
        'equal_role_team_win_rate':metric(wins.mean(axis=(0,1))),
        'roles':{role:{'expected_score':metric(paired[i]),
                     'candidate_alone_team_win_rate':metric(wins[i,0]),
                     'candidate_on_other_two_team_win_rate':metric(wins[i,1])}
                 for i,role in enumerate(ROLES)},
        'team_mirror':{
            'win_rate':metric(wins[0].mean(axis=0)),
            'expected_score':metric(normalized[0].mean(axis=0)),
            'landlord_win_rate':metric(wins[0,0]),
            'farmer_team_win_rate':metric(wins[0,1])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--step',type=int,required=True)
    ap.add_argument('--douzero-src',type=Path,required=True)
    ap.add_argument('--weights-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--deals',type=int,default=1024)
    ap.add_argument('--chunk-deals',type=int,default=8)
    ap.add_argument('--seed',type=int,default=411001)
    ap.add_argument('--baseline',choices=('BEST',),default='BEST')
    ap.add_argument('--resnet-src',type=Path,required=True)
    ap.add_argument('--shard-index',type=int,default=0)
    ap.add_argument('--shard-count',type=int,default=1)
    args = ap.parse_args()
    if args.deals<=0 or args.deals%args.chunk_deals:
        raise ValueError('deals must be positive and divisible by chunk size')
    if not 0<=args.shard_index<args.shard_count or args.deals%(args.chunk_deals*args.shard_count):
        raise ValueError('invalid shard or uneven chunk distribution')
    torch.set_num_threads(2)
    sys.path.insert(0,str(args.douzero_src.resolve()))
    from .douzero_resnet_adapter import load_best_models, make_observation
    get_resnet_obs=make_observation(args.resnet_src)
    from douzero.env import game as game_module, move_detector
    from douzero.env.env import get_obs, DummyAgent
    checkpoint = args.run_dir/f'policy_{args.step:07d}.msgpack'
    checkpoint_hash = file_hash(checkpoint)
    cfg = json.loads((args.run_dir/'config.json').read_text())
    model = CompactMoveTransformer(**cfg['model'])
    params = serialization.msgpack_restore(checkpoint.read_bytes())
    source_commit = subprocess.check_output(['git','-C',str(args.douzero_src),
        'rev-parse','HEAD'],text=True).strip()
    manifest = json.loads((args.weights_root/'manifest.json').read_text())
    identity = {'protocol':PROTOCOL,'checkpoint_step':args.step,
        'checkpoint_sha256':checkpoint_hash,'run_dir':str(args.run_dir.resolve()),
        'official_engine_commit':source_commit,'resnet_commit':manifest['commit'],
        'resnet_source_sha256':{f:file_hash(args.resnet_src/'Douzero_Resnet'/f) for f in ('douzero/dmc/models_res.py','douzero/env/env_res.py')},'seed':args.seed,'deals':args.deals,
        'chunk_deals':args.chunk_deals,'role_order':ROLES,
        'decision':'deterministic API body-then-conditional-wing argmax',
        'scoring':'QOJ raw focus-seat score; landlord divided by 2; equal role mean',
        'uncertainty':'5000 paired whole-deal bootstrap draws; six legs stay together',
        'limitations':['Fixed bid 3; bidding strength is not measured',
            'Publisher-released ResNet best weights pinned by commit and SHA-256',
            'QOJ linear bomb/spring rewards differ from DouZero exponential ADP'],
        'source_sha256':{f:file_hash(Path(__file__).with_name(f)) for f in
            ('compare_douzero_resnet.py','douzero_resnet_adapter.py','model_v4.py','env_v4.py','env_v2.py',
             'actions.py','compare_v3_v2_all_roles.py')}}
    args.out.mkdir(parents=True,exist_ok=True)
    encode = make_move_encoder(move_detector)
    n = args.chunk_deals
    size = 6*n
    focus_roles = np.repeat(np.arange(3),2*n)
    complement = np.tile(np.repeat([False,True],n),3)

    @jax.jit
    def initial_states(seeds, first):
        states = env.batch_reset(jax.vmap(jax.random.PRNGKey)(seeds))
        states = states._replace(turn=first)
        states,_,_,bad = env.batch_step(states,j.full((n,),env.encode_bid(3)),
            jax.vmap(jax.random.PRNGKey)(seeds+1))
        return states,bad

    @jax.jit
    def choose(states):
        obs = jax.vmap(env.observe)(states)
        body,wing,_ = jax.lax.cond(j.max(states.hist_len)<=88,
            lambda _:model.apply({'params':params},obs,memory_length=88),
            lambda _:model.apply({'params':params},obs),None)
        return jax.vmap(greedy_v2_one)(states,body,wing)

    keys = jax.random.split(jax.random.PRNGKey(0),size)
    results = {}
    for baseline in (args.baseline,):
        weights_dir = args.weights_root/'weights'/baseline
        models, weight_hashes = load_best_models(weights_dir,args.resnet_src)
        for role in ROLES:
            if weight_hashes[role]!=manifest['files'][baseline][role]['sha256']:
                raise RuntimeError(f'weight hash mismatch: {baseline} {role}')
        binding = {**identity,'baseline':baseline,'weight_sha256':weight_hashes,
            'weights_provenance':f"{manifest['source']}/tree/{manifest['commit']}/Douzero_Resnet/baseline/best"}
        # JSON round trip normalizes tuples for exact resume comparisons.
        binding = json.loads(json.dumps(binding))
        folder = args.out/baseline
        folder.mkdir(exist_ok=True)
        chunks = []
        started = time.monotonic()
        for offset in range(args.shard_index*n,args.deals,n*args.shard_count):
            path = folder/f'chunk_{offset:05d}.json'
            if path.exists():
                chunk = json.loads(path.read_text())
                if chunk['identity']!=binding or chunk['deal_offset']!=offset:
                    raise RuntimeError(f'cached chunk mismatch: {path}')
                chunks.append(chunk)
                continue
            atomic_json(args.out/'status.json',{'state':'evaluating','pid':os.getpid(),
                'baseline':baseline,'checkpoint_step':args.step,'completed_deals':len(chunks)*n,
                'total_deals':args.deals//args.shard_count,'shard_index':args.shard_index,
                'deal_offset':offset,'time':time.time()})
            seeds = j.asarray([args.seed+(offset+i)*9973 for i in range(n)],j.int32)
            states,bad = initial_states(seeds,j.arange(offset,offset+n,dtype=j.int32)%3)
            if bool(j.any(bad)): raise RuntimeError('initial bid rejected')
            initial_host = jax.device_get(states)
            landlords = np.tile(np.asarray(states.landlord),6)
            states = jax.tree_util.tree_map(lambda x:j.concatenate((x,)*6),states)
            games=[]
            for i in range(size):
                one = jax.tree_util.tree_map(lambda x:x[i%n],initial_host)
                ll = int(one.landlord)
                players = {role:DummyAgent(role) for role in ROLES}
                game = game_module.GameEnv(players)
                game.card_play_init({**{role:values_from_counts(one.hands[(ll+r)%3])
                    for r,role in enumerate(ROLES)},
                    'three_landlord_cards':values_from_counts(one.bottom)})
                check_hands(one,game,ll)
                games.append(game)
            raw_scores=np.zeros(size,np.float64)
            won=np.zeros(size,np.float64)
            finished=np.zeros(size,bool)
            for tick in range(256):
                if finished.all(): break
                # Audit role rotation before selecting moves.
                turns=np.asarray(states.turn)
                acting_roles=(turns-landlords)%3
                own=np.where(complement,acting_roles!=focus_roles,acting_roles==focus_roles)
                chosen=np.zeros(size,np.int32)
                moves=[None]*size
                if np.any(own&~finished):
                    local_actions=np.asarray(choose(states))
                    for i in np.flatnonzero(own&~finished):
                        chosen[i]=local_actions[i]
                        moves[i]=values_from_packed(int(chosen[i]))
                for role_index,role in enumerate(ROLES):
                    group=np.flatnonzero(~own&~finished&(acting_roles==role_index))
                    observations=[]; counts=[]; scored=[]
                    for i in group:
                        info=games[i].game_infoset
                        if len(info.legal_actions)==1:
                            moves[i]=info.legal_actions[0]
                        else:
                            obs=get_resnet_obs(info);observations.append(obs)
                            counts.append(len(info.legal_actions));scored.append(i)
                    if observations:
                        with torch.inference_mode():
                            scores=models[role](torch.from_numpy(np.concatenate(
                                [o['z_batch'] for o in observations])),
                                torch.from_numpy(np.concatenate([o['x_batch'] for o in observations])),
                                return_value=True)['values'][:,0].numpy()
                        at=0
                        for i,count in zip(scored,counts):
                            moves[i]=games[i].game_infoset.legal_actions[int(scores[at:at+count].argmax())]
                            at+=count
                    for i in group: chosen[i]=encode(tuple(moves[i]))
                for i in np.flatnonzero(~finished):
                    game=games[i]
                    if game.acting_player_position!=ROLES[acting_roles[i]]:
                        raise RuntimeError(f'turn mismatch at game {i}')
                    if moves[i] not in game.game_infoset.legal_actions:
                        raise RuntimeError(f'V4 action illegal in official engine: {moves[i]}')
                    # Preserve the body interpretation emitted by our policy;
                    # fail if the official detector assigns a different body.
                    if own[i] and encode(tuple(moves[i]))!=int(chosen[i]):
                        raise RuntimeError(f'body interpretation mismatch: {moves[i]}')
                    game.players[game.acting_player_position].set_action(moves[i])
                    game.step()
                ns,reward,done,bad=env.batch_step(states,j.asarray(chosen),keys)
                if np.any(np.asarray(bad)&~finished):
                    raise RuntimeError('local engine rejected an official move')
                host_hands=np.asarray(ns.hands)
                done=np.asarray(done); reward=np.asarray(reward)
                for i in np.flatnonzero(~finished):
                    game=games[i]; ll=landlords[i]
                    for r,role in enumerate(ROLES):
                        if not np.array_equal(host_hands[i,(ll+r)%3],
                            counts_from_values(game.info_sets[role].player_hand_cards)):
                            raise RuntimeError(f'hand mismatch game {i}, {role}')
                    if bool(done[i])!=game.game_over:
                        raise RuntimeError('terminal mismatch')
                    if game.game_over:
                        focus=(ll+focus_roles[i])%3
                        raw_scores[i]=reward[i,focus]*(-1 if complement[i] else 1)
                        candidate_landlord=(focus_roles[i]==0)^bool(complement[i])
                        won[i]=(game.get_winner()=='landlord')==candidate_landlord
                        if (raw_scores[i]>0)!=bool(won[i]):
                            raise RuntimeError('winner/reward mismatch')
                states=ns; finished|=done
            if not finished.all(): raise RuntimeError('game exceeded full move limit')
            chunk={'identity':binding,'deal_offset':offset,'deals':n,
                'focus_raw_scores':raw_scores.reshape(3,2,n).tolist(),
                'candidate_team_wins':won.reshape(3,2,n).tolist(),
                'invalid_actions':0,'hand_terminal_winner_checks':'passed'}
            atomic_json(path,chunk);chunks.append(chunk)
            print(json.dumps({'baseline':baseline,'completed_deals':len(chunks)*n,
                'deal_offset':offset,'shard_index':args.shard_index,
                'seconds':round(time.monotonic()-started,1)}),flush=True)
        summary=bootstrap_summary(chunks,args.seed)
        results[baseline]={'identity':binding,'summary':summary,'invalid_actions':0,
            'execution':{'shard_index':args.shard_index,'shard_count':args.shard_count},
            'runtime_seconds':time.monotonic()-started}
        atomic_json(folder/'summary.json',results[baseline])
        atomic_json(args.out/'comparison.json',results)
        print(json.dumps({'baseline':baseline,'summary':summary}),flush=True)
    if file_hash(checkpoint)!=checkpoint_hash:
        raise RuntimeError('checkpoint changed during evaluation')
    atomic_json(args.out/'status.json',{'state':'complete','checkpoint_step':args.step,
        'baselines':list(results),'shard_index':args.shard_index,
        'shard_count':args.shard_count,'time':time.time()})


if __name__=='__main__': main()
