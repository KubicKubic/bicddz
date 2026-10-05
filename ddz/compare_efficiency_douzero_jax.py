"""Frozen policy vs EdwardPooh released best ResNet weights, common deals and six role legs.

Training JAX environment and publisher ResNet inference run on the GPU.
The official legal-action generator is retained without candidate truncation.
--audit-engine enables the original second engine and exact public-input audits.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

# env_v4/actions create JAX arrays at import time. The persistent cache is
# initialized once, so selecting its directory in main() is already too late.
# A controller-provided shared cache takes precedence over the CLI directory.
if __name__ == '__main__':
    import argparse as _early_argparse
    _early_parser = _early_argparse.ArgumentParser(add_help=False)
    _early_parser.add_argument('--jax-cache')
    _early_args, _ = _early_parser.parse_known_args()
    if _early_args.jax_cache:
        os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _early_args.jax_cache)

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import time
import copy
from . import candidates_v3 as catalogue
from types import SimpleNamespace
from .douzero_jax import (convert_weights, batch_public_observation,
    candidate_values, padded_candidates, cards_array_numpy)

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
from .model_efficiency import EfficientMoveTransformer as InteractionMoveTransformer
from .policy_v5 import greedy

ROLES = ('landlord', 'landlord_down', 'landlord_up')
PROTOCOL = 'edwardpooh_resnet2_best_fixed_bid3_six_role_legs_qoj_catalogue_v3'
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


def make_qoj_filter(encode):
    """Publisher weights choose among moves legal under the training rules.

    The official generator allows a rocket among single attachments, whereas
    the training/QOJ rules prohibit it. Filter before scoring; never replace a
    chosen action or discard the affected deal.
    """
    @lru_cache(maxsize=65536)
    def allowed(action):
        packed = encode(action)
        body = packed % N_BODY
        b = BODIES[body]
        if b.wing_unit != 1:
            return True
        wing = counts_from_values(action)-COUNTS[body]
        return bool(np.all(wing <= 3) and not (wing[13] and wing[14]))
    return lambda actions:[a for a in actions if allowed(tuple(a))]


def make_catalogue_order(encode):
    """Complete training legal set, preferring the original publisher order.

    Some publisher-generated planes are rejected by its own detector. The
    catalogue provides explicit valid body/wing interpretations in those cases.
    Every catalogue move is retained, including moves absent from the publisher.
    """
    keys=tuple(cards.tobytes() for cards in catalogue.CARDS)
    @lru_cache(maxsize=131072)
    def original_key(action):
        key=counts_from_values(action).astype(np.uint8).tobytes()
        try:
            preferred=encode(action)
        except RuntimeError:
            preferred=None
        return key,preferred
    def ordered(ids, original):
        groups = {}
        for k in ids:
            groups.setdefault(keys[k], []).append(int(k))
        result = []
        for action in original:
            key,preferred = original_key(tuple(action))
            group = groups.pop(key, None)
            if group is None:
                continue
            if preferred is not None:
                group.sort(key=lambda k: int(catalogue.ACTION[k]) != preferred)
            result.extend(group)
        for group in groups.values():
            result.extend(group)
        if len(result) != len(ids):
            raise RuntimeError('Catalogue ordering dropped a legal interpretation')
        return np.asarray(result, np.int32)
    return ordered


def make_cached_legals(game_module, ordered):
    """Memoize deterministic public/own-hand queries across mirrored games."""
    @lru_cache(maxsize=65536)
    def query(hand, last_body, rival):
        ids=catalogue.legal_ids(hand,1,last=last_body,last_seat=-1,turn=0)
        light=SimpleNamespace(acting_player_position='landlord',
            info_sets={'landlord':SimpleNamespace(player_hand_cards=values_from_counts(hand))},
            card_play_action_seq=[list(rival)] if rival else [])
        return ordered(ids,game_module.GameEnv.get_legal_card_play_actions(light))
    def legal(hand, last_body, last_seat, turn, sequence):
        leading=last_body==0 or last_seat==turn
        rival=sequence[-1] if sequence and sequence[-1] else (sequence[-2] if len(sequence)>1 else [])
        return query(tuple(map(int,hand)),0 if leading else int(last_body),tuple(rival))
    return legal


def make_cached_catalogue():
    @lru_cache(maxsize=65536)
    def query(hand,last_body):
        return catalogue.legal_ids(hand,1,last=last_body,last_seat=-1,turn=0)
    def legal(hand,last_body,last_seat,turn):
        leading=last_body==0 or last_seat==turn
        return query(tuple(map(int,hand)),0 if leading else int(last_body))
    return legal


def bootstrap_summary(chunks, seed, confidence=.95, rounds=5000, batch=32):
    if not 0 < confidence < 1 or rounds < 100 or batch < 1:
        raise ValueError('Invalid bootstrap parameters')
    raw = np.concatenate([np.asarray(c['focus_raw_scores']) for c in chunks],axis=2)
    wins = np.concatenate([np.asarray(c['candidate_team_wins']) for c in chunks],axis=2)
    # Complementary leg reward is minus the focus seat's reward. Landlord /2
    # gives equal stake to each of the three roles.
    normalized = raw / np.asarray([2.,1.,1.])[:,None,None]
    paired = normalized.mean(axis=1)
    rng = np.random.default_rng(seed+1)
    sources = [paired.mean(axis=0),wins.mean(axis=(0,1))]
    for i in range(3):
        sources.extend((paired[i],wins[i,0],wins[i,1]))
    sampled = np.empty((len(sources),rounds))
    for start in range(0,rounds,batch):
        count = min(batch,rounds-start)
        indices = rng.integers(raw.shape[2],size=(count,raw.shape[2]))
        for i,values in enumerate(sources):
            sampled[i,start:start+count] = values[indices].mean(axis=1)
    bounds = [(1-confidence)*50,(1+confidence)*50]
    cursor = [0]
    def metric(values):
        index = cursor[0]; cursor[0] += 1
        return {'mean':float(values.mean()),
                'ci95':np.percentile(sampled[index],bounds).tolist()}
    result = {'deals':raw.shape[2], 'games':int(raw.shape[2]*6),
        'equal_role_expected_score':metric(paired.mean(axis=0)),
        'equal_role_team_win_rate':metric(wins.mean(axis=(0,1))),
        'roles':{role:{'expected_score':metric(paired[i]),
                     'candidate_alone_team_win_rate':metric(wins[i,0]),
                     'candidate_on_other_two_team_win_rate':metric(wins[i,1])}
                 for i,role in enumerate(ROLES)}}
    # Reuse identical statistics rather than constructing another large draw.
    samples = (sampled[3]+sampled[4])/2
    result['team_mirror'] = {
        'win_rate':{'mean':float(wins[0].mean()),'ci95':np.percentile(samples,bounds).tolist()},
        'expected_score':result['roles']['landlord']['expected_score'],
        'landlord_win_rate':result['roles']['landlord']['candidate_alone_team_win_rate'],
        'farmer_team_win_rate':result['roles']['landlord']['candidate_on_other_two_team_win_rate']}
    if confidence != .95:
        # Retain compatibility fields; explicitly identify their actual coverage.
        result['confidence_level'] = confidence
    return result


def deal_seed_values(seed, offset, count):
    # Global identifiers stay independent of block size and remain int32-safe.
    return ((seed+(offset+np.arange(count,dtype=np.int64))*9973)%(2**31-1)).astype(np.int32)


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
    ap.add_argument('--require-a100',action='store_true',
                    help='Use the explicitly authorized local single A100 for policy inference')
    ap.add_argument('--jax-cache',type=Path)
    ap.add_argument('--trace-actions',action='store_true')
    ap.add_argument('--policy-microbatch-deals',type=int,default=16)
    ap.add_argument('--audit-engine', action='store_true')
    ap.add_argument('--deal-offset',type=int,default=0)
    args = ap.parse_args()
    if args.deals<=0 or args.deals%args.chunk_deals:
        raise ValueError('deals must be positive and divisible by chunk size')
    if args.chunk_deals % args.policy_microbatch_deals:
        raise ValueError('chunk size must be divisible by policy microbatch size')
    if args.deal_offset < 0:
        raise ValueError('Negative global deal offset')
    if not 0<=args.shard_index<args.shard_count or args.deals%(args.chunk_deals*args.shard_count):
        raise ValueError('invalid shard or uneven chunk distribution')
    torch.set_num_threads(2)
    if args.jax_cache:
        jax.config.update('jax_compilation_cache_dir',str(args.jax_cache))
    hardware = None
    if args.require_a100:
        devices = jax.local_devices()
        if jax.default_backend() != 'gpu' or len(devices) != 1 or 'A100' not in devices[0].device_kind:
            raise RuntimeError('Local evaluation requires exactly one A100')
        hardware = {'policy_backend':'cuda','policy_device':devices[0].device_kind,
                    'policy_devices':1,'douzero_backend':'jax_cuda_fp32',
                    'authorization':'explicit user request for local single-A100 evaluation'}
        print(json.dumps({'hardware':hardware}),flush=True)
    sys.path.insert(0,str(args.douzero_src.resolve()))
    from .douzero_resnet_adapter import load_best_models, make_observation
    get_resnet_obs=make_observation(args.resnet_src)
    from douzero.env import game as game_module, move_detector
    from douzero.env.env import get_obs, DummyAgent
    checkpoint = args.run_dir/f'policy_{args.step:07d}.msgpack'
    checkpoint_hash = file_hash(checkpoint)
    cfg = json.loads((args.run_dir/'config.json').read_text())
    model = InteractionMoveTransformer(**cfg['model'])
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
            'Both policies use training/QOJ rules: no bomb or rocket among attachments',
            'QOJ linear bomb/spring rewards differ from DouZero exponential ADP'],
        'runtime':'jax_resnet_microbatch_catalogue_v3',
        'policy_microbatch_deals':args.policy_microbatch_deals, 'audit_engine':args.audit_engine,
        'legal_actions':'Full training/QOJ catalogue including explicit ambiguous-plane interpretations; original publisher order used for exact score ties',
        'tie_fallback':'publisher CPU FP32 when top-two gap <= 0.0005*max(1,max_abs_score)',
        'source_sha256':{f:file_hash(Path(__file__).with_name(f)) for f in
            ('compare_efficiency_douzero_jax.py','douzero_jax.py','douzero_resnet_adapter.py','model_v5.py','model_efficiency.py','policy_v5.py','env_v4.py','env_v2.py','candidates_v3.py',
             'actions.py','compare_v3_v2_all_roles.py')}}
    if hardware is not None:
        identity['hardware'] = hardware
    if args.deal_offset:
        identity['global_deal_offset'] = args.deal_offset
    args.out.mkdir(parents=True,exist_ok=True)
    encode = make_move_encoder(move_detector)
    ordered_ids = make_catalogue_order(encode)
    exact_legals = make_cached_legals(game_module,ordered_ids)
    fast_legals = make_cached_catalogue()
    catalogue_values = [values_from_counts(cards) for cards in catalogue.CARDS[:catalogue.N_PLAY]]
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

    def choose_micro(states, policy_params):
        obs = jax.vmap(env.observe)(states)
        body,wing,_ = jax.lax.cond(j.max(states.hist_len)<=88,
            lambda _:model.apply({'params':policy_params},obs,memory_length=88),
            lambda _:model.apply({'params':policy_params},obs),None)
        return greedy(states,body,wing)

    @jax.jit
    def choose(states, policy_params):
        # Keep each six-leg/16-deal inference group identical to the original
        # evaluator. BF16 kernels and the 88/192 history branch depend on batch
        # shape; increasing game concurrency must not silently change policy.
        micro=args.policy_microbatch_deals
        groups=n//micro
        def arrange(x):
            shape=x.shape[1:]
            return x.reshape((6,groups,micro)+shape).swapaxes(0,1).reshape((groups,6*micro)+shape)
        small=jax.tree_util.tree_map(arrange,states)
        actions=jax.lax.map(lambda s:choose_micro(s,policy_params),small)
        return actions.reshape((groups,6,micro)).swapaxes(0,1).reshape(size)

    @jax.jit
    def actor_metadata(states):
        return states.turn, states.hands[j.arange(size), states.turn], states.last, states.last_seat

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
        jax_models = {role:convert_weights(models[role]) for role in ROLES}
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
            global_offset = args.deal_offset+offset
            seeds = j.asarray(deal_seed_values(args.seed,global_offset,n))
            states,bad = initial_states(seeds,j.arange(global_offset,global_offset+n,dtype=j.int32)%3)
            if bool(j.any(bad)): raise RuntimeError('initial bid rejected')
            initial_host = jax.device_get(states)
            landlords = np.tile(np.asarray(states.landlord),6)
            states = jax.tree_util.tree_map(lambda x:j.concatenate((x,)*6),states)
            games=[]
            if args.audit_engine:
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
            sequences = [[] for _ in range(size)]
            action_trace=[]
            fallback_games = 0
            score_max_error = 0.
            tick_seconds = {'metadata':0., 'policy':0., 'legals':0., 'douzero':0., 'step':0., 'audit':0.}
            raw_scores=np.zeros(size,np.float64)
            won=np.zeros(size,np.float64)
            finished=np.zeros(size,bool)
            for tick in range(256):
                if finished.all(): break
                started_tick = time.monotonic()
                turns, actor_hands, last_bodies, last_seats = jax.device_get(actor_metadata(states))
                tick_seconds['metadata'] += time.monotonic()-started_tick
                acting_roles=(turns-landlords)%3
                own=np.where(complement,acting_roles!=focus_roles,acting_roles==focus_roles)
                chosen=np.zeros(size,np.int32)
                moves=[None]*size
                at = time.monotonic()
                if np.any(own&~finished):
                    local_actions=np.asarray(choose(states,params))
                    for i in np.flatnonzero(own&~finished):
                        chosen[i]=local_actions[i]
                        moves[i]=values_from_packed(int(chosen[i]))
                tick_seconds['policy'] += time.monotonic()-at
                at = time.monotonic()
                legal_actions = [None]*size
                legal_catalogue = [None]*size
                for i in np.flatnonzero(~finished):
                    ids = fast_legals(actor_hands[i],last_bodies[i],last_seats[i],turns[i])
                    legal_catalogue[i] = ids
                    legal_actions[i] = [catalogue_values[k] for k in ids]
                tick_seconds['legals'] += time.monotonic()-at
                at = time.monotonic()
                state_z, state_x = batch_public_observation(states)
                host_observations = None
                for role_index,role in enumerate(ROLES):
                    group=np.flatnonzero(~own&~finished&(acting_roles==role_index))
                    counts=[]; scored=[]; ids=[]; action_counts=[]
                    for i in group:
                        legal = legal_actions[i]
                        if len(legal)==1:
                            moves[i]=legal[0]
                        else:
                            counts.append(len(legal)); scored.append(i)
                            ids.extend([i]*len(legal))
                            action_counts.extend(catalogue.CARDS[legal_catalogue[i]])
                    if scored:
                        packed_ids, packed_counts = padded_candidates(ids, action_counts)
                        scores=np.asarray(candidate_values(jax_models[role],state_z,state_x,
                            j.asarray(packed_ids),j.asarray(packed_counts)))[:len(ids)]
                        cursor=0
                        decisions=[]; references=[]; reference_games=[]
                        for i,count in zip(scored,counts):
                            values=scores[cursor:cursor+count]
                            selected = int(values.argmax())
                            top = np.partition(values, -2)[-2:]
                            near_tie = abs(top[1]-top[0]) <= 5e-4*max(1.,float(np.abs(values).max()))
                            decisions.append((i,count,cursor,selected,near_tie))
                            if args.audit_engine or near_tie:
                                reference_games.append(len(decisions)-1)
                                if host_observations is None:
                                    host_observations=jax.device_get((state_z,state_x))
                                host_z,host_x=host_observations
                                ac=cards_array_numpy(action_counts[cursor:cursor+count])
                                z=np.concatenate((ac[:,None],np.broadcast_to(host_z[i],(count,39,54))),axis=1)
                                x=np.broadcast_to(host_x[i],(count,15)).copy()
                                if args.audit_engine:
                                    info=copy.copy(games[i].game_infoset)
                                    info.legal_actions=legal_actions[i]
                                    publisher=get_resnet_obs(info)
                                    if not (np.array_equal(z,publisher['z_batch']) and np.array_equal(x,publisher['x_batch'])):
                                        raise RuntimeError('Publisher public-observation mismatch')
                                references.append((z,x))
                            cursor+=count
                        reference_values={}
                        if references:
                            with torch.inference_mode():
                                cpu_values=models[role](
                                    torch.from_numpy(np.concatenate([r[0] for r in references])),
                                    torch.from_numpy(np.concatenate([r[1] for r in references])),
                                    return_value=True)['values'][:,0].numpy()
                            cursor=0
                            for k in reference_games:
                                count=decisions[k][1]
                                reference_values[k]=cpu_values[cursor:cursor+count]
                                cursor+=count
                        for k,(i,count,cursor,selected,near_tie) in enumerate(decisions):
                            if k in reference_values:
                                reference=reference_values[k]
                                values=scores[cursor:cursor+count]
                                if args.audit_engine:
                                    score_max_error=max(score_max_error,float(np.max(np.abs(reference-values))))
                                    if not near_tie and int(reference.argmax()) != selected:
                                        raise RuntimeError('JAX ResNet argmax mismatch beyond fallback threshold')
                                if near_tie:
                                    selected=int(reference.argmax()); fallback_games+=1
                                    tied=np.flatnonzero(reference==reference.max())
                                    if len(tied)>1:
                                        # The network sees cards, so different
                                        # explicit plane bodies can have the
                                        # same value. Preserve publisher tie
                                        # order without generating every move
                                        # in Python for every decision.
                                        preferred=exact_legals(actor_hands[i],last_bodies[i],last_seats[i],turns[i],sequences[i][-2:])
                                        tied_ids=set(map(int,legal_catalogue[i][tied]))
                                        chosen_id=next(int(k) for k in preferred if int(k) in tied_ids)
                                        selected=int(np.flatnonzero(legal_catalogue[i]==chosen_id)[0])
                            moves[i]=legal_actions[i][selected]
                            chosen[i]=catalogue.ACTION[legal_catalogue[i][selected]]
                    for i in group:
                        if len(legal_actions[i])==1:
                            chosen[i]=catalogue.ACTION[legal_catalogue[i][0]]
                tick_seconds['douzero'] += time.monotonic()-at
                at = time.monotonic()
                for i in np.flatnonzero(~finished):
                    if int(chosen[i]) not in catalogue.ACTION[legal_catalogue[i]]:
                        raise RuntimeError(f'Policy action absent from exact training catalogue: {moves[i]}')
                    if args.audit_engine:
                        from .actions import interpretations, beats
                        body=int(chosen[i])%N_BODY
                        cards=counts_from_values(moves[i])
                        if body and body not in interpretations(cards):
                            raise RuntimeError('Independent CPU interpretation audit failed')
                        leading=last_bodies[i]==0 or last_seats[i]==turns[i]
                        if (body==0 and leading) or (body!=0 and not leading and not beats(body,int(last_bodies[i]))):
                            raise RuntimeError('Independent CPU follow-rule audit failed')
                    sequences[i].append(moves[i])
                    if args.audit_engine:
                        game=games[i]
                        if game.acting_player_position!=ROLES[acting_roles[i]]:
                            raise RuntimeError(f'turn mismatch at game {i}')
                        game.players[game.acting_player_position].set_action(moves[i])
                        game.step()
                tick_seconds['audit'] += time.monotonic()-at
                at = time.monotonic()
                if args.trace_actions:
                    action_trace.append(chosen.tolist())
                ns,reward,done,bad=env.batch_step(states,j.asarray(chosen),keys)
                done,reward,bad = jax.device_get((done,reward,bad))
                if np.any(bad&~finished):
                    raise RuntimeError('Training JAX engine rejected an official move: '+str([(int(i),moves[i],int(chosen[i])) for i in np.flatnonzero(bad&~finished)]))
                for i in np.flatnonzero(done&~finished):
                    focus=(landlords[i]+focus_roles[i])%3
                    raw_scores[i]=reward[i,focus]*(-1 if complement[i] else 1)
                    won[i]=raw_scores[i]>0
                tick_seconds['step'] += time.monotonic()-at
                at = time.monotonic()
                if args.audit_engine:
                    host_hands=np.asarray(ns.hands)
                    for i in np.flatnonzero(~finished):
                        game=games[i]; ll=landlords[i]
                        for r,role in enumerate(ROLES):
                            if not np.array_equal(host_hands[i,(ll+r)%3],
                                counts_from_values(game.info_sets[role].player_hand_cards)):
                                raise RuntimeError(f'hand mismatch game {i}, {role}')
                        if bool(done[i])!=game.game_over:
                            raise RuntimeError('terminal mismatch')
                        if game.game_over:
                            candidate_landlord=(focus_roles[i]==0)^bool(complement[i])
                            if bool(won[i]) != ((game.get_winner()=='landlord')==candidate_landlord):
                                raise RuntimeError('winner/reward mismatch')
                tick_seconds['audit'] += time.monotonic()-at
                states=ns; finished|=done
            if not finished.all(): raise RuntimeError('game exceeded full move limit')
            chunk={'identity':binding,'deal_offset':offset,'deals':n,
                'focus_raw_scores':raw_scores.reshape(3,2,n).tolist(),
                'candidate_team_wins':won.reshape(3,2,n).tolist(),
                'invalid_actions':0,
                'hand_terminal_winner_checks':'passed' if args.audit_engine else 'training JAX; complete QOJ catalogue checked',
                'near_tie_cpu_fallbacks':fallback_games,'score_max_error':score_max_error,
                'timings':tick_seconds}
            if args.trace_actions: chunk['action_trace']=action_trace
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
