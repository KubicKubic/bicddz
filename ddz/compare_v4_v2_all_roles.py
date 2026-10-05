"""Direct V4 candidate-policy versus V2 complete-move-policy matches."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from flax import serialization

from . import env_v2 as env
from .actions import N_BODY, WINGS, WUNIT
from .elo_ladder import atomic_json, sha256
from .model_v2 import FullAttentionMoveTransformer
from .model_v4 import CompactMoveTransformer
from .evaluate_compact import ROLE_NAMES, paired_scores


PROTOCOL = 'ddz_v4_vs_v2_all_seats_direct_v1'
WN = j.asarray(WINGS)
WU = j.asarray(WUNIT)


def greedy_v2_one(state, body_logits, wing_logits):
    """Match V2 API's greedy body then legal body-conditioned wings."""
    index = j.argmax(body_logits).astype(j.int32)

    def bid(_):
        return env.encode_bid(index - N_BODY)

    def move(_):
        body = index
        state0 = state._replace(pending=body, wing_left=WN[body],
            wing_min=j.int32(0), wings=j.zeros(15, j.int32))

        def choose(i, carry):
            current, ranks = carry

            def wing(carry):
                current, ranks = carry
                legal = env.legal_wings(current)
                rank = j.argmax(j.where(legal, wing_logits[body], -1e9)).astype(j.int32)
                new_wings = current.wings.at[rank].add(WU[body])
                current = current._replace(wings=new_wings,
                    wing_left=current.wing_left - 1,
                    wing_min=rank + j.where(WU[body] == 2, 1, 0))
                return current, ranks.at[i].set(rank)

            return jax.lax.cond(i < WN[body], wing, lambda x: x, carry)

        _, ranks = jax.lax.fori_loop(0, 5, choose,
            (state0, j.full(5, 15, j.int32)))
        return env.encode_move(body, ranks)

    return jax.lax.cond(index >= N_BODY, bid, move, None)


def make_evaluator(v4_model,v2_model,chunk_deals,forced_bid,memory_limit=88):
    from .evaluate_compact import make_role_evaluator
    return make_role_evaluator(v4_model,chunk_deals,forced_bid,memory_limit,v2_model)


def _interval(per_deal, rounds, seed):
    values = np.asarray(per_deal, np.float64)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(values), size=(rounds, len(values)))
    samples = values[chosen].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {'mean': float(values.mean()), 'low': float(low), 'high': float(high),
            'deals': len(values)}


def compare(v4_run, v4_step, v2_run, v2_step, output, deals=288,
            chunk_deals=12, seed=410500, bootstrap_rounds=2000):
    v4_run, v2_run, output = Path(v4_run), Path(v2_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    v4_path = v4_run / f'policy_{v4_step:07d}.msgpack'
    v2_path = v2_run / f'policy_{v2_step:07d}.msgpack'
    hashes = {'v4': sha256(v4_path), 'v2': sha256(v2_path)}
    protocol = {'name': PROTOCOL, 'v4_step': v4_step, 'v2_step': v2_step,
        'checkpoint_sha256': hashes, 'chunk_deals': chunk_deals,
        'seed': seed, 'bootstrap_rounds': bootstrap_rounds,
        'main_metric': 'natural auction; 3 focus seats x complementary seat assignments',
        'diagnostics': 'forced first bid 3; landlord/2, next, door; equal-role mean',
        'reward': 'raw terminal API score, zero-sum, with bombs and springs',
        'action': 'greedy deterministic decisions for both models'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('output directory has another frozen comparison protocol')
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)
    v4_cfg = json.loads((v4_run / 'config.json').read_text())
    v2_cfg = json.loads((v2_run / 'config.json').read_text())
    v4_model = CompactMoveTransformer(**v4_cfg['model'])
    v2_model = FullAttentionMoveTransformer(**v2_cfg['model'])
    v4_params = serialization.msgpack_restore(v4_path.read_bytes())
    v2_params = serialization.msgpack_restore(v2_path.read_bytes())
    evaluators = (make_evaluator(v4_model, v2_model, chunk_deals, False),
                  make_evaluator(v4_model, v2_model, chunk_deals, True))
    natural_legs, fixed_legs = [], []
    for offset in range(0, deals, chunk_deals):
        path = output / f'deals_{offset:05d}.json'
        if path.exists():
            chunk = json.loads(path.read_text())
            if (chunk['protocol'] != PROTOCOL or chunk['checkpoint_sha256'] != hashes
                    or chunk['offset'] != offset or chunk['seed'] != seed):
                raise RuntimeError(f'comparison chunk identity mismatch: {path}')
            natural_legs.append(np.asarray(chunk['natural_leg_scores'], np.float64))
            fixed_legs.append(np.asarray(chunk['fixed_leg_scores'], np.float64))
            continue
        atomic_json(output / 'status.json', {'state': 'evaluating',
            'deal_offset': offset, 'deals': deals, 'time': time.time()})
        chunk_seed = (seed + offset * 10007) % (2**31 - 1)
        mode_legs = []
        for evaluator in evaluators:
            legs, finished, invalid, decisions = jax.device_get(
                evaluator(v4_params, v2_params, chunk_seed, offset))
            if not bool(np.all(finished)) or bool(np.any(invalid)):
                raise RuntimeError(f'direct comparison failed at {offset}: '
                    f'finished={np.count_nonzero(finished)}, invalid={np.sum(invalid)}, '
                    f'decisions={decisions}')
            mode_legs.append(np.asarray(legs, np.float64))
        atomic_json(path, {'protocol': PROTOCOL, 'checkpoint_sha256': hashes,
            'seed': seed, 'offset': offset, 'natural_leg_scores': mode_legs[0].tolist(),
            'fixed_leg_scores': mode_legs[1].tolist()})
        natural_legs.append(mode_legs[0])
        fixed_legs.append(mode_legs[1])
        print(json.dumps({'deals_complete': offset + chunk_deals, 'deals': deals}), flush=True)
    natural = np.concatenate(natural_legs, axis=2)
    fixed = np.concatenate(fixed_legs, axis=2)
    natural_roles, natural_total = paired_scores(natural, deals, False)
    fixed_roles, fixed_total = paired_scores(fixed, deals, True)
    summary = {'protocol': protocol, 'deals': deals, 'games': 12 * deals,
        'natural_comprehensive': _interval(natural_total, bootstrap_rounds, seed + 1),
        'natural_nominal_seats': {name: _interval(natural_roles[i], bootstrap_rounds, seed + 1)
            for i, name in enumerate(ROLE_NAMES)},
        'fixed_balanced': _interval(fixed_total, bootstrap_rounds, seed + 1),
        'fixed_roles': {name: _interval(fixed_roles[i], bootstrap_rounds, seed + 1)
            for i, name in enumerate(ROLE_NAMES)},
        'natural_paired_scores': natural_total.tolist(),
        'fixed_paired_scores': fixed_total.tolist(),
        'finished_at_unix': time.time()}
    atomic_json(output / 'comparison.json', summary)
    atomic_json(output / 'status.json', {'state': 'ready', 'deals': deals,
        'games': 12 * deals, 'time': time.time()})
    print(json.dumps({'natural_comprehensive': summary['natural_comprehensive'],
        'fixed_balanced': summary['fixed_balanced'],
        'fixed_roles': summary['fixed_roles']}), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4-run', required=True)
    ap.add_argument('--v4-step', type=int, required=True)
    ap.add_argument('--v2-run', required=True)
    ap.add_argument('--v2-step', type=int, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--deals', type=int, default=288)
    ap.add_argument('--chunk-deals', type=int, default=12)
    ap.add_argument('--seed', type=int, default=410500)
    ap.add_argument('--bootstrap-rounds', type=int, default=2000)
    args = ap.parse_args()
    if args.deals <= 0 or args.deals % args.chunk_deals or args.chunk_deals % 3:
        raise ValueError('deals must divide into chunks balanced over three first bidders')
    compare(args.v4_run, args.v4_step, args.v2_run, args.v2_step,
            args.output, args.deals, args.chunk_deals, args.seed,
            args.bootstrap_rounds)


if __name__ == '__main__':
    main()
