"""Role-mirrored PPO versus DouZero on the official DouZero play engine.

DouZero does not bid, so the comparison starts after a fixed three-point bid.
The PPO observation is mirrored through the local QOJ environment. A mismatch
between the engines fails closed rather than silently changing a player's move.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import torch
from flax import serialization

from . import env
from .actions import BID_OFFSET, BODIES, COUNTS, N_BODY, WING_OFFSET, interpretations
from .model import MemoryTransformer

RANK_VALUES = [*range(3, 15), 17, 20, 30]
VALUE_RANKS = {value: rank for rank, value in enumerate(RANK_VALUES)}
TYPE_NAMES = ('pass', 'single', 'pair', 'trio', 'bomb', 'rocket', 'trio1',
              'trio2', 'straight', 'pairs', 'plane', 'plane1', 'plane2',
              'four2', 'four22')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def values_from_counts(counts):
    return sorted(value for rank, count in enumerate(np.asarray(counts))
                  for value in [RANK_VALUES[rank]] * int(count))


def counts_from_values(values):
    return np.bincount([VALUE_RANKS[value] for value in values], minlength=15)


def state_from_move(state, action, detector, step_fn):
    """Apply the official move interpretation to the local state."""
    if not action:
        choices = [0]
        counts = np.zeros(15, np.int32)
    else:
        counts = counts_from_values(action)
        detected = detector.get_move_type(action)
        kind = detected['type']
        if kind >= len(TYPE_NAMES):
            raise RuntimeError(f'DouZero cannot classify its own move: {action}')
        name = TYPE_NAMES[kind]
        start = VALUE_RANKS[detected['rank']] if 'rank' in detected else None
        length = detected.get('len', 1)
        choices = [i for i in interpretations(counts)
                   if BODIES[i].type == name and BODIES[i].length == length
                   and (start is None or BODIES[i].rank - length + 1 == start)]
    if len(choices) != 1:
        raise RuntimeError(f'non-unique or missing move interpretation: {action}, {choices}')
    body = choices[0]
    state, reward, done, invalid = step_fn(state, body)
    if bool(invalid):
        raise RuntimeError(f'local rule rejected official body: {action}, {BODIES[body]}')
    if BODIES[body].wings:
        remaining = counts - COUNTS[body]
        if BODIES[body].wing_unit == 1:
            wings = [r for r, n in enumerate(remaining) for _ in range(int(n))]
        else:
            wings = [r for r, n in enumerate(remaining) if n == 2]
        if len(wings) != BODIES[body].wings:
            raise RuntimeError(f'wrong wing count: {action}, {BODIES[body]}')
        for rank in wings:
            state, reward, done, invalid = step_fn(state, WING_OFFSET + rank)
            if bool(invalid):
                raise RuntimeError(f'local rule rejected official wing: {action}, rank {rank}')
    return state, np.asarray(reward), bool(done)


def check_hands(local_state, official_game, landlord):
    for name, seat in (('landlord', landlord),
                       ('landlord_down', (landlord + 1) % 3),
                       ('landlord_up', (landlord + 2) % 3)):
        official = counts_from_values(official_game.info_sets[name].player_hand_cards)
        local = np.asarray(local_state.hands[seat])
        if not np.array_equal(local, official):
            raise RuntimeError(f'hand mismatch for {name}: {official} versus {local}')


def make_local_actor(model, params, memory_length=96):
    def make_choose(length):
        @jax.jit
        def choose(state):
            obs = env.observe(state)
            obs = jax.tree_util.tree_map(lambda x: x[None], obs)
            logits, _ = model.apply({'params': params}, obs,
                                    memory_length=length)
            return logits[0].argmax()
        return choose
    choose_short = make_choose(memory_length)
    choose_full = make_choose(env.HISTORY)

    @jax.jit
    def step(state, action):
        return env.step(state, action, jax.random.PRNGKey(0))

    def act(state):
        initial_hand = np.asarray(state.hands[state.turn])
        current = state
        for _ in range(12):
            choose = choose_short if int(current.hist_len) <= memory_length else choose_full
            action = int(choose(current))
            current, _, _, invalid = step(current, action)
            if bool(invalid):
                raise RuntimeError(f'PPO chose illegal action {action}')
            if int(current.pending) < 0:
                played = initial_hand - np.asarray(current.hands[state.turn])
                return values_from_counts(played)
        raise RuntimeError('PPO move did not complete in 12 internal decisions')
    return act, step


def load_douzero_models(weights_dir, model_dict):
    models, hashes = {}, {}
    for role in ('landlord', 'landlord_down', 'landlord_up'):
        path = weights_dir / f'{role}.ckpt'
        weights = torch.load(path, map_location='cpu', weights_only=True)
        model = model_dict[role]()
        model.load_state_dict(weights, strict=True)
        model.eval()
        models[role] = model
        hashes[role] = file_hash(path)
    return models, hashes


def play(initial_state, local_landlord, actor, models, game_module, obs_fn,
         detector, step_fn):
    landlord = int(initial_state.landlord)
    position_seat = {'landlord': landlord,
                     'landlord_down': (landlord + 1) % 3,
                     'landlord_up': (landlord + 2) % 3}
    local_roles = {'landlord'} if local_landlord else {'landlord_down', 'landlord_up'}
    current = [initial_state]
    last_action = {}

    class Player:
        def __init__(self, role):
            self.role = role

        def act(self, infoset):
            if int(current[0].turn) != position_seat[self.role]:
                raise RuntimeError(f'turn mismatch at {self.role}')
            if self.role in local_roles:
                action = actor(current[0])
            else:
                if len(infoset.legal_actions) == 1:
                    action = infoset.legal_actions[0]
                else:
                    obs = obs_fn(infoset)
                    with torch.inference_mode():
                        scores = models[self.role](torch.from_numpy(obs['z_batch']),
                            torch.from_numpy(obs['x_batch']), return_value=True)['values']
                    action = infoset.legal_actions[int(torch.argmax(scores[:, 0]))]
            if action not in infoset.legal_actions:
                raise RuntimeError(f'{self.role} chose a move illegal in official engine: {action}')
            last_action[self.role] = action
            return action

    players = {role: Player(role) for role in position_seat}
    game = game_module.GameEnv(players)
    deal = {role: values_from_counts(initial_state.hands[seat])
            for role, seat in position_seat.items()}
    deal['three_landlord_cards'] = values_from_counts(initial_state.bottom)
    game.card_play_init(deal)
    check_hands(current[0], game, landlord)
    for decisions in range(1, 321):
        role = game.acting_player_position
        game.step()
        current[0], reward, done = state_from_move(current[0], last_action[role],
                                                   detector, step_fn)
        check_hands(current[0], game, landlord)
        if done != game.game_over:
            raise RuntimeError('terminal mismatch between official and local games')
        if game.game_over:
            winner = game.get_winner()
            local_score = float(reward[landlord])
            selected = local_score if local_landlord else -local_score
            local_won = winner == 'landlord' if local_landlord else winner == 'farmer'
            if (selected > 0) != local_won:
                raise RuntimeError('winner mismatch between official and local games')
            return {'score': selected, 'win': bool(local_won),
                    'landlord_win': winner == 'landlord',
                    'bombs': game.get_bomb_num(), 'decisions': decisions}
    raise RuntimeError('game did not end within 320 full moves')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--douzero-src', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--weights-provenance', required=True)
    parser.add_argument('--deals', type=int, default=48)
    parser.add_argument('--seed', type=int, default=410923)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.deals <= 0:
        raise ValueError('deals must be positive')
    sys.path.insert(0, str(args.douzero_src.resolve()))
    from douzero.dmc.models import model_dict
    from douzero.env import game as game_module, move_detector
    from douzero.env.env import get_obs

    torch.set_num_threads(2)
    models, weight_hashes = load_douzero_models(args.weights_dir, model_dict)
    checkpoint = args.run_dir / f'policy_{args.step:07d}.msgpack'
    params = serialization.msgpack_restore(checkpoint.read_bytes())
    config = json.loads((args.run_dir / 'config.json').read_text())
    actor, step_fn = make_local_actor(MemoryTransformer(**config['model']), params)
    @jax.jit
    def reset_and_bid(seed):
        state = env.reset(jax.random.PRNGKey(seed))
        state, _, _, invalid = env.step(state, BID_OFFSET + 3, jax.random.PRNGKey(seed + 1))
        return state, invalid

    result = {'protocol': 'official_douzero_engine_common_deals_fixed_bid3_role_mirror_v1',
              'local_step': args.step, 'local_checkpoint_sha256': file_hash(checkpoint),
              'douzero_source': str(args.douzero_src.resolve()),
              'weights_provenance': args.weights_provenance,
              'douzero_weights_sha256': weight_hashes,
              'deals': args.deals, 'seed': args.seed,
              'scoring': 'QOJ raw team score; win rate is the primary cross-rule metric',
              'limitations': ['No bidding', 'Official DouZero move generator and engine',
                              'QOJ linear bomb and spring score differs from DouZero ADP reward',
                              'Community-redistributed weights, not independently authenticated'],
              'games': []}
    started = time.monotonic()
    for index in range(args.deals):
        state, invalid = reset_and_bid(args.seed + index * 9973)
        if bool(invalid):
            raise RuntimeError('fixed three-point bid rejected')
        pair = []
        for local_landlord in (True, False):
            pair.append(play(state, local_landlord, actor, models, game_module,
                             get_obs, move_detector, step_fn))
        result['games'].append({'deal_index': index,
                                'local_landlord': pair[0],
                                'local_farmers': pair[1]})
        if (index + 1) % 8 == 0 or index + 1 == args.deals:
            print(json.dumps({'completed_deals': index + 1,
                              'seconds': round(time.monotonic() - started, 1)}), flush=True)
    paired = np.array([(game['local_landlord']['score'] +
                        game['local_farmers']['score']) / 2
                       for game in result['games']], np.float64)
    rng = np.random.default_rng(args.seed + 1)
    sampled = paired[rng.integers(len(paired), size=(5000, len(paired)))].mean(axis=1)
    result['summary'] = {'games': 2 * args.deals,
        'local_expected_team_score': float(paired.mean()),
        'score_95ci': np.percentile(sampled, [2.5, 97.5]).tolist(),
        'local_win_rate': float(np.mean([g[side]['win'] for g in result['games']
                                         for side in ('local_landlord', 'local_farmers')])),
        'landlord_win_rate': float(np.mean([g['local_landlord']['win'] for g in result['games']])),
        'farmer_team_win_rate': float(np.mean([g['local_farmers']['win'] for g in result['games']])),
        'seconds': time.monotonic() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
