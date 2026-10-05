"""Six-seat complementary matches on each deal, with or without natural bidding."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from functools import partial
import numpy as np
import jax
import jax.numpy as j

from . import env_v2 as env
from . import candidates_v3 as moves


ROLE_NAMES = ('landlord', 'landlord_next', 'door')


def paired_scores(scores, deals, forced_bid):
    """Return three per-deal complementary scores and their equal-role mean."""
    legs = np.asarray(scores, np.float64).reshape(3, 2, deals)
    paired = legs.mean(axis=1)
    if forced_bid:
        paired = paired / np.asarray([2., 1., 1.])[:, None]
    return paired, paired.mean(axis=0)


def make_role_evaluator(model, chunk_deals, forced_bid, memory_limit=88):
    """Evaluate A versus B in each seat and its complementary two-seat team.

    The six legs share each initial deal, first bidder, and per-decision random
    keys. For forced_bid, the first bidder becomes landlord. For natural bidding,
    focus seats are relative to the first bidder; all three final roles remain
    covered, while auction decisions contribute to the score.
    """
    games = 6 * chunk_deals
    role = np.repeat(np.arange(3, dtype=np.int32), 2 * chunk_deals)
    complement = np.tile(np.repeat(np.arange(2, dtype=np.bool_), chunk_deals), 3)
    leg_deal = np.tile(np.arange(chunk_deals, dtype=np.int32), 6)

    @partial(jax.jit, static_argnames=('memory_length',))
    def forward(params, obs, ids, mask, memory_length):
        return model.apply({'params': params}, obs, ids, mask,
                           memory_length=memory_length)[0]

    def evaluate(params_a, params_b, seed, deal_offset):
        key = jax.random.PRNGKey(seed)
        key, deal_key, bid_key = jax.random.split(key, 3)
        states = env.batch_reset(jax.random.split(deal_key, chunk_deals))
        first = (j.arange(chunk_deals, dtype=j.int32) + deal_offset) % 3
        states = states._replace(turn=first)
        invalid_initial = np.zeros(chunk_deals, np.int32)
        if forced_bid:
            states, _, _, bad = env.batch_step(
                states, j.full((chunk_deals,), env.encode_bid(3)),
                jax.random.split(bid_key, chunk_deals))
            invalid_initial = np.asarray(bad, np.int32)
        states = jax.tree_util.tree_map(lambda x: j.concatenate((x,) * 6, axis=0), states)
        focus = (np.asarray(first)[leg_deal] + role) % 3
        finished = np.zeros(games, np.bool_)
        scores = np.zeros(games, np.float32)
        invalid = np.tile(invalid_initial, 6)
        for decision in range(256):
            if finished.all():
                break
            ids_np, mask_np, _ = moves.batch_legal(states)
            ids = j.asarray(ids_np)
            mask = j.asarray(mask_np)
            obs = jax.vmap(env.observe)(states)
            memory = memory_limit if int(j.max(states.hist_len)) <= memory_limit else env.HISTORY
            logits_a = forward(params_a, obs, ids, mask, memory)
            logits_b = forward(params_b, obs, ids, mask, memory)
            actions_a = np.asarray(j.argmax(logits_a, axis=-1))
            actions_b = np.asarray(j.argmax(logits_b, axis=-1))
            turns = np.asarray(states.turn)
            owns_a = np.where(complement, turns != focus, turns == focus)
            chosen = np.where(owns_a, actions_a, actions_b)
            actions = j.asarray(moves.ACTION[ids_np[np.arange(games), chosen]])
            key, step_key = jax.random.split(key)
            # Identical per-deal randomness in all six legs also covers redeals.
            step_keys = jnp_repeat_keys(step_key, chunk_deals)
            next_states, reward, done, bad = env.batch_step(states, actions, step_keys)
            terminal = np.asarray(done) & ~finished
            focus_reward = np.asarray(reward)[np.arange(games), focus]
            scores = np.where(terminal, np.where(complement, -focus_reward, focus_reward), scores)
            invalid += np.asarray(bad & j.asarray(~finished), np.int32)
            finished |= np.asarray(done)
            states = next_states
        return scores, finished, invalid, decision + 1

    return evaluate


def jnp_repeat_keys(key, chunk_deals):
    return j.tile(jax.random.split(key, chunk_deals), (6, 1))
