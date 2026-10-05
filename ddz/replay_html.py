"""Generate self-contained HTML replays of model-versus-baseline DDZ games."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from flax import serialization

from . import env
from .evaluate import baseline
from .model import MemoryTransformer

RANKS = ('3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '小王', '大王')
TYPE_ZH = {
    'pass': '不出', 'single': '单张', 'pair': '对子', 'trio': '三张',
    'trio1': '三带一', 'trio2': '三带二', 'straight': '顺子',
    'pairs': '连对', 'plane': '飞机', 'plane1': '飞机带单',
    'plane2': '飞机带对', 'four2': '四带二', 'four22': '四带两对',
    'bomb': '炸弹', 'rocket': '王炸',
}


def packed(s):
    return {name: getattr(s, name) for name in (
        'hands', 'bottom', 'turn', 'phase', 'landlord', 'bid', 'bombs',
        'redeals', 'hist_len', 'last', 'last_seat', 'done')}


def run_games(model, params, count, seed, mode):
    seats = j.arange(count) % 3

    @jax.jit
    def play(params, key):
        key, deal_key = jax.random.split(key)
        state = jax.vmap(env.reset)(jax.random.split(deal_key, count))
        initial = packed(state)

        def tick(carry, _):
            state, key, finished, invalid = carry
            key, action_key, step_key = jax.random.split(key, 3)
            obs = jax.vmap(env.observe)(state)
            logits, _ = model.apply({'params': params}, obs)
            learner = j.argmax(logits, axis=-1).astype(j.int32)
            if mode == 'selfplay':
                action = learner
            else:
                rivals = jax.vmap(baseline)(state, jax.random.split(action_key, count))
                action = j.where(state.turn == seats, learner, rivals)
            next_state, reward, done, bad = jax.vmap(env.step)(
                state, action, jax.random.split(step_key, count))
            active = ~finished
            invalid = invalid + (bad & active).astype(j.int32)
            trace = {'state': packed(next_state), 'reward': reward, 'active': active}
            return (next_state, key, finished | done, invalid), trace

        (state, _, finished, invalid), trace = jax.lax.scan(
            tick, (state, key, j.zeros(count, j.bool_), j.zeros(count, j.int32)),
            None, length=320)
        return initial, state.history, state.hist_len, trace, finished, invalid

    return jax.device_get(play(params, jax.random.PRNGKey(seed)))


def decode_event(raw):
    from .actions import TYPES
    raw = np.asarray(raw)
    kind = int(raw[15])
    seat = int(raw[16])
    event = {'kind': kind, 'seat': seat, 'cards': raw[:15].astype(int).tolist()}
    if kind == 1:
        event['text'] = f'{"不叫" if not int(raw[20]) else f"叫 {int(raw[20])} 分"}'
    elif kind == 2:
        event['text'] = f'成为地主 · {int(raw[20])} 分，获得底牌'
    elif kind == 3:
        event['type'] = TYPE_ZH[TYPES[int(raw[17])]]
        event['text'] = event['type']
    elif kind == 4:
        event['text'] = '不出'
    elif kind == 5:
        event['text'] = '三家不叫，重新发牌'
    else:
        raise ValueError(f'unknown event kind {kind}')
    return event


def snapshot(source, game, tick=None):
    def get(name):
        value = source[name]
        return value[game] if tick is None else value[tick, game]
    return {
        'hands': get('hands').astype(int).tolist(),
        'bottom': get('bottom').astype(int).tolist(),
        **{name: int(get(name)) for name in (
            'turn', 'phase', 'landlord', 'bid', 'bombs', 'redeals',
            'hist_len', 'last', 'last_seat')},
        'done': bool(get('done')),
    }


def build_replay(initial, history, trace, game, seat, seed, mode):
    frames = [dict(snapshot(initial, game), step=0, events=[], score=None)]
    previous = 0
    for tick in range(trace['active'].shape[0]):
        if not bool(trace['active'][tick, game]):
            break
        state = snapshot(trace['state'], game, tick)
        length = state['hist_len']
        if length < previous:
            raise RuntimeError('history length went backward')
        if length > previous or state['done']:
            events = [decode_event(history[game, i]) for i in range(previous, length)]
            reward = trace['reward'][tick, game].astype(int).tolist() if state['done'] else None
            frames.append(dict(state, step=tick + 1, events=events, score=reward))
        previous = length
    final = frames[-1]
    if not final['done'] or final['score'] is None:
        raise RuntimeError(f'game {game + 1} did not finish')
    if sum(final['score']) != 0 or not any(sum(hand) == 0 for hand in final['hands']):
        raise RuntimeError(f'game {game + 1} has inconsistent final score or hands')
    if final['hist_len'] != len([event for frame in frames for event in frame['events']]):
        raise RuntimeError(f'game {game + 1} lost a public event')
    return {
        'number': game + 1, 'seed': seed, 'bot_seat': seat, 'mode': mode,
        'opponent': 'selfplay' if mode == 'selfplay' else 'fixed_shedding_v1', 'frames': frames,
        'bot_score': final['score'][seat], 'landlord': final['landlord'],
        'winner': '地主' if final['score'][final['landlord']] > 0 else '农民',
        'events': final['hist_len'], 'decisions': final['step'],
    }


def make_index(replays, checkpoint_name, mode):
    cards = []
    selfplay = mode == 'selfplay'
    for game in replays:
        won = game['bot_score'] > 0
        badge = ('观察座位获胜' if won else '观察座位落败') if selfplay else ('模型获胜' if won else '模型落败')
        cards.append(f'''<a class="game" href="game_{game['number']:02d}.html">
          <span class="num">{game['number']:02d}</span><span class="badge {'win' if won else 'loss'}">{badge}</span>
          <strong>第 {game['number']} 局</strong><span>{'观察' if selfplay else '模型'}座位 {game['bot_seat']} · {game['winner']}胜</span>
          <small>{game['events']} 个公开事件 · 原始积分 {game['bot_score']:+d}</small><span class="arrow">观看回放 ↗</span>
        </a>''')
    description = (f'模型快照 {checkpoint_name} 的三个座位都由同一模型控制，每局轮换一个观察座位。'
                   if selfplay else f'模型快照 {checkpoint_name} 与固定启发式对手的本地模拟对局。')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>斗地主 · 十局回放</title>
<style>body{{margin:0;background:#09130f;color:#f4ecd6;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:56px 24px 80px}}.eyebrow{{color:#9bd4af;letter-spacing:.22em;font-size:12px;font-weight:800}}h1{{font:700 clamp(40px,7vw,76px)/1.05 Georgia,serif;margin:12px 0 20px}}p{{color:#aabbb0;max-width:680px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:42px}}.game{{display:grid;gap:8px;text-decoration:none;color:inherit;padding:25px;border:1px solid #355643;border-radius:20px;background:linear-gradient(145deg,#173a2b,#11291f);transition:transform .2s,border-color .2s}}.game:hover{{transform:translateY(-4px);border-color:#d7b56c}}.num{{font:700 50px Georgia,serif;color:#d9ba76}}.badge{{justify-self:start;border-radius:20px;padding:4px 9px;font-size:12px;background:#365c42;color:#b4edbd}}.loss{{background:#51392e;color:#f4c8aa}}strong{{font-size:24px}}small,.game>span:not(.num):not(.badge):not(.arrow){{color:#aabbb0}}.arrow{{margin-top:10px;color:#e9ca84}}</style></head><body><main><div class="eyebrow">DDZ · LOCAL SIMULATION REPLAYS</div><h1>{'十局模型自对弈' if selfplay else '十局斗地主'}</h1><p>{description}每局都有可自动播放、暂停、逐步查看和调速的独立 HTML 页面；这些不是线上真人对局。</p><div class="grid">{''.join(cards)}</div></main></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--mode', choices=('baseline', 'selfplay'), default='baseline')
    args = parser.parse_args()
    if args.games != 10:
        raise ValueError('this renderer produces exactly ten game pages')
    cfg = json.loads(Path(args.config).read_text())
    checkpoint = Path(args.checkpoint)
    model = MemoryTransformer(**cfg['model'])
    params = serialization.msgpack_restore(checkpoint.read_bytes())
    initial, history, lengths, trace, finished, invalid = run_games(
        model, params, args.games, args.seed, args.mode)
    if not np.all(finished) or np.any(invalid):
        raise RuntimeError(f'incomplete or invalid games: finished={finished}, invalid={invalid}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name('replay_template.html').read_text()
    replays = []
    for i in range(args.games):
        game = build_replay(initial, history, trace, i, i % 3, args.seed, args.mode)
        if game['events'] != int(lengths[i]):
            raise RuntimeError('final history mismatch')
        replays.append(game)
        payload = json.dumps(game, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
        (output / f'game_{i + 1:02d}.html').write_text(
            template.replace('__GAME_DATA__', payload), encoding='utf-8')
    (output / 'index.html').write_text(make_index(replays, checkpoint.name, args.mode), encoding='utf-8')
    manifest = {
        'format': 'self-contained interactive HTML replay', 'source': 'local DDZ simulator',
        'opponent': 'selfplay' if args.mode == 'selfplay' else 'fixed_shedding_v1',
        'mode': args.mode, 'games': args.games, 'seed': args.seed,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'results': [{key: game[key] for key in (
            'number', 'bot_seat', 'bot_score', 'landlord', 'winner', 'events', 'decisions')}
                    for game in replays],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output.resolve()), 'results': manifest['results']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
