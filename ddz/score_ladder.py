"""Common-deal, role-mirrored expected raw-score ladder every 500 updates.

The ordinate is the candidate model's expected zero-sum team payoff per game
against the step-0 policy. A candidate controls one landlord or both farmers;
the two games for each deal are averaged before inference.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from flax import serialization

from . import env
from .actions import BID_OFFSET
from .elo_ladder import atomic_json, checkpoint_steps, sha256
from .model import MemoryTransformer

PROTOCOL = 'ddz_common_deal_expected_raw_team_score_v2'


def make_score_evaluator(model, chunk_deals):
    """Return each candidate role's actual terminal API team payoff."""
    batch = 2 * chunk_deals
    a_is_landlord = j.concatenate((j.ones(chunk_deals, j.bool_),
                                   j.zeros(chunk_deals, j.bool_)))

    def logits(params, obs):
        variables = {'params': params}
        return jax.lax.cond(j.max(obs.hist_len) <= 96,
            lambda _: model.apply(variables, obs, memory_length=96)[0],
            lambda _: model.apply(variables, obs)[0], None)

    @jax.jit
    def evaluate(params_a, params_b, seed, deal_offset):
        key = jax.random.PRNGKey(seed)
        key, deal_key, bid_key = jax.random.split(key, 3)
        states = jax.vmap(env.reset)(jax.random.split(deal_key, chunk_deals))
        states = states._replace(turn=(j.arange(chunk_deals, dtype=j.int32) + deal_offset) % 3)
        states, _, _, bid_invalid = jax.vmap(env.step)(
            states, j.full((chunk_deals,), BID_OFFSET + 3, j.int32),
            jax.random.split(bid_key, chunk_deals))
        states = jax.tree_util.tree_map(lambda x: j.concatenate((x, x), axis=0), states)
        finished = j.zeros(batch, j.bool_)
        score = j.zeros(batch, j.float32)
        invalid = j.concatenate((bid_invalid, bid_invalid)).astype(j.int32)

        def condition(carry):
            _, _, finished, _, _, step = carry
            return j.any(~finished) & (step < 320)

        def body(carry):
            states, key, finished, score, invalid, step = carry
            key, step_key = jax.random.split(key)
            obs = jax.vmap(env.observe)(states)
            action_a = j.argmax(logits(params_a, obs), axis=-1).astype(j.int32)
            action_b = j.argmax(logits(params_b, obs), axis=-1).astype(j.int32)
            a_turn = j.where(a_is_landlord, states.turn == states.landlord,
                             states.turn != states.landlord)
            actions = j.where(a_turn, action_a, action_b)
            next_states, reward, done, bad = jax.vmap(env.step)(
                states, actions, jax.random.split(step_key, batch))
            landlord_payoff = j.take_along_axis(reward, states.landlord[:, None], axis=1)[:, 0]
            # Farmers' two API payoffs sum to the negative of landlord payoff.
            candidate_payoff = j.where(a_is_landlord, landlord_payoff, -landlord_payoff)
            active = ~finished
            score = j.where(done & active, candidate_payoff, score)
            return next_states, key, finished | done, score, \
                invalid + (bad & active).astype(j.int32), step + 1

        result = jax.lax.while_loop(condition, body,
            (states, key, finished, score, invalid, j.int32(0)))
        return result[3], result[2], result[4], result[5]

    return evaluate


def score_summary(steps, matches, rounds=2000, seed=41):
    """Resample common deal indices across all checkpoints and both roles."""
    if len(steps) == 1:
        return [{'step': 0, 'expected_score': 0., 'score_low': 0.,
                 'score_high': 0., 'change_from_previous': 0.,
                 'change_low': 0., 'change_high': 0., 'win_rate': .5,
                 'games': 0}]
    indexed = {m['candidate_step']: m for m in matches}
    scores = np.stack([np.asarray(indexed[step]['paired_mean_scores'], np.float64)
                       for step in steps[1:]])
    deals = scores.shape[1]
    rng = np.random.default_rng(seed)
    samples = np.empty((rounds, len(steps)), np.float64)
    samples[:, 0] = 0.
    for k in range(rounds):
        chosen = rng.integers(deals, size=deals)
        samples[k, 1:] = scores[:, chosen].mean(axis=1)
    estimate = np.concatenate(([0.], scores.mean(axis=1)))
    lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
    delta_samples = np.diff(samples, axis=1)
    delta_lo, delta_hi = np.percentile(delta_samples, [2.5, 97.5], axis=0)
    rows = []
    for i, step in enumerate(steps):
        match = indexed.get(step)
        rows.append({'step': step, 'expected_score': float(estimate[i]),
            'score_low': float(lower[i]), 'score_high': float(upper[i]),
            'change_from_previous': float(estimate[i] - estimate[i-1]) if i else 0.,
            'change_low': float(delta_lo[i-1]) if i else 0.,
            'change_high': float(delta_hi[i-1]) if i else 0.,
            'win_rate': match['candidate_wins'] / match['games'] if match else .5,
            'games': match['games'] if match else 0})
    return rows


def make_svg(rows):
    width, height = 1100, 560
    left, right, top, bottom = 90, 35, 42, 76
    x0, x1 = rows[0]['step'], rows[-1]['step']
    lo = min(0., *(r['score_low'] for r in rows))
    hi = max(0., *(r['score_high'] for r in rows))
    spacing = max(2., math.ceil((hi - lo) / 8 / 2) * 2)
    ymin = spacing * math.floor((lo - 1) / spacing)
    ymax = spacing * math.ceil((hi + 1) / spacing)
    if ymin == ymax:
        ymax += spacing
    px = lambda step: left + (step - x0) / max(500, x1-x0) * (width-left-right)
    py = lambda value: top + (ymax-value) / (ymax-ymin) * (height-top-bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Expected raw team score versus initial model">',
             '<rect width="100%" height="100%" rx="20" fill="#102a22"/>',
             f'<text x="{width/2}" y="27" text-anchor="middle" fill="#d9e6d8" font-size="15">Expected team score vs step 0 (points/game)</text>']
    for tick in np.arange(ymin, ymax + spacing/2, spacing):
        y = py(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#779384" opacity=".28"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" fill="#b4c9bb" font-size="13">{tick:g}</text>')
    for row in rows:
        x = px(row['step'])
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#779384" opacity=".12"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" fill="#b4c9bb" font-size="12">{row["step"]}</text>')
    parts.append(f'<line x1="{left}" y1="{py(0):.1f}" x2="{width-right}" y2="{py(0):.1f}" stroke="#d9b873" stroke-dasharray="7 7" opacity=".7"/>')
    if len(rows) > 1:
        upper = ' '.join(f'{px(r["step"]):.1f},{py(r["score_high"]):.1f}' for r in rows)
        lower = ' '.join(f'{px(r["step"]):.1f},{py(r["score_low"]):.1f}' for r in reversed(rows))
        line = ' '.join(f'{px(r["step"]):.1f},{py(r["expected_score"]):.1f}' for r in rows)
        parts.append(f'<polygon points="{upper} {lower}" fill="#8dd8a4" opacity=".19"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="#efd18a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    for row in rows:
        x, y = px(row['step']), py(row['expected_score'])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#f7e0aa" stroke="#102a22" stroke-width="3"><title>step {row["step"]}: {row["expected_score"]:+.2f} points/game (95% {row["score_low"]:+.2f} to {row["score_high"]:+.2f})</title></circle>')
    parts.append(f'<text x="{width/2}" y="{height-17}" text-anchor="middle" fill="#d9e6d8" font-size="15">PPO updates (every 500 steps)</text>')
    parts.append('</svg>')
    return ''.join(parts)


def make_html(rows, svg, protocol):
    latest = rows[-1]
    table = ''.join(f'<tr><td>{r["step"]}</td><td><strong>{r["expected_score"]:+.2f}</strong></td><td>{r["score_low"]:+.2f} 至 {r["score_high"]:+.2f}</td><td>{r["change_from_previous"]:+.2f}</td><td>{r["change_low"]:+.2f} 至 {r["change_high"]:+.2f}</td><td>{r["win_rate"]:.1%}</td><td>{r["games"]}</td></tr>' for r in rows)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DDZ · 500 Step 期望积分天梯</title><style>
body{{margin:0;background:#081710;color:#eef3e7;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:48px 22px 80px}}.tag{{color:#a8d9b3;font-size:12px;font-weight:800;letter-spacing:.15em}}h1{{font:700 clamp(36px,6vw,60px)/1.1 Georgia,"Noto Serif SC",serif;margin:10px 0 16px}}p{{color:#abc1af;max-width:950px}}.number{{font:700 52px Georgia,serif;color:#f2d58f}}.chart{{padding:14px;border:1px solid #365544;border-radius:24px;background:#102a22;overflow:auto}}svg{{display:block;min-width:700px;width:100%}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin-top:25px;font-variant-numeric:tabular-nums;white-space:nowrap}}th,td{{padding:12px 10px;border-bottom:1px solid #315243;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:#aaceb1;font-size:13px}}strong,a{{color:#f0d38d}}.note{{margin-top:25px;padding:18px 20px;border-left:3px solid #e9c980;background:#123124}}.links a{{margin-right:20px}}
</style></head><body><main><div class="tag">DDZ · COMMON DEALS · ROLE MIRRORED</div><h1>每 500 Step 的期望积分</h1><div class="number">{latest['expected_score']:+.2f} <small style="font-size:22px">积分/局</small></div><p>最新快照 step {latest['step']}；95% 区间 {latest['score_low']:+.2f} 至 {latest['score_high']:+.2f}。每个快照与初始模型对战，所有快照使用同一批牌。浅色带是按整副牌成对重采样得到的不确定区间。</p><div class="chart">{svg}</div><div class="table"><table><thead><tr><th>训练 step</th><th>期望净积分/局</th><th>95% 区间</th><th>比上一快照</th><th>变化的 95% 区间</th><th>胜率（辅助）</th><th>对局</th></tr></thead><tbody>{table}</tbody></table></div><div class="note"><strong>评测口径</strong><p>每组用 {protocol['deals_per_matchup']} 副牌、{protocol['games_per_matchup']} 场对局；同一副牌交换候选模型的地主与双农民角色。净积分是候选模型控制座位的原始 API 积分之和，包含炸弹、春天等倍数，横轴每隔 500 次 PPO 更新。叫分固定为 3，模型只决策出牌与带牌；初始 step 0 自对弈的角色平衡期望积分固定为 0。胜率仅作辅助，不参与主曲线计算。各快照的区间和相邻变化共用 {protocol['bootstrap_rounds']} 次牌局重采样。</p></div><p class="links"><a href="ratings.csv">下载 CSV</a><a href="ratings.json">查看数据与协议</a><a href="score_curve.svg">下载 SVG</a></p></main></body></html>'''


def update_ladder(run_dir, output, deals=144, chunk_deals=48, seed=410500,
                  bootstrap_rounds=2000, to_step=None):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matches_dir = output / 'matches'
    matches_dir.mkdir(exist_ok=True)
    steps = checkpoint_steps(run_dir, to_step)
    paths = {step: run_dir / f'policy_{step:07d}.msgpack' for step in steps}
    hashes = {step: sha256(path) for step, path in paths.items()}
    protocol = {'name': PROTOCOL, 'interval': 500, 'deals_per_matchup': deals,
        'games_per_matchup': 2*deals, 'chunk_deals': chunk_deals, 'seed': seed,
        'common_deals_across_checkpoints': True, 'bootstrap_rounds': bootstrap_rounds,
        'bidding': 'first bidder calls 3; same post-auction deal in both role directions',
        'actions': 'greedy argmax; candidate plays landlord then both farmers',
        'outcome': 'mean zero-sum team API score per game, including multipliers',
        'comparison': 'direct versus step-0 policy; no Elo conversion',
        'uncertainty': 'common-index paired-deal percentile bootstrap',
        'hardware': 'CPU, isolated from live A100 training'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('output directory has a different frozen score protocol')
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)

    results = []
    missing = []
    for step in steps[1:]:
        path = matches_dir / f'step_{step:07d}_vs_0000000.json'
        if path.exists():
            value = json.loads(path.read_text())
            if (value['protocol'] != PROTOCOL or value['candidate_sha256'] != hashes[step]
                    or value['reference_sha256'] != hashes[0]
                    or len(value['paired_mean_scores']) != deals):
                raise RuntimeError(f'cached matchup identity mismatch: {path}')
            results.append(value)
        else:
            missing.append((step, path))
    if missing:
        cfg = json.loads((run_dir / 'config.json').read_text())
        model = MemoryTransformer(**cfg['model'])
        evaluator = make_score_evaluator(model, chunk_deals)
        baseline = serialization.msgpack_restore(paths[0].read_bytes())
        for step, path in missing:
            candidate = serialization.msgpack_restore(paths[step].read_bytes())
            landlord_scores, farmer_scores = [], []
            start = time.monotonic()
            for offset in range(0, deals, chunk_deals):
                atomic_json(output / 'status.json', {'state': 'evaluating',
                    'candidate': step, 'deal_offset': offset, 'time': time.time()})
                chunk_seed = (seed + offset * 10007) % (2**31-1)
                scores, finished, invalid, decisions = jax.device_get(
                    evaluator(candidate, baseline, chunk_seed, offset))
                if not bool(np.all(finished)) or bool(np.any(invalid)):
                    raise RuntimeError(f'failed step {step} offset {offset}: finished={finished}, invalid={invalid}')
                landlord_scores.extend(np.asarray(scores[:chunk_deals], np.float64).tolist())
                farmer_scores.extend(np.asarray(scores[chunk_deals:], np.float64).tolist())
            landlord = np.asarray(landlord_scores)
            farmer = np.asarray(farmer_scores)
            paired = (landlord + farmer) / 2
            value = {'protocol': PROTOCOL, 'candidate_step': step, 'reference_step': 0,
                'candidate_sha256': hashes[step], 'reference_sha256': hashes[0],
                'seed': seed, 'deals': deals, 'games': 2*deals,
                'candidate_landlord_scores': landlord_scores,
                'candidate_farmer_scores': farmer_scores,
                'paired_mean_scores': paired.tolist(),
                'candidate_expected_score': float(paired.mean()),
                'candidate_wins': int(np.count_nonzero(landlord > 0) + np.count_nonzero(farmer > 0)),
                'seconds': time.monotonic()-start}
            atomic_json(path, value)
            print(json.dumps({'step': step, 'games': value['games'],
                'expected_score': round(value['candidate_expected_score'], 3),
                'seconds': round(value['seconds'], 1)}), flush=True)
            results.append(value)
    results.sort(key=lambda x: x['candidate_step'])
    rows = score_summary(steps, results, bootstrap_rounds, seed+1)
    for row in rows:
        row['checkpoint_sha256'] = hashes[row['step']]
    atomic_json(output / 'ratings.json', {'protocol': protocol, 'steps': steps,
        'rows': rows, 'matchup_files': len(results), 'updated_at_unix': time.time()})
    csv_path = output / 'ratings.csv'
    with (output / 'ratings.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(output / 'ratings.csv.tmp', csv_path)
    svg = make_svg(rows)
    (output / 'score_curve.svg.tmp').write_text(svg)
    os.replace(output / 'score_curve.svg.tmp', output / 'score_curve.svg')
    (output / 'score_curve.html.tmp').write_text(make_html(rows, svg, protocol))
    os.replace(output / 'score_curve.html.tmp', output / 'score_curve.html')
    atomic_json(output / 'status.json', {'state': 'ready', 'latest_step': steps[-1],
        'matchup_files': len(results), 'time': time.time()})
    print(json.dumps({'latest_step': steps[-1], 'expected_score': round(rows[-1]['expected_score'], 3),
        'matchups': len(results), 'output': str(output.resolve())}), flush=True)
    return steps[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deals', type=int, default=144)
    parser.add_argument('--chunk-deals', type=int, default=48)
    parser.add_argument('--seed', type=int, default=410500)
    parser.add_argument('--bootstrap-rounds', type=int, default=2000)
    parser.add_argument('--to-step', type=int)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.deals <= 0 or args.deals % args.chunk_deals or args.chunk_deals % 3:
        raise ValueError('deals must divide into chunks balanced over three first bidders')
    if args.bootstrap_rounds < 100:
        raise ValueError('at least 100 bootstrap rounds required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.watch:
        (output / 'watcher.pid').write_text(str(os.getpid()))
    last_step = None
    while True:
        try:
            steps = checkpoint_steps(args.run_dir, args.to_step)
            missing = any(not (output / 'matches' / f'step_{step:07d}_vs_0000000.json').exists()
                          for step in steps[1:])
            if steps[-1] != last_step or missing or not (output / 'ratings.json').exists():
                last_step = update_ladder(args.run_dir, output, args.deals,
                    args.chunk_deals, args.seed, args.bootstrap_rounds, args.to_step)
        except Exception:
            atomic_json(output / 'status.json', {'state': 'failed', 'time': time.time(),
                'error': traceback.format_exc()})
            raise
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
