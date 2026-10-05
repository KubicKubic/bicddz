"""Role-mirrored, checkpoint-versus-checkpoint DDZ Elo ladder every 500 updates."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import csv
import hashlib
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
from .model import MemoryTransformer

PROTOCOL = 'ddz_role_mirrored_play_elo_v1'
ELO_K = math.log(10) / 400
BASE_ELO = 1000.0


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_steps(run_dir, to_step=None):
    steps = sorted(int(p.stem.split('_')[1]) for p in Path(run_dir).glob('policy_*.msgpack')
                   if int(p.stem.split('_')[1]) % 500 == 0)
    if to_step is not None:
        steps = [step for step in steps if step <= to_step]
    if not steps or steps[0] != 0 or steps != list(range(0, steps[-1] + 1, 500)):
        raise RuntimeError(f'500-step policy snapshot gap: {steps}')
    return steps


def planned_edges(steps):
    indices = range(1, len(steps))
    edges = set()
    for i in indices:
        edges.add((steps[i], steps[i - 1]))
        if i >= 2:
            edges.add((steps[i], 0))
        anchor = 6 * (i // 6)
        if anchor == i:
            anchor -= 6
        if anchor > 0 and anchor != i - 1:
            edges.add((steps[i], steps[anchor]))
    return sorted(edges)


def make_pair_evaluator(model, deals):
    """Each deal is replayed twice, swapping the model that owns landlord."""
    batch = 2 * deals
    a_is_landlord = j.concatenate((j.ones(deals, j.bool_), j.zeros(deals, j.bool_)))

    def logits(params, obs):
        variables = {'params': params}
        return jax.lax.cond(j.max(obs.hist_len) <= 96,
            lambda _: model.apply(variables, obs, memory_length=96)[0],
            lambda _: model.apply(variables, obs)[0], None)

    @jax.jit
    def evaluate(params_a, params_b, seed):
        key = jax.random.PRNGKey(seed)
        key, deal_key, bid_key = jax.random.split(key, 3)
        states = jax.vmap(env.reset)(jax.random.split(deal_key, deals))
        # The same deal and first bidder are used in both role directions.
        states = states._replace(turn=j.arange(deals, dtype=j.int32) % 3)
        states, _, _, bid_invalid = jax.vmap(env.step)(
            states, j.full((deals,), BID_OFFSET + 3, j.int32),
            jax.random.split(bid_key, deals))
        states = jax.tree_util.tree_map(lambda x: j.concatenate((x, x), axis=0), states)
        finished = j.zeros(batch, j.bool_)
        wins = j.zeros(batch, j.bool_)
        invalid = j.concatenate((bid_invalid, bid_invalid)).astype(j.int32)

        def condition(carry):
            _, _, finished, _, _, step = carry
            return j.any(~finished) & (step < 320)

        def body(carry):
            states, key, finished, wins, invalid, step = carry
            key, step_key = jax.random.split(key)
            obs = jax.vmap(env.observe)(states)
            actions_a = j.argmax(logits(params_a, obs), axis=-1).astype(j.int32)
            actions_b = j.argmax(logits(params_b, obs), axis=-1).astype(j.int32)
            a_turn = j.where(a_is_landlord, states.turn == states.landlord,
                             states.turn != states.landlord)
            actions = j.where(a_turn, actions_a, actions_b)
            next_states, reward, done, bad = jax.vmap(env.step)(
                states, actions, jax.random.split(step_key, batch))
            active = ~finished
            landlord_reward = j.take_along_axis(reward, states.landlord[:, None], axis=1)[:, 0]
            a_won = j.where(a_is_landlord, landlord_reward > 0, landlord_reward < 0)
            wins = j.where(done & active, a_won, wins)
            return (next_states, key, finished | done, wins,
                    invalid + (bad & active).astype(j.int32), step + 1)

        result = jax.lax.while_loop(condition, body,
            (states, key, finished, wins, invalid, j.int32(0)))
        return result[3], result[2], result[4], result[5]

    return evaluate


def fit_elo(steps, edges, wins_override=None):
    """Jeffreys-smoothed Bradley-Terry fit, anchored at step 0 = 1000 Elo."""
    positions = {step: i for i, step in enumerate(steps)}
    x = np.zeros(len(steps) - 1, np.float64)
    for _ in range(80):
        all_ratings = np.concatenate(([0.0], x))
        grad = np.zeros_like(x)
        hessian = np.zeros((len(x), len(x)), np.float64)
        for k, edge in enumerate(edges):
            ai = positions[edge['candidate_step']]
            bi = positions[edge['reference_step']]
            wins = sum(edge['pair_points']) if wins_override is None else wins_override[k]
            games = 2 * len(edge['pair_points'])
            delta = np.clip(ELO_K * (all_ratings[ai] - all_ratings[bi]), -35, 35)
            probability = 1.0 / (1.0 + math.exp(-delta))
            residual = ELO_K * ((games + 1) * probability - (wins + 0.5))
            curvature = ELO_K**2 * (games + 1) * probability * (1 - probability)
            if ai:
                grad[ai - 1] += residual
                hessian[ai - 1, ai - 1] += curvature
            if bi:
                grad[bi - 1] -= residual
                hessian[bi - 1, bi - 1] += curvature
            if ai and bi:
                hessian[ai - 1, bi - 1] -= curvature
                hessian[bi - 1, ai - 1] -= curvature
        if not len(x):
            break
        change = np.linalg.solve(hessian + np.eye(len(x)) * 1e-10, grad)
        scale = max(1.0, np.max(np.abs(change)) / 200.0)
        x -= change / scale
        if np.max(np.abs(change)) < 1e-7:
            break
    return BASE_ELO + np.concatenate(([0.0], x))


def bootstrap_intervals(steps, edges, rounds=300, seed=4105):
    """Paired-deal resampling plus a Jeffreys posterior for swept matchups."""
    if not edges:
        return np.full(len(steps), BASE_ELO), np.full(len(steps), BASE_ELO)
    generator = np.random.default_rng(seed)
    samples = np.empty((rounds, len(steps)), np.float64)
    for round_index in range(rounds):
        wins = []
        for edge in edges:
            pair = np.asarray(edge['pair_points'], np.int32)
            sampled = int(generator.choice(pair, size=len(pair), replace=True).sum())
            games = 2 * len(pair)
            probability = generator.beta(sampled + 0.5, games - sampled + 0.5)
            # fit_elo applies Jeffreys smoothing, so invert that transform here.
            # This keeps the posterior draw centred on the same estimate as the
            # fitted point instead of smoothing swept matchups twice.
            wins.append(float(probability * (games + 1) - 0.5))
        samples[round_index] = fit_elo(steps, edges, wins)
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def make_svg(rows):
    width, height = 1080, 540
    left, right, top, bottom = 88, 35, 36, 70
    x0, x1 = rows[0]['step'], rows[-1]['step']
    lo = min(row['elo_low'] for row in rows)
    hi = max(row['elo_high'] for row in rows)
    ymin = 100 * math.floor((lo - 35) / 100)
    ymax = 100 * math.ceil((hi + 35) / 100)
    if ymin == ymax:
        ymax += 100
    px = lambda step: left + (step - x0) / max(500, x1 - x0) * (width - left - right)
    py = lambda rating: top + (ymax - rating) / (ymax - ymin) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="每500次训练更新的相对Elo曲线">',
             '<rect width="100%" height="100%" rx="20" fill="#102a22"/>']
    for tick in range(ymin, ymax + 1, 100):
        y = py(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#779384" opacity=".28"/>')
        parts.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" fill="#b4c9bb" font-size="13">{tick}</text>')
    for row in rows:
        x = px(row['step'])
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#779384" opacity=".12"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" fill="#b4c9bb" font-size="12">{row["step"]}</text>')
    if ymin <= BASE_ELO <= ymax:
        y = py(BASE_ELO)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9b873" stroke-dasharray="7 7" opacity=".7"/>')
    if len(rows) > 1:
        upper = ' '.join(f'{px(r["step"]):.1f},{py(r["elo_high"]):.1f}' for r in rows)
        lower = ' '.join(f'{px(r["step"]):.1f},{py(r["elo_low"]):.1f}' for r in reversed(rows))
        parts.append(f'<polygon points="{upper} {lower}" fill="#8dd8a4" opacity=".18"/>')
        line = ' '.join(f'{px(r["step"]):.1f},{py(r["elo"]):.1f}' for r in rows)
        parts.append(f'<polyline points="{line}" fill="none" stroke="#efd18a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    for row in rows:
        x, y = px(row['step']), py(row['elo'])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#f7e0aa" stroke="#102a22" stroke-width="3"><title>step {row["step"]}: Elo {row["elo"]:.0f} (95% {row["elo_low"]:.0f}–{row["elo_high"]:.0f}), {row["games"]} games</title></circle>')
    parts.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle" fill="#d9e6d8" font-size="15">训练更新（每 500 step）</text>')
    parts.append('</svg>')
    return ''.join(parts)


def make_html(rows, svg, details):
    table = ''.join(f'<tr><td>{r["step"]}</td><td><strong>{r["elo"]:.0f}</strong></td><td>{r["elo_low"]:.0f}–{r["elo_high"]:.0f}</td><td>{r["games"]}</td><td>{r["matchups"]}</td></tr>' for r in rows)
    latest = rows[-1]
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DDZ · 500 Step Elo 天梯</title>
<style>body{{margin:0;background:#081710;color:#eef3e7;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 22px 80px}}.tag{{color:#a8d9b3;font-size:12px;font-weight:800;letter-spacing:.2em}}h1{{font:700 clamp(36px,6vw,64px)/1.08 Georgia,"Noto Serif SC",serif;margin:10px 0 16px}}p{{color:#abc1af;max-width:820px}}.hero{{display:flex;align-items:end;gap:24px;flex-wrap:wrap;margin-bottom:34px}}.number{{font:700 52px Georgia,serif;color:#f2d58f}}.chart{{padding:14px;border:1px solid #365544;border-radius:24px;background:#102a22;overflow:auto}}svg{{display:block;min-width:700px;width:100%}}table{{width:100%;border-collapse:collapse;margin-top:25px;font-variant-numeric:tabular-nums}}th,td{{padding:13px 12px;border-bottom:1px solid #315243;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:#aaceb1;font-size:13px}}strong{{color:#f0d38d}}.note{{margin-top:25px;padding:18px 20px;border-left:3px solid #e9c980;background:#123124}}.links a{{color:#efd18a;margin-right:20px}}@media(max-width:650px){{th,td{{padding:10px 5px;font-size:12px}}}}</style></head><body><main><div class="tag">DDZ · ROLE MIRRORED MODEL LADDER</div><h1>每 500 Step 的 Elo 曲线</h1><div class="hero"><div class="number">{latest['elo']:.0f}</div><div><div>最新快照：step {latest['step']}</div><div style="color:#aaceb1">95% 区间 {latest['elo_low']:.0f}–{latest['elo_high']:.0f}</div></div></div><p>每副牌对战两次，交换候选模型的地主／农民角色；三个座位由同一对模型分队控制。横轴是 PPO 更新次数，step 0 固定为 1000 Elo。线带表示对整副牌成对重采样、并以 Jeffreys 后验平滑全胜／全败结果所得的区间。</p><div class="chart">{svg}</div><table><thead><tr><th>训练 step</th><th>相对 Elo</th><th>95% 区间</th><th>参与对局</th><th>对手配对</th></tr></thead><tbody>{table}</tbody></table><div class="note"><strong>评测口径</strong><p>叫分由脚本固定为 3 分，保持同一副牌和首叫座位；模型只决策出牌与带牌。每个 500-step 快照都与上一个快照、step 0 以及同一 3000-step 区间的锚点连接。Elo 是该对战图上的相对指数，不是线上官方 Rating，也不测叫分能力。每条配对使用 {details['deals']} 副牌、{2 * details['deals']} 场对局；两场交换角色。区间来自 {details['bootstrap_rounds']} 次平滑的成对重采样。</p></div><p class="links"><a href="ratings.csv">下载曲线 CSV</a><a href="ratings.json">查看数据与协议</a><a href="elo_curve.svg">下载 SVG 曲线</a></p></main></body></html>'''


def update_ladder(run_dir, output, deals, seed, bootstrap_rounds, to_step=None):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matches = output / 'matches'
    matches.mkdir(exist_ok=True)
    steps = checkpoint_steps(run_dir, to_step)
    edges = planned_edges(steps)
    checkpoints = {step: run_dir / f'policy_{step:07d}.msgpack' for step in steps}
    hashes = {step: sha256(path) for step, path in checkpoints.items()}
    protocol = {'name': PROTOCOL, 'interval': 500, 'deals_per_matchup': deals,
        'games_per_matchup': 2 * deals, 'seed': seed, 'baseline_elo': BASE_ELO,
        'bidding': 'first bidder calls 3; identical post-auction deal in both role directions',
        'actions': 'greedy argmax; candidate plays landlord vs reference farmers, then roles swap',
        'outcome': 'team win/loss; no draws', 'inference': 'Jeffreys-smoothed Bradley-Terry Elo',
        'uncertainty': f'{bootstrap_rounds} paired-deal bootstrap replicates with Jeffreys beta posterior smoothing',
        'hardware': 'CPU, isolated from live A100 training'}
    prior = output / 'protocol.json'
    if prior.exists() and json.loads(prior.read_text()) != protocol:
        raise RuntimeError('output directory has a different frozen Elo protocol')
    if not prior.exists():
        atomic_json(prior, protocol)
    results = []
    missing = []
    for a, b in edges:
        path = matches / f'step_{a:07d}_vs_{b:07d}.json'
        if path.exists():
            value = json.loads(path.read_text())
            if value['candidate_sha256'] != hashes[a] or value['reference_sha256'] != hashes[b] or value['protocol'] != PROTOCOL or len(value['pair_points']) != deals:
                raise RuntimeError(f'cached matchup identity mismatch: {path}')
            results.append(value)
        else:
            missing.append((a, b, path))
    if missing:
        cfg = json.loads((run_dir / 'config.json').read_text())
        model = MemoryTransformer(**cfg['model'])
        evaluator = make_pair_evaluator(model, deals)
        params = {step: serialization.msgpack_restore(path.read_bytes()) for step, path in checkpoints.items()}
        for a, b, path in missing:
            pair_seed = (seed + a * 10007 + b * 97) % (2**31 - 1)
            atomic_json(output / 'status.json', {'state': 'evaluating', 'candidate': a, 'reference': b, 'time': time.time()})
            start = time.monotonic()
            wins, finished, invalid, steps_used = jax.device_get(evaluator(params[a], params[b], pair_seed))
            if not bool(np.all(finished)) or bool(np.any(invalid)):
                raise RuntimeError(f'failed pair {a} vs {b}: finished={finished}, invalid={invalid}')
            landlord_wins = wins[:deals].astype(int)
            farmer_wins = wins[deals:].astype(int)
            pair_points = (landlord_wins + farmer_wins).tolist()
            value = {'protocol': PROTOCOL, 'candidate_step': a, 'reference_step': b,
                'candidate_sha256': hashes[a], 'reference_sha256': hashes[b],
                'seed': pair_seed, 'deals': deals, 'games': 2 * deals,
                'candidate_landlord_wins': int(landlord_wins.sum()),
                'candidate_farmer_wins': int(farmer_wins.sum()),
                'pair_points': pair_points, 'candidate_total_wins': sum(pair_points),
                'max_decisions': int(steps_used), 'seconds': time.monotonic() - start}
            atomic_json(path, value)
            print(json.dumps({'pair': [a, b], 'wins': value['candidate_total_wins'],
                              'games': value['games'], 'seconds': round(value['seconds'], 2)}), flush=True)
            results.append(value)
    results.sort(key=lambda r: (r['candidate_step'], r['reference_step']))
    ratings = fit_elo(steps, results)
    lower, upper = bootstrap_intervals(steps, results, bootstrap_rounds, seed + 1)
    rows = []
    for i, step in enumerate(steps):
        played = [r for r in results if step in (r['candidate_step'], r['reference_step'])]
        rows.append({'step': step, 'elo': float(ratings[i]), 'elo_low': float(lower[i]),
                     'elo_high': float(upper[i]), 'games': sum(r['games'] for r in played),
                     'matchups': len(played), 'checkpoint_sha256': hashes[step]})
    data = {'protocol': protocol, 'steps': steps, 'rows': rows, 'matchup_files': len(results),
            'updated_at_unix': time.time()}
    atomic_json(output / 'ratings.json', data)
    temp_csv = output / 'ratings.csv.tmp'
    with temp_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('step', 'elo', 'elo_low', 'elo_high', 'games', 'matchups', 'checkpoint_sha256'))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_csv, output / 'ratings.csv')
    svg = make_svg(rows)
    svg_tmp = output / 'elo_curve.svg.tmp'
    svg_tmp.write_text(svg)
    os.replace(svg_tmp, output / 'elo_curve.svg')
    html_tmp = output / 'elo_curve.html.tmp'
    html_tmp.write_text(make_html(rows, svg, {'deals': deals, 'bootstrap_rounds': bootstrap_rounds}))
    os.replace(html_tmp, output / 'elo_curve.html')
    atomic_json(output / 'status.json', {'state': 'ready', 'latest_step': steps[-1],
                                         'matchup_files': len(results), 'time': time.time()})
    print(json.dumps({'latest_step': steps[-1], 'latest_elo': round(rows[-1]['elo'], 1),
                      'matchups': len(results), 'output': str(output.resolve())}), flush=True)
    return steps[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deals', type=int, default=24)
    parser.add_argument('--seed', type=int, default=410500)
    parser.add_argument('--bootstrap-rounds', type=int, default=300)
    parser.add_argument('--to-step', type=int)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.deals <= 0 or args.deals % 3:
        raise ValueError('deals must be a positive multiple of three for exact first-bidder balance')
    if args.bootstrap_rounds < 20:
        raise ValueError('at least 20 bootstrap replicates required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.watch:
        (output / 'watcher.pid').write_text(str(os.getpid()))
    last_step = None
    while True:
        try:
            steps = checkpoint_steps(args.run_dir, args.to_step)
            edges = planned_edges(steps)
            missing = any(not (output / 'matches' / f'step_{a:07d}_vs_{b:07d}.json').exists() for a, b in edges)
            if steps[-1] != last_step or missing or not (output / 'ratings.json').exists():
                last_step = update_ladder(args.run_dir, output, args.deals, args.seed,
                                          args.bootstrap_rounds, args.to_step)
        except Exception:
            atomic_json(output / 'status.json', {'state': 'failed', 'time': time.time(),
                                                 'error': traceback.format_exc()})
            raise
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
