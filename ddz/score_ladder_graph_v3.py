"""V3 expected-score ladder against three recent predecessors.

Each checkpoint also plays step 0 as a stable anchor. The main curve is the
least-squares projection of all pairwise expected raw team scores onto one
additive, zero-sum rating scale fixed at step 0 = 0 points/game.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import csv
import json
import subprocess
import time
import traceback
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from .elo_ladder import atomic_json, checkpoint_steps, sha256
from .model_v3 import CandidateSetTransformer
from .score_ladder_v3 import make_score_evaluator
from .score_ladder import make_svg

PROTOCOL = 'ddz_v3_candidate_set_recent_three_expected_score_graph_v1'


def planned_edges(steps, recent=3):
    edges = set()
    for i in range(1, len(steps)):
        edges.add((steps[i], steps[0]))
        for j in range(max(0, i-recent), i):
            edges.add((steps[i], steps[j]))
    return sorted(edges)


def fit_score_graph(steps, matches, rounds=2000, seed=41, recent=3):
    """Common-index deal bootstrap preserves correlations across all edges."""
    matches = sorted(matches, key=lambda m: (m['candidate_step'], m['reference_step']))
    if not matches:
        return [{'step': steps[0], 'expected_score': 0., 'score_low': 0.,
                 'score_high': 0., 'change_from_previous': 0.,
                 'change_low': 0., 'change_high': 0., 'local_score': 0.,
                 'local_low': 0., 'local_high': 0., 'games': 0,
                 'opponents': []}], 0.
    positions = {step: i for i, step in enumerate(steps)}
    design = np.zeros((len(matches), len(steps)-1), np.float64)
    for k, match in enumerate(matches):
        i, j = positions[match['candidate_step']], positions[match['reference_step']]
        if i:
            design[k, i-1] = 1.
        if j:
            design[k, j-1] = -1.
    if np.linalg.matrix_rank(design) != len(steps)-1:
        raise RuntimeError('score comparison graph is disconnected')
    scores = np.stack([np.asarray(m['paired_mean_scores'], np.float64) for m in matches])
    if not np.isfinite(scores).all() or not np.all([len(x) == scores.shape[1] for x in scores]):
        raise RuntimeError('invalid matchup score arrays')
    projection = np.linalg.pinv(design)
    means = scores.mean(axis=1)
    estimates = np.concatenate(([0.], projection @ means))
    residual_rmse = float(np.sqrt(np.mean(np.square(design @ estimates[1:] - means))))
    rng = np.random.default_rng(seed)
    samples = np.zeros((rounds, len(steps)), np.float64)
    local_samples = np.zeros((rounds, len(steps)), np.float64)
    recent_indices = []
    for i, step in enumerate(steps):
        refs = set(steps[max(0, i-recent):i])
        recent_indices.append([k for k, m in enumerate(matches)
                               if m['candidate_step'] == step and m['reference_step'] in refs])
    for k in range(rounds):
        chosen = rng.integers(scores.shape[1], size=scores.shape[1])
        sampled_means = scores[:, chosen].mean(axis=1)
        samples[k, 1:] = projection @ sampled_means
        for i in range(1, len(steps)):
            local_samples[k, i] = sampled_means[recent_indices[i]].mean()
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    delta_samples = np.diff(samples, axis=1)
    delta_low, delta_high = np.percentile(delta_samples, [2.5, 97.5], axis=0)
    local_low, local_high = np.percentile(local_samples, [2.5, 97.5], axis=0)
    rows = []
    for i, step in enumerate(steps):
        outgoing = [m for m in matches if m['candidate_step'] == step]
        local_score = float(means[recent_indices[i]].mean()) if i else 0.
        rows.append({'step': step, 'expected_score': float(estimates[i]),
            'score_low': float(low[i]), 'score_high': float(high[i]),
            'change_from_previous': float(estimates[i]-estimates[i-1]) if i else 0.,
            'change_low': float(delta_low[i-1]) if i else 0.,
            'change_high': float(delta_high[i-1]) if i else 0.,
            'local_score': local_score, 'local_low': float(local_low[i]),
            'local_high': float(local_high[i]),
            'games': sum(m['games'] for m in outgoing),
            'opponents': sorted(m['reference_step'] for m in outgoing)})
    return rows, residual_rmse


def make_html(rows, svg, protocol, residual_rmse):
    latest = rows[-1]
    table = ''.join(f'<tr><td>{r["step"]}</td><td><strong>{r["expected_score"]:+.2f}</strong></td><td>{r["score_low"]:+.2f} 至 {r["score_high"]:+.2f}</td><td>{r["change_from_previous"]:+.2f}</td><td>{r["change_low"]:+.2f} 至 {r["change_high"]:+.2f}</td><td>{r["local_score"]:+.2f}</td><td>{", ".join(map(str,r["opponents"])) or "—"}</td><td>{r["games"]}</td></tr>' for r in rows)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DDZ · 最近三代积分天梯</title><style>
body{{margin:0;background:#081710;color:#eef3e7;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:48px 22px 80px}}.tag{{color:#a8d9b3;font-size:12px;font-weight:800;letter-spacing:.14em}}h1{{font:700 clamp(36px,6vw,60px)/1.1 Georgia,"Noto Serif SC",serif;margin:10px 0 16px}}p{{color:#abc1af;max-width:1000px}}.number{{font:700 52px Georgia,serif;color:#f2d58f}}.chart{{padding:14px;border:1px solid #365544;border-radius:24px;background:#102a22;overflow:auto}}svg{{display:block;min-width:700px;width:100%}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin-top:25px;font-variant-numeric:tabular-nums;white-space:nowrap}}th,td{{padding:12px 10px;border-bottom:1px solid #315243;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:#aaceb1;font-size:13px}}strong,a{{color:#f0d38d}}.note{{margin-top:25px;padding:18px 20px;border-left:3px solid #e9c980;background:#123124}}.links a{{margin-right:20px}}
</style></head><body><main><div class="tag">DDZ · RECENT 3 CHECKPOINTS · ROLE MIRRORED</div><h1>每 500 Step 的期望积分天梯</h1><div class="number">{latest['expected_score']:+.2f} <small style="font-size:22px">积分/局</small></div><p>最新快照 step {latest['step']}；95% 区间 {latest['score_low']:+.2f} 至 {latest['score_high']:+.2f}。每个版本对战前面最近三个版本，并保留与初始版本的锚点比较。浅色带是整副牌共同重采样得到的不确定区间。</p><div class="chart">{svg}</div><div class="table"><table><thead><tr><th>训练 step</th><th>全图积分评级</th><th>95% 区间</th><th>比上一版本</th><th>变化的 95% 区间</th><th>最近三代平均净分</th><th>对手 step</th><th>对局</th></tr></thead><tbody>{table}</tbody></table></div><div class="note"><strong>评测口径</strong><p>每组配对用 {protocol['deals_per_matchup']} 副共同牌、{protocol['games_per_matchup']} 场对局；同一副牌交换地主与双农民角色。净分是候选模型控制座位的原始 API 积分之和，包含炸弹、春天等倍数。主曲线将所有配对的平均净分用等权最小二乘投影到一条零和积分轴，step 0 固定为 0；原始逐配对结果保存在 matches 目录。拟合残差均方根为 {residual_rmse:.2f} 积分/局。叫分固定为 3，模型只决策出牌与带牌。区间来自 {protocol['bootstrap_rounds']} 次共同牌局重采样。</p></div><p class="links"><a href="ratings.csv">下载 CSV</a><a href="ratings.json">查看数据与协议</a><a href="score_curve.svg">下载 SVG</a><a href="score_curve.png">下载 PNG</a></p></main></body></html>'''


def update_ladder(run_dir, output, deals=144, chunk_deals=48,
                  seed=410500, bootstrap_rounds=2000, to_step=None,
                  shard_index=0, shard_count=1):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matches_dir = output / 'matches'
    matches_dir.mkdir(exist_ok=True)
    steps = checkpoint_steps(run_dir, to_step)
    edges = planned_edges(steps)
    edge_index = {edge: i for i, edge in enumerate(edges)}
    status_path = output / (f'shard_{shard_index}.status.json' if shard_count > 1
                            else 'status.json')
    paths = {step: run_dir / f'policy_{step:07d}.msgpack' for step in steps}
    hashes = {step: sha256(path) for step, path in paths.items()}
    protocol = {'name': PROTOCOL, 'interval': 500, 'recent_predecessors': 3,
        'also_anchor_step_zero': True, 'deals_per_matchup': deals,
        'games_per_matchup': 2*deals, 'chunk_deals': chunk_deals,
        'seed': seed, 'common_deals_across_matchups': True,
        'bootstrap_rounds': bootstrap_rounds,
        'bidding': 'first bidder calls 3; role-mirrored identical post-auction deal',
        'actions': 'greedy argmax; candidate plays landlord and both farmers',
        'outcome': 'candidate mean zero-sum raw API team score per game',
        'rating': 'equal-weight least-squares additive score graph, step 0 = 0',
        'uncertainty': 'common-index paired-deal percentile bootstrap',
        'hardware': 'CPU, isolated from live A100 training'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('output directory has a different frozen graph protocol')
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)
    results = []
    missing = []
    for a, b in edges:
        path = matches_dir / f'step_{a:07d}_vs_{b:07d}.json'
        if path.exists():
            value = json.loads(path.read_text())
            if (value['protocol'] != PROTOCOL or value['candidate_sha256'] != hashes[a]
                    or value['reference_sha256'] != hashes[b]
                    or len(value['paired_mean_scores']) != deals):
                raise RuntimeError(f'cached matchup identity mismatch: {path}')
            results.append(value)
        elif edge_index[(a, b)] % shard_count == shard_index:
            missing.append((a, b, path))
    if missing:
        cfg = json.loads((run_dir / 'config.json').read_text())
        model = CandidateSetTransformer(**cfg['model'])
        evaluator = make_score_evaluator(model, chunk_deals)
        params = {step: serialization.msgpack_restore(path.read_bytes()) for step, path in paths.items()}
        for a, b, path in missing:
            landlord_scores, farmer_scores = [], []
            start = time.monotonic()
            for offset in range(0, deals, chunk_deals):
                atomic_json(status_path, {'state': 'evaluating',
                    'candidate': a, 'reference': b, 'deal_offset': offset,
                    'time': time.time()})
                chunk_seed = (seed + offset*10007) % (2**31-1)
                scores, finished, invalid, decisions = jax.device_get(
                    evaluator(params[a], params[b], chunk_seed, offset))
                if not bool(np.all(finished)) or bool(np.any(invalid)):
                    raise RuntimeError(f'failed step {a} vs {b}, offset {offset}: finished={finished}, invalid={invalid}')
                landlord_scores.extend(np.asarray(scores[:chunk_deals], np.float64).tolist())
                farmer_scores.extend(np.asarray(scores[chunk_deals:], np.float64).tolist())
            landlord = np.asarray(landlord_scores)
            farmer = np.asarray(farmer_scores)
            paired = (landlord + farmer)/2
            value = {'protocol': PROTOCOL, 'candidate_step': a,
                'reference_step': b, 'candidate_sha256': hashes[a],
                'reference_sha256': hashes[b], 'seed': seed,
                'deals': deals, 'games': 2*deals,
                'candidate_landlord_scores': landlord_scores,
                'candidate_farmer_scores': farmer_scores,
                'paired_mean_scores': paired.tolist(),
                'candidate_expected_score': float(paired.mean()),
                'candidate_wins': int(np.count_nonzero(landlord > 0)
                                      + np.count_nonzero(farmer > 0)),
                'seconds': time.monotonic()-start}
            atomic_json(path, value)
            print(json.dumps({'pair': [a, b], 'games': value['games'],
                'expected_score': round(value['candidate_expected_score'], 3),
                'seconds': round(value['seconds'], 1)}), flush=True)
            results.append(value)
    if shard_count > 1:
        atomic_json(status_path, {'state': 'shard_complete',
            'shard_index': shard_index, 'shard_count': shard_count,
            'latest_step': steps[-1], 'time': time.time()})
        print(json.dumps({'shard_complete': shard_index, 'latest_step': steps[-1]}), flush=True)
        return steps[-1]
    results.sort(key=lambda x: (x['candidate_step'], x['reference_step']))
    rows, residual_rmse = fit_score_graph(steps, results, bootstrap_rounds, seed+1)
    for row in rows:
        row['checkpoint_sha256'] = hashes[row['step']]
    atomic_json(output / 'ratings.json', {'protocol': protocol, 'steps': steps,
        'rows': rows, 'matchup_files': len(results),
        'pairwise_residual_rmse': residual_rmse,
        'updated_at_unix': time.time()})
    with (output / 'ratings.csv.tmp').open('w', newline='') as f:
        fieldnames = tuple(key for key in rows[0] if key != 'opponents') + ('opponents',)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({**row, 'opponents': ' '.join(map(str,row['opponents']))} for row in rows)
    os.replace(output / 'ratings.csv.tmp', output / 'ratings.csv')
    svg = make_svg(rows).replace('Expected team score vs step 0 (points/game)',
                                 'Pairwise expected-score rating (points/game)')
    (output / 'score_curve.svg.tmp').write_text(svg)
    os.replace(output / 'score_curve.svg.tmp', output / 'score_curve.svg')
    (output / 'score_curve.html.tmp').write_text(make_html(rows, svg, protocol, residual_rmse))
    os.replace(output / 'score_curve.html.tmp', output / 'score_curve.html')
    subprocess.run(['/root/miniforge/envs/wm/bin/python',
        str(Path(__file__).with_name('render_score_curve_v3.py')),
        str(output / 'ratings.json'),str(output / 'score_curve.png')],check=True)
    atomic_json(output / 'status.json', {'state': 'ready', 'latest_step': steps[-1],
        'matchup_files': len(results), 'time': time.time()})
    print(json.dumps({'latest_step': steps[-1], 'score_rating': round(rows[-1]['expected_score'], 3),
        'matchups': len(results), 'residual_rmse': round(residual_rmse, 3),
        'output': str(output.resolve())}), flush=True)
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
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.deals <= 0 or args.deals % args.chunk_deals or args.chunk_deals % 3:
        raise ValueError('deals must divide into chunks balanced over three first bidders')
    if args.bootstrap_rounds < 100:
        raise ValueError('at least 100 bootstrap rounds required')
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('invalid shard index or count')
    if args.watch and args.shard_count != 1:
        raise ValueError('sharded backfill workers cannot watch')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.watch:
        (output / 'watcher.pid').write_text(str(os.getpid()))
    if args.shard_count > 1:
        (output / f'shard_{args.shard_index}.pid').write_text(str(os.getpid()))
    last_step = None
    while True:
        try:
            steps = checkpoint_steps(args.run_dir, args.to_step)
            edges = planned_edges(steps)
            missing = any(not (output / 'matches' / f'step_{a:07d}_vs_{b:07d}.json').exists()
                          for a, b in edges)
            if steps[-1] != last_step or missing or not (output / 'ratings.json').exists():
                last_step = update_ladder(args.run_dir, output,
                    args.deals, args.chunk_deals, args.seed,
                    args.bootstrap_rounds, args.to_step,
                    args.shard_index, args.shard_count)
        except Exception:
            status_path = output / (f'shard_{args.shard_index}.status.json'
                                    if args.shard_count > 1 else 'status.json')
            atomic_json(status_path, {'state': 'failed',
                'time': time.time(), 'error': traceback.format_exc()})
            raise
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
