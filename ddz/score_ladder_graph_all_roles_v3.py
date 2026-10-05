"""Comprehensive 500-step DDZ ladder: natural auction plus guaranteed roles."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import csv
import json
import subprocess
import time
import traceback
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from .elo_ladder import atomic_json, checkpoint_steps, sha256
from .model_v3 import CandidateSetTransformer
from .score_ladder_all_roles_v3 import ROLE_NAMES, make_role_evaluator, paired_scores
from .score_ladder_graph_v3 import fit_score_graph, planned_edges


PROTOCOL = 'ddz_v3_natural_auction_all_seats_and_forced_all_roles_v1'


def _metric_rows(steps, matches, key, rounds, seed, games_per_deal):
    selected = [{**m, 'paired_mean_scores': m[key],
                 'games': m['deals'] * games_per_deal} for m in matches]
    return fit_score_graph(steps, selected, rounds=rounds, seed=seed)


def _html(rows, protocol):
    latest = rows[-1]
    table = ''.join(
        '<tr><td>{step}</td><td>{expected_score:+.2f}</td><td>{score_low:+.2f} ~ {score_high:+.2f}</td>'
        '<td>{landlord:+.2f}</td><td>{landlord_next:+.2f}</td><td>{door:+.2f}</td>'
        '<td>{fixed_balanced:+.2f}</td><td>{opponents}</td><td>{games}</td></tr>'.format(
            step=r['step'], expected_score=r['expected_score'],
            score_low=r['score_low'], score_high=r['score_high'], games=r['games'],
            landlord=r['roles']['landlord']['expected_score'],
            landlord_next=r['roles']['landlord_next']['expected_score'],
            door=r['roles']['door']['expected_score'],
            fixed_balanced=r['fixed_balanced']['expected_score'],
            opponents=','.join(map(str, r['opponents'])) or '—') for r in rows)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DDZ 全角色综合积分天梯</title><style>
body{{margin:0;background:#081710;color:#eef3e7;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:38px 22px 70px}}h1{{font:700 clamp(31px,5vw,54px)/1.1 system-ui;margin:8px 0 18px}}p{{color:#b5cabb}}.number{{font:700 52px Georgia,serif;color:#f2d58f}}.chart{{padding:10px;border:1px solid #365544;border-radius:20px;background:#102a22}}img{{display:block;width:100%;height:auto}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin-top:25px;white-space:nowrap;font-variant-numeric:tabular-nums}}th,td{{padding:10px 9px;border-bottom:1px solid #315243;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:#aaceb1;font-size:13px}}a{{color:#efd18a;margin-right:20px}}.note{{margin-top:24px;padding:18px;border-left:3px solid #e9c980;background:#123124}}</style></head><body><main>
<div style="color:#a8d9b3;font-size:12px;letter-spacing:.15em">DDZ · ALL SEATS · NATURAL AUCTION</div><h1>每 500 Step 的全角色综合能力</h1><div class="number">{latest['expected_score']:+.2f} <small style="font-size:20px">原始积分/局</small></div>
<p>最新 step {latest['step']}；主指标 95% 区间 {latest['score_low']:+.2f} 至 {latest['score_high']:+.2f}。主曲线让模型自然叫分，逐副牌交换单座位与其余两个座位的控制权，并对三个起始座位等权平均。下图同时列出固定叫分后确保覆盖的地主、地主下家、门板表现。</p>
<div class="chart"><img src="score_curve.png" alt="全角色综合能力及三个角色的期望积分曲线"></div><div class="table"><table><thead><tr><th>step</th><th>自然叫分综合分</th><th>95% 区间</th><th>地主</th><th>地主下家</th><th>门板</th><th>三角色平均</th><th>对手 step</th><th>对局</th></tr></thead><tbody>{table}</tbody></table></div>
<div class="note"><strong>评测口径</strong><p>每组 {protocol['deals_per_matchup']} 副共同牌、{protocol['games_per_matchup']} 局：自然叫分六局，固定叫分六局。每组三个焦点座位，各有候选模型单独控制该座位和控制另两个座位的互补两局。主指标取自然叫分六局的原始 API 积分期望；固定叫分的地主单座位积分除以 2，与两个农民座位同权后绘制角色曲线。均包含叫分（主指标）、压牌、炸弹、春天等实际规则。每个版本对战最近三代和 step 0；全图以等权最小二乘投影，step 0=0。同一批牌跨配对共用，{protocol['bootstrap_rounds']} 次整副牌共同重采样给出 95% 区间。角色曲线是出牌能力诊断，不能代替含叫分的主指标。</p></div>
<p><a href="ratings.csv">CSV</a><a href="ratings.json">完整数据</a><a href="protocol.json">协议</a><a href="score_curve.png">PNG</a></p></main></body></html>'''


def update_ladder(run_dir, output, deals=288, chunk_deals=12,
                  seed=410500, bootstrap_rounds=2000, to_step=None):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matches_dir = output / 'matches'
    matches_dir.mkdir(exist_ok=True)
    chunks_dir = output / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    steps = checkpoint_steps(run_dir, to_step)
    edges = planned_edges(steps)
    paths = {step: run_dir / f'policy_{step:07d}.msgpack' for step in steps}
    hashes = {step: sha256(path) for step, path in paths.items()}
    protocol = {
        'name': PROTOCOL, 'interval': 500, 'recent_predecessors': 3,
        'also_anchor_step_zero': True, 'deals_per_matchup': deals,
        'games_per_matchup': 12 * deals, 'chunk_deals': chunk_deals,
        'seed': seed, 'common_deals_across_matchups': True,
        'bootstrap_rounds': bootstrap_rounds,
        'natural_auction': '6 games per deal, focus seat relative to first bidder',
        'forced_roles': '6 games per deal, first bidder calls 3; focus landlord/next/door',
        'leg_pair': 'A on focus seat, B on other two; then B on focus, A on other two',
        'primary': 'natural auction mean paired candidate raw API score over three focus seats',
        'role_diagnostic': 'paired focus raw score, landlord divided by 2, equal role weights',
        'rating': 'equal-weight least-squares additive score graph, step 0 = 0',
        'uncertainty': 'common-index paired-deal percentile bootstrap',
        'hardware': 'CPU, isolated from live A100 training'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('output directory has a different frozen graph protocol')
    if not protocol_path.exists():
        atomic_json(protocol_path, protocol)

    matches = []
    missing = []
    for a, b in edges:
        path = matches_dir / f'step_{a:07d}_vs_{b:07d}.json'
        if path.exists():
            match = json.loads(path.read_text())
            if (match['protocol'] != PROTOCOL or match['candidate_sha256'] != hashes[a]
                    or match['reference_sha256'] != hashes[b]
                    or len(match['natural_paired_scores']) != deals
                    or any(len(match['fixed_role_normalized_paired_scores'][r]) != deals
                           for r in ROLE_NAMES)):
                raise RuntimeError(f'cached matchup identity mismatch: {path}')
            matches.append(match)
        else:
            missing.append((a, b, path))
    if missing:
        cfg = json.loads((run_dir / 'config.json').read_text())
        model = CandidateSetTransformer(**cfg['model'])
        natural_eval = make_role_evaluator(model, chunk_deals, forced_bid=False)
        fixed_eval = make_role_evaluator(model, chunk_deals, forced_bid=True)

        @lru_cache(maxsize=5)
        def params(step):
            return serialization.msgpack_restore(paths[step].read_bytes())

        for a, b, path in missing:
            natural_legs, fixed_legs = [], []
            start = time.monotonic()
            for offset in range(0, deals, chunk_deals):
                chunk_path = chunks_dir / f'step_{a:07d}_vs_{b:07d}_deal_{offset:05d}.json'
                if chunk_path.exists():
                    cached = json.loads(chunk_path.read_text())
                    if (cached['protocol'] != PROTOCOL or cached['candidate_sha256'] != hashes[a]
                            or cached['reference_sha256'] != hashes[b]
                            or cached['seed'] != seed or cached['offset'] != offset
                            or np.shape(cached['natural_leg_scores']) != (3, 2, chunk_deals)
                            or np.shape(cached['fixed_leg_scores']) != (3, 2, chunk_deals)):
                        raise RuntimeError(f'cached deal chunk identity mismatch: {chunk_path}')
                    natural_legs.append(np.asarray(cached['natural_leg_scores'], np.float64))
                    fixed_legs.append(np.asarray(cached['fixed_leg_scores'], np.float64))
                    continue
                atomic_json(output / 'status.json', {'state': 'evaluating',
                    'candidate': a, 'reference': b, 'deal_offset': offset,
                    'deals_per_matchup': deals, 'time': time.time()})
                chunk_seed = (seed + offset * 10007) % (2**31 - 1)
                mode_legs = []
                for evaluator in (natural_eval, fixed_eval):
                    scores, finished, invalid, decisions = jax.device_get(
                        evaluator(params(a), params(b), chunk_seed, offset))
                    if not bool(np.all(finished)) or bool(np.any(invalid)):
                        raise RuntimeError(f'failed {a} vs {b} offset {offset}: '
                            f'finished={np.count_nonzero(finished)}/{len(finished)}, '
                            f'invalid={np.sum(invalid)}, decisions={decisions}')
                    mode_legs.append(np.asarray(scores, np.float64).reshape(3, 2, chunk_deals))
                atomic_json(chunk_path, {'protocol': PROTOCOL,
                    'candidate_sha256': hashes[a], 'reference_sha256': hashes[b],
                    'seed': seed, 'offset': offset,
                    'natural_leg_scores': mode_legs[0].tolist(),
                    'fixed_leg_scores': mode_legs[1].tolist()})
                natural_legs.append(mode_legs[0])
                fixed_legs.append(mode_legs[1])
            natural = np.concatenate(natural_legs, axis=2)
            fixed = np.concatenate(fixed_legs, axis=2)
            natural_role, natural_total = paired_scores(natural, deals, forced_bid=False)
            fixed_role, fixed_total = paired_scores(fixed, deals, forced_bid=True)
            match = {'protocol': PROTOCOL, 'candidate_step': a, 'reference_step': b,
                'candidate_sha256': hashes[a], 'reference_sha256': hashes[b],
                'seed': seed, 'deals': deals, 'games': 12 * deals,
                'natural_leg_scores': natural.tolist(), 'fixed_leg_scores': fixed.tolist(),
                'natural_paired_scores': natural_total.tolist(),
                'natural_nominal_seat_paired_scores': {
                    name: natural_role[i].tolist() for i, name in enumerate(ROLE_NAMES)},
                'fixed_role_normalized_paired_scores': {
                    name: fixed_role[i].tolist() for i, name in enumerate(ROLE_NAMES)},
                'fixed_role_raw_paired_scores': {
                    name: (fixed_role[i] * (2 if i == 0 else 1)).tolist()
                    for i, name in enumerate(ROLE_NAMES)},
                'fixed_balanced_paired_scores': fixed_total.tolist(),
                'candidate_expected_score': float(natural_total.mean()),
                'seconds': time.monotonic() - start}
            atomic_json(path, match)
            print(json.dumps({'pair': [a, b], 'games': match['games'],
                'natural_expected_score': round(match['candidate_expected_score'], 3),
                'seconds': round(match['seconds'], 1)}), flush=True)
            matches.append(match)
    matches.sort(key=lambda m: (m['candidate_step'], m['reference_step']))
    rows, residual = _metric_rows(steps, matches, 'natural_paired_scores',
                                  bootstrap_rounds, seed + 1, 6)
    fixed_rows, fixed_residual = _metric_rows(steps, matches,
        'fixed_balanced_paired_scores', bootstrap_rounds, seed + 1, 6)
    role_rows = {}
    role_residual = {}
    for name in ROLE_NAMES:
        key = 'fixed_role_normalized_paired_scores'
        selected = [{**m, key: m[key][name]} for m in matches]
        role_rows[name], role_residual[name] = _metric_rows(
            steps, selected, key, bootstrap_rounds, seed + 1, 2)
    for i, row in enumerate(rows):
        row['checkpoint_sha256'] = hashes[row['step']]
        row['games'] = sum(m['games'] for m in matches
                           if m['candidate_step'] == row['step'])
        row['fixed_balanced'] = fixed_rows[i]
        row['roles'] = {name: role_rows[name][i] for name in ROLE_NAMES}
    atomic_json(output / 'ratings.json', {'protocol': protocol, 'steps': steps,
        'rows': rows, 'matchup_files': len(matches),
        'pairwise_residual_rmse': {'natural': residual,
            'fixed_balanced': fixed_residual, **role_residual},
        'updated_at_unix': time.time()})
    csv_path = output / 'ratings.csv'
    with csv_path.with_suffix('.csv.tmp').open('w', newline='') as handle:
        fields = ['step', 'natural_expected_score', 'natural_low', 'natural_high',
                  'fixed_balanced', 'landlord', 'landlord_next', 'door',
                  'games', 'opponents']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'step': row['step'],
                'natural_expected_score': row['expected_score'],
                'natural_low': row['score_low'], 'natural_high': row['score_high'],
                'fixed_balanced': row['fixed_balanced']['expected_score'],
                **{name: row['roles'][name]['expected_score'] for name in ROLE_NAMES},
                'games': row['games'],
                'opponents': ' '.join(map(str, row['opponents']))})
    os.replace(csv_path.with_suffix('.csv.tmp'), csv_path)
    html_path = output / 'score_curve.html'
    html_path.with_suffix('.html.tmp').write_text(_html(rows, protocol))
    os.replace(html_path.with_suffix('.html.tmp'), html_path)
    subprocess.run(['/root/miniforge/envs/wm/bin/python',
        str(Path(__file__).with_name('render_score_curve_all_roles_v3.py')),
        str(output / 'ratings.json'), str(output / 'score_curve.png')], check=True)
    atomic_json(output / 'status.json', {'state': 'ready',
        'latest_step': steps[-1], 'matchup_files': len(matches), 'time': time.time()})
    print(json.dumps({'latest_step': steps[-1],
        'natural_score': round(rows[-1]['expected_score'], 3),
        'matchups': len(matches), 'output': str(output.resolve())}), flush=True)
    return steps[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deals', type=int, default=288)
    parser.add_argument('--chunk-deals', type=int, default=12)
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
            missing = any(not (output / 'matches' /
                f'step_{a:07d}_vs_{b:07d}.json').exists()
                for a, b in planned_edges(steps))
            if steps[-1] != last_step or missing or not (output / 'ratings.json').exists():
                last_step = update_ladder(args.run_dir, output, args.deals,
                    args.chunk_deals, args.seed, args.bootstrap_rounds, args.to_step)
        except Exception:
            atomic_json(output / 'status.json', {'state': 'failed',
                'time': time.time(), 'error': traceback.format_exc()})
            raise
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
