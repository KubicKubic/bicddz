"""Render the all-seat comprehensive score ladder as a standalone PNG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(ratings, out):
    rows = json.loads(Path(ratings).read_text())['rows']
    x = np.asarray([r['step'] for r in rows], float)
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': '#081710',
        'axes.facecolor': '#102a22', 'axes.edgecolor': '#809a89',
        'axes.labelcolor': '#ecf1e9', 'xtick.color': '#c7d7cb',
        'ytick.color': '#c7d7cb', 'text.color': '#ecf1e9'})
    fig, (main, roles) = plt.subplots(2, 1, figsize=(13, 9.2), dpi=160,
                                       sharex=True, gridspec_kw={'height_ratios': [1.1, 1]})
    y = np.asarray([r['expected_score'] for r in rows])
    lo = np.asarray([r['score_low'] for r in rows])
    hi = np.asarray([r['score_high'] for r in rows])
    main.fill_between(x, lo, hi, color='#8dd8a4', alpha=.25,
                      label='95% paired-deal interval')
    main.plot(x, y, color='#efd18a', lw=2.8, marker='o', markersize=7,
              label='Natural-auction comprehensive score')
    for step, score in zip(x, y):
        main.annotate(f'{score:+.2f}', (step, score), xytext=(0, 8),
                      textcoords='offset points', ha='center', color='#f5dfa9')
    main.set_ylabel('Expected raw score / game')
    main.set_title('DDZ V3: all-seat comprehensive ladder (natural bidding)', pad=17)
    main.legend(facecolor='#17382a', edgecolor='#456552', labelcolor='#ecf1e9')

    for name, label, color in [('landlord', 'Landlord / 2', '#efd18a'),
                               ('landlord_next', 'Landlord next', '#8dd8a4'),
                               ('door', 'Door farmer', '#92b9f5')]:
        role = [r['roles'][name] for r in rows]
        value = np.asarray([r['expected_score'] for r in role])
        lower = np.asarray([r['score_low'] for r in role])
        upper = np.asarray([r['score_high'] for r in role])
        roles.plot(x, value, lw=2.3, marker='o', markersize=5, color=color, label=label)
        roles.fill_between(x, lower, upper, color=color, alpha=.12)
    fixed = np.asarray([r['fixed_balanced']['expected_score'] for r in rows])
    roles.plot(x, fixed, lw=2, ls='--', color='#e8b9cb',
               label='Equal-role mean')
    roles.set_title('Guaranteed roles after a fixed 3-point bid', pad=12)
    roles.set_ylabel('Role-normalized score / game')
    roles.set_xlabel('PPO update step')
    roles.legend(facecolor='#17382a', edgecolor='#456552', labelcolor='#ecf1e9',
                 ncol=2)
    for ax in (main, roles):
        ax.axhline(0, color='#d9b873', lw=1, ls='--', alpha=.65)
        ax.grid(axis='y', color='#6f8c79', alpha=.22)
        ax.set_xticks(x)
    roles.set_xlim(-max(50, x[-1] * .025), max(500, x[-1]) + max(50, x[-1] * .025))
    fig.tight_layout(h_pad=2.8)
    path = Path(out)
    temp = path.with_name(path.stem + '.tmp' + path.suffix)
    fig.savefig(temp, facecolor=fig.get_facecolor())
    plt.close(fig)
    temp.replace(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ratings')
    ap.add_argument('output')
    args = ap.parse_args()
    render(args.ratings, args.output)
