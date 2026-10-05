"""Plot direct, paired-deal V3 versus final V2 results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(inputs, output):
    data = [json.loads(Path(path).read_text()) for path in inputs]
    data.sort(key=lambda item: item['protocol']['v3_step'])
    steps = [item['protocol']['v3_step'] for item in data]
    v2_steps = {item['protocol']['v2_step'] for item in data}
    v2_label = f'V2 step {next(iter(v2_steps))}' if len(v2_steps) == 1 else 'V2 references'
    x = np.arange(len(data), dtype=float)
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': '#081710',
        'axes.facecolor': '#102a22', 'axes.edgecolor': '#809a89',
        'axes.labelcolor': '#ecf1e9', 'xtick.color': '#c7d7cb',
        'ytick.color': '#c7d7cb', 'text.color': '#ecf1e9'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    main = [item['natural_comprehensive'] for item in data]
    means = np.array([item['mean'] for item in main])
    lo = means - np.array([item['low'] for item in main])
    hi = np.array([item['high'] for item in main]) - means
    axes[0].bar(x, means, width=.55, color='#efd18a', alpha=.88)
    axes[0].errorbar(x, means, yerr=[lo, hi], fmt='none', ecolor='#f2f1df',
                     capsize=5, lw=1.8)
    for position, value in zip(x, means):
        axes[0].annotate(f'{value:+.2f}', (position, value),
                         xytext=(0, -18 if value < 0 else 7),
                         textcoords='offset points', ha='center')
    axes[0].set_title('Natural-auction comprehensive score', pad=15)
    axes[0].set_ylabel(f'V3 minus {v2_label}: raw points / game')
    names = [('landlord', 'Landlord / 2', '#efd18a'),
             ('landlord_next', 'Landlord next', '#8dd8a4'),
             ('door', 'Door farmer', '#92b9f5')]
    for i, (name, label, color) in enumerate(names):
        role = [item['fixed_roles'][name] for item in data]
        values = np.array([item['mean'] for item in role])
        lower = values - np.array([item['low'] for item in role])
        upper = np.array([item['high'] for item in role]) - values
        position = x + (i - 1) * .23
        axes[1].bar(position, values, width=.21, color=color, label=label)
        axes[1].errorbar(position, values, yerr=[lower, upper], fmt='none',
                         ecolor='#f2f1df', capsize=3, lw=1.1)
    axes[1].set_title('Guaranteed roles after a fixed 3-point bid', pad=15)
    axes[1].set_ylabel(f'V3 minus {v2_label}: role-normalized points')
    axes[1].legend(facecolor='#17382a', edgecolor='#456552',
                   labelcolor='#ecf1e9', fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [f'V3 step {step}\n{data[i]["deals"]} deals'
                          for i, step in enumerate(steps)])
        ax.axhline(0, color='#e9d39d', lw=1.3)
        ax.grid(axis='y', color='#6f8c79', alpha=.2)
    fig.suptitle(f'DDZ direct comparison: V3 versus {v2_label}', fontsize=15)
    fig.tight_layout()
    out = Path(output)
    temp = out.with_name(out.stem + '.tmp' + out.suffix)
    fig.savefig(temp, facecolor=fig.get_facecolor())
    plt.close(fig)
    temp.replace(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('output')
    ap.add_argument('inputs', nargs='+')
    args = ap.parse_args()
    render(args.inputs, args.output)
