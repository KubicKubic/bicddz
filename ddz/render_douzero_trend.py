"""Standalone score chart for the fixed strong DouZero opponent."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    rows = data['rows']
    x = [r['global_step'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, label in [('expected_score','All-role average')]:
        axes[0].errorbar(x,[r[key] for r in rows],
            yerr=[[r[key]-r['low'] for r in rows],[r['high']-r[key] for r in rows]],
            fmt='o-',capsize=4,label=label)
    axes[0].fill_between(x,[r['low'] for r in rows],[r['high'] for r in rows], alpha=.2, label='95% deal bootstrap CI')
    axes[0].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[0].set_ylabel('Expected normalized score / game')
    axes[0].set_title('V5 own-seat GAE vs DouZero ResNet 2.0 best')
    axes[0].legend()
    for role, label in [('landlord','Landlord'),('landlord_down','Landlord next'),('landlord_up','Door')]:
        axes[1].plot(x,[r['roles'][role]['expected_score']['mean'] for r in rows], 'o-', label=label)
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_ylabel('Role normalized score / game')
    axes[1].set_xlabel('Global training updates')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    note='Fixed bid 3; six role/control legs per common deal; landlord score /2. Does not measure bidding.'
    transition=data.get('protocol',{}).get('sampling_transition')
    if transition:
        note+=f'\nNew points: {transition["new_deals"]:,} deals; retained earlier points: {transition["historical_deals"]:,} deals each.'
    fig.text(.01,.01,note,fontsize=9)
    fig.tight_layout(rect=(0,.055 if transition else .035,1,1))
    destination = Path(sys.argv[2])
    tmp = destination.with_name(destination.stem+'.tmp.png')
    fig.savefig(tmp, dpi=170)
    tmp.replace(destination)
    plt.close(fig)


if __name__ == '__main__':
    main()
