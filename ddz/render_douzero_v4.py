"""Render frozen DouZero evaluation statistics without rerunning games."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def interval(metric,percent=False):
    scale=100 if percent else 1
    suffix='%' if percent else ''
    mean=metric['mean']*scale
    lo,hi=[x*scale for x in metric['ci95']]
    return f'{mean:.2f}{suffix} [{lo:.2f}, {hi:.2f}]'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('comparison',type=Path)
    args=ap.parse_args()
    result=json.loads(args.comparison.read_text())
    root=args.comparison.parent
    baselines=result['baselines']
    roles=('landlord','landlord_down','landlord_up')
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.2),gridspec_kw={'width_ratios':[1.6,1]})
    colors=('#248f71','#d58b2b')
    for i,(name,entry) in enumerate(baselines.items()):
        summary=entry['summary']
        metrics=[summary['roles'][r]['expected_score'] for r in roles]
        metrics.append(summary['equal_role_expected_score'])
        mean=np.array([m['mean'] for m in metrics])
        lows=np.array([m['ci95'][0] for m in metrics])
        highs=np.array([m['ci95'][1] for m in metrics])
        x=np.arange(4)+(i-.5)*.26
        axes[0].errorbar(x,mean,yerr=np.array([mean-lows,highs-mean]),
            fmt='o',capsize=5,label=f'DouZero {name}',color=colors[i],linewidth=1.8)
        m=summary['team_mirror']['win_rate'];v=m['mean']*100
        axes[1].errorbar(i,v,yerr=[[v-m['ci95'][0]*100],[m['ci95'][1]*100-v]],
            fmt='o',color=colors[i],capsize=6,markersize=8,linewidth=2)
        axes[1].annotate(f'{v:.2f}%',(i,v),xytext=(12,0),
            textcoords='offset points',va='center',fontsize=11)
    axes[0].axhline(0,color='#5e6570',linestyle='--',linewidth=1)
    axes[0].set_xticks(range(4),['Landlord','Landlord next','Door','Equal-role mean'])
    axes[0].set_ylabel('Expected QOJ points (landlord score / 2)')
    axes[0].set_title('Six control-swapped legs per deal')
    axes[0].legend(frameon=False)
    axes[1].axhline(50,color='#5e6570',linestyle='--',linewidth=1)
    axes[1].set_xticks(range(len(baselines)),[f'DouZero {n}' for n in baselines])
    axes[1].set_ylabel('Our team win rate (%)')
    axes[1].set_title('Our landlord / our two farmers mirror')
    axes[1].set_xlim(-.4,len(baselines)-.35)
    for ax in axes:
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'V4 checkpoint {result["checkpoint_step"]:,} vs DouZero',fontsize=17)
    fig.text(.5,.025,'1,024 common deals per baseline; 6,144 games each; paired whole-deal 95% intervals. Fixed bid 3; community-redistributed weights.',ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.055,1,.94))
    fig.savefig(root/'comparison.png',dpi=180)
    plt.close(fig)
    rows=[]
    for name,entry in baselines.items():
        s=entry['summary']
        rows.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in
            (name,str(s['games']),interval(s['team_mirror']['win_rate'],True),
             interval(s['equal_role_expected_score']),
             *(interval(s['roles'][r]['expected_score']) for r in roles)))+'</tr>')
    provenance=''.join(f'<li>DouZero {name}: <a href="{html.escape(entry["identity"]["weights_provenance"],quote=True)}">retained weight source</a></li>' for name,entry in baselines.items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>V4 vs DouZero</title>
<style>body{{font:16px/1.6 system-ui;background:#f5f7f8;color:#192a28;max-width:1400px;margin:35px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:12px;border-bottom:1px solid #ccd7d4;text-align:right}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>Checkpoint {result['checkpoint_step']:,} vs DouZero ADP and WP</h1>
<p>Frozen checkpoint SHA-256: <code>{result['checkpoint_sha256']}</code></p>
<img src="comparison.png" alt="Role scores and mirrored team win rates with confidence intervals">
<div class="scroll"><table><tr><th>Baseline</th><th>Games</th><th>Team mirror win rate</th><th>Equal-role expected score</th><th>Landlord</th><th>Landlord next</th><th>Door</th></tr>{''.join(rows)}</table></div>
<h2>Protocol</h2><p>Each baseline uses the same 1,024 unseen evaluation deals. Each deal has six games: our model controls one focus role against DouZero on the other two seats, then control is swapped, for each of landlord, landlord next and door. Farmer focus tests include a teammate from the other model. Role scores average the two complementary legs; landlord scores are divided by two to equalize stakes. Zero is the equal-strength score reference.</p>
<p>The team mirror win rate uses the two landlord-focus legs: our landlord against two DouZero farmers, and two of our farmers against a DouZero landlord. Fifty percent is the equal-strength reference. Separate landlord or farmer win rates need not be 50% because the roles have different inherent advantages.</p>
<p>All intervals use 5,000 paired whole-deal bootstrap draws, keeping all six legs together. These are individual comparison intervals, without multiple-comparison adjustment. Sampling uncertainty does not cover weight-provenance or rule-system uncertainty.</p>
<p>Our deployed deterministic body-then-wing policy is unchanged. Every move, hand update, turn, terminal flag and winner was checked against the official DouZero engine; invalid moves were not replaced. All 12,288 games finished with zero invalid actions. Histories beyond the short-memory shortcut use the full history. No opponent's private hand is supplied to either policy.</p>
<h2>Scope and baseline identity</h2><p>The test fixes bidding at three points and evaluates card play. Expected points use our QOJ linear bomb/spring score, which differs from DouZero's exponential ADP reward. The DouZero weights are the retained community redistributions of the original architecture, with strict state-dictionary loading and hash checks; they are not independently authenticated official checkpoints or a test of newer DouZero derivatives.</p>
<ul>{provenance}</ul><p><a href="https://github.com/kwai/DouZero">Official DouZero source</a> · <a href="comparison.json">Full results and identities</a> · <a href="comparison.png">PNG</a></p></html>'''
    (root/'report.html').write_text(page)
    print(root/'report.html')


if __name__=='__main__': main()
