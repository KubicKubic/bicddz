"""Render the V3 expected-score ladder JSON as a standalone PNG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(ratings,out):
    data=json.loads(Path(ratings).read_text())
    rows=data['rows']
    x=np.asarray([r['step'] for r in rows],float)
    y=np.asarray([r['expected_score'] for r in rows],float)
    lo=np.asarray([r['score_low'] for r in rows],float)
    hi=np.asarray([r['score_high'] for r in rows],float)
    plt.rcParams.update({'font.size':12,'figure.facecolor':'#081710',
                         'axes.facecolor':'#102a22','axes.edgecolor':'#809a89',
                         'axes.labelcolor':'#ecf1e9','xtick.color':'#c7d7cb',
                         'ytick.color':'#c7d7cb','text.color':'#ecf1e9'})
    fig,ax=plt.subplots(figsize=(13,6.5),dpi=160)
    ax.fill_between(x,lo,hi,color='#8dd8a4',alpha=.25,label='95% paired-deal interval')
    ax.plot(x,y,color='#efd18a',lw=2.8,marker='o',markersize=7,
            label='Expected team score')
    ax.axhline(0,color='#d9b873',lw=1,ls='--',alpha=.7)
    for step,score in zip(x,y):
        ax.annotate(f'{score:+.2f}',(step,score),xytext=(0,9),
                    textcoords='offset points',ha='center',color='#f5dfa9')
    ax.set_xticks(x)
    ax.set_xlim(-max(50,x[-1]*.025),max(500,x[-1])+max(50,x[-1]*.025))
    span=max(2.,float(np.max(hi)-np.min(lo)))
    ax.set_ylim(float(np.min(lo))-.15*span,float(np.max(hi))+.15*span)
    ax.set_xlabel('PPO update step')
    ax.set_ylabel('Expected raw team score (points/game)')
    ax.set_title('DDZ V3: role-mirrored, recent-checkpoint score ladder',pad=20)
    ax.grid(axis='y',color='#6f8c79',alpha=.22)
    ax.legend(facecolor='#17382a',edgecolor='#456552',labelcolor='#ecf1e9')
    fig.tight_layout()
    path=Path(out); temp=path.with_name(path.stem+'.tmp'+path.suffix)
    fig.savefig(temp,facecolor=fig.get_facecolor())
    plt.close(fig)
    temp.replace(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('ratings'); ap.add_argument('output')
    args=ap.parse_args()
    render(args.ratings,args.output)
