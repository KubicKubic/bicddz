"""Render an existing morphology audit; no checkpoint reload or inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(out):
    out=Path(out);data=json.loads((out/'audit.json').read_text())
    categories=data['parameter_categories'];ffn=data['ffn_activations'];n=data['diagnostics']['positions']
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    allocation=sorted(categories,key=categories.get)
    axes[0,0].barh(allocation,[categories[k]/1e6 for k in allocation])
    axes[0,0].set_xlabel('Parameters (millions)');axes[0,0].set_title('Parameter allocation')
    axes[0,1].bar([x['layer'] for x in ffn],[100*x['score_p90_neurons']/x['neurons'] for x in ffn])
    axes[0,1].tick_params(axis='x',rotation=25)
    axes[0,1].set_ylabel('Neurons covering 90% of proxy energy (%)')
    axes[0,1].set_title('Proxy = activation RMS x output-weight norm')
    ranked=sorted(data['ablations'],key=lambda x:x['body_kl_mean'],reverse=True)[:14]
    axes[1,0].barh([x['name'] for x in ranked][::-1],[x['body_kl_mean'] for x in ranked][::-1])
    axes[1,0].set_xlabel('Mean body KL after zeroing (nats)');axes[1,0].set_title('Functional sensitivity; not playing strength')
    points=[{'name':'baseline','remaining_parameters':data['parameters'],'complete_greedy_change':0},*data['pruning_probes']]
    axes[1,1].plot([x['remaining_parameters']/1e6 for x in points],[100*x['complete_greedy_change'] for x in points],'o-')
    for x in points:
        label='V5 baseline' if x['name']=='baseline' else 'FF width '+x['name'].rsplit('_',1)[-1]
        offset=(-8,10) if x['name']=='baseline' else (8,-18) if x['complete_greedy_change']>.1 else (8,10)
        axes[1,1].annotate(label,(x['remaining_parameters']/1e6,100*x['complete_greedy_change']),
            xytext=offset,textcoords='offset points',ha='right' if x['name']=='baseline' else 'left')
    axes[1,1].set_xlabel('Parameters after structural pruning (millions)')
    axes[1,1].set_ylabel('Complete greedy action changes (%)')
    axes[1,1].set_title('Direct norm pruning; no distillation or retraining')
    fig.suptitle(f"V5 morphology: global step {data['global_step']} | {data['parameters']:,} parameters",fontsize=17)
    max_history=max(data['diagnostics']['history_lengths'])
    fig.text(.02,.01,f'{n} role/history-stratified positions; CPU FP32 reference; maximum sampled history = {max_history}.\nWeight spectra, activation proxies and zeroing sensitivity do not establish retrained model efficiency.',fontsize=10)
    for axis in axes.flat:axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    fig.tight_layout(rect=(0,.045,1,.96))
    destination=out/'parameter_efficiency.png';fig.savefig(destination,dpi=150);plt.close(fig)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    print(render(parser.parse_args().out))
