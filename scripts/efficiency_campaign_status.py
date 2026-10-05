"""Read-only progress for the isolated campaign; uses the standard library."""
import argparse,json,re
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('runs/v5_efficiency_campaign_v1'))
args=ap.parse_args();root=args.root
s=json.loads((root/'campaign_status.json').read_text());task=s.get('task','')
print('CAMPAIGN',s['state'],task)
match=re.fullmatch(r'train_(.+)_rng(\d+)_(\d+)',task)
if match:
    name,seed,limit=match.groups();p=root/'training'/f'{name}_rng{seed}'/'metrics.jsonl'
    if p.exists() and p.stat().st_size:
        x=json.loads(p.read_text().splitlines()[-1])
        print(f"{name} RNG {seed}: {x['iteration']}/{limit}, "
              f"KL {x['kl']:.4f}, applied {x['applied']:.1%}, "
              f"fresh/s {x['decisions_per_second']:.0f}, "
              f"invalid {x['invalid_actions']}, nonfinite {x['nonfinite']:.0f}")
elif task.startswith('eval_'):
    p=root/'evaluations'/task[5:]/'status.json'
    if p.exists():
        x=json.loads(p.read_text());print(x.get('scope',''),x.get('completed_deals',x.get('deals')),'/',x.get('deals'),'unique deals')
elif task.startswith('douzero_'):
    p=root/'douzero'/task[8:]/'status.json'
    if p.exists():
        x=json.loads(p.read_text());print('strong DouZero',x.get('completed_deals',0),'/',x.get('total_deals'),'unique deals')
if s['state']=='failed':print(s.get('error',''),s.get('exit_code',''))
if (root/'selection.json').exists():
    x=json.loads((root/'selection.json').read_text());print('Exploratory selection:',x['winner'],'(confirmation required)')
if s['state']=='complete':
    x=json.loads((root/'result.json').read_text());print('Confirmed benefit:',x['confirmed_benefit'])
