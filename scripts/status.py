"""Read-only concise run status; does not initialize JAX or touch the GPU."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('run',type=Path); a=p.parse_args()
for name in ['latest.json','status.json']:
    path=a.run/name
    if path.exists(): print(name,json.loads(path.read_text()))
path=a.run/'metrics.jsonl'
if path.exists():
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16384)); lines=f.read().splitlines()
    for line in lines[-3:]:
        try: print(json.loads(line))
        except json.JSONDecodeError: pass
