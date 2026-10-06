"""Freeze and queue trimming after the currently owned production segment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ddz.trim_rollout_campaign import revised_config
from ddz.scale_rollout_campaign import read,write,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--tests-proof',type=Path,required=True)
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=args.root.resolve()
    if root.exists():raise RuntimeError('Frozen campaign already exists; inspect rather than repeat')
    tests=read(args.tests_proof)
    if not tests.get('passed') or tests.get('cpu_collective_devices')!=8:
        raise RuntimeError('CPU selection and eight-replica gradient checks required')
    for name,expected in tests['files_sha256'].items():
        if sha(Path(name))!=expected:raise RuntimeError('Tested implementation changed: '+name)
    queue=repo.parent/'Q/tools/cluster/cluster_scheduler.queue';status=read(queue/'status.json')
    if status['phase']!='RUNNING' or status['running_count']!=1 or time.time()-status['updated_at_ns']/1e9>180:
        raise RuntimeError('Healthy persistent worker required')
    active=read(queue/'running'/(status['active_id']+'.json'));argv=shlex.split(active['command'])
    source=Path(read(repo/'runs/production_current.json')['run']);old=source.parent.parent
    if (not any(module in argv for module in ('ddz.cluster_distributed_job','ddz.scale_architecture_v6')) or '--root' not in argv or
        Path(argv[argv.index('--root')+1]).resolve()!=old.resolve()):
        raise RuntimeError('Source is not the currently owned production task')
    production=read(source.parent/'queue_job_status.json')
    if production['state']!='running' or production['queue_id']!=active['id']:
        raise RuntimeError('Current bounded production child identity differs')
    at=int(production['steps']);cfg=revised_config(read(source/'config.json'))
    if '--steps' in argv and int(argv[argv.index('--steps')+1])!=at:
        raise RuntimeError('Queue and production target differ')
    if at+8>=cfg['continuation_updates']:raise RuntimeError('Insufficient remaining rollout budget')
    if any('ddz.trim_rollout_campaign' in read(p)['command'] for p in (queue/'pending').glob('*.json')):
        raise RuntimeError('An advantage-trim campaign is already pending')
    acceptance=read(old/'engineering_READY.json')
    if not acceptance.get('passed') or acceptance.get('nranks')!=8 or not acceptance.get('NCCL_evidence'):
        raise RuntimeError('Accepted source eight-rank/NCCL evidence required')
    with (source/'metrics.jsonl').open() as stream:rows=[json.loads(x) for x in deque(stream,maxlen=100)]
    peak=max(r.get('gpu_peak_memory_bytes',0) for r in rows)
    if not peak:raise RuntimeError('Measured memory gate missing')
    root.mkdir();(root/'production').mkdir()
    shutil.copytree(repo/'ddz',root/'code/ddz',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(args.tests_proof,root/'TESTS_VERIFICATION.json')
    # Keep the original constructor source identity; the full boundary state is
    # migrated separately only when the old task has completed.
    os.link(old/'production/source.msgpack',root/'production/source.msgpack')
    write(root/'production/config.json',cfg)
    files=[p for p in (root/'code').rglob('*') if p.is_file()]
    files += [root/'TESTS_VERIFICATION.json',root/'production/config.json',root/'production/source.msgpack',
              source/'config.json',old/'engineering_READY.json']
    ready={'cpu_verification':{'passed':True,'proof':str(root/'TESTS_VERIFICATION.json')},
        'files_sha256':{str(p):sha(p) for p in files},
        'boundary_source_policy':'completed full checkpoint at the currently owned segment end',
        'resource_gate':{'source_peak_bytes':peak,'estimated_peak_bytes':int(peak*1.1)+2**30,
                         'remote_eight_device_free_memory_check_required':True}}
    write(root/'READY.json',ready)
    request={'old_root':str(old),'at_iteration':at,'queue':str(queue),
        'active_source_queue_id':active['id'],
        'files_sha256':{**ready['files_sha256'],str(root/'READY.json'):sha(root/'READY.json')},
        'resource_gate':ready['resource_gate'],'user_instruction':'retain global top 50% absolute raw advantage',
        'preserve_global_update_target':cfg['updates'],'time':time.time()}
    write(root/'REQUEST.json',request)
    if read(queue/'status.json')['active_id']!=active['id']:
        raise RuntimeError('Worker changed while preparing; inspect before submission')
    command='cd '+shlex.quote(str(root/'code'))+' && '+shlex.join([
        'env','JAX_PLATFORMS=cpu','CUDA_VISIBLE_DEVICES=','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2',
        sys.executable,'-u','-m','ddz.trim_rollout_campaign','--root',str(root)])
    receipt=subprocess.run([str(repo.parent/'Q/tools/cluster/submit_scheduler_task.sh'),command],
                           capture_output=True,text=True,check=True)
    result={'state':'queued','command':command,'submission':receipt.stdout,
            'at_iteration':at,'adv_keep_fraction':.5,'time':time.time()}
    write(root/'SUBMISSION.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
