"""Freeze an 8M DDZ handoff and enqueue it behind the current owned segment."""
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
from ddz.upgrade_v6 import revised_config
from ddz.scale_rollout_campaign import read,write,sha


def supersede_pending(queue,task_id,new_root):
    """Retain a cancellation receipt for one reviewed, unclaimed V6 item."""
    pending=queue/'pending'/(task_id+'.json');record=read(pending)
    argv=shlex.split(record['command'])
    if (record.get('status')!='pending' or record.get('id')!=task_id or
        record.get('command_sha256')!=__import__('hashlib').sha256(record['command'].encode()).hexdigest() or
        'ddz.scale_architecture_v6' not in argv or '--root' not in argv or
        (queue/'running'/pending.name).exists() or (queue/'interrupted'/pending.name).exists()):
        raise RuntimeError('Superseded V6 pending item identity differs')
    old=Path(argv[argv.index('--root')+1])
    write(queue/'interrupted'/pending.name,{**record,'status':'interrupted','finished_at_ns':time.time_ns(),
        'exit_code':None,'reason':'USER_REQUESTED_V6_ARCHITECTURE_AUDIT_FIXES','replacement_root':str(new_root)})
    pending.unlink()
    write(old/'SUPERSEDED.json',{'queue_id':task_id,'replacement_root':str(new_root),
        'reason':'audited checkpoint restore, CUDA backward/masks, publication and continuation gates',
        'never_started':True,'time':time.time()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--cpu-proof',type=Path,required=True);ap.add_argument('--tests-proof',type=Path,required=True)
    ap.add_argument('--resume-proof',type=Path,required=True);ap.add_argument('--cuda-proof',type=Path,required=True)
    ap.add_argument('--supersede-task')
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=args.root.resolve()
    if root.exists():raise RuntimeError('Frozen campaign already exists; inspect instead of resubmitting')
    cpu=read(args.cpu_proof);tests=read(args.tests_proof)
    resume=read(args.resume_proof);cuda=read(args.cuda_proof)
    if not cpu.get('passed') or cpu['parameters']!=8_014_192 or not tests.get('passed'):
        raise RuntimeError('CPU/tests verification gate failed')
    if (not resume.get('passed') or not resume.get('cold_restore_exact') or resume.get('parameters')!=8_014_192 or
        not resume.get('fp32_source_function') or not cuda.get('passed') or
        not cuda.get('training_signatures',{}).get('passed') or not cuda.get('attention_mask_parity',{}).get('passed')):
        raise RuntimeError('Complete-checkpoint resume and local CUDA engineering proofs required')
    queue=repo.parent/'Q/tools/cluster/cluster_scheduler.queue'
    status=read(queue/'status.json')
    if status['phase']!='RUNNING' or status['running_count']!=1 or time.time()-status['updated_at_ns']/1e9>180:
        raise RuntimeError('Healthy persistent queue worker required')
    active=read(queue/'running'/(status['active_id']+'.json'))
    argv=shlex.split(active['command'])
    source_run=Path(read(repo/'runs/production_current.json')['run'])
    old=source_run.parent.parent
    if ('ddz.cluster_distributed_job' not in argv or '--root' not in argv or
        Path(argv[argv.index('--root')+1]).resolve()!=old.resolve()):
        raise RuntimeError('Current queue task is not the owned DDZ production segment')
    at=int(argv[argv.index('--steps')+1])
    cfg=read(source_run/'config.json');target=revised_config(cfg)
    if target!=cpu['config']:raise RuntimeError('CPU-tested architecture/protocol differs')
    if target!=read(args.resume_proof.parent/'config.json') or cuda['source_iteration']!=resume['source_iteration']:
        raise RuntimeError('Resume/CUDA engineering protocol differs')
    expected=[(target.get('memory_limit',88),target['per_gpu_minibatch']),
              (192,min(1024,target['per_gpu_minibatch']))]
    actual=[(r['memory_length'],r['per_gpu_minibatch']) for r in cuda['training_signatures']['signatures']]
    if actual!=expected:raise RuntimeError('CUDA backward signature sizes differ')
    pending=[read(p) for p in (queue/'pending').glob('*.json')
             if 'ddz.scale_architecture_v6' in read(p)['command']]
    if pending and (len(pending)!=1 or pending[0]['id']!=args.supersede_task):
        raise RuntimeError('A V6 handoff is already pending; specify its exact reviewed supersession id')
    if args.supersede_task and not pending:raise RuntimeError('Reviewed superseded task is no longer pending')
    with (source_run/'metrics.jsonl').open() as stream:
        rows=[json.loads(line) for line in deque(stream,maxlen=600)]
    peak=max(r.get('gpu_peak_memory_bytes',0) for r in rows)
    if not peak:raise RuntimeError('Measured source memory gate missing')
    previous=read(old/'engineering_READY.json')
    if not previous.get('passed') or previous.get('nranks')!=8 or not previous.get('NCCL_evidence'):
        raise RuntimeError('Source lacks eight-rank NCCL engineering acceptance')
    root.mkdir();(root/'production').mkdir()
    shutil.copytree(repo/'ddz',root/'code/ddz',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(args.cpu_proof,root/'CPU_VERIFICATION.json')
    shutil.copyfile(args.tests_proof,root/'TESTS_VERIFICATION.json')
    shutil.copyfile(args.resume_proof,root/'RESUME_VERIFICATION.json')
    shutil.copyfile(args.cuda_proof,root/'CUDA_VERIFICATION.json')
    write(root/'production/config.json',target)
    files=[p for p in (root/'code').rglob('*') if p.is_file()]
    files += [root/'CPU_VERIFICATION.json',root/'TESTS_VERIFICATION.json',root/'RESUME_VERIFICATION.json',
              root/'CUDA_VERIFICATION.json',root/'production/config.json',source_run/'config.json']
    ready={'cpu_verification':{'passed':True,'proof':str(root/'CPU_VERIFICATION.json')},
        'files_sha256':{str(p):sha(p) for p in files},'boundary_source_policy':'pin completed full checkpoint at owned segment end',
        'resource_gate':{'source_peak_bytes':peak,'estimated_peak_bytes':2*peak+3*2**30,
            'remote_capacity_free_memory_check_required':True,
            'hardware_evidence':'existing accepted eight A100-80GB ranks and NCCL source engineering log'}}
    write(root/'READY.json',ready)
    request={'old_root':str(old),'at_iteration':at,'queue':str(queue),
        'files_sha256':{**ready['files_sha256'],str(root/'READY.json'):sha(root/'READY.json')},
        'resource_gate':ready['resource_gate'],'active_source_queue_id':active['id'],
        'user_instruction':'Scale useful attention depth and width to approximately 8M',
        'preserve_global_update_target':200000,'time':time.time()}
    write(root/'REQUEST.json',request)
    if args.supersede_task:
        current=read(queue/'status.json');training=read(source_run/'status.json')
        if (current['phase']!='RUNNING' or current['active_id']!=active['id'] or
            training['state']!='running' or training['iteration']>=at-2):
            raise RuntimeError('Source boundary is too close or worker changed; inspect claims before supersession')
        supersede_pending(queue,args.supersede_task,root)
    command='cd '+shlex.quote(str(root/'code'))+' && '+shlex.join([
        'env','JAX_PLATFORMS=cpu','CUDA_VISIBLE_DEVICES=','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2',
        sys.executable,'-u','-m','ddz.scale_architecture_v6','--root',str(root)])
    helper=repo.parent/'Q/tools/cluster/submit_scheduler_task.sh'
    result=subprocess.run([str(helper),command],text=True,capture_output=True,check=True)
    receipt={'state':'queued','command':command,'submission':result.stdout,
        'parameters':8_014_192,'at_iteration':at,'time':time.time()}
    write(root/'SUBMISSION.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
