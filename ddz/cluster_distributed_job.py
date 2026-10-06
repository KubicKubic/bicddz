"""Immutable queue task: verify inputs, run eight GPUs, retain NCCL proof.

Successful production chunks append their next bounded chunk through the queue
helper. Failed chunks never retry. No bridge commands or unrelated tasks change.
"""
import argparse,hashlib,json,math,os,re,shlex,subprocess,sys,time
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--phase',choices=['engineering','production'],required=True)
    ap.add_argument('--steps',type=int,default=32);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=args.root.resolve();phase=root/args.phase
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Queue worker required')
    ready=read(phase/'READY.json')
    if not ready['cpu_verification']['passed']:raise RuntimeError('CPU verification gate failed')
    for path,expected in ready['files_sha256'].items():
        if sha(Path(path))!=expected:raise RuntimeError('Frozen dependency changed '+path)
    if args.phase=='production' and not read(root/'engineering_READY.json')['passed']:
        raise RuntimeError('Eight-GPU engineering verification required')
    task_id=os.environ['Q_SCHEDULER_QUEUE_TASK_ID'];log=phase/(task_id+'.log')
    cfg=read(phase/'config.json')
    start_iteration=read(phase/'training'/'latest.json')['iteration'] if args.resume else 0
    if args.steps<=start_iteration:raise RuntimeError('Continuation must contain fresh rollout rounds')
    argv=[sys.executable,'-u','-m','ddz.train_distributed_v5','--config',str(phase/'config.json'),
          '--source',str(phase/'source.msgpack'),'--out',str(phase/'training'),'--steps',str(args.steps)]
    if args.resume:argv.append('--resume')
    env={**os.environ,'JAX_PLATFORMS':'cuda','NCCL_DEBUG':'INFO',
         'JAX_COMPILATION_CACHE_DIR':str(phase/'jax_cache'),
         'XLA_PYTHON_CLIENT_MEM_FRACTION':'.80','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'8',
         'PYTHONPATH':str(root/'code')}
    write(phase/'queue_job_status.json',{'state':'running','queue_id':task_id,'pid':os.getpid(),
          'time':time.time(),'steps':args.steps})
    with log.open('w') as stream:
        child=subprocess.Popen(argv,cwd=root/'code',env=env,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
        code=child.wait()
    if code:
        write(phase/'queue_job_status.json',{'state':'failed','queue_id':task_id,'exit_code':code,'time':time.time()})
        raise RuntimeError('Eight-GPU child failed; explicit new queue item required')
    content=log.read_text();nccl=[x for x in content.splitlines() if re.search(r'NCCL.*nranks[ =]+8\b',x)]
    if not nccl or 'nranks=8;' not in content:
        raise RuntimeError('Missing real eight-rank NCCL log evidence')
    rows=[json.loads(x) for x in (phase/'training'/'metrics.jsonl').read_text().splitlines()]
    current=[r for r in rows if r['iteration']>start_iteration]
    expected_decisions=cfg['envs']*cfg['horizon']
    if ([r['iteration'] for r in current]!=list(range(start_iteration+1,args.steps+1)) or
        any(r['invalid_actions'] or r['nonfinite'] or
        r['nranks']!=8 or r['fresh_decisions']!=expected_decisions or
        r.get('replica_parameter_max_difference',0)!=0 or r.get('ema_replica_parameter_max_difference',0)!=0 or
        not all(isinstance(v,(int,float)) and math.isfinite(v) for v in r.values()) for r in current)):
        raise RuntimeError('Eight-GPU scale/health/synchronization gate failed')
    warm=[r for r in current if r['iteration']>current[0]['iteration']+3]
    if not warm:warm=current
    proof={'passed':True,'queue_id':task_id,'nranks':8,'NCCL_evidence':nccl[:8],
           'parameters':read(phase/'training'/'status.json')['parameters'],
           'global_envs':cfg['envs'],'global_minibatch':cfg['ppo']['minibatch'],
           'fresh_decisions_per_update':expected_decisions,'tested_updates':len(current),
           'fresh_decisions':sum(r['fresh_decisions'] for r in current),
           'total_seconds':sum(r['seconds'] for r in current),
           'warm_fresh_decisions_per_second':sum(r['fresh_decisions'] for r in warm)/sum(r['seconds'] for r in warm),
           'invalid_actions':0,'nonfinite':0,'replica_parameter_max_difference':0,
           'log':str(log),'time':time.time()}
    write(phase/(task_id+'.verification.json'),proof)
    write(phase/'queue_job_status.json',{'state':'complete','queue_id':task_id,'time':time.time(),**proof})
    if args.phase=='engineering':write(root/'engineering_READY.json',proof)
    else:
        write(root.parent/'production_current.json',{'state':'running' if args.steps<cfg['continuation_updates'] else 'completed',
            'run':str(phase/'training'),'nranks':8,'global_iteration':cfg['global_source_iteration']+args.steps,
            'proof':str(phase/(task_id+'.verification.json')),'time':time.time()})
        if args.steps<cfg['continuation_updates']:
            target=min(args.steps+1000,cfg['continuation_updates'])
            command=shlex.join([sys.executable,'-u','-m','ddz.cluster_distributed_job','--root',str(root),
                               '--phase','production','--steps',str(target),'--resume'])
            command='cd '+shlex.quote(str(root/'code'))+' && '+command
            helper=root.parent.parent.parent/'Q'/'tools'/'cluster'/'submit_scheduler_task.sh'
            receipt=subprocess.run([str(helper),command],check=True,text=True,capture_output=True)
            write(phase/(task_id+'.followup.json'),{'target':target,'command':command,'submission':receipt.stdout,'time':time.time()})
            print(receipt.stdout,flush=True)


if __name__=='__main__':main()
