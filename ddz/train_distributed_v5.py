"""Eight-device synchronous PPO: independent rollouts, one shared policy/Adam.

Global envs and minibatches are eight times the single-device sizes. Masked
losses, advantage normalization, KL rejection and adaptive value gradients use
all replicas. Complete variable histories and complete actions are retained.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.80')
os.environ.setdefault('NCCL_DEBUG','INFO')
import argparse,copy,hashlib,json,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from . import env_v4 as env
from .train_efficiency import create
from .optimizer_efficiency import sampled_lr
from .ppo_efficiency import ArenaState,make_rollout,make_update,make_diagnostic
from .train_v2 import atomic,checkpoint
from .train_v3 import balanced_vf_coefficient
from .ema_weights import update_weights,validate_decay


def merge_env(tree):
    return jax.tree_util.tree_map(lambda x:np.asarray(x).reshape((-1,)+x.shape[2:]),tree)


def shard_env(tree,ranks):
    return jax.tree_util.tree_map(lambda x:j.asarray(x).reshape((ranks,-1)+x.shape[1:]),tree)


def replica_difference(params):
    arrays=jax.device_get(params)
    return max(float(np.max(np.abs(x-x[0]))) for x in jax.tree_util.tree_leaves(arrays))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if os.environ.get('Q_CLUSTER_TASK')!='1':raise RuntimeError('Eight-GPU runs require the persistent queue worker')
    if jax.default_backend()!='gpu' or jax.local_device_count()!=8 or jax.process_count()!=1:
        raise RuntimeError('Requires exactly eight GPUs in the scheduler allocation')
    devices=jax.local_devices();ranks=8
    probe=jax.pmap(lambda x:jax.lax.psum(x,'replicas'),axis_name='replicas')(j.arange(8,dtype=j.float32))
    np.testing.assert_array_equal(np.asarray(probe),np.full(8,28.))
    print('nranks=8; GPU collective verified; NCCL initialization must also be present in log',flush=True)
    cfg=json.loads(args.config.read_text());raw=args.source.read_bytes()
    source=serialization.msgpack_restore(raw);source_sha=hashlib.sha256(raw).hexdigest()
    if cfg['envs']!=8*cfg['per_gpu_envs'] or cfg['ppo']['minibatch']!=8*cfg['per_gpu_minibatch']:
        raise ValueError('Global rollout and batch sizes must be eight times each replica')
    if cfg['ppo']['epochs']!=1 or cfg['horizon']<=0 or cfg['per_gpu_envs']<=0:
        raise ValueError('Positive rollout sizes and one PPO epoch required')
    if cfg['horizon']!=source['config']['horizon'] and not args.resume:
        raise ValueError('A changed horizon requires an explicitly migrated checkpoint')
    if cfg.get('pool_probability',0):raise ValueError('Unconfirmed opponent pool is disabled')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.msgpack').exists() and not args.resume:raise RuntimeError('Explicit resume required')
    saved=serialization.msgpack_restore((out/'latest.msgpack').read_bytes()) if args.resume else None
    if saved and (saved['config']!=cfg or saved['runtime']['source_sha256']!=source_sha):
        raise RuntimeError('Distributed resume identity mismatch')
    jax.config.update('jax_compilation_cache_dir',str(out.parent/'jax_cache'))
    model,ts=create(cfg,source)
    parameters=sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    if parameters>5_500_000:raise ValueError('Parameter limit exceeded')
    b=cfg['per_gpu_envs'];p=cfg['ppo'];memory=cfg.get('memory_limit',88)
    base_key=j.asarray(source['key'])
    runtime=copy.deepcopy(source['runtime']);runtime.update(source_sha256=source_sha,
        source_checkpoint=str(args.source.resolve()),source_iteration=int(source['iteration']),
        source_adam_step=int(source['train']['step']),stable_updates=0,
        distributed={'nranks':8,'per_gpu_envs':b,'per_gpu_minibatch':cfg['per_gpu_minibatch'],
                     'rank_zero_environment_preserved':True,'other_ranks':'fresh distinct seed-41 streams'})
    start=0
    if saved:
        ts=serialization.from_state_dict(ts,saved['train']);runtime=saved['runtime'];start=int(saved['iteration'])
        flat=serialization.from_state_dict(env.batch_reset(jax.random.split(base_key,cfg['envs'])),saved['env'])
        games=shard_env(flat,8);keys=j.asarray(saved['key']);a=runtime['arena']
        arena=ArenaState(games,*[j.asarray(a[k]).reshape((8,b)) for k in ('pool_id','focus','complement','use_pool')])
        if games.turn.shape!=(8,b) or keys.shape!=(8,2):
            raise ValueError('Migrated environment or random-stream shape differs')
    else:
        initial=env.batch_reset(jax.random.split(jax.random.PRNGKey(cfg['seed']),b))
        original=serialization.from_state_dict(initial,source['env'])
        if original.turn.shape!=(b,):raise ValueError('The source must be a single-device full checkpoint')
        keys=j.stack([base_key]+[jax.random.fold_in(base_key,i) for i in range(1,8)])
        states=[original]+[env.batch_reset(jax.random.split(jax.random.fold_in(base_key,100+i),b)) for i in range(1,8)]
        games=jax.tree_util.tree_map(lambda *x:j.stack(x),*states)
        arena=ArenaState(games,j.zeros((8,b),j.int32),j.zeros((8,b),j.int32),j.zeros((8,b),j.bool_),j.zeros((8,b),j.bool_))
    ts=jax.device_put_replicated(ts,devices)
    ema_decay=validate_decay(cfg.get('ema_decay',0.999))
    saved_ema=saved.get('ema') if saved else None
    if saved_ema is not None and saved_ema['decay']!=ema_decay:
        raise ValueError('EMA decay differs from the saved checkpoint')
    if saved_ema is None:
        ema_params=jax.tree_util.tree_map(lambda x:x,ts.params);ema_updates=0
    else:
        ema_params=jax.device_put_replicated(saved_ema['params'],devices)
        ema_updates=int(saved_ema['updates'])
    update_ema=jax.pmap(lambda old,new:update_weights(old,new,ema_decay))
    arena=jax.device_put_sharded([jax.tree_util.tree_map(lambda x:x[i],arena) for i in range(8)],devices)
    keys=jax.device_put_sharded(list(keys),devices)
    pool=jax.tree_util.tree_map(lambda x:x[0][None],ts.params)
    rollout=jax.pmap(make_rollout(model,cfg['horizon'],memory,0.,p.get('random_action_prob',0.)),
                     axis_name='replicas',in_axes=(0,0,0,None))
    local_p={**p,'minibatch':cfg['per_gpu_minibatch']}
    long_p={**local_p,'minibatch':min(1024,local_p['minibatch'])}
    def updater(options,length):
        return jax.pmap(make_update(model,options,length,'replicas'),axis_name='replicas',
                        in_axes=(0,0,0,0,None,None,None))
    update_short=updater(local_p,memory);update_long=updater(long_p,env.HISTORY)
    def diagnostic(options,length):
        return jax.pmap(make_diagnostic(model,options,length,'replicas'),axis_name='replicas')
    diag_short=diagnostic(local_p,memory);diag_long=diagnostic(long_p,env.HISTORY)
    split_keys=jax.pmap(lambda k:jax.random.split(k,2))
    lr=sampled_lr(cfg,int(source['train']['step']));stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    atomic(out/'config.json',json.dumps(cfg,indent=2).encode());atomic(out/'pid',str(os.getpid()).encode())
    print(json.dumps({'nranks':8,'devices':[d.device_kind for d in devices],'parameters':parameters,
          'global_envs':cfg['envs'],'fresh_decisions_per_update':cfg['envs']*cfg['horizon'],
          'global_minibatch':p['minibatch'],'source_sha256':source_sha,'config':cfg}),flush=True)
    def save(it):
        runtime['arena']={k:np.asarray(getattr(arena,k)).reshape(-1) for k in ('pool_id','focus','complement','use_pool')}
        one=jax.tree_util.tree_map(lambda x:x[0],ts)
        ema_one=jax.tree_util.tree_map(lambda x:x[0],ema_params)
        checkpoint(out,one,merge_env(jax.device_get(arena.games)),np.asarray(keys),it,cfg,runtime,
                   ema={'params':ema_one,'decay':ema_decay,'updates':ema_updates})
        # Existing local evaluation owners keep following their original path.
        # Publish only new policy files there; historical checkpoints stay intact.
        publication=out.parent.parent/'policy_publication.json'
        if publication.exists():
            publication_cfg=json.loads(publication.read_text())
            if Path(publication_cfg['run_dir']).resolve()!=out.resolve():
                raise RuntimeError('Policy publication source differs from active training')
            destination=Path(publication_cfg['destination'])/f'policy_{it:07d}.msgpack'
            target=(out/f'policy_{it:07d}.msgpack').resolve()
            if destination.exists() or destination.is_symlink():
                if destination.resolve()!=target:raise RuntimeError('Policy publication would overwrite a snapshot')
            else:destination.symlink_to(target)
    if start==0:save(0)
    with (out/'metrics.jsonl').open('a',buffering=1) as stream:
        for it in range(start+1,args.steps+1):
            begin=time.monotonic();arena,keys,tr,last,stats=rollout(ts.params,arena,keys,pool)
            jax.block_until_ready(stats);roll_s=time.monotonic()-begin
            short=int(np.asarray(stats)[:,:,3].max())<=memory
            update=update_short if short else update_long;local_mb=local_p['minibatch'] if short else long_p['minibatch']
            split=split_keys(keys);keys,uk=split[:,0],split[:,1]
            vf=runtime['vf_coef'];used_lr=float(lr(it-1))
            entropy=p['entropy_start']+(p['entropy_end']-p['entropy_start'])*min(1,
                (cfg['global_source_iteration']+it)/p['entropy_updates'])
            ts,m=update(ts,tr,last,uk,j.float32(entropy),j.float32(vf),j.float32(used_lr))
            jax.block_until_ready(m);train_s=time.monotonic()-begin-roll_s
            row={k:float(np.asarray(v)[0]) for k,v in m.items()}
            # Rollout clock: exactly one EMA tick per completed rollout, even
            # when the optimizer rejects some or all of its minibatches.
            ema_params=update_ema(ema_params,ts.params);jax.block_until_ready(ema_params)
            ema_updates+=1
            stable=row['nonfinite']==0 and row['kl']<2*p['target_kl'] and row['applied']>.1
            runtime['stable_updates']=runtime['stable_updates']+1 if stable else 0
            if it%p.get('vf_balance_every',10)==0:
                split=split_keys(keys);keys,dk=split[:,0],split[:,1]
                diagnosis=(diag_short if short else diag_long)(ts.params,tr,last,dk)
                row.update({k:float(np.asarray(v)[0]) for k,v in diagnosis.items()})
                if it>=p.get('vf_balance_after',20) and runtime['stable_updates']>=5:
                    runtime['vf_coef']=balanced_vf_coefficient(vf,row['policy_grad_l2'],row['value_grad_l2'],p)
            if it==start+1 or it%cfg['save_every']==0 or it==args.steps:
                row['replica_parameter_max_difference']=replica_difference(ts.params)
                if row['replica_parameter_max_difference']!=0:raise RuntimeError('Policy replicas diverged')
                row['ema_replica_parameter_max_difference']=replica_difference(ema_params)
                if row['ema_replica_parameter_max_difference']!=0:raise RuntimeError('EMA replicas diverged')
                memory_stats=[d.memory_stats() or {} for d in devices]
                row['gpu_peak_memory_bytes']=max(s.get('peak_bytes_in_use',0) for s in memory_stats)
            elapsed=time.monotonic()-begin;n=cfg['envs']*cfg['horizon'];host_stats=np.asarray(stats)
            row.update(iteration=it,global_iteration=cfg['global_source_iteration']+it,time=time.time(),
                seconds=elapsed,rollout_seconds=roll_s,update_seconds=train_s,nranks=8,
                fresh_decisions=n,learner_decisions=int(host_stats[:,:,2].sum()),
                processed_decisions=8*local_mb*row['applied_minibatches'],decisions_per_second=n/elapsed,
                games=int(host_stats[:,:,0].sum()),invalid_actions=int(host_stats[:,:,1].sum()),
                vf_coef=vf,learning_rate=used_lr,entropy_coef=entropy,history_max=int(host_stats[:,:,3].max()),
                optimizer_step=int(np.asarray(ts.step)[0]),train_minibatch=8*local_mb,
                train_memory_length=memory if short else env.HISTORY)
            row.update(ema_decay=ema_decay,ema_updates=ema_updates)
            if row['invalid_actions'] or row['nonfinite'] or not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError('Distributed training invariant failed')
            stream.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if it%cfg['save_every']==0 or stop[0] or it==args.steps:save(it)
            status={'state':'stopped' if stop[0] else 'completed' if it==args.steps else 'running',
                    'iteration':it,'global_iteration':row['global_iteration'],'pid':os.getpid(),
                    'time':time.time(),'parameters':parameters,'nranks':8}
            atomic(out/'status.json',json.dumps(status).encode())
            if stop[0]:break


if __name__=='__main__':main()
