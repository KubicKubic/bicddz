"""Persistent local DDZ contrasts; shared frozen source, held-out confirmations.

No Q tasks are submitted and no unrelated process is controlled. Failed tasks
are recorded and halt the campaign. Resume is explicit and verified by hashes.
"""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from .elo_ladder import atomic_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,screen_steps=256,confirm_steps=1024):
    from flax import serialization
    root=root.resolve();source=root/'source.msgpack'
    s=serialization.msgpack_restore(source.read_bytes());base=s['config']
    variants={name:copy.deepcopy(base) for name in
        ('baseline','epochs2','own_gae','belief','pool','forced_mask','team_value','explore001','combined')}
    variants['epochs2']['ppo']['epochs']=2
    variants['own_gae']['ppo']['gae_clock']='own'
    variants['belief']['model']['belief_head']=True;variants['belief']['ppo']['belief_coef']=.03
    paths=[str(p.resolve()) for p in sorted((root/'refs').glob('policy_*.msgpack'))]
    hashes={p:digest(p) for p in paths}
    variants['pool'].update(pool_probability=.25,pool_bucket=64,opponent_pool=paths,opponent_pool_sha256=hashes)
    variants['forced_mask']['ppo']['optional_actor_only']=True
    variants['team_value']['model']['team_value']=True
    variants['explore001']['ppo']['random_action_prob']=.01
    c=variants['combined'];c['model'].update(belief_head=True,team_value=True)
    c['ppo'].update(epochs=2,gae_clock='own',belief_coef=.03,optional_actor_only=True,random_action_prob=.01)
    c.update(pool_probability=.25,pool_bucket=64,opponent_pool=paths,opponent_pool_sha256=hashes)
    folder=root/'configs';folder.mkdir(exist_ok=True)
    for name,cfg in variants.items():
        cfg['compilation_cache']=str(root/'jax_cache')
        atomic_json(folder/(name+'.json'),cfg)
    code=root/'code';code.mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parent,code/'ddz',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    sources={str(p.relative_to(code)):digest(p) for p in sorted(code.rglob('*.py'))}
    proof=json.loads((root/'engineering_verification.json').read_text())
    if not proof.get('passed'):raise RuntimeError('engineering READY gate failed')
    manifest={'version':1,'created':time.time(),'source_sha256':digest(source),
        'source_iteration':int(s['iteration']),'source_optimizer_step':int(s['train']['step']),
        'config_sha256':{name:digest(folder/(name+'.json')) for name in variants},
        'frozen_code_sha256':sources,'opponent_sha256':hashes,
        'variants':list(variants),'screen_steps':screen_steps,'confirm_steps':confirm_steps,
        'screen_deals':1024,'confirmation_deals':4096,'douzero_deals':1024,
        'screen_deal_seed':700001,'confirmation_deal_seed':810001,'douzero_deal_seed':911001,
        'training_rng_streams':[0,1],'fresh_decisions_per_step':base['envs']*base['horizon'],
        'selection':'largest screen forced equal-role score versus matched baseline; screen is exploratory',
        'confirmation':'independent deals; both training streams; natural and forced role scopes',
        'promotion':'positive forced score CI in BOTH training streams, no significant natural score regression, '
                    'positive paired strong-DouZero score improvement CI; no illegal/nonfinite samples',
        'limitations':['Two RNG continuations do not establish robustness across arbitrary training seeds',
            'Fixed-budget failures to establish improvement are inconclusive, not proofs of no long-run benefit',
            'No chosen hyperparameter grid; one auxiliary coefficient and one opponent-pool probability'],
        'engineering_proof':proof}
    atomic_json(root/'manifest.json',manifest)
    return manifest


def verify(root,m):
    if digest(root/'source.msgpack')!=m['source_sha256']:raise RuntimeError('source checkpoint changed')
    for name,sha in m['config_sha256'].items():
        if digest(root/'configs'/(name+'.json'))!=sha:raise RuntimeError('config changed '+name)
    for path,sha in m['frozen_code_sha256'].items():
        if digest(root/'code'/path)!=sha:raise RuntimeError('frozen code changed '+path)
    for path,sha in m['opponent_sha256'].items():
        if digest(path)!=sha:raise RuntimeError('frozen opponent changed '+path)


def paired_difference(a,b,seed=19):
    """Bootstrap whole independent deals, paired between candidate and baseline."""
    x=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(20):
        ix=rng.integers(len(x),size=(500,len(x)));draws.extend(x[ix].mean(-1).tolist())
    return {'mean':float(x.mean()),'ci95':np.percentile(draws,[2.5,97.5]).tolist(),'deals':len(x)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--screen-steps',type=int,default=256)
    ap.add_argument('--confirm-steps',type=int,default=1024);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=args.root.resolve();root.mkdir(exist_ok=True,parents=True)
    if args.prepare:
        if (root/'manifest.json').exists():raise RuntimeError('manifest already frozen')
        print(json.dumps(prepare(root,args.screen_steps,args.confirm_steps)));return
    lock=(root/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((root/'manifest.json').read_text());verify(root,m)
    if (root/'campaign_status.json').exists() and not args.resume:raise RuntimeError('explicit --resume required')
    env={**os.environ,'PYTHONPATH':str(root/'code'),'JAX_PLATFORMS':'cuda',
        'CUDA_VISIBLE_DEVICES':'0','XLA_PYTHON_CLIENT_MEM_FRACTION':'.35',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'8'}
    started=time.time();logs=root/'logs';logs.mkdir(exist_ok=True)
    def command(name,module,argv,extra=None):
        status={'state':'running','task':name,'pid':os.getpid(),'started':started,'time':time.time()}
        atomic_json(root/'campaign_status.json',status)
        with (logs/(name+'.log')).open('a') as f:
            p=subprocess.Popen([sys.executable,'-u','-m',module,*map(str,argv)],
                cwd=root/'code',env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
            atomic_json(root/'campaign_status.json',{**status,'child_pid':p.pid})
            code=p.wait()
        if code:
            atomic_json(root/'campaign_status.json',{**status,'state':'failed','exit_code':code,'time':time.time()})
            raise RuntimeError(f'{name} failed; inspect its retained log; explicit resume required')
    def train(name,seed,steps):
        run=root/'training'/f'{name}_rng{seed}'
        if (run/'status.json').exists() and json.loads((run/'status.json').read_text())['iteration']>=steps:return run
        argv=['--config',root/'configs'/(name+'.json'),'--source',root/'source.msgpack',
            '--out',run,'--steps',steps,'--training-seed',seed,'--require-a100']
        if (run/'latest.msgpack').exists():argv+=['--resume']
        command(f'train_{name}_rng{seed}_{steps}','ddz.train_efficiency',argv)
        return run
    def evaluate(name,a,b,steps,deals,seed):
        out=root/'evaluations'/name
        if not (out/'summary.json').exists():
            command('eval_'+name,'ddz.evaluate_efficiency',['--run-a',a,'--step-a',steps,
                '--run-b',b,'--step-b',steps,'--out',out,'--deals',deals,'--chunk',64,'--seed',seed],
                {'XLA_PYTHON_CLIENT_MEM_FRACTION':'.12'})
        return json.loads((out/'summary.json').read_text())
    def douzero(name,run,step):
        out=root/'douzero'/name
        if not (out/'BEST'/'summary.json').exists():
            command('douzero_'+name,'ddz.compare_efficiency_douzero',['--run-dir',run,'--step',step,
                '--douzero-src','/tmp/ddz_douzero_official','--resnet-src','/tmp/ddz_douzero_resnet_2_0',
                '--weights-root',root.parents[0]/'douzero_resnet_2_0_reference','--out',out,
                '--deals',m['douzero_deals'],'--chunk-deals',16,'--seed',m['douzero_deal_seed']],
                {'PYTHONPATH':str(root/'code')+':/tmp/ddz_compare_env/lib/python3.12/site-packages',
                 'XLA_PYTHON_CLIENT_MEM_FRACTION':'.12','OMP_NUM_THREADS':'2'})
        return out
    try:
        # Explicit GPU ownership: never start while the original DDZ trainer is live.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
            except (FileNotFoundError,PermissionError):continue
            if b'-m ddz.train_v5 ' in cmd:raise RuntimeError('original DDZ trainer must first save and stop')
        baseline=train('baseline',0,m['screen_steps']);screen={}
        for name in m['variants'][1:]:
            run=train(name,0,m['screen_steps'])
            screen[name]=evaluate('screen_'+name,run,baseline,m['screen_steps'],m['screen_deals'],m['screen_deal_seed'])
            atomic_json(root/'screen_results.json',screen)
        winner=max(screen,key=lambda x:screen[x]['results']['forced']['mean'])
        atomic_json(root/'selection.json',{'winner':winner,'selection_only':True,
            'screen_mean':screen[winner]['results']['forced']['mean'],'time':time.time()})
        confirmation={}
        for seed in m['training_rng_streams']:
            b=train('baseline',seed,m['confirm_steps']);a=train(winner,seed,m['confirm_steps'])
            confirmation[str(seed)]=evaluate('confirmation_rng'+str(seed),a,b,m['confirm_steps'],
                m['confirmation_deals'],m['confirmation_deal_seed'])
            atomic_json(root/'confirmation_results.json',confirmation)
        baseline_dz=douzero('baseline',root/'training'/'baseline_rng0',m['confirm_steps'])
        candidate_dz=douzero(winner,root/'training'/f'{winner}_rng0',m['confirm_steps'])
        # The evaluator owns per-deal score provenance; preserve it for paired analysis.
        def scores(folder):
            chunks=[json.loads(p.read_text()) for p in sorted((folder/'BEST').glob('chunk_*.json'))]
            x=np.concatenate([np.asarray(c['focus_raw_scores']) for c in chunks],axis=2)
            return (x/np.array([2.,1.,1.])[:,None,None]).mean(axis=(0,1))
        dz_change=paired_difference(scores(candidate_dz),scores(baseline_dz))
        qualified=all(c['results']['forced']['ci95'][0]>0 and c['results']['natural']['ci95'][1]>=0
            for c in confirmation.values()) and dz_change['ci95'][0]>0
        result={'state':'complete','winner':winner,'confirmed_benefit':qualified,
            'confirmation':confirmation,'strong_douzero_paired_change':dz_change,
            'decision':'eligible for separately reviewed promotion' if qualified else
                       'keep production unchanged; tested improvements not established',
            'seconds':time.time()-started,'time':time.time()}
        atomic_json(root/'result.json',result)
        atomic_json(root/'campaign_status.json',{'state':'complete','winner':winner,
            'confirmed_benefit':qualified,'time':time.time()})
    except BaseException as e:
        status=json.loads((root/'campaign_status.json').read_text()) if (root/'campaign_status.json').exists() else {}
        atomic_json(root/'campaign_status.json',{**status,'state':'failed','error':str(e),'time':time.time()})
        raise


if __name__=='__main__':main()
