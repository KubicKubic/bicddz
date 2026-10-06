"""Follow completed raw checkpoints and replace CPU policy weights in place.

Loading and warmup run in one background thread. The game loop commits a
prepared checkpoint between requests/decisions; it never waits for disk or JIT.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from flax import serialization


def atomic_json(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def model_signature(config):
    return {'model':config['model'],
            'gae_clock':config.get('ppo',{}).get('gae_clock','public')}


def compatible_signature(frozen,config):
    if model_signature(config)==model_signature(frozen):return True
    if (config.get('model_family')!='V6' or
        model_signature(config)['gae_clock']!=model_signature(frozen)['gae_clock']):return False
    from .upgrade_v6 import validate_growth
    try:validate_growth(frozen['model'],config['model'])
    except (ValueError,KeyError,TypeError):return False
    return True


def discover(pointer):
    """latest.json is published AFTER full, raw and EMA checkpoint files."""
    run=Path(json.loads(Path(pointer).read_text())['run']).resolve()
    latest=json.loads((run/'latest.json').read_text())
    iteration=latest['iteration']
    if type(iteration) is not int or iteration<0:
        raise ValueError('Invalid checkpoint iteration')
    name=f'policy_{iteration:07d}.msgpack'
    if latest.get('policy')!=name:
        raise ValueError('Checkpoint publication marker has an unexpected policy path')
    config=json.loads((run/'config.json').read_text())
    origin=config.get('global_source_iteration',0)
    if type(origin) is not int or origin<0:
        raise ValueError('Invalid global checkpoint origin')
    policy=run/name
    if not policy.is_file():
        raise FileNotFoundError('Published checkpoint policy is not yet available')
    metadata={'source_run':str(run),'source_checkpoint':str(policy),
        'config_path':str(run/'config.json'),'relative_step':iteration,
        'global_step':origin+iteration,'policy_kind':'raw',
        'model_signature':model_signature(config)}
    return metadata,config


def read_weights(metadata):
    raw=Path(metadata['source_checkpoint']).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    expected=metadata.get('checkpoint_sha256')
    if expected is not None and digest!=expected:
        raise ValueError('Checkpoint SHA-256 mismatch')
    return serialization.msgpack_restore(raw),{**metadata,'checkpoint_sha256':digest}


def resume_paths(root,deployment,event):
    """Recover the last accepted checkpoint after a watchdog/process restart."""
    base=Path(deployment['model_dir'])
    original=(base/'config.json',base/'policy.msgpack')
    path=Path(root)/'active_model.json'
    if not path.exists():return (*original,None)
    try:
        metadata=json.loads(path.read_text())
        config=json.loads(Path(metadata['config_path']).read_text())
        frozen=json.loads(original[0].read_text())
        if (type(metadata['global_step']) is not int or
            metadata['global_step']<deployment['global_step'] or
            metadata.get('policy_kind')!='raw' or
            metadata['model_signature']!=model_signature(config) or
            not compatible_signature(frozen,config)):
            raise ValueError('Recovered model is incompatible or older than frozen release')
        read_weights(metadata)
        return Path(metadata['config_path']),Path(metadata['source_checkpoint']),metadata
    except (OSError,ValueError,KeyError,TypeError) as error:
        event({'event':'model_resume_rejected','error':f'{type(error).__name__}: {error}'})
        return (*original,None)


class CheckpointWatcher:
    def __init__(self,model,root,pointer,active,on_activate,event,poll_seconds=5):
        self.model=model;self.root=Path(root);self.pointer=Path(pointer)
        self.active=dict(active);self.on_activate=on_activate;self.event=event
        self.poll_seconds=max(.05,float(poll_seconds))
        self.ready=queue.Queue(maxsize=1);self.stop=threading.Event()
        self.prepared_step=self.active['global_step']
        self.failed_key=None;self.retry_after=0
        self.thread=threading.Thread(target=self._loop,name='checkpoint-loader',daemon=True)

    def __getattr__(self,name):
        return getattr(self.model,name)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set();self.thread.join(timeout=2)

    def poll_once(self):
        metadata,config=discover(self.pointer)
        step=metadata['global_step']
        if step<=max(self.active['global_step'],self.prepared_step):return False
        policy=Path(metadata['source_checkpoint'])
        stat=policy.stat()
        key=(str(policy),step,stat.st_size,stat.st_mtime_ns)
        if key==self.failed_key and time.monotonic()<self.retry_after:return False
        started=time.monotonic()
        try:
            params,metadata=read_weights(metadata)
            replacement=None
            current_config=getattr(self.model,'config',None)
            if current_config is not None and config['model']!=current_config['model']:
                replacement=self.model.prepare_replacement(params,config)
                params=replacement.params
            else:
                params=self.model.prepare_params(params,config)
        except Exception:
            self.failed_key=key;self.retry_after=time.monotonic()+60
            raise
        metadata['prepared_at']=time.time()
        metadata['prepare_seconds']=time.monotonic()-started
        try:self.ready.get_nowait()
        except queue.Empty:pass
        metadata['architecture_upgrade']=replacement is not None
        self.ready.put_nowait((params,metadata,replacement))
        self.prepared_step=step;self.failed_key=None
        self.event({'event':'model_prepared','global_step':step,
            'checkpoint_sha256':metadata['checkpoint_sha256'],
            'summary':f"raw step={step} validated in {metadata['prepare_seconds']:.3f}s; ready for next decision"})
        return True

    def _loop(self):
        last_error=None;last_error_time=0
        while not self.stop.is_set():
            try:self.poll_once();last_error=None
            except Exception as error:
                message=f'{type(error).__name__}: {error}'
                now=time.monotonic()
                if message!=last_error or now-last_error_time>=60:
                    self.event({'event':'model_update_rejected','error':message,
                        'summary':f'Keep serving raw step={self.active["global_step"]}; {message}'})
                    last_error=message;last_error_time=now
            self.stop.wait(self.poll_seconds)

    def commit_ready(self):
        try:params,metadata,replacement=self.ready.get_nowait()
        except queue.Empty:return False
        if metadata['global_step']<=self.active['global_step']:return False
        metadata={**metadata,'activated_at':time.time()}
        try:atomic_json(self.root/'active_model.json',metadata)
        except OSError as error:
            self.prepared_step=self.active['global_step']
            self.event({'event':'model_update_rejected','error':str(error),
                'summary':'Cannot persist model activation; keep current weights'})
            return False
        # This method is called only by the serving thread, outside inference.
        previous=self.active['global_step']
        if replacement is not None:self.model=replacement
        else:self.model.params=params
        self.active=metadata
        self.on_activate(metadata)
        self.event({'event':'model_activated','previous_global_step':previous,
            'global_step':metadata['global_step'],
            'checkpoint_sha256':metadata['checkpoint_sha256'],
            'summary':f"raw weights {previous} -> {metadata['global_step']}; same worker, no restart"})
        return True

    def decide(self,raw):
        self.commit_ready()
        return self.model.decide(raw)
