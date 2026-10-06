"""Exercise both production CUDA backward signatures before a V6 rollout."""
import time
import json
import numpy as np
import jax
import jax.numpy as j
import optax
from . import env_v4 as env


def attention_mask_parity():
    """Cross attention must retain all 16 queries even with 16 history keys."""
    from flax.linen import dot_product_attention
    from .model_v5 import fused_full_attention
    rows=[]
    lengths=j.asarray([1,3],j.int32)
    mask=(j.arange(16)[None,:]<lengths[:,None])[:,None,None,:]
    for heads in (2,4,6,8,10):
        q,k,v=jax.random.normal(jax.random.PRNGKey(700+heads),(3,2,16,heads,32),dtype=j.bfloat16)
        expected=dot_product_attention(q,k,v,mask=mask,dtype=j.bfloat16)
        for memory in (False,True):
            apply=jax.jit(lambda q,k,v:fused_full_attention(q,k,v,mask,query_is_memory=memory))
            actual=apply(q,k,v)
            valid=(j.arange(16)[None,:]<lengths[:,None]) if memory else j.ones((2,16),j.bool_)
            error=np.max(np.abs(np.asarray(actual.astype(j.float32)-expected.astype(j.float32)))[np.asarray(valid)])
            if not np.isfinite(error) or error>.02:raise RuntimeError('CUDA attention prefix mask disagrees with full-attention reference')
            rows.append({'heads':heads,'history_queries':memory,'max_error':float(error)})
    return {'passed':True,'cases':rows}


def training_signatures(model,params,cfg):
    if jax.default_backend()!='gpu':raise RuntimeError('CUDA preflight requires a GPU')
    rows=[]
    for length,batch in ((cfg.get('memory_limit',88),cfg['per_gpu_minibatch']),
                         (env.HISTORY,min(1024,cfg['per_gpu_minibatch']))):
        print(json.dumps({'v6_backward_preflight':'compiling','memory_length':length,'per_gpu_minibatch':batch}),flush=True)
        state=env.reset(jax.random.PRNGKey(cfg['seed']))
        state=state._replace(phase=j.int32(1),landlord=j.int32(0),bid=j.int32(1))
        one=env.observe(state)
        obs=jax.tree_util.tree_map(lambda x:j.broadcast_to(x,(batch,)+x.shape),one)
        # Include empty, partial, exactly-full and overflowing-short histories.
        lengths=j.asarray([0,1,15,length,env.HISTORY],j.int32)[j.arange(batch)%5]
        history=obs.history.at[:,:,15].set(j.uint8(4))
        history=history.at[:,:,16].set((j.arange(env.HISTORY)[None,:]%3).astype(j.uint8))
        obs=obs._replace(history=history,hist_len=lengths)
        def objective(p,o):
            logits,wing,value=model.apply({'params':p},o,memory_length=length)
            label=j.argmax(o.legal[:,:logits.shape[-1]],axis=-1)
            logp=j.take_along_axis(jax.nn.log_softmax(logits),label[:,None],axis=-1)[:,0]
            return j.mean(-logp+.03*j.square(value[:,0]-.1))+.0001*j.mean(j.square(wing.base))
        start=time.monotonic()
        compiled=jax.jit(jax.value_and_grad(objective)).lower(params,obs).compile()
        allocation=compiled.memory_analysis()
        loss,gradient=compiled(params,obs)
        jax.block_until_ready((loss,gradient))
        finite=all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(gradient))
        norm=float(optax.global_norm(gradient))
        if not finite or not np.isfinite(float(loss)) or not np.isfinite(norm) or norm<=0:
            raise RuntimeError('V6 CUDA forward/backward is nonfinite or has no gradient')
        stats=jax.local_devices()[0].memory_stats() or {}
        rows.append({'memory_length':length,'per_gpu_minibatch':batch,'finite':True,
            'loss':float(loss),'gradient_norm':norm,'seconds':time.monotonic()-start,
            'compiled_temp_bytes':allocation.temp_size_in_bytes,
            'peak_bytes_in_use':stats.get('peak_bytes_in_use',0)})
        print(json.dumps({'v6_backward_preflight':'passed',**rows[-1]}),flush=True)
        del gradient,obs,compiled
        jax.clear_caches()
    return {'passed':True,'backend':jax.default_backend(),
            'device':jax.local_devices()[0].device_kind,'signatures':rows}
