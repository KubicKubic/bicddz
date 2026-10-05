"""Full public-event memory Transformer, ~2M parameters at default dimensions.

All events remain in explicit memory (no lossy history truncation). Re-encoding
raw memory on each PPO update avoids stale recurrent hidden-state targets.
Current rank tokens attend to every prior event, then self-attend. No opponent
hand or unrevealed bottom cards reach the actor OR the critic.
"""
import flax.linen as nn
import jax
import jax.numpy as j
from .actions import N_ACTIONS
from .env import HISTORY

class DenseEmbedding(nn.Module):
    """Small categorical vocabularies use GEMMs, not contended scatter VJPs.

    Public-history padding repeats the same category hundreds of thousands of
    times per minibatch. Gather embedding gradients otherwise serialize GPU
    atomic additions into a few rows. Dense one-hot contraction has the same
    parameter schema and forward semantics, with a dense reduction backward.
    """
    size: int
    width: int
    dtype: object

    @nn.compact
    def __call__(self,indices):
        table=self.param('embedding',nn.initializers.normal(1/self.width**.5),(self.size,self.width))
        return jax.nn.one_hot(indices,self.size,dtype=self.dtype) @ table.astype(self.dtype)


class MemoryTransformer(nn.Module):
    width: int = 192
    layers: int = 3
    heads: int = 6
    ff: int = 640
    bf16: bool = True

    @nn.compact
    def __call__(self,obs,memory_length=HISTORY):
        dt=j.bfloat16 if self.bf16 else j.float32
        dense=lambda n,**kw:nn.Dense(n,dtype=dt,**kw)
        x=dense(self.width,name='rank_in')(obs.ranks)
        x=x+self.param('rank_position',nn.initializers.normal(.02),(15,self.width))
        ctx=dense(self.width,name='context_in')(obs.context)[:,None,:]
        x=j.concatenate((ctx,x),axis=1).astype(dt)
        # The caller may omit masked padding, provided every valid event fits.
        h=obs.history[:,:memory_length].astype(j.int32)
        mem=dense(self.width,name='memory_cards')(h[...,:15].astype(j.float32)/4)
        for col,size in [(15,6),(16,3),(17,15),(18,15),(19,13),(20,4),(21,2)]:
            mem=mem+DenseEmbedding(size,self.width,dtype=dt,name=f'event_{col}')(h[...,col])
        mem=mem+self.param('memory_position',nn.initializers.normal(.02),(HISTORY,self.width))[:memory_length]
        # A learned null event keeps softmax defined before the first action.
        null=self.param('null_memory',nn.initializers.normal(.02),(1,1,self.width))
        mem=j.concatenate((j.broadcast_to(null,(x.shape[0],1,self.width)),mem),axis=1).astype(dt)
        valid=j.concatenate((j.ones((x.shape[0],1),j.bool_),j.arange(memory_length)[None,:]<obs.hist_len[:,None]),axis=1)
        mask=valid[:,None,None,:]
        mem=nn.LayerNorm(dtype=dt,name='memory_norm')(mem)
        for i in range(self.layers):
            q=nn.LayerNorm(dtype=dt,name=f'cross_norm{i}')(x)
            x=x+nn.MultiHeadDotProductAttention(self.heads,dtype=dt,name=f'cross{i}')(q,mem,mask=mask)
            q=nn.LayerNorm(dtype=dt,name=f'self_norm{i}')(x)
            x=x+nn.MultiHeadDotProductAttention(self.heads,dtype=dt,name=f'self{i}')(q)
            q=nn.LayerNorm(dtype=dt,name=f'ff_norm{i}')(x)
            x=x+dense(self.width,name=f'ff_out{i}')(nn.gelu(dense(self.ff,name=f'ff_in{i}')(q)))
        x=nn.LayerNorm(dtype=dt,name='out_norm')(x)
        pooled=j.concatenate((x[:,0],j.mean(x[:,1:],axis=1)),axis=-1)
        actor=nn.gelu(dense(self.width,name='actor_hidden')(pooled))
        critic=nn.gelu(dense(self.width,name='critic_hidden')(pooled))
        actor=nn.gelu(dense(self.width,name='actor_hidden2')(actor))
        critic=nn.gelu(dense(self.width,name='critic_hidden2')(critic))
        logits=dense(N_ACTIONS,name='actor',kernel_init=nn.initializers.orthogonal(.01))(actor).astype(j.float32)
        # Relative-seat expected raw scores. The zero-sum constraint is exact.
        value=dense(3,name='critic')(critic).astype(j.float32)
        value=value-value.mean(axis=-1,keepdims=True)
        return j.where(obs.legal,logits,-1e9),value


def absolute_value(relative,turn):
    return j.take_along_axis(relative,(j.arange(3)[None,:]-turn[:,None])%3,axis=-1)
