"""V5 interaction model, approximately 5M parameters.

Complete variable-length public memory; direct body-to-rank attention; explicit
remaining-card features; context-dependent wing combinations. A V4 warm start
preserves its function by zeroing new residual paths and growing FFNs silently.
"""
import numpy as np
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as j

from .actions import N_BODY, COUNTS, TYPE, RANK, LENGTH, WUNIT, WINGS
from .env_v2 import HISTORY, EVENT_DIM
from .model import DenseEmbedding, absolute_value
from .policy_v5 import WingContext


def prefix_features():
    """Semantic descriptors for all bodies and four bids, independent of state."""
    eye=lambda size,values:np.eye(size,dtype=np.float32)[values]
    body=np.concatenate((COUNTS/4,eye(15,TYPE),eye(15,RANK),
        eye(13,LENGTH),np.stack((WUNIT/2,WINGS/5),axis=-1),
        np.zeros((N_BODY,5))),axis=-1)
    bids=np.zeros((4,body.shape[1]));bids[:,-5:-1]=np.eye(4);bids[:,-1]=1
    return np.concatenate((body,bids)).astype(np.float32)


PREFIX_FEATURES=prefix_features()


def expanded_attention(model,query,key,mask,index,prefix,dtype,attention_fn,memory=False):
    """Keep mature BF16 GEMM shapes; add a separate zero-output head bank.

    Padding a learned projection with new heads changes backend GEMM reduction
    order. Separate banks retain the exact mature computations and temperature.
    Every head still attends to the complete unmasked sequence.
    """
    # Query and key lengths can coincide for cross attention (15 events plus
    # the null token). Shape equality cannot identify history self attention.
    if attention_fn is fused_full_attention:
        attention_fn=partial(attention_fn,query_is_memory=memory)
    options=model.attention_options(memory)
    inherited=model.inherited_memory_layers if memory else model.inherited_layers
    if inherited and index<inherited and options['qkv_features']>model.width:
        base=nn.MultiHeadDotProductAttention(model.heads,dtype=dtype,attention_fn=attention_fn,
            name=f'{prefix}{index}')(query,key,mask=mask)
        width=options['qkv_features']-model.width
        heads=options['num_heads']-model.heads
        if heads<=0 or width//heads!=model.width//model.heads or width%heads:
            raise ValueError('Expanded attention bank must preserve inherited head dimensions')
        added=nn.MultiHeadDotProductAttention(heads,qkv_features=width,out_features=model.width,
            dtype=dtype,attention_fn=attention_fn,name=f'{prefix}_extra{index}')(query,key,mask=mask)
        return base+added
    return nn.MultiHeadDotProductAttention(**options,dtype=dtype,attention_fn=attention_fn,
        name=f'{prefix}{index}')(query,key,mask=mask)


def fused_full_attention(query,key,value,mask=None,query_is_memory=False,**kwargs):
    """cuDNN full attention with exact prefix lengths and masked even padding.

    JAX 0.4.38's cuDNN backward requires even sequence lengths. The extra
    token is backend padding only; it never enters memory or the policy.
    """
    qlength=query.shape[1];klength=key.shape[1]
    kv_lengths=(j.sum(mask[:,0,0,:],axis=-1).astype(j.int32) if mask is not None
                else j.full((query.shape[0],),klength,j.int32))
    q_lengths=kv_lengths if query_is_memory else j.full((query.shape[0],),qlength,j.int32)
    pad=lambda x:j.pad(x,((0,0),(0,x.shape[1]%2),(0,0),(0,0)))
    return jax.nn.dot_product_attention(pad(query),pad(key),pad(value),
        query_seq_lengths=q_lengths,key_value_seq_lengths=kv_lengths,
        implementation='cudnn')[:,:qlength]


class InteractionMoveTransformer(nn.Module):
    width: int = 192
    layers: int = 4
    heads: int = 6
    ff: int = 1536
    bf16: bool = True
    memory_layers: int = 3
    memory_ff: int = 512
    wing_rank: int = 16
    action_width: int = 48
    interaction_width: int = 64
    action_hidden: int = 96
    attention_backend: str = 'standard'
    # Widen attention head banks without changing the residual feature basis.
    # Keeping head_dim fixed permits an exact V5 warm start.
    attention_width: int = 0
    attention_heads: int = 0
    memory_attention_width: int = 0
    memory_attention_heads: int = 0
    inherited_layers: int = 0
    inherited_memory_layers: int = 0
    new_layer_ff: int = 0

    def attention_options(self, memory=False):
        width = (self.memory_attention_width if memory else self.attention_width) or self.width
        heads = (self.memory_attention_heads if memory else self.attention_heads) or self.heads
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('Attention width must be a positive multiple of heads')
        return dict(num_heads=heads, qkv_features=width, out_features=self.width)

    @staticmethod
    def block_order(count, inherited):
        if not inherited:
            return tuple(range(count))
        if not 0 < inherited <= count:
            raise ValueError('Invalid inherited block count')
        # Insert new identity-initialized blocks after early inherited blocks;
        # retain the exact ordering of every old block.
        result=[]
        for i in range(inherited):
            result.append(i)
            if inherited+i<count:
                result.append(inherited+i)
        result.extend(range(2*inherited,count))
        return tuple(result)

    @nn.compact
    def __call__(self, obs, memory_length=HISTORY):
        dt = j.bfloat16 if self.bf16 else j.float32
        if self.attention_backend not in ('standard','cudnn'):
            raise ValueError('unknown attention backend')
        attention_fn=(fused_full_attention if self.attention_backend=='cudnn'
                      and self.bf16 and jax.default_backend()=='gpu'
                      else nn.dot_product_attention)
        dense = lambda n, **kw: nn.Dense(n, dtype=dt, **kw)
        x = dense(self.width, name='rank_in')(obs.ranks)
        x = x + self.param('rank_position', nn.initializers.normal(.02),
                           (15, self.width))
        ctx = dense(self.width, name='context_in')(obs.context)[:, None, :]
        x = j.concatenate((ctx, x), axis=1).astype(dt)

        h = obs.history[:, :memory_length].astype(j.int32)
        mem = dense(self.width, name='memory_cards')(h[..., :15].astype(j.float32) / 4)
        for col, size in [(15, 6), (16, 4), (17, 15), (18, 15),
                          (19, 13), (20, 4), (21, 2)]:
            mem = mem + DenseEmbedding(size, self.width, dtype=dt,
                                        name=f'event_{col}')(h[..., col])
        # The event's full public snapshot is available to every history
        # token. Counts and bids are scaled by their game maxima.
        scale = j.array([20,20,20,3,3,20,2,3,15,15,13,3,2,3,
                         4,4,4,20,20,20,3],j.float32)
        mem = mem + dense(self.width, name='memory_public')(
            h[..., 22:EVENT_DIM].astype(j.float32) / scale)
        mem = mem + self.param('memory_position', nn.initializers.normal(.02),
                               (HISTORY, self.width))[:memory_length]
        null = self.param('null_memory', nn.initializers.normal(.02),
                          (1, 1, self.width))
        mem = j.concatenate((j.broadcast_to(null, (x.shape[0], 1, self.width)),
                             mem), axis=1).astype(dt)
        valid = j.concatenate((j.ones((x.shape[0], 1), j.bool_),
                                j.arange(memory_length)[None, :] < obs.hist_len[:, None]),
                               axis=1)
        # All real events attend to all other real events. Padded keys are
        # masked; padded queries are never read by the current-hand tokens.
        key_mask = valid[:, None, None, :]
        mem = nn.LayerNorm(dtype=dt, name='memory_norm')(mem)
        for i in self.block_order(self.memory_layers, self.inherited_memory_layers):
            q = nn.LayerNorm(dtype=dt, name=f'memory_self_norm{i}')(mem)
            attended = expanded_attention(self,q,q,key_mask,i,'memory_self',dt,attention_fn,memory=True)
            mem = mem + attended
            q = nn.LayerNorm(dtype=dt, name=f'memory_ff_norm{i}')(mem)
            mem = mem + dense(self.width, name=f'memory_ff_out{i}')(
                nn.gelu(dense(self.memory_ff, name=f'memory_ff_in{i}')(q)))

        for i in self.block_order(self.layers, self.inherited_layers):
            q = nn.LayerNorm(dtype=dt, name=f'cross_norm{i}')(x)
            x = x + expanded_attention(self,q,mem,key_mask,i,'cross',dt,attention_fn)
            q = nn.LayerNorm(dtype=dt, name=f'self_norm{i}')(x)
            x = x + expanded_attention(self,q,q,None,i,'self',dt,attention_fn)
            q = nn.LayerNorm(dtype=dt, name=f'ff_norm{i}')(x)
            x = x + dense(self.width, name=f'ff_out{i}')(
                nn.gelu(dense(self.new_layer_ff if self.inherited_layers and i >= self.inherited_layers
                              and self.new_layer_ff else self.ff, name=f'ff_in{i}')(q)))
        x = nn.LayerNorm(dtype=dt, name='out_norm')(x)
        pooled = j.concatenate((x[:, 0], j.mean(x[:, 1:], axis=1)), axis=-1)
        actor = nn.gelu(dense(self.width, name='actor_hidden')(pooled))
        critic = nn.gelu(dense(self.width, name='critic_hidden')(pooled))
        actor = nn.gelu(dense(self.width, name='actor_hidden2')(actor))
        critic = nn.gelu(dense(self.width, name='critic_hidden2')(critic))

        # Exact prefix set: impossible bodies/bids are masked on device.
        # Descriptors are encoded once per forward, shared by the entire batch.
        keys=dense(self.action_width,name='prefix_keys')(j.asarray(PREFIX_FEATURES))
        query=dense(self.action_width,name='prefix_query',
                    kernel_init=nn.initializers.zeros)(actor)
        scores=j.einsum('bd,ad->ba',query.astype(j.float32),keys.astype(j.float32))
        scores=scores/self.action_width**.5
        legal=obs.legal[:,:N_BODY+4]
        attention=jax.nn.softmax(j.where(legal,scores,-1e9),axis=-1)
        summary=attention.astype(dt)@keys
        actor=actor+dense(self.width,name='prefix_context',
                         kernel_init=nn.initializers.zeros)(summary)

        body_logits = dense(N_BODY + 4, name='actor',
                            kernel_init=nn.initializers.orthogonal(.01))(actor).astype(j.float32)
        body_logits=body_logits+scores
        # Each legal body queries all contextual card ranks directly, including
        # history-derived rank features. Only 309 prefixes are materialized.
        dim=self.interaction_width
        rank_keys=dense(dim,name='action_rank_keys')(x[:,1:])
        rank_values=dense(dim,name='action_rank_values')(x[:,1:])
        body_keys=nn.gelu(dense(dim,name='action_body_embed')(
            j.asarray(PREFIX_FEATURES[:N_BODY])))
        queries=dense(dim,name='action_body_queries')(body_keys)
        rank_attention=jax.nn.softmax(j.einsum('ad,bkd->bak',
            queries.astype(j.float32),rank_keys.astype(j.float32))/dim**.5,axis=-1)
        attended=j.einsum('bak,bkd->bad',rank_attention.astype(dt),rank_values)
        state_key=dense(dim,name='action_state')(actor)
        post_body=j.maximum(obs.ranks[:,None,:,0]*4-j.asarray(COUNTS)[None],0)/4
        fused=attended+body_keys[None]+state_key[:,None]
        action_hidden=nn.gelu(dense(self.action_hidden,name='action_fusion')(
            j.concatenate((fused,post_body.astype(dt)),axis=-1)))
        body_delta=dense(1,name='actor_candidate_output',
            kernel_init=nn.initializers.zeros)(action_hidden)[...,0].astype(j.float32)
        body_logits=body_logits.at[:,:N_BODY].add(body_delta)
        body_logits = j.where(obs.legal[:, :N_BODY + 4], body_logits, -1e9)
        # Low-rank action features yield a separate preference for each
        # (body, wing rank) without a huge 309 x 15 dense actor layer.
        factors = dense(self.wing_rank * 15, name='wing_state')(actor)
        factors = factors.reshape((actor.shape[0], self.wing_rank, 15))
        body_factors = self.param('body_wing_factors',
                                  nn.initializers.normal(.02),
                                  (N_BODY, self.wing_rank))
        body_factors=body_factors+dense(self.wing_rank,name='prefix_wing',
            kernel_init=nn.initializers.zeros)(j.asarray(PREFIX_FEATURES[:N_BODY]))
        wing_logits = j.einsum('brk,ar->bak', factors.astype(j.float32),
                               body_factors.astype(j.float32))
        # Conditional wing head: all expensive event/state encoding happens
        # once. Subsequent wing choices use these compact contextual features.
        wing_state=dense(dim,name='wing_query')(actor)
        wing_body=dense(dim,name='wing_body')(body_keys)
        remain=self.param('wing_remaining_kernel',nn.initializers.normal(.05),(15,dim))
        output=self.param('wing_output',nn.initializers.zeros,(dim,))
        broadcast=lambda a:j.broadcast_to(a,(actor.shape[0],)+a.shape)
        wing=WingContext(wing_logits,rank_values.astype(j.float32),
            wing_state.astype(j.float32),broadcast(wing_body.astype(j.float32)),
            broadcast(remain),broadcast(output))
        value = dense(3, name='critic')(critic).astype(j.float32)
        legal_action_summary=j.einsum('ba,bad->bd',attention[:,:N_BODY].astype(dt),action_hidden)
        value=value+dense(3,name='critic_action',kernel_init=nn.initializers.zeros)(
            legal_action_summary).astype(j.float32)
        value = value - value.mean(axis=-1, keepdims=True)
        return body_logits, wing, value


__all__ = ['InteractionMoveTransformer', 'absolute_value']
