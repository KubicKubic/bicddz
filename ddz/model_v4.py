"""Full public memory and a compressed, exact complete-action policy.

The legal action tree shares 309 body prefixes and conditional wing ranks.
One small state query attends legal prefix descriptors, rather than applying
three decoder layers to every expanded complete action. New branches start
at zero, preserving every V2 policy and value output after weight transfer.
"""
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as j

from .actions import N_BODY, COUNTS, TYPE, RANK, LENGTH, WUNIT, WINGS
from .env_v2 import HISTORY, EVENT_DIM
from .model import DenseEmbedding, absolute_value


def prefix_features():
    """Semantic descriptors for all bodies and four bids, independent of state."""
    eye=lambda size,values:np.eye(size,dtype=np.float32)[values]
    body=np.concatenate((COUNTS/4,eye(15,TYPE),eye(15,RANK),
        eye(13,LENGTH),np.stack((WUNIT/2,WINGS/5),axis=-1),
        np.zeros((N_BODY,5))),axis=-1)
    bids=np.zeros((4,body.shape[1]));bids[:,-5:-1]=np.eye(4);bids[:,-1]=1
    return np.concatenate((body,bids)).astype(np.float32)


PREFIX_FEATURES=prefix_features()


class CompactMoveTransformer(nn.Module):
    width: int = 192
    layers: int = 3
    heads: int = 6
    ff: int = 640
    bf16: bool = True
    memory_layers: int = 1
    memory_ff: int = 384
    wing_rank: int = 16
    action_width: int = 48

    @nn.compact
    def __call__(self, obs, memory_length=HISTORY):
        dt = j.bfloat16 if self.bf16 else j.float32
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
        for i in range(self.memory_layers):
            q = nn.LayerNorm(dtype=dt, name=f'memory_self_norm{i}')(mem)
            attended = nn.MultiHeadDotProductAttention(
                self.heads, dtype=dt, name=f'memory_self{i}')(q, q, mask=key_mask)
            mem = mem + attended
            q = nn.LayerNorm(dtype=dt, name=f'memory_ff_norm{i}')(mem)
            mem = mem + dense(self.width, name=f'memory_ff_out{i}')(
                nn.gelu(dense(self.memory_ff, name=f'memory_ff_in{i}')(q)))

        for i in range(self.layers):
            q = nn.LayerNorm(dtype=dt, name=f'cross_norm{i}')(x)
            x = x + nn.MultiHeadDotProductAttention(
                self.heads, dtype=dt, name=f'cross{i}')(q, mem, mask=key_mask)
            q = nn.LayerNorm(dtype=dt, name=f'self_norm{i}')(x)
            x = x + nn.MultiHeadDotProductAttention(self.heads, dtype=dt,
                                                     name=f'self{i}')(q)
            q = nn.LayerNorm(dtype=dt, name=f'ff_norm{i}')(x)
            x = x + dense(self.width, name=f'ff_out{i}')(
                nn.gelu(dense(self.ff, name=f'ff_in{i}')(q)))
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
        value = dense(3, name='critic')(critic).astype(j.float32)
        value = value - value.mean(axis=-1, keepdims=True)
        return body_logits, wing_logits, value


__all__ = ['CompactMoveTransformer', 'absolute_value']
