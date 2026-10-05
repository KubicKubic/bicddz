"""Full-history attention and one-forward complete-move policy heads.

Every valid public event attends every other valid public event before the
current hand queries the memory. Event positions encode chronology; all
events in the memory have already happened, so bidirectional attention does
not expose future information. The policy emits body and body-conditioned wing
scores in one forward pass; the resulting complete move is one environment
action.
"""
import flax.linen as nn
import jax
import jax.numpy as j

from .actions import N_BODY
from .env_v2 import HISTORY, EVENT_DIM
from .model import DenseEmbedding, absolute_value


class FullAttentionMoveTransformer(nn.Module):
    width: int = 192
    layers: int = 3
    heads: int = 6
    ff: int = 640
    bf16: bool = True
    memory_layers: int = 1
    memory_ff: int = 384
    wing_rank: int = 16

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

        body_logits = dense(N_BODY + 4, name='actor',
                            kernel_init=nn.initializers.orthogonal(.01))(actor).astype(j.float32)
        body_logits = j.where(obs.legal[:, :N_BODY + 4], body_logits, -1e9)
        # Low-rank action features yield a separate preference for each
        # (body, wing rank) without a huge 309 x 15 dense actor layer.
        factors = dense(self.wing_rank * 15, name='wing_state')(actor)
        factors = factors.reshape((actor.shape[0], self.wing_rank, 15))
        body_factors = self.param('body_wing_factors',
                                  nn.initializers.normal(.02),
                                  (N_BODY, self.wing_rank))
        wing_logits = j.einsum('brk,ar->bak', factors.astype(j.float32),
                               body_factors.astype(j.float32))
        value = dense(3, name='critic')(critic).astype(j.float32)
        value = value - value.mean(axis=-1, keepdims=True)
        return body_logits, wing_logits, value


__all__ = ['FullAttentionMoveTransformer', 'absolute_value']
