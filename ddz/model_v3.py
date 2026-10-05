"""Full-history attention with an explicit legal complete-move token set.

Every valid public event attends every other valid public event before the
current hand queries the memory. Event positions encode chronology; all
events in the memory have already happened, so bidirectional attention does
not expose future information. The state queries the legal-move set, then each
complete-move token cross-attends to the contextualized state. Padding tokens
are encoded as pass but masked from attention and policy normalization.
"""
import flax.linen as nn
import jax
import jax.numpy as j

from .env_v2 import HISTORY, EVENT_DIM
from .model import DenseEmbedding, absolute_value
from . import candidates_v3 as moves


class CandidateSetTransformer(nn.Module):
    width: int = 192
    layers: int = 3
    heads: int = 6
    ff: int = 640
    bf16: bool = True
    memory_layers: int = 1
    memory_ff: int = 384
    candidate_layers: int = 3
    candidate_ff: int = 512

    @nn.compact
    def __call__(self, obs, candidate_ids, candidate_mask, memory_length=HISTORY):
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
        ids=candidate_ids.astype(j.int32)
        counts=j.take(j.asarray(moves.CARDS),ids,axis=0).astype(j.float32)/4
        bodies=j.take(j.asarray(moves.BODY_CARDS),ids,axis=0).astype(j.float32)/4
        wings=j.take(j.asarray(moves.WING_CARDS),ids,axis=0).astype(j.float32)/4
        candidate=dense(self.width,name='candidate_cards')(
            j.concatenate((counts,bodies,wings),axis=-1))
        for name,table,size in [('type',moves.TYPE_OF,15),
                                ('rank',moves.RANK_OF,15),
                                ('length',moves.LENGTH_OF,13),
                                ('bid',moves.BID_OF,4),
                                ('is_bid',moves.IS_BID.astype(j.uint8),2)]:
            candidate=candidate+DenseEmbedding(size,self.width,dtype=dt,
                name=f'candidate_{name}')(j.take(j.asarray(table),ids,axis=0))
        candidate=nn.LayerNorm(dtype=dt,name='candidate_in_norm')(candidate)

        # The state explicitly queries the available complete-move set.
        query=nn.LayerNorm(dtype=dt,name='set_query_norm')(x[:, :1])
        set_mask=candidate_mask[:,None,None,:]
        summary=nn.MultiHeadDotProductAttention(self.heads,dtype=dt,
            name='set_cross')(query,candidate,mask=set_mask)
        x=x.at[:,0].add(summary[:,0])

        # Candidate queries are independent; no O(N^2) candidate self-attention.
        # All legal choices are scored in one pass, with pass only as padding.
        for i in range(self.candidate_layers):
            q=nn.LayerNorm(dtype=dt,name=f'candidate_cross_norm{i}')(candidate)
            candidate=candidate+nn.MultiHeadDotProductAttention(
                self.heads,dtype=dt,name=f'candidate_cross{i}')(q,x)
            q=nn.LayerNorm(dtype=dt,name=f'candidate_ff_norm{i}')(candidate)
            candidate=candidate+dense(self.width,name=f'candidate_ff_out{i}')(
                nn.gelu(dense(self.candidate_ff,name=f'candidate_ff_in{i}')(q)))
        candidate=nn.LayerNorm(dtype=dt,name='candidate_out_norm')(candidate)
        logits=dense(1,name='candidate_score',
            kernel_init=nn.initializers.orthogonal(.01))(candidate)[...,0].astype(j.float32)
        logits=j.where(candidate_mask,logits,-1e9)
        pooled=j.concatenate((x[:,0],j.mean(x[:,1:16],axis=1)),axis=-1)
        critic=nn.gelu(dense(self.width,name='critic_hidden')(pooled))
        critic=nn.gelu(dense(self.width,name='critic_hidden2')(critic))
        value = dense(3, name='critic')(critic).astype(j.float32)
        value = value - value.mean(axis=-1, keepdims=True)
        return logits,value


__all__ = ['CandidateSetTransformer', 'absolute_value']
