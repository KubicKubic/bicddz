"""Optional V5 sample-efficiency branches; base policy parameter names are unchanged.

Opponent cards are supervised labels only: this module receives the same public
Observation as V5 and never receives full environment hands.
"""
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as j
from .model import DenseEmbedding
from .env_v2 import HISTORY, EVENT_DIM
from .actions import N_BODY, COUNTS
from .policy_v5 import WingContext
from .model_v5 import InteractionMoveTransformer, PREFIX_FEATURES, fused_full_attention


class EfficientMoveTransformer(InteractionMoveTransformer):
    belief_head: bool = False
    team_value: bool = False

    @nn.compact
    def __call__(self, obs, memory_length=HISTORY, return_aux=False):
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
        for i in range(self.memory_layers):
            q = nn.LayerNorm(dtype=dt, name=f'memory_self_norm{i}')(mem)
            attended = nn.MultiHeadDotProductAttention(
                self.heads, dtype=dt, attention_fn=attention_fn,
                name=f'memory_self{i}')(q, q, mask=key_mask)
            mem = mem + attended
            q = nn.LayerNorm(dtype=dt, name=f'memory_ff_norm{i}')(mem)
            mem = mem + dense(self.width, name=f'memory_ff_out{i}')(
                nn.gelu(dense(self.memory_ff, name=f'memory_ff_in{i}')(q)))

        for i in range(self.layers):
            q = nn.LayerNorm(dtype=dt, name=f'cross_norm{i}')(x)
            x = x + nn.MultiHeadDotProductAttention(
                self.heads, dtype=dt, attention_fn=attention_fn,
                name=f'cross{i}')(q, mem, mask=key_mask)
            q = nn.LayerNorm(dtype=dt, name=f'self_norm{i}')(x)
            x = x + nn.MultiHeadDotProductAttention(self.heads, dtype=dt, attention_fn=attention_fn,
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
        if self.team_value:
            landlord = j.argmax(obs.context[:, 3:6], axis=-1)
            landlord_value = j.take_along_axis(value, landlord[:, None], axis=-1)
            projected = j.where(j.arange(3)[None] == landlord[:, None], landlord_value, -landlord_value / 2)
            value = j.where(obs.context[:, 6:7] > .5, value, projected)
        beliefs = None
        if self.belief_head:
            hidden = nn.gelu(dense(128, name='belief_hidden')(pooled))
            beliefs = dense(150, name='belief_output',
                kernel_init=nn.initializers.orthogonal(.01))(hidden).astype(j.float32)
            beliefs = beliefs.reshape((-1, 2, 15, 5))
            valid_counts = (j.arange(15)[:, None] < 13) | (j.arange(5)[None] <= 1)
            beliefs = j.where(valid_counts[None, None], beliefs, -1e9)
        return (body_logits, wing, value, beliefs) if return_aux else (body_logits, wing, value)
