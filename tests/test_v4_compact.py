"""Complete-action equivalence, preserved information and functional transfer."""
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from ddz import env_v2 as old,env_v4 as fast,candidates_v3 as moves,policy_v2 as policy
from ddz.model_v2 import FullAttentionMoveTransformer
from ddz.model_v4 import CompactMoveTransformer
from ddz.train_v4 import warm_start_v2


def assert_tree_equal(a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))


def test_fast_atomic_step_matches_original_over_complete_random_games():
    batch=12
    initial=old.batch_reset(jax.random.split(jax.random.PRNGKey(414),batch))
    @jax.jit
    def check(states,key):
        def tick(carry,_):
            states,key,equal,bad=carry
            key,ak,sk,rk=jax.random.split(key,4)
            mask=jax.vmap(old.legal)(states)[:,:309+4]
            logits=j.where(mask,j.zeros(mask.shape),-1e9)
            actions,_,_=policy.sample(states,logits,j.zeros((batch,309,15)),jax.random.split(ak,batch))
            keys=jax.random.split(sk,batch)
            a=jax.vmap(old.step)(states,actions,keys)
            b=jax.vmap(fast.step)(states,actions,keys)
            eq=j.stack([j.all(x==y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b))]).all()
            ns,reward,done,invalid=b
            fresh=old.batch_reset(jax.random.split(rk,batch))
            ns=jax.tree_util.tree_map(lambda x,z:j.where(done.reshape((batch,)+(1,)*(x.ndim-1)),z,x),ns,fresh)
            return (ns,key,equal&eq,bad+j.sum(invalid)),None
        return jax.lax.scan(tick,(states,key,j.array(True),j.int32(0)),None,length=160)[0]
    _,_,equal,bad=check(initial,jax.random.PRNGKey(415))
    assert bool(equal) and int(bad)==0


def test_compressed_distribution_covers_and_normalizes_all_complete_actions():
    state=old.reset(jax.random.PRNGKey(416))
    hand=np.array([1,1,1,1,1,3,3,3,3,1,1,1,0,0,0],np.int32)
    state=state._replace(hands=state.hands.at[0].set(j.asarray(hand)),turn=j.int32(0),
        phase=j.int32(1),landlord=j.int32(0),bid=j.int32(1))
    ids=moves.legal_ids(hand,1);actions=j.asarray(moves.ACTION[ids])
    mask=old.legal(state)[:313]
    logits=j.where(mask,j.sin(j.arange(313)*.2),-1e9)
    wings=j.cos(j.arange(309*15).reshape(309,15)*.11)
    logp,_=jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,wings,a)))(actions)
    np.testing.assert_allclose(j.sum(j.exp(logp)),1,rtol=2e-5)
    assert np.isfinite(np.asarray(logp)).all()
    a=jax.jit(jax.vmap(lambda action:old.step(state,action,jax.random.PRNGKey(417))))(actions)
    b=jax.jit(jax.vmap(lambda action:fast.step(state,action,jax.random.PRNGKey(417))))(actions)
    assert_tree_equal(a,b)
    assert not np.any(np.asarray(b[3]))
    # Invalid card multiplicity, noncanonical wings and bad bids roll back.
    invalid=j.array([old.encode_move(1,j.array([0,15,15,15,15])),old.encode_bid(4),0])
    for action in invalid:
        assert_tree_equal(old.step(state,action,jax.random.PRNGKey(418)),
                          fast.step(state,action,jax.random.PRNGKey(418)))


def test_transfer_preserves_policy_value_and_history_with_trainable_new_branches(tmp_path):
    cfg=dict(width=48,layers=1,heads=3,ff=96,memory_ff=96,bf16=False)
    old_model=FullAttentionMoveTransformer(**cfg)
    model=CompactMoveTransformer(**cfg,action_width=16)
    state=old.reset(jax.random.PRNGKey(419))._replace(phase=j.int32(1),landlord=j.int32(0))
    state=old.event(state,3,1,j.array([1]+[0]*14),1)
    state=old.event(state,4,2)
    batch=lambda s:jax.tree_util.tree_map(lambda x:x[None],old.observe(s))
    obs=batch(state)
    source=old_model.init(jax.random.PRNGKey(420),obs)['params']
    target=model.init(jax.random.PRNGKey(421),obs)['params']
    path=tmp_path/'v2.msgpack';path.write_bytes(serialization.to_bytes(source))
    transferred,copied=warm_start_v2(target,path)
    assert len(copied)==len(jax.tree_util.tree_leaves(source))
    original=old_model.apply({'params':source},obs)
    for x,y in zip(original,model.apply({'params':transferred},obs)):
        np.testing.assert_allclose(x,y,atol=1e-6)
    for x,y in zip(original,model.apply({'params':transferred},obs,memory_length=4)):
        np.testing.assert_allclose(x,y,atol=1e-6)
    padded=batch(state._replace(history=state.history.at[50,0].set(4)))
    for x,y in zip(original,model.apply({'params':transferred},padded)):
        np.testing.assert_allclose(x,y,atol=1e-6)
    reversed_state=state._replace(history=state.history.at[0].set(state.history[1]).at[1].set(state.history[0]))
    reversed_out=model.apply({'params':transferred},batch(reversed_state))
    assert np.max(np.abs(np.asarray(original[2]-reversed_out[2])))>1e-7
    def loss(params):
        logits,wings,_=model.apply({'params':params},obs)
        return -jax.nn.log_softmax(logits)[0,j.argmax(original[0][0])]+j.square(wings).mean()
    grads=jax.grad(loss)(transferred)
    for name in ('prefix_query','prefix_context','prefix_wing'):
        assert np.linalg.norm(np.asarray(grads[name]['kernel']))>0


def test_device_rollout_update_checkpoint_and_role_pairing(tmp_path):
    from ddz.train_v4 import create,checkpoint
    from ddz.ppo_v4 import make_rollout,make_update
    from ddz.evaluate_compact import make_role_evaluator,paired_scores
    cfg={'seed':41,'envs':4,'horizon':8,'updates':10,
        'model':dict(width=32,layers=1,heads=2,ff=64,memory_ff=64,
                     bf16=False,action_width=16),
        'ppo':{'gamma':1.,'lambda':.95,'clip':.2,'lr':1e-4,'lr_min':1e-5,
               'warmup':1,'minibatch':16,'epochs':1,'target_kl':.03,
               'max_grad_norm':1.,'weight_decay':1e-4}}
    model,ts=create(cfg);states=fast.batch_reset(jax.random.split(jax.random.PRNGKey(422),4))
    key=jax.random.PRNGKey(423);roll=make_rollout(model,8,8)
    states,key,tr,last,stats=roll(ts.params,states,key)
    assert int(stats[:,1].sum())==0
    logp,_=jax.vmap(jax.vmap(policy.score_one))(tr.state,
        *jax.vmap(lambda s:model.apply({'params':ts.params},jax.vmap(old.observe)(s),memory_length=8)[:2])(tr.state),tr.action)
    np.testing.assert_allclose(logp,tr.logp,atol=1e-5)
    trained,metrics=make_update(model,cfg['ppo'],8)(ts,tr,last,key,j.float32(.02),j.float32(.03))
    assert float(metrics['nonfinite'])==0 and float(metrics['applied'])>0
    assert all(np.isfinite(float(x)) for x in metrics.values())
    runtime={'vf_coef':.03,'stable_updates':1}
    checkpoint(tmp_path,trained,states,key,1,cfg,runtime)
    restored=serialization.msgpack_restore((tmp_path/'latest.msgpack').read_bytes())
    new_ts=serialization.from_state_dict(trained,restored['train'])
    new_states=serialization.from_state_dict(states,restored['env'])
    assert_tree_equal(roll(trained.params,states,key),roll(new_ts.params,new_states,j.asarray(restored['key'])))
    evaluate=make_role_evaluator(model,3,True,8)
    scores,finished,bad,_=evaluate(ts.params,ts.params,424,0)
    assert np.all(finished) and not np.any(bad)
    _,paired=paired_scores(scores,3,True)
    np.testing.assert_array_equal(paired,0)
