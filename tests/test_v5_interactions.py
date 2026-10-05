"""Exercise migration, conditional complete-action probabilities and resident PPO."""
import copy
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from ddz.env_v2 import event, legal_wings
from ddz import env_v4 as env, policy_v5 as policy, policy_v2 as prior
from ddz.model_v4 import CompactMoveTransformer
from ddz.model_v5 import InteractionMoveTransformer
from ddz.upgrade_v5 import warm_start_v4, upgrade_state


def batch(state):
    return jax.tree_util.tree_map(lambda x:x[None],env.observe(state))


def test_grown_depth_preserves_policy_value_history_and_has_live_gradients(tmp_path):
    cfg=dict(width=32,layers=1,heads=2,ff=64,memory_layers=1,memory_ff=64,
             bf16=False,wing_rank=4,action_width=16)
    old=CompactMoveTransformer(**cfg)
    new=InteractionMoveTransformer(**{**cfg,'layers':2,'memory_layers':3,
        'ff':96,'memory_ff':96},interaction_width=16,action_hidden=24)
    state=env.reset(jax.random.PRNGKey(510))._replace(phase=j.int32(1),landlord=j.int32(0))
    state=event(state,3,1,j.array([1]+[0]*14),1)
    state=event(state,4,2)
    obs=batch(state)
    source=old.init(jax.random.PRNGKey(511),obs)['params']
    target=new.init(jax.random.PRNGKey(512),obs)['params']
    path=tmp_path/'policy.msgpack';path.write_bytes(serialization.to_bytes(source))
    params,copied=warm_start_v4(target,path)
    assert len(copied)==len(jax.tree_util.tree_leaves(source))
    a=old.apply({'params':source},obs)
    out=new.apply({'params':params},obs)
    for x,y in zip(a,(out[0],out[1].base,out[2])):
        np.testing.assert_allclose(x,y,atol=2e-6)
    for changed in (obs,batch(state._replace(history=state.history.at[50,0].set(4)))):
        z=new.apply({'params':params},changed,memory_length=4)
        for x,y in zip(out,(z[0],z[1],z[2])):
            for xx,yy in zip(jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)):
                np.testing.assert_allclose(xx,yy,atol=2e-6)
    reverse=state._replace(history=state.history.at[0].set(state.history[1]).at[1].set(state.history[0]))
    assert np.max(np.abs(np.asarray(new.apply({'params':params},batch(reverse))[2]-out[2])))>1e-7
    one=jax.tree_util.tree_map(lambda x:x[0],out[1])
    for seed in range(8):
        key=jax.random.PRNGKey(seed)
        x=prior.sample_one(state,a[0][0],a[1][0],key)
        y=policy.sample_one(state,out[0][0],one,key)
        for xx,yy in zip(x,y):np.testing.assert_allclose(xx,yy,atol=2e-6)
    def loss(p):
        logits,_,values=new.apply({'params':p},obs,memory_length=4)
        return -jax.nn.log_softmax(logits)[0,j.argmax(out[0][0])]+j.square(values-j.array([[1.,-1.,0.]])).sum()
    grads=jax.grad(loss)(params)
    for name in ('actor_candidate_output','critic_action','memory_self1','memory_self2','self1','cross1'):
        assert sum(float(j.linalg.norm(x)) for x in jax.tree_util.tree_leaves(grads[name]))>0,name


def test_conditional_wings_normalize_exact_complete_action_space():
    from ddz import candidates_v3 as moves
    from ddz.actions import WINGS,WUNIT,COUNTS
    state=env.reset(jax.random.PRNGKey(514))
    hand=np.array([1,1,1,1,1,3,3,3,3,1,1,1,0,0,0],np.int32)
    state=state._replace(hands=state.hands.at[0].set(j.asarray(hand)),turn=j.int32(0),
        phase=j.int32(1),landlord=j.int32(0),bid=j.int32(1))
    keys=jax.random.split(jax.random.PRNGKey(515),6)
    context=policy.WingContext(jax.random.normal(keys[0],(309,15)),
        jax.random.normal(keys[1],(15,16)),jax.random.normal(keys[2],(16,)),
        jax.random.normal(keys[3],(309,16)),jax.random.normal(keys[4],(15,16)),
        jax.random.normal(keys[5],(16,)))
    logits=j.where(env.legal(state)[:313],j.sin(j.arange(313)*.2),-1e9)
    actions=j.asarray(moves.ACTION[moves.legal_ids(hand,1)])
    lp,_=jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,context,a)))(actions)
    np.testing.assert_allclose(j.exp(lp).sum(),1,rtol=3e-5)
    samples=jax.jit(jax.vmap(lambda k:policy.sample_one(state,logits,context,k)))(jax.random.split(keys[0],128))
    scored=jax.jit(jax.vmap(lambda a:policy.score_one(state,logits,context,a)))(samples[0])
    np.testing.assert_allclose(scored[0],samples[1],atol=2e-6)
    assert np.isin(np.asarray(samples[0]),np.asarray(actions)).all()
    bodies=[i for i in range(309) if WINGS[i]>=2 and bool(env.legal(state)[i])]
    assert bodies
    body=bodies[0];pending=prior._pending(state,j.int32(body))
    choices=np.flatnonzero(np.asarray(legal_wings(pending)))
    assert len(choices)>=2
    left=prior._advance(pending,j.int32(body),j.int32(choices[0]))
    right=prior._advance(pending,j.int32(body),j.int32(choices[1]))
    common=np.asarray(legal_wings(left)&legal_wings(right))
    diff=np.asarray(policy.wing_scores(left,context,body)-policy.wing_scores(right,context,body))[common]
    # Different earlier wings change relative preferences, beyond legality.
    assert len(diff)>=2 and np.ptp(diff)>1e-5


def test_optimizer_migration_and_resident_ppo_checkpoint(tmp_path):
    from ddz.train_v4 import create as create4
    from ddz.train_v5 import create,checkpoint
    from ddz.ppo_v5 import make_rollout,make_update
    cfg={'seed':41,'envs':2,'horizon':4,'updates':10,
        'model':dict(width=32,layers=1,heads=2,ff=64,memory_ff=64,
                     bf16=False,wing_rank=4,action_width=16),
        'ppo':{'gamma':1.,'lambda':.95,'clip':.2,'lr':1e-4,'lr_min':1e-5,
               'warmup':1,'minibatch':8,'epochs':1,'target_kl':.03,
               'max_grad_norm':1.,'weight_decay':1e-4}}
    _,old=create4(cfg)
    old=old.apply_gradients(grads=jax.tree_util.tree_map(lambda x:j.ones_like(x)*.01,old.params))
    source=tmp_path/'v4.msgpack';source.write_bytes(serialization.to_bytes(old.params))
    target=copy.deepcopy(cfg);target['model'].update(layers=2,memory_layers=2,ff=96,memory_ff=96,
        interaction_width=16,action_hidden=24)
    from flax.traverse_util import flatten_dict
    origin={'source_step':int(old.step),
        'source_shapes':{'.'.join(k):list(v.shape) for k,v in flatten_dict(old.params).items()}}
    model,new=create(target,origin)
    migrated,_=upgrade_state(new,{'train':serialization.to_state_dict(old)},source)
    assert int(migrated.step)==int(old.step)==1
    assert int(migrated.opt_state[1][0].count)==int(old.opt_state[1][0].count)==1
    np.testing.assert_array_equal(migrated.opt_state[1][0].mu['actor']['kernel'],old.opt_state[1][0].mu['actor']['kernel'])
    assert not np.any(np.asarray(migrated.opt_state[1][0].mu['actor_candidate_output']['kernel']))
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(516),2));key=jax.random.PRNGKey(517)
    roll=make_rollout(model,4,4)
    states,key,tr,last,stats=roll(migrated.params,states,key)
    assert not int(stats[:,1].sum())
    assert int(stats[:,4].max())>=int(tr.state.hist_len.max())
    trained,metrics=make_update(model,target['ppo'],4)(migrated,tr,last,key,j.float32(.002),j.float32(.03))
    fixed,fixed_metrics=make_update(model,target['ppo'],4,fixed_memory_length=True)(
        migrated,tr,last,key,j.float32(.002),j.float32(.03))
    for x,y in zip(jax.tree_util.tree_leaves(trained),jax.tree_util.tree_leaves(fixed)):
        np.testing.assert_allclose(x,y,atol=2e-6)
    assert float(metrics['nonfinite'])==0 and float(metrics['applied'])>0
    assert all(np.isfinite(float(x)) for x in metrics.values())
    checkpoint(tmp_path,trained,states,key,1,target,{'vf_coef':.03,'optimizer_origin':origin})
    payload=serialization.msgpack_restore((tmp_path/'latest.msgpack').read_bytes())
    _,empty=create(target,payload['runtime']['optimizer_origin'])
    restored=serialization.from_state_dict(empty,payload['train'])
    s=serialization.from_state_dict(states,payload['env'])
    a=roll(trained.params,states,key);b=roll(restored.params,s,j.asarray(payload['key']))
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x,y)


def test_new_adam_coordinates_match_fresh_adam_and_old_coordinates_are_unchanged():
    import optax
    from ddz.optimizer_v5 import age_corrected_adam
    params={'w':j.ones(4)};mask={'w':j.array([False,False,True,True])}
    base=optax.scale_by_adam();fresh=base.init(params)
    old=base.init(params)._replace(count=j.int32(1_000_000),
        mu={'w':j.array([.2,-.3,0.,0.])},nu={'w':j.array([.1,.2,0.,0.])})
    tx=age_corrected_adam(mask,1_000_000);state=old
    for i in range(15):
        grads={'w':j.sin(j.arange(4)+i*.2)+.1}
        mature,old=base.update(grads,old)
        new,fresh=base.update(grads,fresh)
        actual,state=tx.update(grads,state)
        np.testing.assert_array_equal(actual['w'][:2],mature['w'][:2])
        np.testing.assert_allclose(actual['w'][2:],new['w'][2:],rtol=2e-6,atol=1e-7)
        assert int(state.count)==1_000_001+i


def test_api_complete_move_and_all_role_evaluator(tmp_path):
    import json
    from ddz.api_v5 import Policy
    from ddz.evaluate_interaction import make_role_evaluator,paired_scores
    from ddz.actions import BODIES,physical_cards
    cfg={'model':dict(width=32,layers=2,heads=2,ff=64,memory_ff=64,
        memory_layers=2,bf16=False,wing_rank=4,action_width=16,
        interaction_width=16,action_hidden=24)}
    model=InteractionMoveTransformer(**cfg['model'])
    initial=env.reset(jax.random.PRNGKey(520))
    params=model.init(jax.random.PRNGKey(521),batch(initial))['params']
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    weights=tmp_path/'policy.msgpack';weights.write_bytes(serialization.to_bytes(params))
    api=Policy(config,weights)
    # A synthetic public snapshot after an immediate three-point auction.
    state,_,_,bad=env.step(initial,env.encode_bid(3),jax.random.PRNGKey(522))
    assert not bool(bad)
    def ids(counts):
        return [r*4+i if r<13 else r+39 for r,n in enumerate(counts) for i in range(int(n))]
    raw={'seat':int(state.turn),'turn':int(state.turn),'phase':'playing','version':1,
        'players':[{'count':int(h.sum())} for h in state.hands],
        'hand':ids(state.hands[state.turn]),'bottom':ids(state.bottom),'landlord':int(state.landlord),
        'bid':3,'bombs':0,'redeals':0,'leading':True,'last':None,
        'log':[{'kind':'bid','seat':int(state.landlord),'value':3},{'kind':'landlord','seat':int(state.landlord)}]}
    # Ensure the physical bottom cards really belong to the landlord's hand.
    used=set(raw['bottom']);remaining=np.asarray(state.hands[state.landlord]-state.bottom)
    raw['hand']=raw['bottom']+[
        card for r,n in enumerate(remaining) for card in
        [r*4+i if r<13 else r+39 for i in range(4 if r<13 else 1) if (r*4+i if r<13 else r+39) not in used][:int(n)]]
    kind,result=api.decide(raw)
    assert kind=='play' and set(result['cards'])<=set(raw['hand'])
    evaluate=make_role_evaluator(model,3,True,8)
    scores,done,bad,_=evaluate(params,params,523,0)
    assert np.all(done) and not np.any(bad)
    _,paired=paired_scores(scores,3,True)
    np.testing.assert_array_equal(paired,0)
