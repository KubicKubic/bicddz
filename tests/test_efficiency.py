"""Independent trace references, exact migration, KL gates and label isolation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import jax
import jax.numpy as j
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState
from ddz import env_v4 as env, policy_v5 as policy
from ddz.actions import N_BODY,BODIES
from ddz.ppo_efficiency import (own_seat_gae,belief_targets,forced_one,prepare_targets,
    ArenaState,Transition,make_rollout,make_update)
from ddz.optimizer_efficiency import birth_corrected_adam,coordinate_births,sampled_lr
from tests.test_v5_exploration import small_config


def reference_own(values,rewards,dones,turns,gamma,lam):
    t,b,_=values.shape;adv=np.zeros((t,b));target=np.zeros((t,b));valid=np.zeros((t,b),bool)
    for lane in range(b):
        ends=list(np.flatnonzero(dones[:,lane]));starts=[0]+[x+1 for x in ends]
        if not ends or ends[-1]!=t-1:ends.append(t-1)
        for start,end in zip(starts,ends):
            terminal=bool(dones[end,lane])
            for seat in range(3):
                indices=[i for i in range(start,end+1) if turns[i,lane]==seat]
                future_adv=0
                for k in range(len(indices)-1,-1,-1):
                    i=indices[k];has_next=k+1<len(indices)
                    if not has_next and not terminal:
                        target[i,lane]=values[i,lane,seat];continue
                    stop=indices[k+1] if has_next else end+1
                    r=sum(gamma**(u-i)*rewards[u,lane,seat] for u in range(i,stop))
                    nv=values[stop,lane,seat] if has_next else 0
                    a=r+gamma**(stop-i)*nv-values[i,lane,seat]
                    if has_next:a+=gamma**(stop-i)*lam*future_adv
                    adv[i,lane]=a;target[i,lane]=a+values[i,lane,seat];valid[i,lane]=True;future_adv=a
    return adv,target,valid


@pytest.mark.parametrize('gamma,lam',[(1.,.95),(1.,1.),(.9,.7)])
def test_own_seat_gae_matches_independent_episodic_reference(gamma,lam):
    rng=np.random.default_rng(7);values=rng.normal(size=(13,3,3)).astype(np.float32)
    turns=np.array([[0,1,2],[0,2,0],[1,0,1],[2,1,2],[0,2,0],[1,0,1],
        [2,0,2],[0,1,0],[1,2,1],[2,0,2],[0,1,0],[1,2,1],[2,0,2]],np.int32)
    rewards=np.zeros_like(values);dones=np.zeros((13,3),bool)
    for t,b in [(3,0),(8,0),(5,1),(12,1),(7,2)]:
        dones[t,b]=True;rewards[t,b]=np.array([2,-1,-1])*(t+1)
    expected=reference_own(values,rewards,dones,turns,gamma,lam)
    actual=jax.jit(lambda:own_seat_gae(j.asarray(values),j.asarray(rewards),j.asarray(dones),j.asarray(turns),gamma,lam))()
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-6)
    # Other heads see different private observations; own targets ignore them.
    altered=values.copy()
    for t in range(13):
        for b in range(3):
            for seat in range(3):
                if seat!=turns[t,b]:altered[t,b,seat]=10000
    changed=own_seat_gae(j.asarray(altered),j.asarray(rewards),j.asarray(dones),j.asarray(turns),gamma,lam)
    for x,y in zip(actual,changed):np.testing.assert_array_equal(x,y)


def test_own_gae_terminal_monte_carlo_and_unresolved_tail():
    values=j.arange(8*1*3,dtype=j.float32).reshape(8,1,3)/7
    rewards=j.zeros_like(values).at[4,0].set(j.array([12.,-6.,-6.]))
    dones=j.array([[False],[False],[False],[False],[True],[False],[False],[False]])
    turns=j.array([[0],[1],[2],[0],[1],[2],[0],[1]])
    a,t,valid=own_seat_gae(values,rewards,dones,turns,1.,1.)
    expected=np.array([12,-6,-6,12,-6])
    np.testing.assert_allclose(t[:5,0],expected,atol=1e-6)
    assert valid[:5].all() and not valid[5:].any()


def test_three_generations_of_adam_ages_and_sample_clock_lr():
    params={'w':j.ones(4),'aux':j.ones(2)};source={'w':j.ones(4)}
    origin={'source_step':100,'source_shapes':{'w':[2]}}
    births=coordinate_births(params,source,origin,1000)
    np.testing.assert_array_equal(births['w'],[0,0,100,100]);assert births['aux']==1000
    base=optax.scale_by_adam();fresh=base.init(params)
    state=base.init(params)._replace(count=j.int32(1000),
        mu={'w':j.array([.2,.3,.1,-.2]),'aux':j.zeros(2)},
        nu={'w':j.array([.1,.2,.05,.08]),'aux':j.zeros(2)})
    from ddz.optimizer_v5 import age_corrected_adam
    legacy=age_corrected_adam({'w':j.array([False,False,True,True]),'aux':False},100)
    legacy_state=state;tx=birth_corrected_adam(births)
    for i in range(8):
        g=jax.tree_util.tree_map(lambda x:j.sin(x+i)+.1,params)
        actual,state=tx.update(g,state);old,legacy_state=legacy.update(g,legacy_state);new,fresh=base.update(g,fresh)
        np.testing.assert_allclose(actual['w'],old['w'],rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(actual['aux'],new['aux'],rtol=2e-6,atol=1e-7)
    cfg=small_config();cfg['updates']=10000
    from ddz.train_v2 import learning_rate_schedule
    expected=learning_rate_schedule(cfg)
    for epochs in (1,2,4):
        variant=copy.deepcopy(cfg);variant['ppo']['epochs']=epochs
        lr=sampled_lr(variant,100)
        for fresh in (0,1,100):np.testing.assert_allclose(lr(fresh),expected(100+fresh),rtol=1e-7)


def test_belief_labels_do_not_change_policy_inputs_and_value_team_constraint(tmp_path):
    from ddz.train_v5 import create as original_create
    from ddz.train_efficiency import create
    cfg=small_config();old_model,old=original_create(cfg)
    state=env.reset(jax.random.PRNGKey(8))._replace(phase=j.int32(1),landlord=j.int32(0),turn=j.int32(0))
    one=jax.tree_util.tree_map(lambda x:x[None],state)
    source={'train':serialization.to_state_dict(old),'runtime':{}}
    new_cfg=copy.deepcopy(cfg);new_cfg['model'].update(belief_head=True,team_value=False)
    model,ts=create(new_cfg,source)
    obs=jax.vmap(env.observe)(one)
    expected=old_model.apply({'params':old.params},obs,memory_length=8)
    actual=model.apply({'params':ts.params},obs,memory_length=8,return_aux=True)
    for x,y in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual[:3])):
        np.testing.assert_array_equal(x,y)
    assert actual[3].shape==(1,2,15,5)
    changed=one._replace(hands=one.hands.at[:,1].set(one.hands[:,2]).at[:,2].set(one.hands[:,1]))
    # Equal hand sizes: exchanging hidden rank composition does not change obs.
    for x,y in zip(jax.tree_util.tree_leaves(obs),jax.tree_util.tree_leaves(jax.vmap(env.observe)(changed))):
        np.testing.assert_array_equal(x,y)
    assert not np.array_equal(belief_targets(one),belief_targets(changed))
    for k,seat in enumerate((1,2)):np.testing.assert_array_equal(belief_targets(one)[:,k],one.hands[:,seat])
    new_cfg['model']['team_value']=True
    team,tts=create(new_cfg,source)
    projected=team.apply({'params':tts.params},obs,memory_length=8)[2]
    np.testing.assert_array_equal(projected[:,1],projected[:,2])
    np.testing.assert_allclose(projected.sum(-1),0,atol=1e-7)
    # Unknown landlord must retain the unconstrained auction predictions.
    before=jax.vmap(env.observe)(one._replace(landlord=j.array([-1]),phase=j.array([0])))
    np.testing.assert_array_equal(team.apply({'params':tts.params},before,memory_length=8)[2],
        model.apply({'params':ts.params},before,memory_length=8)[2])
    # Experimental checkpoints remain consumable by the public-info bot.
    from ddz.api_v5 import Policy
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(new_cfg))
    weights=tmp_path/'policy.msgpack';weights.write_bytes(serialization.to_bytes(tts.params))
    bot=Policy(config_path,weights)
    actual=bot.forward(state)
    expected=team.apply({'params':tts.params},obs)
    for x,y in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)


class ToyModel:
    belief_head=False
    def __init__(self,calls=None):self.calls=calls
    def apply(self,variables,obs,memory_length=None,return_aux=False):
        if self.calls is not None:
            jax.debug.callback(lambda _:self.calls.append(1),j.sum(obs.hist_len),ordered=True)
        w=variables['params']['w'];b=obs.context.shape[0]
        logits=j.where(obs.legal[:,:313],w*j.sin(j.arange(313)*.3),-1e9)
        logits=j.broadcast_to(logits,(b,313))
        ctx=policy.WingContext(j.zeros((b,309,15)),j.zeros((b,15,1)),j.zeros((b,1)),
            j.zeros((b,309,1)),j.zeros((b,15,1)),j.zeros((b,1)))
        v=j.broadcast_to(w*j.array([1.,-1.,0.]),(b,3))
        return (logits,ctx,v,None) if return_aux else (logits,ctx,v)


def toy_rollout():
    model=ToyModel();params={'w':j.float32(.1)}
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(9),4))
    arena=ArenaState(states,j.zeros(4,j.int32),j.zeros(4,j.int32),j.zeros(4,j.bool_),j.zeros(4,j.bool_))
    pool=jax.tree_util.tree_map(lambda x:x[None],params)
    result=make_rollout(model,4,8)(params,arena,jax.random.PRNGKey(10),pool)
    return model,params,pool,result


def test_kl_blocks_current_update_and_really_skips_later_gradient_computation():
    _,params,_,result=toy_rollout();_,key,tr,last,_=result
    calls=[];model=ToyModel(calls)
    cfg={**small_config()['ppo'],'epochs':3,'minibatch':4}
    ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    mismatch=tr._replace(logp=tr.logp-3)
    updated,m=make_update(model,cfg,8)(ts,mismatch,last,key,j.float32(.002),j.float32(.03),j.float32(.001))
    jax.block_until_ready(m);jax.effects_barrier()
    assert int(updated.step)==0 and int(m['applied_minibatches'])==0
    assert int(m['evaluated_minibatches'])==1 and len(calls)==1
    assert float(m['kl'])>.03 and float(m['evaluated_fraction'])==pytest.approx(1/12)
    np.testing.assert_array_equal(updated.params['w'],params['w'])


def test_foreign_actions_do_not_affect_actor_loss_and_pool_never_leaks_labels():
    model,params,pool,result=toy_rollout();arena,key,tr,last,_=result
    foreign=tr._replace(learner=j.zeros_like(tr.learner),logp=j.full_like(tr.logp,10000))
    cfg=small_config()['ppo'];ts=TrainState.create(apply_fn=model.apply,params=params,tx=optax.scale(-1.))
    _,m=make_update(model,cfg,8)(ts,foreign,last,key,j.float32(.002),j.float32(.03),j.float32(.001))
    assert float(m['policy_loss'])==float(m['kl'])==float(m['entropy'])==0
    assert float(m['nonfinite'])==0 and float(m['actor_eligible_fraction'])==0
    arena=arena._replace(use_pool=j.ones(4,j.bool_),focus=j.array([0,1,2,0]))
    # Exercise subset bucket overflow fallback: bucket one, many rival seats.
    _,_,mixed,_,stats=make_rollout(model,8,16,1.,pool_bucket=1)(params,arena,key,pool)
    assert int(stats[:,1].sum())==0 and (~mixed.learner).any() and mixed.learner.any()
    a,target,vm,am,valid=prepare_targets(mixed,j.zeros((4,3)),{**cfg,'gae_clock':'own'})
    assert not np.any(np.asarray(am & ~mixed.learner))


def test_forced_complete_actions_use_legality_not_near_one_policy_probability():
    state=env.reset(jax.random.PRNGKey(11))._replace(turn=j.int32(0),phase=j.int32(1),landlord=j.int32(0),bid=j.int32(3))
    rocket=next(i for i,b in enumerate(BODIES) if b.type=='rocket')
    follow=state._replace(last=j.int32(rocket),last_seat=j.int32(1))
    pass_action=env.encode_move(j.int32(0),j.full(5,15,j.int32))
    assert bool(forced_one(follow,pass_action))
    action=policy.greedy_one(state,*_toy_outputs_for_state(state))
    assert not bool(forced_one(state,action))


def _toy_outputs_for_state(state):
    obs=jax.tree_util.tree_map(lambda x:x[None],env.observe(state))
    l,c,_=ToyModel().apply({'params':{'w':j.float32(100)}},obs)
    return l[0],jax.tree_util.tree_map(lambda x:x[0],c)


def test_full_efficiency_checkpoint_and_cli_combined_resume(tmp_path):
    from ddz.train_v5 import create as original_create
    from ddz.train_v2 import checkpoint
    cfg=small_config();cfg['save_every']=1
    model,ts=original_create(cfg)
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(12),2))
    checkpoint(tmp_path,ts,states,jax.random.PRNGKey(13),0,cfg,{'vf_coef':.03,'stable_updates':0})
    source=tmp_path/'latest.msgpack';pool=tmp_path/'policy_0000000.msgpack'
    variant=copy.deepcopy(cfg);variant['model'].update(belief_head=True,team_value=True)
    variant['ppo'].update(epochs=2,gae_clock='own',belief_coef=.03,optional_actor_only=True)
    variant.update(pool_probability=1.,opponent_pool=[str(pool)])
    config=tmp_path/'variant.json';config.write_text(json.dumps(variant));out=tmp_path/'run'
    def run(*args):
        result=subprocess.run([sys.executable,'-m','ddz.train_efficiency','--config',str(config),
            '--source',str(source),'--out',str(out),*args],cwd=Path(__file__).resolve().parents[1],
            env={**os.environ,'JAX_PLATFORMS':'cpu'},capture_output=True,text=True,timeout=240)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    run('--steps','1');first=serialization.msgpack_restore((out/'latest.msgpack').read_bytes())
    run('--steps','2','--resume');second=serialization.msgpack_restore((out/'latest.msgpack').read_bytes())
    assert first['iteration']==1 and second['iteration']==2
    assert second['runtime']['source_iteration']==0 and second['runtime']['training_seed']==0
    assert second['train']['step']>=first['train']['step']
    rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
    assert all(r['invalid_actions']==r['nonfinite']==0 and np.isfinite(r['belief_loss']) for r in rows)
    assert rows[0]['learning_rate']==pytest.approx(float(sampled_lr(variant,0)(0)))
    assert second['train']['step']>first['train']['step']
    assert not np.array_equal(first['train']['params']['belief_output']['kernel'],
                              second['train']['params']['belief_output']['kernel'])
    # Frozen-pool identities are enforced before any resumed rollout.
    pool.write_bytes(pool.read_bytes()+b'changed')
    result=subprocess.run([sys.executable,'-m','ddz.train_efficiency','--config',str(config),
        '--source',str(source),'--out',str(out),'--steps','3','--resume'],
        cwd=Path(__file__).resolve().parents[1],env={**os.environ,'JAX_PLATFORMS':'cpu'},
        capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'pool changed' in result.stderr


def test_evaluation_deals_unique_and_invariant_to_chunk_partition():
    from ddz.evaluate_efficiency import unique_deals
    whole,_,keys=unique_deals(500,0,12)
    parts=[unique_deals(500,k,3)[0] for k in range(0,12,3)]
    joined=jax.tree_util.tree_map(lambda *xs:j.concatenate(xs),*parts)
    for x,y in zip(jax.tree_util.tree_leaves(whole),jax.tree_util.tree_leaves(joined)):
        np.testing.assert_array_equal(x,y)
    assert len({x.tobytes() for x in np.asarray(whole.hands)})==12
    assert len({x.tobytes() for x in np.asarray(keys)})==12
    forced,bad,_=unique_deals(500,0,12,True)
    assert not bad.any() and (forced.landlord==j.arange(12)%3).all()
    assert (forced.phase==1).all()


def test_unique_common_deals_identical_policy_cancels_all_six_legs():
    from ddz.evaluate_efficiency import make_evaluator,summarize
    model=ToyModel();params={'w':j.float32(.1)}
    scores,done,bad,t=make_evaluator(model,model,3,True,192)(params,params,500,0)
    assert done.all() and not bad.any() and int(t)<=256
    np.testing.assert_array_equal(np.asarray(scores).mean(axis=1),0)
    summary=summarize(np.asarray(scores),True,500,100)
    assert summary['mean']==0 and summary['ci95']==[0,0]


def test_frozen_pool_selects_correct_model_in_all_groups_including_overflow():
    model=ToyModel();learner={'w':j.float32(.2)};pool={'w':j.array([.1,-.1,0.])}
    states=env.batch_reset(jax.random.split(jax.random.PRNGKey(401),4))
    ids=j.array([0,1,2,2]);arena=ArenaState(states,ids,(states.turn+1)%3,
        j.zeros(4,j.bool_),j.ones(4,j.bool_))
    _,_,tr,_,stats=make_rollout(model,1,8,1.,pool_bucket=1)(learner,arena,jax.random.PRNGKey(402),pool)
    expected=[]
    for k in range(4):
        one=jax.tree_util.tree_map(lambda x:x[k:k+1],states)
        l,c,_=model.apply({'params':{'w':pool['w'][ids[k]]}},jax.vmap(env.observe)(one))
        expected.append(policy.greedy(one,l,c)[0])
    np.testing.assert_array_equal(tr.action[0],expected)
    assert not tr.learner.any() and not stats[:,1].any()
