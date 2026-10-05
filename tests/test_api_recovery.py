import copy
import json
from pathlib import Path
import numpy as np
import pytest
from ddz.api_v5 import from_api
from ddz import env_v2 as env
from ddz.api import RetryDecision
from ddz.api_recovery import safe_decision
from ddz.actions import N_BODY


def real_redeal_state():
    return json.loads((Path(__file__).parent/'fixtures/qoj_redeal_landlord_reset_26894.json').read_text())


def test_real_redeal_reset_retains_training_inputs_and_current_deal_history():
    raw=real_redeal_state()
    parsed=from_api(raw)
    assert int(parsed.redeals)==1
    assert int(parsed.hist_len)==2
    assert np.all(np.asarray(parsed.history[:2,33])==1)
    retained=copy.deepcopy(raw); retained['redeals']=1
    for x,y in zip(env.observe(parsed),env.observe(from_api(retained))):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('field,value',[('bid',2),('bombs',1),('redeals',2)])
def test_redeal_compatibility_does_not_hide_other_state_corruption(field,value):
    raw=real_redeal_state();raw[field]=value
    with pytest.raises(ValueError):from_api(raw)


def test_bidding_redeal_counter_and_forced_bid_remain_strict():
    raw=real_redeal_state();raw.update(phase='bidding',bottom=None,landlord=None,bid=0,
        redeals=3,must_bid=True,leading=True,last=None)
    raw['players']=[{'count':17} for _ in range(3)]
    raw['log']=[]
    for _ in range(3):
        raw['log'] += [{'kind':'bid','seat':seat,'value':0} for seat in range(3)]
        raw['log'].append({'kind':'redeal'})
    raw['log'] += [{'kind':'bid','seat':seat,'value':0} for seat in (1,2)]
    parsed=from_api(raw)
    assert int(parsed.redeals)==3
    assert not bool(env.legal(parsed)[N_BODY])
    raw['redeals']=0
    with pytest.raises(ValueError,match='redeal'):from_api(raw)


def test_recovery_forced_bid_and_following_pass():
    raw=real_redeal_state()
    assert safe_decision(raw,None)==('pass',{'version':raw['version']})
    raw.update(phase='bidding',bid=0,must_bid=True)
    assert safe_decision(raw,None)==('bid',{'version':raw['version'],'value':1})
    raw['must_bid']=False
    assert safe_decision(raw,None)==('bid',{'version':raw['version'],'value':0})


def test_recovery_lead_uses_legal_hints_and_exposed_suits_first():
    raw=real_redeal_state();raw.update(leading=True,hand=[4,5,6,7,12],bottom=[6,7])
    class Client:
        def request(self,path):return 200,{'hints':[[4,5]]}
    assert safe_decision(raw,Client())==('play',{'version':raw['version'],'cards':[6,7]})


def test_missing_leading_hints_refreshes_instead_of_an_illegal_pass():
    raw=real_redeal_state();raw['leading']=True
    class Client:
        def request(self,path):return 200,{'hints':[]}
    with pytest.raises(RetryDecision):safe_decision(raw,Client())


def test_stale_leading_hints_refreshes_without_restarting():
    raw=real_redeal_state();raw.update(leading=True,hand=[0])
    class Client:
        def request(self,path):return 200,{'hints':[[53]]}
    with pytest.raises(RetryDecision):safe_decision(raw,Client())


def test_retry_decision_reuses_live_policy_and_refreshes_state(monkeypatch):
    from ddz.api import run
    monkeypatch.setattr('ddz.api.time.sleep',lambda _:None)
    state={'phase':'playing','turn':0,'seat':0,'version':1,'players':[{'auto':False}]}
    fresh={**state,'version':2}
    finished={'phase':'finished','result':{'deltas':[2,-1,-1]}}
    responses=[(200,{'game':9}),(200,{'state':state}),(200,{'state':fresh}),
        (200,{'state':finished})]
    calls=[]
    class Client:
        def request(self,path,payload=None):
            calls.append((path,payload));return responses.pop(0)
    class Policy:
        def decide(self,state):
            if state['version']==1:raise RetryDecision()
            return 'pass',{'version':state['version']}
    run(Client(),Policy(),once=True)
    assert calls[2]==('/games/9',None)
    assert calls[-1]==('/games/9/pass',{'version':2})
