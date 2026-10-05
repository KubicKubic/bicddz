"""Fault injection for live play; no network service or training is required."""
import copy
import io
import json
import urllib.error
import pytest
from ddz.api import (APIError,Client,ProtocolError,run)
from ddz.api_recovery import GuardedPolicy
from ddz.api_dashboard import public_snapshot,render_html,render_terminal,clip
from ddz.api_supervisor import stalled,terminate

ACTIVE={'id':9,'phase':'playing','turn':0,'seat':0,'version':1,
        'players':[{'auto':False}]*3,'hand':[0,1,4], 'leading':False,'bid':1}
FINISHED={'id':9,'phase':'finished','result':{'deltas':[2,-1,-1]}}

class ScriptClient:
    def __init__(self,responses):self.responses=list(responses);self.calls=[];self.recoveries=[]
    def request(self,path,payload=None):
        self.calls.append((path,payload))
        answer=self.responses.pop(0)
        if isinstance(answer,Exception):raise answer
        return answer
    def on_recovery(self,error,game):self.recoveries.append((type(error).__name__,game))

class Model:
    def __init__(self):self.versions=[]
    def decide(self,state):
        self.versions.append(state['version']);return 'pass',{'version':state['version']}

@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):monkeypatch.setattr('ddz.api.time.sleep',lambda _:None)

@pytest.mark.parametrize('status,body',[(429,b'busy'),(500,b'{"error":"server"}'),
    (502,b'<html>gateway</html>'),(503,b'[]'),(504,b'')])
def test_edge_transients_are_recoverable_even_without_json(monkeypatch,status,body):
    def fail(*args,**kwargs):
        raise urllib.error.HTTPError('http://test',status,'error',{'Retry-After':'999'},io.BytesIO(body))
    monkeypatch.setattr('urllib.request.urlopen',fail)
    code,data=Client('http://test','test-secret').request('/me')
    assert code==status and data['retry_after']==5
    assert 'test-secret' not in str(data)

@pytest.mark.parametrize('body',[b'{"state":',b'[]',b'null',b'<html>error</html>'])
def test_malformed_success_is_not_a_process_failure(monkeypatch,body):
    class Response(io.BytesIO):status=200
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:Response(body))
    with pytest.raises(ProtocolError):Client('http://test','secret').request('/games/9')

@pytest.mark.parametrize('fault',[(502,{'error':'bad gateway'}),
    ProtocolError('truncated JSON'),TimeoutError('timed out'),
    (200,{'state':{'phase':'playing'}}),
    (200,{'state':{'unchanged':True,'version':1}})])
def test_invalid_initial_state_refreshes_using_same_policy(fault):
    client=ScriptClient([(200,{'game':9}),fault,(200,{'state':ACTIVE}),
        (200,{'state':FINISHED})]);model=Model()
    run(client,model,once=True)
    assert client.calls[1:3]==[('/games/9',None)]*2
    assert model.versions==[1] and len(client.recoveries)==1

@pytest.mark.parametrize('fault',[TimeoutError('POST timeout'),
    ProtocolError('truncated POST response'),(503,{'error':'busy'})])
def test_uncertain_post_commit_fetches_state_instead_of_replaying(fault):
    client=ScriptClient([(200,{'game':9}),(200,{'state':ACTIVE}),fault,
        (200,{'state':FINISHED})]);model=Model()
    run(client,model,once=True)
    assert client.calls[-2:]==[('/games/9/pass',{'version':1}),('/games/9',None)]
    assert model.versions==[1]

def test_404_stale_game_rejoins_lobby_and_next_game():
    client=ScriptClient([(200,{'game':9}),APIError(404,'game missing'),
        (200,{'game':10}),(200,{'state':FINISHED})])
    run(client,None,once=True)
    assert [p for p,_ in client.calls]==['/me','/games/9','/me','/games/10']

def test_authentication_failure_stops_for_operator_attention():
    client=ScriptClient([APIError(401,'invalid credentials')])
    with pytest.raises(APIError):run(client,None,once=True)
    assert len(client.calls)==1

def test_same_version_rejection_uses_legal_fallback_and_new_version_returns_to_model():
    calls=[]
    class BadModel(Model):
        def decide(self,state):
            self.versions.append(state['version']);return 'play',{'version':state['version'],'cards':[53]}
    fresh={**ACTIVE,'version':2}
    client=ScriptClient([(200,{'game':9}),(200,{'state':ACTIVE}),
        (409,{'error':'cards outside hand','state':ACTIVE}),
        (200,{'state':fresh}),(200,{'state':FINISHED})])
    model=BadModel();guard=GuardedPolicy(model,client,lambda *args:calls.append(args))
    run(client,guard,once=True)
    assert [p for p,_ in client.calls[-3:]]==['/games/9/play','/games/9/pass','/games/9/play']
    assert model.versions==[1,2]
    assert [c[3] for c in calls]==['model','fallback','model']

def test_parameter_rejection_also_refreshes_then_falls_back():
    client=ScriptClient([(200,{'game':9}),(200,{'state':ACTIVE}),APIError(400,'bad choice'),
        (200,{'state':ACTIVE}),(200,{'state':FINISHED})]);model=Model()
    run(client,GuardedPolicy(model,client),once=True)
    assert client.calls[-2:]==[('/games/9',None),('/games/9/pass',{'version':1})]
    assert model.versions==[1]

@pytest.mark.parametrize('error',[ValueError('bad log'),KeyError('new pattern'),
    IndexError('oversized history'),TypeError('null pattern')])
def test_adapter_faults_do_not_restart_or_leak_hidden_cards(error):
    class BrokenModel:
        def decide(self,state):raise error
    calls=[];guard=GuardedPolicy(BrokenModel(),None,lambda *a:calls.append(a))
    assert guard.decide(ACTIVE)==('pass',{'version':1})
    assert calls[0][3]=='fallback' and calls[0][5].startswith(type(error).__name__)

def test_changed_version_conflict_recomputes_normally_without_fallback():
    client=ScriptClient([(200,{'game':9}),(200,{'state':ACTIVE}),
        (409,{'state':{**ACTIVE,'version':2}}),(200,{'state':FINISHED})]);model=Model()
    run(client,GuardedPolicy(model,client),once=True)
    assert model.versions==[1,2]

def test_unchanged_poll_refreshes_clock_and_wrong_version_fetches_full_state():
    waiting={**ACTIVE,'turn':1}
    client=ScriptClient([(200,{'game':9}),(200,{'state':waiting}),
        (200,{'state':{'unchanged':True,'version':99}}),
        (200,{'state':ACTIVE}),(200,{'state':FINISHED})])
    run(client,Model(),once=True)
    assert client.calls[2:4]==[('/games/9?version=1',None),('/games/9',None)]

def test_dashboard_escapes_untrusted_text_and_contains_only_public_model_inputs():
    raw={**ACTIVE,'players':[{'username':'<script>boom</script>','count':17,'auto':False}],
        'hands':[[53]],'fairness':{'deals':'SECRET_DECK'},'chat':['SECRET_CHAT'],
        'log':[{'kind':'pass','seat':1}], 'landlord':0}
    state=public_snapshot(raw)
    assert not {'hands','fairness','chat'}&state.keys()
    status={'pid':12,'game':9,'last_http_completed':100,'state':'in_game'}
    decision={'game':9,'version':1,'endpoint':'pass','source':'model','payload':{},
        'inference':{'game':9,'version':1,'value_by_seat':[2,-1,-1]}}
    page=render_html(status,state,decision)
    assert '<script>boom</script>' not in page and '&lt;script&gt;' in page
    assert 'SECRET_DECK' not in page and '不是胜率' in page
    text=render_terminal(status,state,decision,140,24,now=101)
    assert 'S0=+2.000' in text and 'S1 不出' in text
    assert '\x1b' not in render_terminal(status,{**state,'hand':[]},decision)
    assert clip('你好 abc',5)=='你好 '
    assert clip('a\x1b[2J',50)=='a[2J'

def test_watchdog_distinguishes_warmup_live_heartbeat_and_hung_worker():
    assert not stalled({'pid':4,'state':'warming'},4,100,144)
    assert stalled({'pid':4,'state':'warming'},4,100,146)
    assert not stalled({'pid':4,'state':'in_game','last_progress':145},4,100,150)
    assert stalled({'pid':4,'state':'in_game','last_progress':130},4,100,150)
    assert not stalled({'pid':3,'state':'in_game','last_progress':0},4,145,150)

def test_dashboard_prefers_own_value_with_team_derived_scores_over_other_raw_heads():
    inference={'game':9,'version':1,'seat':0,'value_training_clock':'own',
        'value_by_seat':[-99.,99.,0.],'display_value_by_seat':[2.,-4.,2.]}
    decision={'inference':inference}
    text=render_terminal({'game':9},ACTIVE,decision,140,28)
    assert 'S0=+2.000' in text and 'S1=-4.000' in text and 'S0=-99.000' not in text
    page=render_html({'game':9},ACTIVE,decision)
    assert '阵营推导' in page and '-99.000' not in page

def test_supervisor_terminates_only_its_owned_process_group():
    import subprocess,sys
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
    try:
        terminate(process);assert process.poll() is not None
    finally:
        if process.poll() is None:process.kill();process.wait()
