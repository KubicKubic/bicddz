import copy
import numpy as np
import jax
import jax.numpy as j
import pytest
from ddz import env
from ddz.actions import BODIES,TYPES,BID_OFFSET,COUNTS,interpretations,physical_cards
from ddz.api import from_api,Policy,run

def ids(counts):
    return [4*r+i if r<13 else r+39 for r,n in enumerate(counts) for i in range(int(n))]

def snapshot(s):
    s=jax.device_get(s); logs=[]
    pools=[[4*r+i for i in range(4)] if r<13 else [r+39] for r in range(15)]
    for e in s.history[:s.hist_len]:
        kind={1:'bid',2:'landlord',3:'play',4:'pass',5:'redeal'}[int(e[15])]
        record={'kind':kind,'seat':int(e[16]),'t':0,'auto':bool(e[21])}
        if kind in ('bid','landlord'): record['value']=int(e[20])
        if kind=='landlord': record['cards']=ids(e[:15])
        if kind=='play':
            cards=[]
            for r,n in enumerate(e[:15]):
                for _ in range(n): cards.append(pools[r].pop(0))
            record['cards']=cards
            record['pattern']={'type':TYPES[e[17]],'rank':int(e[18]),'len':int(e[19])}
        if kind=='redeal': pools=[[4*r+i for i in range(4)] if r<13 else [r+39] for r in range(15)]
        logs.append(record)
    hand=[c for r,n in enumerate(s.hands[s.turn]) for c in pools[r][:n]]
    last=None if s.last==0 else {'seat':int(s.last_seat),'pattern':{
        'type':BODIES[s.last].type,'rank':BODIES[s.last].rank,'len':BODIES[s.last].length}}
    return {'id':1,'version':int(s.hist_len)+1,'seat':int(s.turn),'turn':int(s.turn),
        'phase':['bidding','playing','finished'][s.phase],'players':[
            {'count':int(s.hands[i].sum()),'auto':False,'bid':None if s.bids[i]<0 else int(s.bids[i])} for i in range(3)],
        'hand':hand,'hands':None,'landlord':None if s.landlord<0 else int(s.landlord),
        'bottom':None if s.phase==0 else ids(s.bottom),'bid':int(s.bid),'bombs':int(s.bombs),
        'redeals':int(s.redeals),'must_bid':bool(s.redeals>=3 and s.bid_count==2 and s.bid==0),
        'leading':bool(s.last==0 or s.last_seat==s.turn),'last':last,'log':logs}

def test_api_training_feature_parity_and_private_fields_ignored():
    step=jax.jit(env.step); legal=jax.jit(env.legal)
    rng=np.random.default_rng(123)
    s=env.reset(jax.random.PRNGKey(3))
    for t in range(250):
        if s.pending<0 and not s.done:
            raw=snapshot(s); parsed=from_api(raw)
            for x,y in zip(env.observe(s),env.observe(parsed)): np.testing.assert_allclose(x,y,atol=1e-6)
            # Finished-game disclosures, identities, fairness and chat cannot leak.
            raw['hands']=[[0]*20]*3; raw['fairness']={'deals':[{'deck':list(range(54))}]}
            for x,y in zip(env.observe(parsed),env.observe(from_api(raw))): np.testing.assert_array_equal(x,y)
        actions=np.flatnonzero(np.asarray(legal(s)))
        a=int(rng.choice(actions)); s,r,d,bad=step(s,j.int32(a),jax.random.PRNGKey(t))
        assert not bad
        if d: s=env.reset(jax.random.PRNGKey(t+50))

def test_api_choice_uses_full_local_move_and_physical_ids():
    s=env.reset(jax.random.PRNGKey(1))._replace(turn=j.int32(0),phase=j.int32(1),
       landlord=j.int32(0),bid=j.int32(3),hands=j.array([[3,3,2]+[0]*12,[0]*15,[0]*15]))
    raw=snapshot(s)
    body=next(i for i,b in enumerate(BODIES) if (b.type,b.rank,b.length)==('plane1',1,2))
    p=object.__new__(Policy)
    calls=[body,309+4+2,309+4+2]
    def forward(s):
        a=calls.pop(0); assert env.legal(s)[a]
        return j.zeros((1,328)).at[0,a].set(1),j.zeros((1,3))
    p.forward=forward
    endpoint,payload=p.decide(raw)
    assert endpoint=='play' and payload['choice']=='plane1:1:2'
    assert len(payload['cards'])==8 and len(set(payload['cards']))==8
    assert set(payload['cards'])<=set(raw['hand'])


def test_revealed_bottom_suits_are_played_first_for_equal_ranks():
    hand=[4,5,6,7,12]
    need=np.zeros(15,np.int32);need[1]=2
    assert physical_cards(hand,need,[7,6])==[6,7]
    assert physical_cards(hand,need,[7])==[4,7]

def test_conflict_auto_and_authoritative_state():
    s={'phase':'playing','turn':0,'seat':0,'version':1,'players':[{'auto':True}]*3}
    fresh=copy.deepcopy(s); fresh['version']=2
    fresh['players'][0]['auto']=False
    conflict=copy.deepcopy(fresh); conflict['version']=3
    finished={'phase':'finished','result':{'deltas':[2,-1,-1]}}
    responses=[(200,{'game':9}),(200,{'state':s}),(200,{'state':fresh}),
               (409,{'state':conflict}),(200,{'state':finished})]
    calls=[]
    class Client:
        def request(self,path,payload=None):
            calls.append((path,payload)); return responses.pop(0)
    class Policy:
        def decide(self,state): return 'pass',{'version':state['version']}
    run(Client(),Policy(),once=True)
    assert calls[2]==('/games/9/auto',{'version':1,'on':False})
    assert calls[3]==('/games/9/pass',{'version':2})
    assert calls[4]==('/games/9/pass',{'version':3})


def test_idle_account_queues_and_keeps_heartbeat(monkeypatch):
    monkeypatch.setattr('ddz.api.time.sleep',lambda _:None)
    responses=[(200,{'game':None,'queued':None}),
               (200,{'game':None,'queued':'single'}),
               (200,{'game':None,'queued':'single'}),
               (200,{'game':10,'queued':None}),
               (200,{'state':{'phase':'finished','result':{'deltas':[2,-1,-1]}}})]
    calls=[]
    class Client:
        def request(self,path,payload=None):
            calls.append((path,payload)); return responses.pop(0)
    run(Client(),None,mode='single',once=True)
    assert calls==[('/me',None),('/queue',{'mode':'single'}),('/me',None),
                   ('/me',None),('/games/10',None)]


def test_cancel_auto_while_another_player_is_acting():
    waiting={'phase':'playing','turn':1,'seat':0,'version':1,
        'players':[{'auto':True},{'auto':False},{'auto':False}]}
    finished={'phase':'finished','result':{'deltas':[2,-1,-1]}}
    responses=[(200,{'game':9}),(200,{'state':waiting}),(200,{'state':finished})]
    calls=[]
    class Client:
        def request(self,path,payload=None):
            calls.append((path,payload)); return responses.pop(0)
    run(Client(),None,once=True)
    assert calls[-1]==('/games/9/auto',{'version':1,'on':False})


def test_immediate_match_from_queue_response_is_used_without_an_extra_heartbeat():
    responses=[(200,{'game':None,'queued':None}),(200,{'game':9,'queued':None}),
        (200,{'state':{'phase':'finished','result':{'deltas':[2,-1,-1]}}})]
    calls=[]
    class Client:
        def request(self,path,payload=None):
            calls.append((path,payload)); return responses.pop(0)
    run(Client(),None,mode='match',once=True)
    assert calls==[('/me',None),('/queue',{'mode':'match'}),('/games/9',None)]


def test_match_continues_nine_rounds_then_requeues(monkeypatch):
    monkeypatch.setattr('ddz.api.time.sleep',lambda _:None)
    responses=[]
    for round_number in range(1,10):
        game=100+round_number
        responses.extend([(200,{'game':game,'queued':None}),
            (200,{'state':{'phase':'finished','result':{'deltas':[2,-1,-1]},
                'match':{'id':4,'round':round_number,'rounds':9,'finished':round_number==9}}})])
    responses.extend([(200,{'game':None,'queued':None}),
                      (200,{'game':None,'queued':'match'})])
    calls=[]
    class EndOfScript(Exception): pass
    class Client:
        def request(self,path,payload=None):
            if not responses: raise EndOfScript
            calls.append((path,payload)); return responses.pop(0)
    with pytest.raises(EndOfScript): run(Client(),None,mode='match',once=False)
    assert [path for path,_ in calls if path.startswith('/games/')]==[
        f'/games/{game}' for game in range(101,110)]
    assert calls[-2:]==[('/me',None),('/queue',{'mode':'match'})]
    assert sum(path=='/queue' for path,_ in calls)==1

def test_http_client_json_auth_and_409():
    import threading
    from http.server import BaseHTTPRequestHandler,HTTPServer
    from ddz.api import Client
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            import json
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((self.path,self.headers.get('Authorization'),payload))
            self.send_response(409); self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(b'{"error":"stale","state":{"version":12}}')
        def log_message(self,*args): pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        c=Client(f'http://127.0.0.1:{server.server_port}/api/v1/doudizhu','test-token')
        code,result=c.request('/games/3/bid',{'version':11,'value':2})
        assert code==409 and result['state']['version']==12
        assert received==[('/api/v1/doudizhu/games/3/bid','Bearer test-token',{'version':11,'value':2})]
    finally: server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize('status,body,expected',[
    (403,b'error code: 1010','API HTTP 403: non-JSON response'),
    (401,b'{"error":"invalid test-secret"}','API HTTP 401: invalid [REDACTED]'),
])
def test_http_client_edge_errors_and_credentials_redacted(status,body,expected):
    import threading
    from http.server import BaseHTTPRequestHandler,HTTPServer
    from ddz.api import Client
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status); self.end_headers(); self.wfile.write(body)
        def log_message(self,*args): pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        client=Client(f'http://127.0.0.1:{server.server_port}','test-secret')
        with pytest.raises(RuntimeError) as error: client.request('/me')
        assert str(error.value)==expected
        assert 'test-secret' not in str(error.value)
    finally:
        server.shutdown(); thread.join(); server.server_close()
