"""Exercise the actual deployed worker's startup and logging with a fake API."""
import json
from pathlib import Path
import runpy
import shutil
import pytest

WORKER=Path(__file__).resolve().parents[1]/'deploy/qoj/worker.py'

def launch_fake_worker(root,monkeypatch,ledger):
    shutil.copyfile(WORKER,root/'worker_v6.py')
    (root/'key').write_text('fake-test-key')
    config={'token_file':str(root/'key'),'base':'http://test','username':'Fortune',
        'tmux_session':'test','source_checkpoint':'fake.msgpack','global_step':1,
        'checkpoint_sha256':'fake'}
    (root/'deployment.json').write_text(json.dumps(config))
    (root/'results.jsonl').write_bytes(ledger)
    class Client:
        def __init__(self,*args):pass
        def request(self,path,payload=None):
            if path=='/info':return 200,{'match_rounds':9}
            if path=='/me':return 200,{'username':'Fortune','game':None,'score':0}
            raise AssertionError(path)
    class Model:
        def __init__(self,*args):pass
    monkeypatch.setattr('ddz.api.Client',Client)
    monkeypatch.setattr('ddz.api_v5.Policy',Model)
    monkeypatch.setattr('ddz.api.run',lambda *a,**kw:None)
    return runpy.run_path(str(root/'worker_v6.py'))

@pytest.mark.parametrize('tail',[b'{"game":',b'{"game":2,"username":"\xe4\xbd',
    b'{"game":2,"our_delta":"oops"}',b'{"game":2,"match":{"finished":true}}'])
def test_torn_or_invalid_result_record_is_quarantined_and_worker_still_starts(tmp_path,monkeypatch,tail):
    good=b'{"game":1,"our_delta":3}\n'
    state=launch_fake_worker(tmp_path,monkeypatch,good+tail)
    try:
        status=json.loads((tmp_path/'status.json').read_text())
        assert status['state']=='ready' and status['games_completed']==1 and status['score_delta']==3
        backups=list(tmp_path.glob('results.damaged.*.jsonl'))
        assert len(backups)==1 and backups[0].read_bytes()==good+tail
        assert (tmp_path/'results.jsonl').read_bytes()==good
        assert 'LEDGER_REPAIR' in (tmp_path/'console.log').read_text()
        assert 'fake-test-key' not in (tmp_path/'events.jsonl').read_text()
    finally:state['lock'].close()

def test_duplicate_result_does_not_double_count_and_missing_final_newline_is_repaired(tmp_path,monkeypatch):
    row=b'{"game":1,"our_delta":3}'
    state=launch_fake_worker(tmp_path,monkeypatch,row+b'\n'+row)
    try:
        status=json.loads((tmp_path/'status.json').read_text())
        assert status['games_completed']==1 and status['score_delta']==3
        assert (tmp_path/'results.jsonl').read_bytes().endswith(b'\n')
        assert not list(tmp_path.glob('results.damaged.*.jsonl'))
    finally:state['lock'].close()
