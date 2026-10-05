"""Persistent QOJ rated play with bounded network recovery and model diagnostics."""
import fcntl
import http.client
import json
import os
from pathlib import Path
import time
import urllib.error
from ddz.api import (APIError,RetryDecision,TemporaryAPIError,Client,run,
                     checked_request,response_state)
from ddz.api_v5 import Policy
from ddz.api_recovery import GuardedPolicy
from ddz.api_dashboard import action_text,clean,public_snapshot

root=Path(os.environ.get('DDZ_MATCH_ROOT',Path(__file__).resolve().parent)).resolve()
cfg=json.loads(Path(os.environ.get('DDZ_DEPLOYMENT_FILE',root/'deployment.json')).read_text())
model_dir=Path(cfg.get('model_dir',root))
token=Path(cfg['token_file']).read_text().strip()
lock=(root/'account.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(75)

def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)

def event(record):
    record={'time':time.time(),**record}
    # All records are structured; credentials and transport headers are excluded.
    with (root/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
    detail=record.get('summary') or record.get('error') or record.get('path') or record.get('game','')
    with (root/'console.log').open('a') as stream:
        stamp=time.strftime('%H:%M:%S',time.gmtime(record['time']))
        stream.write(f"{stamp} {record['event'].upper():<15} {clean(detail)}\n")

finished_games=set();finished_matches=set();score_delta=0
results_path=root/'results.jsonl'
if results_path.exists():
    valid=[];damaged=[]
    for index,encoded in enumerate(results_path.read_bytes().splitlines()):
        try:
            line=encoded.decode('utf-8')
            result=json.loads(line)
            if not isinstance(result,dict) or type(result.get('game')) is not int:
                raise ValueError('Invalid result record')
            if type(result.get('our_delta',0)) is not int:
                raise ValueError('Invalid result score')
            match=result.get('match')
            if match is not None and (not isinstance(match,dict) or
                (match.get('finished') and type(match.get('id')) is not int)):
                raise ValueError('Invalid result match')
        except (ValueError,TypeError) as error:
            damaged.append({'line':index+1,'error':str(error)});continue
        valid.append(line)
        if result['game'] in finished_games:continue
        finished_games.add(result['game'])
        score_delta+=result.get('our_delta',0)
        match=result.get('match')
        if match and match.get('finished'):finished_matches.add(match['id'])
    if damaged:
        backup=root/f'results.damaged.{time.time_ns()}.jsonl'
        backup.write_bytes(results_path.read_bytes())
        temporary=root/'results.repair.tmp'
        temporary.write_text('\n'.join(valid)+'\n');temporary.replace(results_path)
        event({'event':'ledger_repair','summary':f'Retained valid rows; damaged original saved to {backup}',
            'errors':damaged})
    elif results_path.stat().st_size and not results_path.read_bytes().endswith(b'\n'):
        with results_path.open('a') as stream:stream.write('\n')

status={'state':'warming','pid':os.getpid(),'tmux_session':cfg['tmux_session'],
    'username':cfg['username'],'mode':'match','backend':'cpu','client_version':8,
    'checkpoint':cfg['source_checkpoint'],'global_step':cfg['global_step'],
    'checkpoint_sha256':cfg['checkpoint_sha256'],'accepted_actions':0,'conflicts':0,
    'games_completed':len(finished_games),'matches_completed':len(finished_matches),
    'score_delta':score_delta,'fallback_decisions':0,'recoveries':0}

def publish(**fields):
    status.update(fields,time=time.time(),last_progress=time.time())
    atomic_json(root/'status.json',status)

class LoggedClient(Client):
    def on_recovery(self,error,game):
        message=str(error).replace(token,'[REDACTED]')
        publish(state='recovering' if status['state']=='warming' else status['state'],
            recoveries=status['recoveries']+1,last_recovery=message,
            last_recovery_time=time.time())
        event({'event':'recovery','game':game,'error':message})

    def request(self,path,payload=None):
        started=time.monotonic()
        try:code,data=super().request(path,payload)
        except (APIError,RetryDecision,OSError,urllib.error.URLError,http.client.HTTPException) as error:
            latency=time.monotonic()-started
            status['last_request_seconds']=latency
            event({'event':'transport_error','path':path,'action':payload,'request_seconds':latency,
                'error':str(error).replace(token,'[REDACTED]')})
            raise
        latency=time.monotonic()-started
        status.update(last_request_seconds=latency,last_http_completed=time.time())
        if payload is not None:
            if path.startswith('/games/'):
                if code==200:status['accepted_actions']+=1
                elif code==409:status['conflicts']+=1
            publish()
            event({'event':'request','path':path,'code':code,'action':payload,
                'error':data.get('error'),'request_seconds':latency,
                'summary':f"POST {path} HTTP={code} {latency*1000:.0f}ms {action_text(path.rsplit('/',1)[-1],payload)} error={data.get('error')}"})
        if path in ('/me','/queue') and code==200:
            if data.get('username') is not None and data['username']!=cfg['username']:
                raise APIError(401,'Unexpected authenticated account')
            publish(state='in_game' if data.get('game') else 'queued' if data.get('queued') else 'lobby',
                game=data.get('game'),queued=data.get('queued'),rating=data.get('rating',status.get('rating')),
                score=data.get('score',status.get('score')))
        raw=data.get('state')
        if isinstance(raw,dict) and not raw.get('unchanged'):
            raw=response_state(data)
            if type(raw.get('id')) is not int or type(raw.get('version')) is not int:
                raise RetryDecision('Full state lacks game ID/version')
            match=raw.get('match') or {}
            if not isinstance(match,dict):raise RetryDecision('Invalid match metadata')
            seat=raw.get('seat');players=raw.get('players') or []
            own=players[seat] if type(seat) is int and 0<=seat<len(players) else {}
            changed=(raw.get('id'),raw.get('version'))!=(status.get('game'),status.get('version'))
            publish(state='in_game',game=raw.get('id',status.get('game')),phase=raw['phase'],
                version=raw.get('version'),seat=seat,landlord=raw.get('landlord'),turn=raw.get('turn'),
                our_auto=own.get('auto'),remaining_ms=raw.get('remaining'),match_id=match.get('id'),
                match_round=match.get('round'),match_rounds=match.get('rounds'))
            atomic_json(root/'api_state.json',public_snapshot(raw))
            if changed:
                event({'event':'state','game':raw.get('id'),'version':raw.get('version'),
                    'summary':f"game={raw.get('id')} v{raw.get('version')} {raw['phase']} turn=S{raw.get('turn')} own=S{seat} cards={[p.get('count') for p in players]} auto={own.get('auto')} remaining={raw.get('remaining')}ms HTTP={latency*1000:.0f}ms"})
            if raw['phase']=='finished' and raw.get('id') not in finished_games:
                if type(seat) is not int or not 0<=seat<len(raw['result']['deltas']):
                    raise RetryDecision('Finished state has no own settlement')
                result={'game':raw['id'],'result':raw['result'],'match':raw.get('match'),
                    'time':time.time(),'global_step':cfg['global_step'],
                    'our_delta':raw['result']['deltas'][seat],
                    'our_auto_actions':sum(entry.get('seat')==seat and
                        entry.get('kind') in ('bid','play','pass') and entry.get('auto',False)
                        for entry in raw.get('log',()))}
                with results_path.open('a') as stream:
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                finished_games.add(raw['id']);status['score_delta']+=result['our_delta']
                if match.get('finished'):finished_matches.add(match['id'])
                publish(state='between_games',games_completed=len(finished_games),
                    matches_completed=len(finished_matches),last_result=result)
                event({'event':'game_finished',**result,
                    'summary':f"game={raw['id']} score={result['our_delta']:+} our_auto_actions={result['our_auto_actions']} match={match.get('id')} round={match.get('round')}/{match.get('rounds')}"})
        elif time.time()-status.get('time',0)>=2:
            if isinstance(raw,dict):status['remaining_ms']=raw.get('remaining')
            publish()
            event({'event':'heartbeat','summary':f"game={status.get('game')} v{status.get('version')} turn=S{status.get('turn')} remaining={status.get('remaining_ms')}ms HTTP={code}/{latency*1000:.0f}ms"})
        return code,data

def decision_record(raw,endpoint,payload,source,seconds,reason):
    inference=model.last_inference if source=='model' else None
    record={'event':'decision','game':raw['id'],'version':raw['version'],'endpoint':endpoint,
        'payload':payload,'decision_seconds':seconds,'remaining_ms':raw.get('remaining'),
        'source':source,'reason':reason,'inference':inference,'time':time.time()}
    if source=='fallback':
        snapshot=root/f"invalid_state_{raw['id']}_{raw['version']}.json"
        atomic_json(snapshot,public_snapshot(raw))
        publish(fallback_decisions=status['fallback_decisions']+1,last_parser_error=reason,
            last_invalid_state=str(snapshot))
    atomic_json(root/'last_decision.json',record)
    # Store large input tensors once in last_decision.json; concise values in logs.
    logged={k:v for k,v in record.items() if k!='inference'}
    values=None if inference is None else inference['display_value_by_seat']
    logged['value_by_seat']=values
    logged['raw_value_by_seat']=None if inference is None else inference['value_by_seat']
    if source=='model' and model.last_bid_diagnostics:logged.update(model.last_bid_diagnostics)
    logged['summary']=f"game={raw['id']} v{raw['version']} {source} {seconds*1000:.1f}ms {action_text(endpoint,payload)} value={values} remaining={raw.get('remaining')}ms"+(f' reason={reason}' if reason else '')
    event(logged)
    publish(last_decision_game=raw['id'],last_decision_version=raw['version'],
        last_decision_seconds=seconds,last_value_by_seat=values)

publish()
try:
    model=Policy(model_dir/'config.json',model_dir/'policy.msgpack')
    model.capture_observation=True
    client=LoggedClient(cfg['base'],token)
    # Startup transient failures reuse the already compiled model too.
    while True:
        try:
            code,info=checked_request(client,'/info')
            code,me=checked_request(client,'/me')
            if me.get('username')!=cfg['username']:raise APIError(401,'Unexpected authenticated account')
            if type(info.get('match_rounds')) is not int:raise RetryDecision('Missing server match metadata')
            break
        except APIError as error:
            if error.code==401:raise
            client.on_recovery(error,None);time.sleep(1)
        except (RetryDecision,OSError,urllib.error.URLError,http.client.HTTPException) as error:
            client.on_recovery(error,None)
            time.sleep(error.delay if isinstance(error,TemporaryAPIError) else 1)
    publish(state='ready',server_match_rounds=info['match_rounds'])
    event({'event':'started','summary':f"{cfg['username']} client=8 CPU V5 step={cfg['global_step']} rounds={info['match_rounds']} warmup completed"})
    run(client,GuardedPolicy(model,client,decision_record),mode='match',once=False)
except Exception as error:
    message=str(error).replace(token,'[REDACTED]')
    publish(state='failed',error=message)
    event({'event':'failed','error':message})
    if isinstance(error,APIError) and error.code==401:raise SystemExit(78)
    raise
