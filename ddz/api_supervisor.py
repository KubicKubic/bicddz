"""Own one worker, recover hangs, and render a read-only terminal / HTML view."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .api_dashboard import load_json,render_terminal,render_html

def stalled(status,pid,started,now,warmup=45,timeout=12):
    if status.get('pid')!=pid or status.get('state')=='warming':
        return now-started>warmup
    return now-status.get('last_progress',status.get('time',started))>timeout

def terminate(process):
    # The worker owns a new process group; do not signal unrelated jobs.
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()

def run(root,worker):
    root=Path(root); process=None; next_start=0; started=0; stop=False; last_html=0
    tty=sys.stdout.isatty()
    def note(message):
        with (root/'supervisor.log').open('a') as stream:
            stream.write(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message+'\n')
    def request_stop(*args):
        nonlocal stop
        stop=True
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,request_stop)
    if tty:sys.stdout.write('\033[?25l\033[2J');sys.stdout.flush()
    try:
        while not stop:
            now=time.time()
            if process is None and now>=next_start:
                with (root/'worker.log').open('ab',buffering=0) as stream:
                    process=subprocess.Popen([sys.executable,'-u',str(root/worker)],
                        cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                started=now;note(f'worker started pid={process.pid}')
            status=load_json(root/'status.json')
            if process is not None:
                code=process.poll()
                if code is not None:
                    note(f'worker exited pid={process.pid} code={code}')
                    if code in (75,78):return code
                    process=None;next_start=now+2
                elif stalled(status,process.pid,started,now):
                    note(f'watchdog stale progress; terminate owned worker pid={process.pid}')
                    terminate(process);process=None;next_start=now+1
            state=load_json(root/'api_state.json');decision=load_json(root/'last_decision.json')
            size=shutil.get_terminal_size((120,28))
            try:text=render_terminal(status,state,decision,size.columns,size.lines,now)
            except (ValueError,TypeError,KeyError,IndexError) as error:
                text=f'DDZ dashboard recovering from incomplete display data: {error}'
            if tty:
                lines=text.splitlines()
                title='\033[1;36m'+lines[0]+'\033[0m' if lines else ''
                sys.stdout.write('\033[H'+title+'\033[K\n'+'\n'.join(line+'\033[K' for line in lines[1:])+'\033[J')
                sys.stdout.flush()
            if now-last_html>=2:
                try:
                    temporary=root/'dashboard.html.tmp'
                    temporary.write_text(render_html(status,state,decision))
                    temporary.replace(root/'dashboard.html')
                except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
                    note(f'dashboard write error: {error}')
                last_html=now
            time.sleep(.5)
    finally:
        if process is not None:terminate(process)
        if tty:sys.stdout.write('\033[0m\033[?25h\n');sys.stdout.flush()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',default='worker_v6.py')
    args=parser.parse_args();raise SystemExit(run(args.root,args.worker))

if __name__=='__main__':main()
