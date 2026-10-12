"""Flushed stage messages, heartbeat JSON and bounded waits; stdlib only."""
import contextlib
import datetime
import faulthandler
import json
import os
from pathlib import Path
import threading
import time


def log(message):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}] '
          f'job={os.getenv("SLURM_JOB_ID","local")} {message}', flush=True)


@contextlib.contextmanager
def stage(name, *, timeout=None, tag='data'):
    started = time.monotonic()
    stop = threading.Event()
    root = os.getenv('C3D_LATENT_PROGRESS_DIR')
    path = None
    if root:
        path = Path(root)/f'{os.getenv("SLURM_JOB_ID","local")}_{os.getenv("SLURM_ARRAY_TASK_ID","main")}_{tag}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
    def report(state):
        row = {'stage':name,'state':state,'elapsed_seconds':round(time.monotonic()-started,1),
               'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'job_id':os.getenv('SLURM_JOB_ID','local'),'pid':os.getpid()}
        if path:
            tmp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
            tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)
        return row
    interval = max(.05,float(os.getenv('LATENT_HEARTBEAT_SECONDS','15')))
    def beat():
        while not stop.wait(interval):
            log(f'RUNNING {name} ({time.monotonic()-started:.0f}s)')
            try: report('running')
            except OSError as exc: log(f'Progress file unavailable: {exc}')
    log('START '+name)
    report('running')
    thread = threading.Thread(target=beat,daemon=True)
    thread.start()
    seconds = float(timeout if timeout is not None else os.getenv('LATENT_STAGE_TIMEOUT_SECONDS','1800'))
    # One active stage per process. On timeout emit all thread stacks and exit;
    # Slurm dependencies then prevent downstream work from starting.
    if seconds > 0: faulthandler.dump_traceback_later(seconds,exit=True)
    state = 'complete'
    try:
        yield
    except BaseException:
        state = 'failed'
        raise
    finally:
        if seconds > 0: faulthandler.cancel_dump_traceback_later()
        stop.set();thread.join(timeout=1)
        report(state)
        log(f'{state.upper()} {name} ({time.monotonic()-started:.1f}s)')
