"""Per-rank startup stages and a native traceback watchdog (no CUDA work)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import time


class StartupMonitor:
    def __init__(self, run_dir):
        self.run = Path(run_dir)
        self.rank = int(os.getenv('RANK', '0'))
        self.timeout = float(os.getenv('AE_STARTUP_TIMEOUT_SECONDS', '900'))
        if self.timeout <= 0:
            raise ValueError('AE_STARTUP_TIMEOUT_SECONDS must be positive')

    def _record(self, stage, state, elapsed, error=None):
        stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        print(f'[AE startup {stamp} rank={self.rank}] {stage}: {state} '
              f'elapsed={elapsed:.2f}s' + (f' error={error}' if error else ''), flush=True)
        record = dict(stage=stage, state=state, elapsed_seconds=elapsed,
                      updated_unix=time.time(), rank=self.rank, pid=os.getpid(),
                      job_id=os.getenv('SLURM_JOB_ID', 'local'),
                      timeout_seconds=self.timeout)
        if error: record['error'] = error
        path = self.run / 'control' / f'startup_rank_{self.rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)
        if state == 'FAILED':
            failure = path.with_name(f'startup_failure_rank_{self.rank}.json')
            tmp = failure.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(failure)

    @contextmanager
    def stage(self, name):
        # This C-level watchdog still works if Python is blocked in native code.
        # torchrun tears down sibling ranks when this process exits with code 1.
        start = time.monotonic()
        faulthandler.dump_traceback_later(self.timeout, file=2, exit=True)
        try:
            self._record(name, 'START', 0.)
            yield
        except BaseException as exc:
            self._record(name, 'FAILED', time.monotonic()-start, repr(exc))
            raise
        else:
            self._record(name, 'DONE', time.monotonic()-start)
        finally:
            faulthandler.cancel_dump_traceback_later()

    def batches(self, loader):
        # Diagnose first read each epoch; later batches retain the usual logging.
        with self.stage('first volume batch / DataLoader'):
            iterator = iter(loader)
            batch = next(iterator, None)
        if batch is not None:
            yield batch
        yield from iterator
