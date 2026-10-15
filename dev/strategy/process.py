"""Bounded experiment children; never terminate another GPU workload."""
import os
from pathlib import Path
import signal
import subprocess
import time


def wait_process(command, log, held, timeout):
    with Path(log).open('x') as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
            pass_fds=(held.fileno(),), start_new_session=True)
        start = time.monotonic()
        try:
            while True:
                remaining = timeout-(time.monotonic()-start)
                if remaining <= 0:
                    if p.poll() is None:
                        if os.getpgid(p.pid) != p.pid: raise RuntimeError('owned process group changed')
                        os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    return dict(rc=p.returncode, timeout=True, seconds=time.monotonic()-start)
                try:
                    return dict(rc=p.wait(timeout=min(30, remaining)), timeout=False,
                                seconds=time.monotonic()-start)
                except subprocess.TimeoutExpired:
                    print(f'STRATEGY_CUBLAS_WAIT log={log} seconds={time.monotonic()-start:.1f}', flush=True)
        finally:
            if p.poll() is None:
                if os.getpgid(p.pid) != p.pid: raise RuntimeError('owned process group changed')
                os.killpg(p.pid, signal.SIGKILL); p.wait()
