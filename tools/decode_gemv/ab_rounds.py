#!/usr/bin/env python3
"""Run qgemv_bench arms under Asys on one idle PPU, alternating arms per round.

For each round r and each arm (in alternating order), wait until the physical device has
no foreign compute process, run one Asys capture of the arm over the case list, export
SQLite, attribute kernels, and record whether a foreign process appeared during the
capture (contaminated rounds are repeated, never mixed into results).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def foreign(dev):
    text = subprocess.run(['ppu-smi'], capture_output=True, text=True).stdout
    found = []
    for m in re.finditer(r'^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+\S+\s+(.+?)\s+(\d+)MiB\s*\|', text, re.M):
        if int(m[1]) == dev:
            found.append((int(m[2]), m[3].strip()))
    return found


def own_pids():
    pids = {os.getpid()}
    try:
        out = subprocess.run(['ps', '-e', '-o', 'pid=,ppid='], capture_output=True, text=True).stdout
        tree = {}
        for line in out.split('\n'):
            parts = line.split()
            if len(parts) == 2:
                tree.setdefault(int(parts[1]), []).append(int(parts[0]))
        stack = [os.getpid()]
        while stack:
            p = stack.pop()
            for c in tree.get(p, []):
                if c not in pids:
                    pids.add(c)
                    stack.append(c)
    except Exception:
        pass
    return pids


def wait_idle(dev, max_wait, quiet=60):
    start = time.time()
    clear_since = None
    while True:
        busy = [p for p in foreign(dev) if p[0] not in own_pids()]
        now = time.time()
        if not busy:
            clear_since = clear_since or now
            if now - clear_since >= quiet:
                return True
        else:
            clear_since = None
        if now - start > max_wait:
            return False
        time.sleep(5)


class Monitor(threading.Thread):
    def __init__(self, dev):
        super().__init__(daemon=True)
        self.dev, self.seen, self.stop = dev, set(), False

    def run(self):
        while not self.stop:
            mine = own_pids()
            for pid, name in foreign(self.dev):
                if pid not in mine:
                    self.seen.add((pid, name))
            time.sleep(2)


def capture(arm, package, cases, out_dir, dev, bench_args, env):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / 'results.jsonl'
    if results.exists():
        results.unlink()
    base = out_dir / 'trace'
    cmd = ['asys', 'profile', '-t', 'hggc', '--force-overwrite', 'true', '-o', str(base),
           sys.executable, str(HERE / 'qgemv_bench.py'), '--package', package, '--cases', str(cases),
           '--out', str(results), *bench_args]
    mon = Monitor(dev)
    mon.start()
    t0 = time.time()
    with open(out_dir / 'bench.log', 'w') as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    mon.stop = True
    elapsed = time.time() - t0
    rep = next(iter(sorted(out_dir.glob('trace*.asysrep'))), None)
    if rc or rep is None:
        return dict(status='CAPTURE_FAIL', rc=rc, seconds=elapsed)
    sqlite = out_dir / 'trace.sqlite'
    subprocess.run(['asys', 'export', '--force-overwrite', 'true', '-o', str(sqlite), str(rep)], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    parsed = out_dir / 'parsed.jsonl'
    with open(out_dir / 'parse.log', 'w') as log:
        subprocess.run([sys.executable, str(HERE / 'parse_asys.py'), '--results', str(results), '--sqlite',
                        str(sqlite), '--out', str(parsed)], stdout=log, stderr=subprocess.STDOUT, check=True)
    sqlite.unlink()  # parsed rows + .asysrep are retained; SQLite is reproducible from the report
    return dict(status='OK', seconds=elapsed, foreign=sorted(mon.seen), report=str(rep))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='append', required=True, help='name=package_dir')
    parser.add_argument('--cases', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='0', help='device or comma list; the first idle one is used for ALL rounds')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--max-wait', type=float, default=6 * 3600)
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--quiet', type=float, default=45)
    parser.add_argument('bench_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arms = [a.split('=', 1) for a in args.arm]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = [int(d) for d in str(args.device).split(',')]
    chosen = out / 'device.txt'
    if chosen.exists():
        device = int(chosen.read_text())
    else:
        start = time.time()
        device = None
        while device is None:
            for d in devices:
                # Instant idle probe first (max_wait=0 needs quiet=0), then the quiet period.
                if wait_idle(d, 0, 0) and wait_idle(d, args.quiet, args.quiet):
                    device = d
                    break
            if device is None:
                if time.time() - start > args.max_wait:
                    print('no device became idle; stopping', flush=True)
                    sys.exit(3)
                time.sleep(10)
        chosen.write_text(str(device))
    args.device = device
    print('device', device, flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    env.setdefault('QUACTLIZE_PPU_CACHE_DIR', '/dev/shm/qgemv/jit-cache')
    journal = out / 'journal.jsonl'
    bench_args = [a for a in args.bench_args if a != '--']
    for r in range(args.rounds):
        order = arms if r % 2 == 0 else arms[::-1]
        for name, package in order:
            target = out / f'round{r}' / name
            if (target / 'DONE').exists():
                continue
            for attempt in range(args.retries):
                if not wait_idle(args.device, args.max_wait, args.quiet):
                    print('device never idle; stopping', flush=True)
                    sys.exit(3)
                rec = capture(name, package, Path(args.cases), target, args.device, bench_args, env)
                rec.update(round=r, arm=name, attempt=attempt, time=time.strftime('%FT%T'))
                with open(journal, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                print(json.dumps(rec), flush=True)
                if rec['status'] == 'OK' and not rec['foreign']:
                    (target / 'DONE').write_text(json.dumps(rec) + '\n')
                    break
                bad = target.with_name(f'{name}.contaminated{attempt}')
                if bad.exists():
                    shutil.rmtree(bad)
                target.rename(bad)
            else:
                print(f'round {r} arm {name}: no clean capture', flush=True)
                sys.exit(4)


if __name__ == '__main__':
    main()
