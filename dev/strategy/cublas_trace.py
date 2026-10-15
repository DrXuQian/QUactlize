"""No-Graph cuBLAS measurement: synchronized HGTX ranges, GPU activity times.

Events around the provider are an auxiliary API-span check, never a GEMM or
reducer timer. Every GPU kernel is retained, including unclassified entries.
"""
from contextlib import contextmanager
import ctypes as C
import json
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import time
import uuid

from dev.strategy.components import stage_role
from quactlize.runtime.compiler import sha

PREFIX = 'STRATEGY_CUBLAS/'
PHASES = ('input', 'provider', 'output')


def label(round_, copy, phase):
    if phase not in PHASES or min(round_, copy) < 0:
        raise ValueError('invalid cuBLAS range identity')
    return f'{PREFIX}{round_}/{copy}/{phase}'


class Ranges:
    def __init__(self, sdk):
        path = Path(sdk)/'asight/lib/libhgToolsExt.so'
        self.lib = C.CDLL(str(path.resolve(strict=True)))
        self.push = self.lib.hgtxRangePushA
        self.push.argtypes, self.push.restype = [C.c_char_p], C.c_int
        self.pop = self.lib.hgtxRangePop
        self.pop.argtypes, self.pop.restype = [], C.c_int
        self.identity = dict(path=str(path), sha256=sha(path))

    @contextmanager
    def range(self, name, sync):
        # No pending work from another sample may enter this range. Pop only
        # after device completion: CPU/GPU time containment is then explicit.
        sync()
        self.push(name.encode())
        try:
            yield
            sync()
        finally:
            self.pop()


def profile_command(asys, report, application):
    return [str(asys), 'profile', '--trace', 'hggc,hgtx', '--hggc-trace-set', 'kernel-activity',
            '--sample', 'none', '--python-sampling', 'false', '--wait', 'primary',
            '--kill', 'sigterm', '--show-output', 'true', '--force-overwrite', 'false',
            '--session-new', 'quactlize-cublas-'+uuid.uuid4().hex,
            '--output', str(report), *application]


def wait_process(command, log, held, timeout):
    """Timeout only this fresh process group, including its profiler worker."""
    with Path(log).open('x') as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
            pass_fds=(held.fileno(),), start_new_session=True)
        start = time.monotonic()
        try:
            while True:
                remaining = timeout-(time.monotonic()-start)
                if remaining <= 0:
                    # start_new_session makes this PID the owned process-group
                    # leader. No pkill/device reset/shared profiler shutdown.
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


def parse_trace(path, expected):
    """Read *device* activities only. Never fall back to host API durations.

    Each marked phase is stream-synchronized before range end. Missing ranges,
    empty GPU activity/calibration, overlaps, or partial device intervals fail
    closed. No kernel-count equality is used as a tuning-completion oracle.
    """
    if not expected or len({r['label'] for r in expected}) != len(expected):
        raise ValueError('expected range inventory is empty/duplicate')
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        columns = lambda table: {r[1] for r in con.execute('PRAGMA table_info('+table+')')}
        kc = columns('HGPTI_ACTIVITY_KIND_KERNEL')
        rc = columns('HGTX_EVENTS')
        required = {'start', 'end', 'demangledName', 'deviceId', 'contextId', 'streamId',
                    'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ'}
        if not required <= kc or not {'start', 'end', 'text', 'textId', 'globalTid', 'eventType'} <= rc:
            raise ValueError('Asys GPU kernel/HGTX schema missing; no host-timing fallback: '+
                             repr(dict(kernel=sorted(kc), ranges=sorted(rc))))
        strings = dict(con.execute('SELECT id,value FROM StringIds'))
        ranges = []
        for row in con.execute('SELECT * FROM HGTX_EVENTS'):
            name = row['text'] if row['text'] is not None else strings.get(row['textId'])
            if isinstance(name, str) and name.startswith(PREFIX):
                if row['eventType'] not in (59, 60, 70, 71) or row['end'] is None or row['end'] <= row['start']:
                    raise ValueError('incomplete measured HGTX range: '+name)
                ranges.append(dict(label=name, start=row['start'], end=row['end'], tid=row['globalTid']))
        if (len(ranges) != len(expected) or {r['label'] for r in ranges} != {r['label'] for r in expected}):
            raise ValueError(f'Asys measured range coverage differs: {len(ranges)}/{len(expected)}')
        if len({r['label'] for r in ranges}) != len(ranges):
            raise ValueError('duplicate measured HGTX range')
        ranges.sort(key=lambda r: r['start'])
        if len({r['tid'] for r in ranges}) != 1 or any(
                a['end'] > b['start'] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('measured ranges overlap or use multiple host threads')
        kernels = list(con.execute('SELECT rowid,* FROM HGPTI_ACTIVITY_KIND_KERNEL ORDER BY start,end,rowid'))
        if not kernels:
            raise ValueError('empty device kernel activity; check Asys time calibration')
        expected_by_name = {r['label']: r for r in expected}
        result, devices, contexts, streams = [], set(), set(), set()
        for span in ranges:
            measured = []
            for k in kernels:
                if k['end'] <= span['start'] or k['start'] >= span['end']:
                    continue
                if not span['start'] <= k['start'] < k['end'] <= span['end']:
                    raise ValueError('GPU interval crosses synchronized HGTX boundary; no timing admitted')
                name = strings.get(k['demangledName'])
                if not name:
                    raise ValueError('GPU kernel lacks a resolved name')
                devices.add(k['deviceId']); contexts.add(k['contextId']); streams.add(k['streamId'])
                measured.append(dict(name=name, role=stage_role(name, 1),
                    symbol=strings.get(k['mangledName']) if 'mangledName' in kc else None,
                    grid=[k['grid'+d] for d in 'XYZ'], block=[k['block'+d] for d in 'XYZ'],
                    shared_bytes={c: k[c] for c in ('staticSharedMemory', 'dynamicSharedMemory') if c in kc},
                    start_ns=k['start'], end_ns=k['end'], duration_us=(k['end']-k['start'])/1000))
            if not measured:
                raise ValueError('no GPU kernel in measured range '+span['label'])
            if any(a['end_ns'] > b['start_ns'] for a, b in zip(measured, measured[1:])):
                raise ValueError('overlapping kernel activities cannot be composed as serial work')
            item = expected_by_name[span['label']]
            if item['label'] != label(item['round'], item['sample'], item['phase']):
                raise ValueError('expected range label/fields disagree')
            if item['phase'] != 'provider' and len(measured) != 1:
                raise ValueError('adapter range has unexpected kernel count')
            result.append(dict(**item, kernels=measured,
                kernel_sum_us=sum(k['duration_us'] for k in measured),
                device_span_us=(measured[-1]['end_ns']-measured[0]['start_ns'])/1000,
                host_range_us=(span['end']-span['start'])/1000))
        if len(devices) != 1 or len(contexts) != 1 or len(streams) != 1:
            raise ValueError('measured work escaped the single device/context/stream contract')
    return dict(status='PASS', ranges=result, sqlite_sha256=sha(path),
                device=list(devices)[0], context=list(contexts)[0], stream=list(streams)[0],
                scope='INDIVIDUAL_GPU_ACTIVITIES_FROM_ORDINARY_CALLS', cuda_graph=False,
                warmup_excluded=True, host_launch_cost_added=False)


def summarize_trace(trace):
    """Retain all internal stages even if cuBLAS changes its dispatch."""
    inventory, totals = {}, {p: [] for p in PHASES}
    for row in trace['ranges']:
        totals[row['phase']].append(row['kernel_sum_us'])
        for k in row['kernels']:
            key = json.dumps([row['phase'], k['name'], k['symbol'], k['grid'], k['block'],
                              k['shared_bytes']], sort_keys=True)
            item = inventory.setdefault(key, dict(phase=row['phase'],
                **{f: k[f] for f in ('name', 'symbol', 'role', 'grid', 'block', 'shared_bytes')}, samples_us=[]))
            item['samples_us'].append(k['duration_us'])
    for item in inventory.values():
        item['instances'] = len(item['samples_us'])
        item['median_us'] = statistics.median(item['samples_us'])
    return dict(kernels=list(inventory.values()),
        phase_kernel_sum_median_us={p: statistics.median(v) for p, v in totals.items()},
        cost_scope='SUM_OF_GPU_ACTIVITIES_NOT_API_SPAN_NOT_MODEL_E2E',
        unknown_kernels=[i['name'] for i in inventory.values() if i['role'] == 'unclassified'])
