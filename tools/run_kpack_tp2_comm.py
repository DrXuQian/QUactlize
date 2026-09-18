#!/usr/bin/env python3
"""Isolate TP2 local arithmetic from the unchanged caller collective."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def cases():
    return [(arm, count, extension) for extension in ('inherited', 'disabled')
            for arm, count in ((('copy', 512), ('copy', 3072), ('copy', 32768), ('raw', 512), ('kpack', 512))
                               if extension == 'inherited' else (('copy', 512), ('kpack', 512)))]


def evidence(text, arm, count, rc):
    local = re.findall(r'^KPACK_TP2_COMM_LOCAL arm=(\w+) rank=([01]) replay=([012]) error=(\S+) synchronized=1 status=PASS$', text, re.M)
    sums = re.findall(r'^KPACK_TP2_COMM_SUM arm=(\w+) rank=([01]) replay=([012]) error=(\S+) status=PASS$', text, re.M)
    complete = set((arm, str(rank), str(replay)) for rank in (0, 1) for replay in range(3))
    pre_reduce = {(a, r, i) for a, r, i, _ in local}
    begin = re.findall(r'^KPACK_TP2_COMM_REDUCE_BEGIN arm=(\w+) count=(\d+) replay=([012])$', text, re.M)
    first_locals_pass = {(arm, '0', '0'), (arm, '1', '0')} <= pre_reduce
    fallback = bool(re.search(r'falling back to internal AllReduce|NCCL disabled|unknown GGML_CUDA_ALLREDUCE', text))
    def valid_error(value):
        try:
            return math.isfinite(float(value)) and (float(value) == 0 if arm == 'copy' else 0 <= float(value) < .02)
        except ValueError:
            return False
    success = (rc == 0 and len(local) == len(sums) == 6 and pre_reduce == complete and
               {(a, r, i) for a, r, i, _ in sums} == complete and
               all(valid_error(error) for *_, error in local + sums) and
               begin == [(arm, str(count), str(i)) for i in range(3)] and not fallback and
               text.count(f'KPACK_TP2_COMM PASS arm={arm} count={count} replays=3\n') == 1)
    if success:
        status = 'PASS'
    elif fallback:
        status = 'UNEXPECTED_COMMUNICATION_FALLBACK'
    elif first_locals_pass and begin:
        status = 'COLLECTIVE_FAILED_AFTER_LOCAL_PASS'
    elif 'KPACK_TP2_COMM_LOCAL_BEGIN' in text:
        status = 'LOCAL_PRODUCER_OR_ORACLE_FAILED'
    else:
        status = 'SETUP_FAILED'
    if rc == 0 and not success:
        status = 'COVERAGE_FAILED'
    return dict(status=status, local_records=len(local), sum_records=len(sums),
                local_before_collective_pass=first_locals_pass,
                extension_invalid_function='drv_extension.cc' in text and 'invalid device function' in text,
                libraries=sorted(set(re.findall(r'^KPACK_TP2_COMM_LIBRARY path=(.+)$', text, re.M))))


def run(binary, output, timeout=180):
    if os.environ.get('GGML_CUDA_ALLREDUCE') not in (None, 'nccl'):
        raise ValueError('communication diagnostic requires the existing NCCL path; unset GGML_CUDA_ALLREDUCE or set nccl')
    output.mkdir(parents=True, exist_ok=False)
    base = os.environ.copy()
    base.update(NCCL_DEBUG='INFO', PCCL_DEBUG='INFO')
    keys = ('CUDA_VISIBLE_DEVICES', 'PPU_SDK', 'LD_LIBRARY_PATH', 'GGML_CUDA_ALLREDUCE',
            'PCCL_ENABLE_EXT_KERNEL', 'PCCL_EXT_KERNEL_PLUGIN', 'PCCL_ALGO', 'PCCL_PROTO',
            'QUACTLIZE_KPACK_EXECUTION', 'QUACTLIZE_PPU_PACK_LIBRARY', 'QUACTLIZE_PPU_BUNDLE')
    (output / 'environment.json').write_text(json.dumps({k: base.get(k) for k in keys}, indent=2) + '\n')
    results = []
    for arm, count, extension in cases():
        name = f'{arm}-{count}-{extension}'
        command = [str(binary), '--tp2-comm', arm, str(count)]
        env = dict(base)
        if extension == 'disabled':
            env['PCCL_ENABLE_EXT_KERNEL'] = '0'
        log = output / (name + '.log')
        print(f'KPACK_TP2_COMM_START case={name} log={log}', flush=True)
        started = time.monotonic()
        timed_out = False
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = process.wait()
        result = dict(arm=arm, count=count, extension=extension, rc=rc,
                      seconds=time.monotonic()-started, timeout=timed_out, command=command,
                      **evidence(log.read_text(errors='replace'), arm, count, rc))
        if timed_out:
            result['status'] = 'TIMEOUT'
        results.append(result)
        print('KPACK_TP2_COMM_RESULT ' + json.dumps(result), flush=True)
        (output / 'summary.json').write_text(json.dumps(dict(status='RUNNING', cases=results), indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',
        device_admission='PENDING', production_communication='UNCHANGED', cases=results), indent=2) + '\n')
    print(f'KPACK_TP2_COMM_DONE cases={len(results)} verdict=DIAGNOSTIC_COMPLETE device_admission=PENDING results={output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    run(args.binary.resolve(strict=True), args.output, args.timeout)
