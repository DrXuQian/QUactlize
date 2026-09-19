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


def cases(wrapper_ab=False):
    result = [(arm, count, extension, 'none') for extension in ('inherited', 'disabled')
            for arm, count in ((('copy', 512), ('copy', 3072), ('copy', 32768), ('raw', 512), ('kpack', 512))
                               if extension == 'inherited' else (('copy', 512), ('kpack', 512)))]
    if wrapper_ab:
        result += [('copy', 512, 'inherited', 'local'), ('copy', 512, 'inherited', 'global'),
                   ('kpack', 512, 'inherited', 'global')]
    return result


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
                wrapper_scopes=re.findall(r'^KPACK_TP2_COMM_WRAPPER scope=(\w+) path=(.+)$', text, re.M),
                global_symbols=sorted(set(re.findall(r'^KPACK_TP2_COMM_SYMBOL (.+)$', text, re.M))),
                libraries=sorted(set(re.findall(r'^KPACK_TP2_COMM_LIBRARY path=(.+)$', text, re.M))))


def wrapper_verdict(results):
    if len(results) != 10:
        return 'NOT_RUN'
    if [(r['arm'], r['count'], r['extension'], r['wrapper_scope']) for r in results] != cases(True):
        return 'COVERAGE_FAILED'
    if any(r['status'] != 'PASS' for r in results[:8]):
        return 'CANDIDATE_NOT_ADMITTED'
    if any(len(r['wrapper_scopes']) != 1 or r['wrapper_scopes'][0][0] != r['wrapper_scope']
           for r in results[7:]):
        return 'COVERAGE_FAILED'
    if len({r['wrapper_scopes'][0][1] for r in results[7:]}) != 1:
        return 'COVERAGE_FAILED'
    if any(r['wrapper_scopes'] for r in results[:7]):
        return 'COVERAGE_FAILED'
    if not any('phase=after-wrapper name=hggcLaunchKernel library=' in s and 'libhggc_wrapper.so' not in s
               for s in results[7]['global_symbols']):
        return 'COVERAGE_FAILED'
    for r in results[8:]:
        if not (r['status'] == 'COLLECTIVE_FAILED_AFTER_LOCAL_PASS' and r['extension_invalid_function'] and
                any('phase=after-wrapper name=hggcLaunchKernel library=' in s and 'libhggc_wrapper.so' in s
                    for s in r['global_symbols'])):
            return 'CANDIDATE_PASS_GLOBAL_NEGATIVE_NOT_REPRODUCED'
    return 'GLOBAL_WRAPPER_CAUSAL_CANDIDATE_PASS'


def q4_evidence(text, arm, rc):
    if arm in ('kpack','meta'):
        selected = re.findall(r'^\[quactlize-plan\] tensor=tp-weight op=grouped route=gemv reader=simt-reuse q=12 rows=2 n=512 k=512 variant=0 columns=4 warps=4 values=4 split=1 policy=11 activation=BF16 scale_resident=0 experts=4 channels=2 topk=2$', text, re.M)
        devices = re.findall(r'^\[quactlize-device-plan\] tensor=tp-weight device=([01]) op=grouped q=12 rows=2 n=512 k=512 experts=4 selected=1$', text, re.M)
        if len(selected) != 2 or sorted(devices) != ['0','1']:
            return dict(status='SELECTION_COVERAGE_FAILED')
    if arm == 'meta':
        mismatch = re.findall(r'^KPACK_TP2_MISMATCH q=12 experts=4 tokens=1 split=K replay=(\d+) relative=(\S+)$', text, re.M)
        complete = 'KPACK_TP2_SINGLE PASS q=12 experts=4 tokens=1 split=K replays=3\n'
        status = 'PASS' if rc == 0 and text.count(complete) == 1 else 'SETUP_OR_COVERAGE_FAILED'
        if rc != 0 and mismatch:
            status = 'META_NUMERICAL_FAILED'
        return dict(status=status, mismatch=mismatch)
    result = evidence(text, arm, 1024, rc)
    shape = 'KPACK_TP2_COMM_SHAPE q=12 n=512 global_k=1024 local_k=512 experts=4 tokens=1 topk=2\n'
    result['weights'] = re.findall(r'^KPACK_TP2_COMM_WEIGHT rank=([01]) q=12 bytes=589824 hash=([0-9a-f]{16})$', text, re.M)
    result['oracles'] = re.findall(r'^KPACK_TP2_COMM_ORACLE rank=([01]) replay=([012]) input_hash=([0-9a-f]{16}) ids_hash=([0-9a-f]{16}) golden_hash=([0-9a-f]{16})$', text, re.M)
    result['local_errors'] = re.findall(r'^KPACK_TP2_COMM_LOCAL arm=\w+ rank=([01]) replay=([012]) error=(\S+) synchronized=1 status=(PASS|FAIL)$', text, re.M)
    if (text.count(shape) != 1 or len(result['weights']) != 2 or
            {r for r, _ in result['weights']} != {'0','1'}):
        result['status'] = 'FIXTURE_IDENTITY_FAILED'
    if result['status'] == 'PASS' and (len(result['oracles']) != 6 or
            {(r,i) for r,i,*_ in result['oracles']} != {(str(r),str(i)) for r in (0,1) for i in range(3)}):
        result['status'] = 'COVERAGE_FAILED'
    return result


def q4_verdict(results):
    if [r['arm'] for r in results] != ['raw','kpack','meta']:
        return 'COVERAGE_FAILED'
    raw, native, meta = results
    if raw['status'] != 'PASS':
        return 'RAW_CONTROL_NOT_ADMITTED'
    if (sorted(raw.get('weights',[])) != sorted(native.get('weights',[])) or
            not native.get('oracles') or
            any(tuple(x) not in {tuple(y) for y in raw['oracles']} for x in native['oracles'])):
        return 'FIXTURE_IDENTITY_FAILED'
    if native['status'] == 'LOCAL_PRODUCER_OR_ORACLE_FAILED':
        return 'LOCAL_KPACK_PACKING_OR_COMPUTE_FAILED'
    if native['status'] != 'PASS':
        return 'LOCAL_OR_COLLECTIVE_NOT_ADMITTED'
    if meta['status'] == 'META_NUMERICAL_FAILED':
        return 'META_PATH_ONLY_FAILURE'
    return 'FRESH_PROCESS_NOT_REPRODUCED' if meta['status'] == 'PASS' else 'META_COVERAGE_FAILED'


def run_q4(binary, output, timeout=180):
    if os.environ.get('GGML_CUDA_ALLREDUCE') not in (None, 'nccl'):
        raise ValueError('keep the original PCCL path for the Q4 local diagnostic')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, QUACTLIZE_KPACK_COMPUTE='bf16', QUACTLIZE_KPACK_ROUTE='auto')
    keys = ('CUDA_VISIBLE_DEVICES','PPU_SDK','LD_LIBRARY_PATH','QUACTLIZE_PPU_BUNDLE',
            'QUACTLIZE_KPACK_EXECUTION','QUACTLIZE_KPACK_COMPUTE','QUACTLIZE_KPACK_ROUTE')
    (output/'environment.json').write_text(json.dumps({k:env.get(k) for k in keys},indent=2)+'\n')
    results = []
    for arm in ('raw','kpack','meta'):
        command = [str(binary),'--tp2-q4-meta'] if arm == 'meta' else [str(binary),'--tp2-q4-local',arm]
        log = output/(arm+'.log')
        print(f'KPACK_TP2_Q4_START arm={arm} log={log}',flush=True)
        started = time.monotonic()
        timed_out = False
        with log.open('w') as stream:
            process = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                rc = process.wait()
            except BaseException:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                raise
        result = dict(arm=arm,rc=rc,timeout=timed_out,seconds=time.monotonic()-started,command=command,
                      **q4_evidence(log.read_text(errors='replace'),arm,rc))
        if timed_out: result['status']='TIMEOUT'
        results.append(result)
        (output/'summary.json').write_text(json.dumps(dict(status='RUNNING',cases=results),indent=2)+'\n')
        print('KPACK_TP2_Q4_RESULT '+json.dumps(result),flush=True)
    verdict = q4_verdict(results)
    (output/'summary.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',verdict=verdict,
        device_admission='PENDING',scope='Q4_E4_TOP2_M1_K_SPLIT_ONLY',cases=results),indent=2)+'\n')
    print(f'KPACK_TP2_Q4_DONE verdict={verdict} results={output}',flush=True)


def run(binary, output, timeout=180, wrapper_ab=False):
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
    for arm, count, extension, scope in cases(wrapper_ab):
        name = f'{arm}-{count}-{extension}' + (f'-wrapper-{scope}' if scope != 'none' else '')
        command = [str(binary), '--tp2-comm', arm, str(count)]
        if scope != 'none':
            command += [scope]
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
        result = dict(arm=arm, count=count, extension=extension, wrapper_scope=scope, rc=rc,
                      seconds=time.monotonic()-started, timeout=timed_out, command=command,
                      **evidence(log.read_text(errors='replace'), arm, count, rc))
        if timed_out:
            result['status'] = 'TIMEOUT'
        results.append(result)
        print('KPACK_TP2_COMM_RESULT ' + json.dumps(result), flush=True)
        (output / 'summary.json').write_text(json.dumps(dict(status='RUNNING', cases=results), indent=2) + '\n')
    verdict = wrapper_verdict(results)
    (output / 'summary.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE', wrapper_verdict=verdict,
        device_admission='PENDING', production_communication='UNCHANGED', cases=results), indent=2) + '\n')
    print(f'KPACK_TP2_WRAPPER_VERDICT verdict={verdict}', flush=True)
    print(f'KPACK_TP2_COMM_DONE cases={len(results)} verdict=DIAGNOSTIC_COMPLETE device_admission=PENDING results={output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--wrapper-ab', action='store_true', help='add local/global wrapper-only controls and the historical K-pack negative')
    parser.add_argument('--q4-local', action='store_true', help='isolate Q4 E4/top2 local GEMV, collective and fresh Meta graph')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    if args.q4_local and args.wrapper_ab:
        parser.error('select one bounded diagnostic')
    if args.q4_local: run_q4(args.binary.resolve(strict=True),args.output,args.timeout)
    else: run(args.binary.resolve(strict=True), args.output, args.timeout, args.wrapper_ab)
