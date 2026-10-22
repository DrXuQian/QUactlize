#!/usr/bin/env python3
"""Q4 KV only: three fixed routes, ordinary component and complete-call timers."""
import argparse
from dataclasses import asdict
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_point.bench import WORKLOAD, SF, qz, summarize, trace_contract
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ARMS = ('native-fp16', 'sf', 'full-bf16')


def plan():
    return dict(workload=asdict(WORKLOAD), arms=list(ARMS), sf=asdict(SF),
        full_dequant_config=12, full_workspace_bytes=4 << 20,
        accumulator='FP32_ALL_ARMS', external_io='FP32',
        cache_modes=['REPEATED_ADDRESSES', 'ROTATING_WEIGHTS_RESIDENT_F32_INPUT'],
        timers=['ISOLATED_STAGE_EVENTS', 'SEQUENCE_STAGE_EVENTS', 'COMPLETE_CALL_EVENTS'],
        timing_note='GPU_EVENT_API_INTERVALS_NOT_PROFILER_KERNEL_DURATIONS',
        allocation_note='NATIVE_ROTATING_SEQUENCE_HAS_PER_COPY_SCRATCH; QZ_REUSES_SCRATCH; REPEATED_MODE_SAME_RESIDENCY_CLASS',
        scope='ONE_POINT_DIAGNOSTIC_NOT_SWEEP_OR_MODEL_PERFORMANCE_ADMISSION',
        requested_clock_mhz=1700, production_changed=False, performance_admitted=False,
        cuda_graph_timing=False, cublas_graph=False, profiler=False)


def native_stage_names(inventory):
    names = []
    for stage in inventory:
        name = stage.get('name', stage['symbol'])
        if stage['entry'] == 'PROVIDER' and stage['symbol'] == 'cublasGemmEx':
            role = 'main'
        elif 'dequantize' in name:
            role = 'expansion'
        elif any(s in name for s in ('convert_cont_ppu_kernel<float, __half>',
                                    'fp32_to_fp16', 'convert_f32_to_f16')):
            role = 'input'
        elif any(s in name for s in ('convert_cont_ppu_kernel<__half, float>',
                                    'fp16_to_fp32', 'convert_f16_to_f32')):
            role = 'output'
        else:
            raise ValueError('native stage requires explicit classification: '+name)
        names.append(role)
    if (len(names) != 4 or set(names[:2]) != {'input', 'expansion'} or
            names[2:] != ['main', 'output']):
        raise ValueError('native four-stage inventory/order differs: '+repr(names))
    return names


def build(args):
    from dev.native_components.build import run_commands
    from dev.prefill_point.build import build_native
    from tools.run_native_components import environment
    from dev.strategy.prefill_build import build as helpers
    from tools.build_kpack_dequant import build as dequant
    target = args.output/'build'
    target.mkdir()
    reuse = getattr(args, 'reuse_build', None)
    print('PREFILL_POINT_BUILD '+('profile host wrappers only; device_payload=REUSED' if reuse else
          'native adapter + one SF parent + dequant/helpers')+'; llama_rebuild=0', flush=True)
    adapter, binary, receipt = build_native(args, target/'native', environment(args))
    if not reuse:
        helpers(args.sdk, target/'helpers')
        dequant(args.sdk, target/'dequant')
    include = args.sdk/'include'
    if not (include/'hggc_runtime_api.h').is_file():
        include = args.sdk/'targets/x86_64-linux/include'
    source = ROOT/'dev/prefill_point/timer.cpp'
    command = ['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-Wl,-z,defs',
        '-I'+str(include), str(source), '-L'+str(args.sdk/'lib'),
        '-L'+str(args.sdk/'targets/x86_64-linux/lib'),
        '-lhggc_wrapper', '-lhggcrt1', '-lhggc', '-o', str(target/'libpoint-timer.so')]
    run_commands([command], ROOT, target/'point-timer.log', env=environment(args), stages=['point timer'])
    data = dict(adapter=str(adapter), adapter_receipt=receipt, carrier=str(binary),
        carrier_sha256=sha(binary), timer_sha256=sha(target/'libpoint-timer.so'),
        timer_source_sha256=sha(source), timer_command=command)
    atomic_json(target/'receipt.json', data)
    return data


def native(args, receipt):
    from dev.native_components.fixture import weights, activation
    from tools.run_kpack_llama_ab import point_request
    from tools.run_native_components import environment
    w = getattr(args, 'workload', WORKLOAD)
    row = dict(id=w.key, workload=asdict(w), cohort='prefill')
    row = point_request(row, args.l2_bytes)
    row['input_fixture'] = activation(row)
    raw, fixture = weights(row['workload'], args.output/'fixture')
    runtime = next((p for p in (args.sdk/'lib/libhggc_wrapper.so',
                    args.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if p.is_file()), None)
    if runtime is None: raise ValueError('SDK runtime wrapper missing')
    out = args.output/'native'
    out.mkdir()
    request = dict(runtime_library=str(runtime), adapter_library=receipt['adapter'],
        authority=json.loads((args.output/'authority.json').read_text()), points=[row],
        raw_fixture=str(raw), raw_fixture_receipt=fixture, rounds=args.rounds,
        samples=args.samples, point_probe=True, trace_only=getattr(args, 'trace_only', False),
        cohort_prefill_probe=getattr(args, 'cohort_prefill_probe', False),
        model_prefill_probe=getattr(args, 'model_prefill_probe', False))
    atomic_json(out/'request.json', request)
    command = [receipt['carrier'], str(out/'request.json'), str(out)]
    atomic_json(out/'command.json', command)
    # C++ carrier performs independent oracle, two-input replay comparison,
    # missing-producer and last-weight-copy negatives before any timing.
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment(args))
        wait(proc, 'native-fp16', out/'run.log')
    data = json.loads((out/(w.key+'.json')).read_text())
    if data['correctness'] != 'PASS': raise ValueError('native correctness failed')
    names = native_stage_names(data['inventory'])
    trace_only = getattr(args, 'trace_only', False)
    data.update(status='PASS', arm='native-fp16', stages=names,
                allocation='PER_WEIGHT_COPY_NATIVE_POOL_AND_OUTPUT; ISOLATED_CASTS_USE_COPY0',
                summary=None if trace_only else summarize(data['point_timings'], names))
    if trace_only:
        data['trace'] = trace_contract(row['copies'])
        data['scope'] = 'ASYS_KERNEL_DURATIONS_NOT_EVENT_TIMER'
    atomic_json(args.output/'results/native-fp16.json', data)
    return data


def wait(proc, arm, log):
    last = time.monotonic()
    while proc.poll() is None:
        time.sleep(.5)
        if time.monotonic()-last >= 30:
            print(f'PREFILL_POINT_WAIT arm={arm} log={log}', flush=True); last=time.monotonic()
    if proc.returncode:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-8192))
            text = stream.read().decode(errors='replace')
        raise RuntimeError(f'{arm} rc={proc.returncode}; {log}\n{text}')


def package(output):
    # No weights, object files, libraries, JIT trees or full build logs.
    files = [output/'plan.json', output/'authority.json', output/'summary.json', output/'error.json',
             output/'build/receipt.json', output/'native/request.json', output/'native/command.json']
    files += list((output/'results').glob('*.json')) + list(output.glob('clock-*.json'))
    files += list(output.glob('*.log')) + list((output/'native').glob('*.log'))
    files += list((output/'build').rglob('*.log'))
    files += list((output/'native').glob('*.inventory.json'))
    archive = output/'review.tgz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(set(files)):
            if not path.is_file() or path.is_symlink(): continue
            with path.open('rb') as stream:
                if path.suffix == '.log': stream.seek(max(0, path.stat().st_size-8192))
                raw = stream.read()
            item = tarfile.TarInfo(str(path.relative_to(output)))
            item.size = len(raw); tar.addfile(item, io.BytesIO(raw))
    print('results='+str(archive), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--llama-source', type=Path, default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='1')
    p.add_argument('--smi', type=Path)
    p.add_argument('--l2-bytes', type=int, default=67108864)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--samples', type=int, default=17)
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    p.add_argument('--trace-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--reuse-build', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.device.isdigit() or a.rounds < 3 or a.samples < 3 or a.l2_bytes <= 0:
        p.error('one device, positive verified L2, at least three rounds/samples required')
    a.output = a.output.resolve()
    if (a.trace_only or a.reuse_build) and not a.worker:
        p.error('trace/reuse options belong to tools/profile_prefill_point.py')
    if a.worker:
        if a.worker == 'native-fp16':
            native(a, json.loads((a.output/'build/receipt.json').read_text()))
        else: qz(a, a.worker)
        return 0
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output/'plan.json', plan())
    print('PREFILL_POINT_PLAN '+json.dumps(plan()), flush=True)
    if a.plan_only: return 0
    if not a.confirm_idle: p.error('--confirm-idle is required; run only on an idle PPU')
    a.sdk = a.sdk.resolve(strict=True)
    a.llama_source = a.llama_source.resolve(strict=True)
    a.llama_build = (a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve(strict=True)
    from tools.run_native_components import environment
    from tools.run_kpack_llama_ab import device
    from dev.strategy.refresh_clock import request_lock, observe
    env = environment(a)
    env.update(CUDA_VISIBLE_DEVICES=a.device, GGML_CUDA_CUBLAS_ACCUMULATOR='f32',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Child dynamic loaders receive the clean SDK env before startup.
    os.environ.update({k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PPU_SDK', 'CUDA_HOME')})
    status = dict(status='INCOMPLETE', results={}, production_changed=False, performance_admitted=False)
    try:
        identity = device(a)
        a.l2_bytes = identity['l2']['bytes']
        pci = identity['device']['pci']
        smi = a.smi or next((path for path in (Path(shutil.which('ppu-smi') or '/nonexistent'),
                            a.sdk/'bin/ppu-smi', a.sdk/'ppu-smi') if path.is_file()), None)
        if smi is None or not smi.is_file():
            raise ValueError('ppu-smi unavailable; cannot request 1700MHz')
        with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = build(a)
            clock = request_lock(smi, pci, 1700)
            atomic_json(a.output/'clock-lock.json', clock)
            if clock['returncode']: raise ValueError('1700MHz request failed; see clock-lock.json')
            atomic_json(a.output/'authority.json', dict(device=identity, build=receipt,
                source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
                    if (ROOT/'.git').exists() else os.environ.get('PREFILL_POINT_SOURCE_SHA', 'FROZEN_GIT_ARCHIVE'),
                actlize_head=os.environ.get('PREFILL_POINT_ACTLIZE_SHA', 'SEE_BUILD_HEADER_IDENTITIES'),
                plan=plan(), frequency_normalization=False, idle='OPERATOR_CONFIRMED'))
            (a.output/'results').mkdir()
            for arm in ARMS:
                atomic_json(a.output/('clock-'+arm+'-before.json'), observe(smi, pci, 1700, advisory=True))
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', arm,
                    '--sdk', str(a.sdk), '--output', str(a.output), '--device', a.device,
                    '--llama-source', str(a.llama_source), '--llama-build', str(a.llama_build),
                    '--rounds', str(a.rounds), '--samples', str(a.samples), '--l2-bytes', str(a.l2_bytes)]
                print(f'PREFILL_POINT_START arm={arm} sweep=0', flush=True)
                with (a.output/(arm+'.log')).open('x') as log:
                    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    wait(proc, arm, a.output/(arm+'.log'))
                atomic_json(a.output/('clock-'+arm+'-after.json'), observe(smi, pci, 1700, advisory=True))
                data = json.loads((a.output/'results'/(arm+'.json')).read_text())
                status['results'][arm] = data['summary']
                print('PREFILL_POINT_RESULT '+json.dumps(dict(arm=arm, **data['summary'])), flush=True)
            status['status'] = 'PASS'
            atomic_json(a.output/'summary.json', status)
    except BaseException as exc:
        atomic_json(a.output/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        atomic_json(a.output/'summary.json', status)
        raise
    finally: package(a.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
