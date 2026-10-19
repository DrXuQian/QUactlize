#!/usr/bin/env python3
"""Q6 incumbent/candidate only: eight points, paired cold timings and optional ACU."""
import argparse
import ctypes as C
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.q6_metadata import ARMS
from dev.gemv_simt.build_q6_metadata import build, inputs
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def workloads():
    def w(m, n=512, k=2048, compute='bf16', experts=256, channels=8, router='cluster'):
        indexed = experts > 1
        return Workload(14, n, k, m, operation='indexed' if indexed else 'dense', compute=compute,
                        experts=experts, topk=8 if indexed else 1, channels=channels if indexed else 1,
                        storage='f32', input_stride=k, output_stride=n, router=router if indexed else '')
    return [w(8), w(1, router='real'), w(2, channels=1, router='real'), w(8, router='spread'),
            w(1, experts=1, compute='f16'), w(8, experts=1, compute='f16'),
            w(8, n=1024, k=5120, experts=32, router='real'),
            w(2, k=512, experts=16, channels=1, compute='f16', router='real')]


class Kernel:
    def __init__(self, b, library):
        from quactlize.execution.native import Call, SimtCallV2, Arrangement
        self.b = b
        self.lib = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        self.fn = self.lib.q6_metadata_run
        self.fn.argtypes, self.fn.restype = [C.POINTER(SimtCallV2), C.POINTER(Arrangement), C.c_int], C.c_int
        p = b.p
        call = Call(version=1, size=C.sizeof(Call), qtype=14, n=p['n'], k=p['k'], experts=p['experts'],
                    rows=b.rows, mode=p['mode'], input_type=1, channels=p['channels'], topk=p['topk'],
                    a_row_stride=p['k'], a_token_stride=p['channels']*p['k'], ids_stride=p['topk'],
                    out_row_stride=p['n'], a=b.a.ptr, ids=b.ids.ptr if b.ids else None,
                    output=b.out.ptr, stream=b.r.stream.value)
        self.calls = []
        for plane in b.ring.planes:
            request = SimtCallV2(Call.from_buffer_copy(call), int(p['compute'] == 'bf16'))
            for name, ptr in plane.items(): setattr(request.call, name, ptr)
            self.calls.append(request)

    def launch(self, arm, copy=0):
        return self.fn(C.byref(self.calls[copy]), C.byref(self.b.arr), arm)


def raw_equal(first, second):
    import numpy as np
    if not np.array_equal(first.view('<u4'), second.view('<u4')):
        bad = np.flatnonzero(first.view('<u4').ravel() != second.view('<u4').ravel())
        raise ValueError(f'candidate differs from same-order baseline: bad={len(bad)} first={int(bad[0])}')


def measure(a, w, record, pattern_model=None):
    import numpy as np
    from contextlib import ExitStack
    from dev.gemv_simt.q6_metadata_fixture import Weights
    from dev.gemv_simt.access import pattern
    from quactlize.execution.simt_codegen import Config
    from dev.strategy.cold_decode import Ring
    from dev.smallm_closure.bench import Context
    from dev.strategy.indexed import point, RuntimeView
    from dev.strategy.isolated import Captured
    from dev.strategy.components import statistics_checked
    from dev.strategy.pilot_native import scale_activation_data
    from quactlize.runtime.native import SDK, checked
    sdk = SDK(a.sdk)
    with ExitStack() as cleanup:
        weights = Weights(w.n, w.k, w.experts)
        ring = Ring(sdk, w, a.l2_bytes, weights=weights);cleanup.callback(ring.close)
        p = point(w);p['mode'] = 2 if w.operation == 'indexed' else 0
        b = Context(sdk, ring, p);cleanup.callback(b.close)
        core = Kernel(b, a.output/'build'/record['image'])
        rt = RuntimeView(b)
        proofs = []
        for repeat in range(3):
            b.update(repeat)
            if repeat == 2 and w.compute == 'bf16':
                scale_activation_data(b.data, 2**20);b.a.upload(b.data['a'])
                if np.max(np.abs(b.data['a'])) <= 65504: raise ValueError('missing BF16 range test')
            for copy in (0, ring.copies-1):
                outputs = []
                for arm in range(2):
                    b.out.poison();checked(core.launch(arm, copy), 'Q6 numerical')
                    proofs.append(b.check());outputs.append(b.out.read((b.rows, w.n)).copy())
                raw_equal(*outputs)
        b.update(0);b.out.poison()
        try: b.check()
        except ValueError as exc:
            if not str(exc).startswith('typed oracle mismatch:'): raise
        else: raise ValueError('omitted producer passed')
        for arm in range(2):
            b.a.upload(np.zeros_like(b.data['a']));b.out.poison()
            checked(core.launch(arm), 'zero A');rt.sync()
            if np.any(b.out.read((b.rows, w.n))): raise ValueError('zero A failed')
        b.update(0)
        for name, buf in ring.objects[-1].items():
            try:
                ring.r.fill(buf.ptr, 0, buf.size);sdk.synchronize(ring.r.stream)
                for arm in range(2):
                    checked(core.launch(arm, ring.copies-1), 'plane negative')
                    try: b.check()
                    except ValueError as exc:
                        if not str(exc).startswith('typed oracle mismatch:'): raise
                    else: raise ValueError('plane negative passed: '+name)
                    checked(core.launch(arm, 0), 'copy isolation');b.check()
            finally:
                rt.sync();buf.upload(weights.planes[name])
        patterns = []
        bases = {name: int(ptr)%128 for name, ptr in ring.planes[0].items()}
        bases['A'] = b.a.ptr%128
        for warp in (range(8) if pattern_model else (0, 7)):
            patterns.append((pattern_model or pattern)(14, Config(3,4,8,8,1), w.n, w.k, bases=bases, warp=warp))
        captures, graphs, nodes = [], [], []
        for arm in range(2):
            for copy in range(ring.copies): checked(core.launch(arm, copy), 'excluded ring warmup')
            b.check()
            capture = Captured(rt, [lambda i=i,arm=arm:core.launch(arm,i) for i in range(ring.copies)])
            cleanup.callback(capture.close);captures.append(capture)
            expected = record.get('kernel_names', ['::register_reuse<','::q6_register_fields('])[arm]
            if len(capture.groups)!=1 or expected not in capture.descriptions[0]['name']:
                raise ValueError('unexpected Q6 kernel inventory')
            nodes.append(capture.descriptions[0])
            graph = capture.isolate(0, weight_ring=True, repeats=16)
            cleanup.callback(graph.close);graphs.append(graph)
        # Captured call parameters must observe changed inputs, not warmup output.
        b.update(1)
        outputs = []
        for graph in graphs:
            b.out.poison();graph.run();rt.sync();proofs.append(b.check())
            outputs.append(b.out.read((b.rows,w.n)).copy())
        raw_equal(*outputs)
        b.update(0)
        samples = [[], []]
        if a.profile_arm is not None:
            from tools.profile_kpack_gpu_compact import AcuRange
            rt.sync()
            with AcuRange(sdk):
                checked(core.launch(a.profile_arm), 'one profiled Q6 call');rt.sync()
            proofs.append(b.check())
            timing = []
        else:
            for rnd in range(6):
                for arm in ((0,1) if rnd%2==0 else (1,0)):
                    samples[arm].append([graphs[arm].sample() for _ in range(5)])
                    proofs.append(b.check())
                print(f'Q6_METADATA_ROUND point={a.child} round={rnd+1}/6', flush=True)
            timing = [statistics_checked(v) for v in samples]
        for buffer in ring.buffers: buffer.guard()
        result = dict(status='PASS', workload=asdict(w), numerical=proofs, raw_equal=True,
                      zero_input='PASS', omitted_producer='DETECTED', plane_negatives=list(ring.objects[-1]),
                      fixture=ring.receipt, patterns=patterns, nodes=nodes, timings=timing,
                      calls_per_graph=[g.count for g in graphs], profiled=a.profile_arm is not None,
                      production_changed=False, scope='Q6_S1_SINGLE_KERNEL_OLD_NEW_NOT_MODEL_OR_NATIVE')
        if timing:
            result['delta_pct'] = 100*(timing[1]['median_us']/timing[0]['median_us']-1)
            result['modeled_weight_mbu_pct'] = [ring.receipt['minimum_read_bytes']/t['median_us']/27000 for t in timing]
            result['mbu_peak_gbps_assumption'] = 2700
    return result


def child(a, source_inputs=inputs, pattern_model=None):
    from tools.run_kpack_llama_ab import device
    from quactlize.runtime.native import sdk_identity
    authority = json.loads((a.output/'authority.json').read_text())
    manifest = json.loads((a.output/'build/manifest.json').read_text())
    if (device(a)!=authority['device'] or harness_identity(ROOT)!=authority['harness'] or
            source_inputs()!=manifest['sources'] or manifest.get('sdk')!=sdk_identity(a.sdk)):
        raise ValueError('source, SDK or physical device changed')
    w = workloads()[a.child]
    record = next(r for r in manifest['records'] if r['compute']==w.compute)
    if sha(a.output/'build'/record['image'])!=record['sha256']: raise ValueError('binary changed')
    result = measure(a,w,record,pattern_model=pattern_model)
    result.update(authority_sha256=digest(authority), library_sha256=record['sha256'])
    atomic_json(a.result, result)
    print('Q6_METADATA_POINT '+json.dumps(dict(point=a.child,status='PASS',delta_pct=result.get('delta_pct'))),flush=True)
    return 0


def execute(command, log):
    atomic_json(log.with_suffix('.command.json'), command)
    start = last = time.monotonic()
    with log.open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                time.sleep(1)
                if time.monotonic()-last>=30:
                    print(f'Q6_METADATA_WAIT seconds={time.monotonic()-start:.1f} log={log}',flush=True)
                    last=time.monotonic()
        except BaseException:
            child.terminate();child.wait();raise
    atomic_json(log.with_suffix('.process.json'), dict(rc=child.returncode,seconds=time.monotonic()-start))
    if child.returncode:
        print(log.read_text(errors='replace')[-5000:],flush=True)
    return child.returncode


def collect(a, *, builder=build, arms=ARMS, driver=None, point_indices=None,
            profile_arms=(0,1), profile_rounds=1):
    from tools.run_kpack_llama_ab import device
    from dev.native_components.profile import acu_command,check_report
    a.output.mkdir(parents=True,exist_ok=False)
    indices=list(range(len(workloads()))) if point_indices is None else list(point_indices)
    if not indices or len(set(indices))!=len(indices) or any(i not in range(len(workloads())) for i in indices):
        raise ValueError('invalid or duplicate point inventory')
    if a.acu and indices[0]!=0: raise ValueError('profile requires point0 first')
    if not isinstance(profile_rounds,int) or not 1<=profile_rounds<=3:
        raise ValueError('profile rounds must be 1..3')
    if len(set(profile_arms))!=len(profile_arms) or any(i not in (0,1) for i in profile_arms):
        raise ValueError('invalid profile arm inventory')
    driver=Path(__file__).resolve() if driver is None else Path(driver).resolve()
    plan = dict(points=[asdict(workloads()[i]) for i in indices],point_indices=indices,arms=arms,rounds=6,samples=5,
                native=False,production_changed=False,configs=1,profile_rounds=profile_rounds,
                profile_arms=list(profile_arms))
    atomic_json(a.output/'plan.json',plan)
    if a.plan_only:
        print('Q6_METADATA_PLAN '+json.dumps(plan));return 0
    if not a.confirm_idle: raise ValueError('confirm one otherwise idle GPU')
    actual = device(a);a.l2_bytes=actual['l2']['bytes']
    authority = dict(device=actual,harness=harness_identity(ROOT))
    atomic_json(a.output/'authority.json',authority)
    lock = (Path('/tmp')/('quactlize-components-'+digest(actual['device']['pci'])[:24]+'.lock')).open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        builder(a.sdk,a.output/'build')
        rows=[];reports=[]
        for done,i in enumerate(indices,1):
            destination=a.output/f'point-{i}.json'
            command=[sys.executable,'-u',str(driver),'--sdk',str(a.sdk),
                     '--output',str(a.output),'--l2-bytes',str(a.l2_bytes),'--child',str(i),'--result',str(destination)]
            rc=execute(command,a.output/f'point-{i}.log')
            row=json.loads(destination.read_text()) if rc==0 and destination.is_file() else dict(status='FAIL',rc=rc)
            rows.append(row)
            print(f'Q6_METADATA_PROGRESS completed={done}/{len(indices)} point={i} status={row["status"]} remaining_continue=1',flush=True)
        if a.acu and rows[0]['status']=='PASS':
            for rnd in range(profile_rounds):
                order=profile_arms if rnd%2==0 else tuple(reversed(profile_arms))
                for arm in order:
                    suffix=str(arm) if profile_rounds==1 else f'r{rnd}-a{arm}'
                    target=a.output/f'profile-{suffix}'
                    command=[sys.executable,'-u',str(ROOT/'tools/profile_q6_metadata.py'),
                             '--source',str(ROOT),'--results',str(a.output),'--sdk',str(a.sdk),
                             '--l2-bytes',str(a.l2_bytes),'--point','0',
                             '--result',str(a.output/f'profile-{suffix}.json'),'--arm',str(arm)]
                    print(f'Q6_PROFILE_PROGRESS round={rnd+1}/{profile_rounds} arm={arm}',flush=True)
                    rc=execute(acu_command(a.acu,target,command,graph_profiling='none'),a.output/f'acu-{suffix}.log')
                    try:
                        if rc: raise ValueError(f'ACU child rc={rc}')
                        paths=[p for p in (target,target.with_suffix('.acurep')) if p.is_file()]
                        if len(paths)!=1: raise ValueError('missing or ambiguous ACU report')
                        raw=a.output/f'profile-{suffix}.raw.csv'
                        if execute([str(a.acu),'--import',str(paths[0]),'--page','raw','--csv'],raw):
                            raise ValueError('ACU import failed')
                        reports.append(dict(status='PASS',round=rnd,arm=arm,raw_csv=str(raw),
                                            report=str(paths[0]),report_sha256=sha(paths[0]),
                                            **check_report(raw.read_text(errors='replace'),rows[0]['nodes'][arm])))
                    except Exception as exc:
                        reports.append(dict(status='FAIL',round=rnd,arm=arm,error=str(exc)))
        status='PASS' if all(r['status']=='PASS' for r in rows+reports) else 'INCOMPLETE'
        summary=dict(status=status,points=rows,profiles=reports,production_changed=False,
                     performance_admitted=False,scope='EXPERIMENT_NOT_SELECTOR_PROMOTION')
        atomic_json(a.output/'summary.json',summary)
    print('Q6_METADATA_DONE '+json.dumps(dict(status=status,points=len(rows),profiles=len(reports))),flush=True)
    return int(status!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--acu',type=Path)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--child',type=int,choices=range(len(workloads())),help=argparse.SUPPRESS)
    p.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--profile-arm',type=int,choices=(0,1),help=argparse.SUPPRESS)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    if a.child is not None:
        if a.result is None:p.error('child needs a result path')
        return child(a)
    try:
        return collect(a)
    finally:
        if a.output.is_dir():
            archive=a.output/'review.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                for path in sorted(a.output.rglob('*')):
                    if path.is_file() and not path.is_symlink() and (path.suffix in ('.json','.log','.csv','.txt','.acurep')
                            or path.name in ('profile-0','profile-1')):
                        tar.add(path,arcname=str(path.relative_to(a.output)),recursive=False)
            print('results='+str(archive),flush=True)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
