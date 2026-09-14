#!/usr/bin/env python3
"""Admit per-call SF and full-BF16 composition, not a new config sweep.

The two grouped geometries are the full-dequant winners in the measured
policy. The gate uses every output, changing GPU routes and A between graph
replays. Compilation and the first eager provider invocation are not timed.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dequant.native import Call as Weight, config_ids
from quactlize.dispatch.native import Request
from quactlize.execution.native import Arrangement, arrangement
from quactlize.prefill.native import Call, Prefill, selector
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from tools.kpack_bf16_fixture import Oracle, as_float
from tools.kpack_dequant_fixture import bf16, compare, fixture
from tools.run_kpack_gemv_gate import Resources, metadata_oracle
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify
from dev.gemv_ppu.decode_sweep import real_ids


def workloads():
    return [dict(q=q, n=1024, k=5120, experts=1, m=4096) for q in range(10, 15)] + [
        dict(q=11, n=n, k=512, experts=256, m=32768) for n in (2048, 3072)]


def routing(m, experts, channels, repeat):
    if experts == 1:
        rows = np.arange(m, dtype='i4')
        return rows, rows, np.array([0, m], dtype='i4'), np.zeros(m, dtype='i4'), rows
    ids = real_ids(m//8).reshape(-1)
    # Include a sparse set with many empty experts, then change it. These
    # are valid directories, not a host-selected production route.
    if repeat == 1:
        ids = (np.arange(m)*7 % 16 + 31).astype('i4')
    else:
        ids = ((ids + 37*repeat) % experts).astype('i4')
    order = np.argsort(ids, kind='stable').astype('i4')
    a_rows = (np.arange(m)//8*channels + np.arange(m)%8%channels).astype('i4')
    offsets = np.r_[0, np.bincount(ids, minlength=experts).cumsum()].astype('i4')
    return a_rows[order], order, offsets, ids, a_rows


def require_guards(raw, byte, size):
    if raw[:size] != bytes([byte])*size or raw[-size:] != bytes([byte])*size:
        raise ValueError('composition output/workspace guard changed')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sf_proof(sdk, resources, runtime, w, planes, pointers, arr):
    n, k, e, q = (w[x] for x in ('n', 'k', 'experts', 'q'))
    gold = metadata_oracle(planes['units'], q, n, k, e)
    size = gold[0].nbytes
    scale, zero = resources.alloc(size), resources.alloc(size)
    fn = runtime.library.quactlize_kpack_dequant_v1
    fn.argtypes, fn.restype = [C.POINTER(Weight), C.POINTER(Arrangement)], C.c_int
    call = Weight(version=1, size=C.sizeof(Weight), qtype=q, n=n, k=k, experts=e,
        operation=0, output=scale, zero=zero, output_bytes=size, stream=resources.stream.value,
        low_bytes=planes['low'].nbytes, high_bytes=planes['high'].nbytes,
        unit_bytes=planes['units'].nbytes, **pointers)
    records = []
    for config in config_ids(q, 0, generation=4):
        call.config = config
        resources.fill(scale, 0x7e, size); resources.fill(zero, 0x7e, size)
        checked(fn(C.byref(call), C.byref(arr)), 'measured SF expansion')
        sdk.synchronize(resources.stream)
        for pointer, expected in zip((scale, zero), gold):
            got = np.frombuffer(sdk.download(pointer, size), dtype='<u2')
            if not np.array_equal(got, expected.view('<u2').reshape(-1)):
                raise ValueError(f'SF metadata differs: q={q} config={config}')
        records.append(dict(config=config, cells=2*gold[0].size, bad=0))
    return records


def run_case(args, sdk, w, channels, planes, weight_gold, oracle):
    q, n, k, e, m = (w[x] for x in ('q', 'n', 'k', 'experts', 'm'))
    arr = arrangement(q)
    req = Request(1, C.sizeof(Request), q, 0 if e==1 else 2, m, n, k, e,
                  m if e==1 else m//8, arr.mapping_id)
    selected = selector(args.bundle)(req)
    if selected is None:
        raise ValueError('measured prefill knot is absent')
    if selected.route != 2:
        raise ValueError('expected full-BF16 winner changed')
    full_config = selected.dequant_config
    r = Resources(sdk)
    runtime = Prefill(args.bundle)
    graph, instance = C.c_void_p(), C.c_void_p()
    record = dict(workload=w, channels=channels, automatic_route=selected.route,
        config=full_config, scope='COMPOSED_RUNTIME_FUNCTIONAL_NOT_MODEL_PERFORMANCE',
        input_type='F32', provider_operands='BF16', compute='FP32', output_type='F32')
    try:
        pointers = {name:r.upload(planes[name]) if planes[name].size else None for name in ('low','high','units')}
        sdk.synchronize(None)
        record['sf'] = sf_proof(sdk, r, runtime, w, planes, pointers, arr)
        a_rows = m if e==1 else m//8*channels
        astride, ostride = k+16, n+16
        a = r.alloc(a_rows*astride*4)
        output = r.alloc(m*ostride*4+512)
        src, dst, bounds = (r.alloc(m*4), r.alloc(m*4), r.alloc((e+1)*4)) if e>1 else (None, None, None)
        weight = Weight(version=1, size=C.sizeof(Weight), qtype=q, n=n, k=k, experts=e,
            operation=1, config=full_config, stream=r.stream.value,
            low_bytes=planes['low'].nbytes, high_bytes=planes['high'].nbytes,
            unit_bytes=planes['units'].nbytes, **pointers)
        call = Call(version=1, size=C.sizeof(Call), weight=weight, m=m, device=0, a_rows=a_rows,
            a=a, output=output+256, a_stride=astride, output_stride=ostride,
            src_rows=src, dst_rows=dst, offsets=bounds)
        call.workspace_bytes = runtime.query(call, arr)
        workspace = r.alloc(call.workspace_bytes+512)
        call.workspace = workspace+256
        handle = runtime.prepare(call, arr, args.sdk)
        image = runtime.image(handle)
        record['provider'] = dict(kind='cublas' if e==1 else 'deepgemm', library=str(image), sha256=sha(image))
        if e>1:
            receipt = image.parent/f'quactlize-launch-m{m}.json'
            provider = json.loads(receipt.read_text())
            if provider['files']['kernel.so'] != sha(image) or provider['shape'] != [m,n,k,e]:
                raise ValueError('resolved DeepGEMM provider receipt differs')
            record['provider']['receipt'] = provider
        coefficients = None
        current_ids = None
        current_arows = None

        def write(pointer, array):
            array = np.ascontiguousarray(array)
            checked(sdk.lib.hggcMemcpy(pointer, array.ctypes.data, array.nbytes, 1), 'fixture H2D')

        def reset(repeat):
            nonlocal coefficients, current_ids, current_arows
            a_bits, coefficients = oracle.activations(a_rows)
            values = as_float(a_bits)
            # Change every replay while preserving a factorized independent
            # reference. Values are BF16-exact but caller storage is FP32.
            factor = -1. if repeat%2 else 0.5 if repeat==2 else 1.
            values *= factor; coefficients *= factor
            host_a = np.full((a_rows,astride), 17, dtype='f4'); host_a[:,:k] = values
            from_, to, offsets, current_ids, current_arows = routing(m,e,channels,repeat)
            write(a, host_a)
            if e>1:
                write(src, from_); write(dst, to); write(bounds, offsets)
            sdk.synchronize(None)
            r.fill(output, 0xA5, m*ostride*4+512)
            r.fill(workspace, 0xA5, call.workspace_bytes+512)

        def check():
            sdk.synchronize(r.stream)
            status = int(np.frombuffer(sdk.download(runtime.status_pointer(handle),4), dtype='i4')[0])
            if status:
                raise ValueError(f'composition device status={status}')
            raw = sdk.download(output, m*ostride*4+512)
            require_guards(raw, 0xA5, 256)
            matrix_bits = np.frombuffer(raw[256:-256], dtype='<u4').reshape(m,ostride)
            if np.any(matrix_bits[:,n:] != 0xA5A5A5A5):
                raise ValueError('output row padding overwritten')
            matrix = matrix_bits[:,:n].view('<f4')
            error = oracle.error(bf16(matrix), coefficients[current_arows], current_ids)
            if not np.isfinite(error) or error>=0.005:
                raise ValueError(f'composed BF16 dot error={error}')
            if np.any(matrix_bits[:,:n] & 0xffff):
                raise ValueError('F32 output is not the provider BF16 result')
            if sdk.download(workspace,256) != bytes([0xA5])*256 or sdk.download(
                    call.workspace+call.workspace_bytes,256) != bytes([0xA5])*256:
                raise ValueError('workspace guard overwritten')
            # Check every expanded expert and every untouched expert. This is
            # a source-bound internal layout test (weights begin at offset 0).
            active = set(map(int, current_ids))
            expert_bytes = n*k*2
            for expert in range(e):
                got = np.frombuffer(sdk.download(call.workspace+expert*expert_bytes, expert_bytes), dtype='<u2').reshape(n,k)
                if expert in active:
                    compare(got, weight_gold[expert])
                elif np.any(got != 0xA5A5):
                    raise ValueError('full dequant touched an inactive expert')
            return dict(error=error, active_experts=len(active), inactive_experts_untouched=e-len(active),
                        status=status, output_cells=m*n, output_and_workspace_guards='PASS')

        reset(0)
        runtime.run(handle, r.stream)
        record['eager'] = check()
        # At this point library loading, compilation and provider first use
        # have all completed. Only current GPU inputs enter the graph.
        checked(sdk.lib.hggcStreamBeginCapture(r.stream,0), 'begin composition capture')
        runtime.run(handle, r.stream)
        checked(sdk.lib.hggcStreamEndCapture(r.stream,C.byref(graph)), 'end composition capture')
        checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance),graph,0), 'composition graph instantiate')
        replays = []
        for repeat in range(3):
            reset(repeat)
            checked(sdk.lib.hggcGraphLaunch(instance,r.stream), 'composition graph replay')
            replays.append(check())
        record['graph_replays'] = replays
        # Zero-code plane is an independent negative. It must disagree with
        # the original GGUF oracle, even though its output can stay nonzero.
        r.fill(pointers['low'], 0, planes['low'].nbytes)
        runtime.run(handle, r.stream)
        sdk.synchronize(r.stream)
        raw = sdk.download(call.output, m*ostride*4)
        wrong = np.frombuffer(raw,dtype='<f4').reshape(m,ostride)[:,:n]
        planted = oracle.error(bf16(wrong), coefficients[current_arows], current_ids)
        if not np.isfinite(planted) or planted <= 0.005:
            raise ValueError('zeroed-code negative did not disagree with the independent oracle')
        record['zeroed_code_error'] = planted
        record['status'] = 'PASS'
        return record
    finally:
        sdk.synchronize(r.stream)
        if instance:
            checked(sdk.lib.hggcGraphExecDestroy(instance), 'destroy composition graph instance')
        if graph:
            checked(sdk.lib.hggcGraphDestroy(graph), 'destroy composition graph')
        runtime.close(); r.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify(args.bundle, sdk=args.sdk)
    sdk = SDK(args.sdk); graph_bind(sdk)
    summary = dict(schema='quactlize.prefill-composition-gate.v1', device=device_identity(sdk),
        bundle_manifest_sha256=sha(args.bundle/'manifest.json'), cases=[],
        scope='LIBRARY_COMPOSITION_NOT_LLAMA_MODEL_ADMISSION')
    start = time.monotonic()
    for w in workloads():
        label = f'q{w["q"]}-n{w["n"]}-k{w["k"]}-e{w["experts"]}'
        print('KPACK_PREFILL_COMPOSITION fixture='+label, flush=True)
        try:
            planes, gold = fixture(w['q'],w['n'],w['k'],w['experts'],1,
                progress=lambda done,total:print(f'KPACK_PREFILL_COMPOSITION fixture={label} experts={done}/{total}',flush=True))
            oracle = Oracle(gold)
            for channels in ((1,) if w['experts']==1 else (1,8)):
                record = run_case(args,sdk,w,channels,planes,gold,oracle)
                summary['cases'].append(record)
                save(args.output/f'{label}-ch{channels}.json', record)
                print(f'KPACK_PREFILL_COMPOSITION case={label} channels={channels} status=PASS elapsed_s={time.monotonic()-start:.1f}', flush=True)
        except Exception as error:
            traceback.print_exc()
            record = dict(workload=w, status='FAIL', error=str(error))
            summary['cases'].append(record)
            save(args.output/f'{label}.failure.json', record)
        save(args.output/'result.json', summary)
    summary['seconds'] = time.monotonic()-start
    summary['status'] = 'PASS' if len(summary['cases'])==9 and all(x['status']=='PASS' for x in summary['cases']) else 'FAIL'
    save(args.output/'result.json', summary)
    print('KPACK_PREFILL_COMPOSITION verdict='+summary['status']+' result='+str(args.output/'result.json'), flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
