#!/usr/bin/env python3
"""PPU dense dynamic-call gate: real bound/dynamic kernels, two streams, graphs.

This is a correctness gate, not a timing sweep. Full-BF16 provider composition
is separately admitted by the provider/integration tests.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from quactlize.dispatch.native import Dispatch, Request, receipt
from quactlize.execution.native import arrangement
from quactlize.runtime.native import Call, SDK, checked
from tools.run_kpack_decode_io import fixture, output_check
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.verify_kpack_dispatch import verify


def run_case(args, sdk, w, route, m, typed):
    arr = arrangement(w.q)
    jit = None if not args.jit_cache else dict(python=sys.executable,
        helper=ROOT/'tools/kpack_jit.py', sdk=args.sdk, cache=args.jit_cache)
    d = Dispatch(args.bundle, jit=jit)
    resources, graphs = [], []
    try:
        request = Request(1, C.sizeof(Request), w.q, route, m, w.n, w.k, 1, m, arr.mapping_id)
        choice = d.query_dense_io(request, 1, False) if typed else d.query(
            w.q, route, m, w.n, w.k, 1, m, arr.mapping_id)
        if choice is None:
            raise ValueError('selected test point has no prepared parent; coverage is incomplete')
        plan = d.prepare_dynamic(choice)
        dtype = '<f4' if typed else '<f2'
        scalar = np.dtype(dtype).itemsize
        calls, controls, baselines = [], [], []
        for slot in range(2):
            r = Resources(sdk); resources.append(r)
            planes = {name: r.upload(data) if data.size else None for name, data in w.planes.items()}
            output = r.alloc(m*w.n*scalar+32)
            workspace = r.alloc(choice.workspace_bytes+32)
            call = Call(version=1, size=C.sizeof(Call), m=m, n=w.n, k=w.k, experts=1,
                group_size=arr.group_size, device=choice.device, compute_units=choice.compute_units,
                mapping_id=arr.mapping_id, a=r.alloc(m*w.k*scalar), low=planes['low'], high=planes['high'],
                metadata=planes['scale' if route==1 else 'units'],
                zero=planes['zero'] if route==1 and w.q!=8 else None, output=output+16,
                workspace=workspace+16 if choice.workspace_bytes else None,
                workspace_bytes=choice.workspace_bytes, stream=r.stream.value)
            r.fill(output, 0xa5, m*w.n*scalar+32); r.fill(workspace, 0xa5, choice.workspace_bytes+32)
            control = Call.from_buffer_copy(call)
            control.output = r.alloc(m*w.n*scalar)
            control.workspace = r.alloc(choice.workspace_bytes) if choice.workspace_bytes else None
            baseline = d.prepare_dense_io(choice, control, 1) if typed else d.prepare(choice, control)
            calls.append((call, output, workspace)); controls.append(control); baselines.append(baseline)
        sdk.synchronize(None)
        # Initialize all A buffers before the excluded eager warmup/capture.
        for r, (call, _, _) in zip(resources, calls):
            r.fill(call.a, 0, m*w.k*scalar)
        sdk.synchronize(None)
        for r, (call, _, _) in zip(resources, calls):
            checked(d.run_dynamic(plan, call), 'dynamic warmup')
            sdk.synchronize(r.stream)
            graphs.append(Replay(sdk, r.stream, lambda c=call: d.run_dynamic(plan, c), 1))
        errors = []
        for repeat in range(6):
            golds = []
            for slot, (call, _, _) in enumerate(calls):
                values = np.random.default_rng(500+repeat*2+slot).uniform(-.125, .125, (m, 4)).astype('f2')
                source = np.ascontiguousarray(values[:, w.categories[0]], dtype=dtype)
                checked(sdk.lib.hggcMemcpy(call.a, source.ctypes.data, source.nbytes, 1), 'changed A')
                golds.append((values.astype('f8') @ w.sums[0], np.abs(values.astype('f8')) @ w.abs_sums[0]))
            sdk.synchronize(None)
            # Both streams enqueue before either is synchronized. A later
            # invocation must not rewrite parameters captured for stream 0.
            for slot, (call, _, _) in enumerate(calls):
                checked(baselines[slot](), 'bound control')
                checked(graphs[slot]() if repeat%2==0 else d.run_dynamic(plan, call), 'dynamic/graph')
            for slot, (call, output, workspace) in enumerate(calls):
                sdk.synchronize(resources[slot].stream)
                actual = sdk.download(call.output, m*w.n*scalar)
                if actual != sdk.download(controls[slot].output, m*w.n*scalar):
                    raise ValueError('bound/dynamic output bits differ')
                got = np.frombuffer(actual, dtype=dtype).reshape(m, w.n)
                errors.append(output_check(got, *golds[slot]))
                for pointer, size in ((output, m*w.n*scalar+32), (workspace, choice.workspace_bytes+32)):
                    raw = sdk.download(pointer, size)
                    if raw[:16] != b'\xa5'*16 or raw[-16:] != b'\xa5'*16:
                        raise ValueError('output/workspace guard changed')
        invalid = Call.from_buffer_copy(calls[0][0]); invalid.n += 256
        if d.run_dynamic(plan, invalid) == 0:
            raise ValueError('changed geometry was accepted')
        return dict(q=w.q, route=route, m=m, typed=typed, status='PASS', choice=receipt(choice),
                    streams=2, changing_address_sets=2, iterations=6, errors=errors,
                    bound_dynamic='BITWISE_EQUAL', graph_replays=6, guards='PASS', timing_valid=False)
    finally:
        for r in resources:
            sdk.synchronize(r.stream)
        for graph in graphs:
            graph.close()
        d.close()
        for r in resources:
            r.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jit-cache', type=Path, help='explicit selected-parent JIT before capture')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = verify(args.bundle, sdk=args.sdk)
    sdk = SDK(args.sdk); graph_bind(sdk)
    results = []
    for q in (8, 10, 11, 12, 13, 14):
        w = fixture(q, 1024, 5120)
        for route in ((1,) if q==8 else (0, 1)):
            for m, typed in ((1, False), (128, False), (1, True), (8, True)):
                try:
                    result = run_case(args, sdk, w, route, m, typed)
                except Exception as exc:
                    traceback.print_exc()
                    result = dict(q=q, route=route, m=m, typed=typed, status='FAIL', error=str(exc))
                results.append(result)
                print('KPACK_DYNAMIC_GATE '+json.dumps(result, allow_nan=False), flush=True)
                (args.output/'summary.json').write_text(json.dumps(dict(
                    status='PASS' if len(results)==44 and all(r['status']=='PASS' for r in results) else 'INCOMPLETE',
                    expected=44, completed=len(results), results=results, bundle=manifest.get('source'),
                    scope='DYNAMIC_DENSE_F16_AND_F32_ENDPOINTS_NOT_FULL_PROVIDER_OR_BF16_ADMISSION',
                    timing_valid=False), indent=2, allow_nan=False)+'\n')
    return 0 if all(r['status']=='PASS' for r in results) else 1


if __name__=='__main__':
    raise SystemExit(main())
