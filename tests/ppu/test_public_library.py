"""Real-device public ABI coverage; run via run_tests.sh --suite accuracy.

No model, llama.cpp checkout, six-library bundle, timing sweep or profiler.
"""
import ctypes as C
import json

import numpy as np
import pytest

from tools.run_kpack_grouped_decode_probe import Replay
from .support import (Arena, Candidate, DENSE_M, Info, Options, Params, Public,
                      QTYPES, check_output, packed)
from quactlize.dispatch.dense_prefill import EVIDENCE


def helper_followup_points():
    # Immutable measured scope, selected through the public ABI, not a forced
    # experimental launcher. Includes both retained controls as well as winners.
    data=json.loads(EVIDENCE.read_text())
    review='d3eec044e46f3beff68dc1fc5ce6371faaed7232aa73e673d91fb81f7e1f4128'
    historical=[p for p in data.get('superseded_points',data['points'])
                if p.get('admission',{}).get('review_sha256')==review]
    keys={p['key'] for p in historical}
    points=[p for p in data['points'] if p['key'] in keys]
    assert len(points)==15, 'helper followup public coverage changed'
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


def m_span_boundary_points():
    # Exercise public preparation at each new high-M boundary without another
    # performance sweep or a model. Prefer small matrices for routine CI.
    data=json.loads(EVIDENCE.read_text())
    points=[min((p for p in data['points'] if p['shape'][0]==q and p['shape'][1]==m),
                key=lambda p:(p['shape'][2]*p['shape'][3],p['key']))
            for q,m in ((8,3072),(12,5120),(14,8192))]
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


@pytest.mark.parametrize("q", QTYPES)
@pytest.mark.parametrize("pair", [False, True])
def test_device_pack_all_formats_and_experts(package, sdk, q, pair):
    with Arena(sdk) as arena:
        packed(arena, Public(package), q, experts=3, pair=pair, bit_patterns=True)


@pytest.mark.parametrize("q", QTYPES)
@pytest.mark.parametrize("m", DENSE_M)
@pytest.mark.parametrize("compute", [0, 1], ids=["f16", "bf16"])
def test_selected_dense_dynamic_bindings(package, sdk, q, m, compute, record_property):
    run_dense(package, sdk, q, m, compute, record_property)


@pytest.mark.parametrize("q", [10, 11, 12, 13, 14])
def test_full_dequant_provider_ordinary_calls(package, sdk, q, record_property):
    # This is not a cuBLAS graph/timer test. Explicit route admission is logged.
    run_dense(package, sdk, q, 128, 1, record_property, full=True)


@pytest.mark.parametrize('shape,recipe',helper_followup_points()+m_span_boundary_points(),
                         ids=lambda x: str(x) if isinstance(x,list) else '')
def test_admitted_helper_followup_public_calls(package,sdk,shape,recipe,record_property):
    from dev.prefill_point.typed import DenseLibrary
    from quactlize.dispatch.table import Workload
    # Existing host-only device guard: this fixed helper row exists on ZW810
    # only. Other supported cards test execution, not a ZW810 policy claim.
    query=DenseLibrary(package/'quactlize-runtime/libquactlize_ppu_prefill.so')
    known=Workload(qtype=12,m=2048,n=1024,k=5120,operation='dense',compute='f16',storage='f32')
    measured_device=query.choice(known,0,allow_miss=True) is not None
    record_property('selection_domain','ZW810_EXACT' if measured_device else 'OTHER_DEVICE_FALLBACK')
    q,m,n,k=shape
    run_dense(package,sdk,q,m,0,record_property,n=n,k=k,expected=recipe,measured_device=measured_device)


def run_dense(package, sdk, q, m, compute, record_property, *, full=False,n=256,k=512,expected=None,measured_device=True):
    lib = Public(package)
    with Arena(sdk) as arena:
        arr, sizes, weight_ptrs, gold = packed(arena, lib, q,n=n,k=k)
        n, k = gold.shape[1:]
        padding = 8 if m <= 8 else 0
        p = Params(1, C.sizeof(Params), q, n, k, m, k + padding, n + padding, arr, sizes)
        admission = Params.from_buffer_copy(p); admission.m = 0
        assert lib.supported(C.byref(admission)) == 1
        assert lib.supported(C.byref(p)) == 1
        admission.qtype = 9
        assert lib.supported(C.byref(admission)) == 0
        context, handle = C.c_void_p(), C.c_void_p()
        options, info = Options(1, C.sizeof(Options), compute, 0), Info(1, C.sizeof(Info))
        lib.check(lib.create(0, C.byref(options), C.byref(context)))
        graph = None
        try:
            if full:
                candidate = Candidate(1, C.sizeof(Candidate))
                candidate.kind = candidate.route = 3
                candidate.dequant_config = 5
                lib.check(lib.candidate(context, C.byref(p), *weight_ptrs, C.byref(candidate), C.byref(handle), C.byref(info)))
                assert info.route == 3 and info.compute_type == 1
            else:
                lib.check(lib.prepare(context, C.byref(p), *weight_ptrs, C.byref(handle), C.byref(info)))
            record_property("plan", f"route={info.route} compute={info.compute_type} split={info.split} config={info.config.decode()}")
            assert handle.value and info.config and info.split >= 1
            if expected and measured_device:
                assert info.policy==20 and info.compute_type==0, 'expected ZW810 exact F16 recipe'
                route=expected['route']; assert info.route==(3 if route==2 else route)
                if route==2:
                    name=f'full-f16:cublas:typed-dq{expected["dequant"]}:io{expected["io"]}'
                    if expected.get('dequant_first'): name+=':dq-first'
                    assert info.split==1
                else:
                    name=f'{expected["tc"]["symbol"]}:dq{expected["dequant"]}:io{expected["io"]}'
                    assert info.split==expected['tc']['split']
                assert info.config.decode()==name, 'public recipe does not match admitted winner'
            elif expected:
                assert info.policy!=20, 'ZW810 exact policy must not run on another device'
            if expected:
                record_property('oracle','ALL_OUTPUTS_FINITE; 9x33_SPOTS_FULL_K_FP64_DOT; ALL_PACKED_BYTES; TWO_STREAMS')
            def check(got,source):
                if not expected:
                    return check_output(got,source,gold[0],info.compute_type)
                assert np.isfinite(got).all(), 'unwritten or nonfinite public output'
                rows=np.linspace(0,m-1,min(m,9),dtype=int)
                cols=np.linspace(0,n-1,min(n,33),dtype=int)
                return check_output(got[np.ix_(rows,cols)],source[rows],gold[0,cols],info.compute_type)
            streams = [arena.stream(), arena.stream()]
            inputs = [arena.buffer(m * p.input_stride * 4) for _ in streams]
            outputs = [arena.buffer(m * p.output_stride * 4) for _ in streams]
            scratch = [arena.buffer(info.workspace_bytes) for _ in streams]
            def launch(index):
                return lib.run(handle, inputs[index].ptr, outputs[index].ptr,
                               scratch[index].ptr, info.workspace_bytes, streams[index])
            for repeat in range(3):
                sources = []
                for i in range(2):
                    source = np.random.default_rng(q * 31 + m + repeat * 7 + i).uniform(-.13, .13, (m, p.input_stride)).astype("f4")
                    if compute and m == 1:
                        source *= 1_000_000  # Exercise range beyond F16, without clipping.
                    inputs[i].write(source); outputs[i].poison()
                    sources.append(source[:, :k])
                arena.sync()
                if repeat == 1 and info.route != 3:
                    graph = Replay(sdk, streams[0], lambda: launch(0), 1)
                for i in range(2):
                    lib.check(graph() if i == 0 and graph else launch(i))
                arena.sync()
                for i in range(2):
                    image = outputs[i].bytes()
                    got = np.frombuffer(image, dtype="f4").reshape(m, p.output_stride)
                    check(got[:, :n], sources[i])
                    if padding:
                        assert np.all(got[:, n:].view("u4") == 0xffffffff), "output stride padding overwritten"
                    inputs[i].guard(); scratch[i].guard()
            # An omitted/zero producer must not pass against retained golden output.
            inputs[0].write(np.zeros((m, p.input_stride), "f4")); outputs[0].poison()
            arena.sync(); lib.check(graph() if graph else launch(0)); arena.sync()
            got = np.frombuffer(outputs[0].bytes(), dtype="f4").reshape(m, p.output_stride)[:, :n]
            assert np.all(got == 0) and np.isfinite(got).all()
            with pytest.raises(ValueError):
                check(got, sources[0])
            # Validate the actual workspace contract before an enqueue.
            if info.workspace_bytes:
                assert lib.run(handle, inputs[0].ptr, outputs[0].ptr, scratch[0].ptr,
                               info.workspace_bytes - 1, streams[0]) != 0
        finally:
            arena.sync()
            if graph:
                graph.close()
            lib.destroy(context)
