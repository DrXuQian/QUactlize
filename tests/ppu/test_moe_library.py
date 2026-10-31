"""Complete public MoE accuracy plus low-level witnesses. No llama or timers."""
import ctypes as C
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quactlize.execution.native import Arrangement, Call, SimtCallV2, SimtConfig, Sizes, bind_simt_compute
from quactlize.runtime.native import checked
from ci.support.graph import Replay
from .support import Arena, DECODE_M, Public, QTYPES, check_output, packed


@pytest.mark.parametrize('q',QTYPES)
@pytest.mark.parametrize('channels',[1,2])
def test_public_indexed_projection(package,sdk,q,channels):
    from ci.support.moe import Params,Info,Public as MoE
    from ci.support.fixture import compare
    public=Public(package);api=MoE(public.lib);context=C.c_void_p()
    api.check(api.create(0,C.byref(context)))
    try:
        with Arena(sdk) as arena:
            arr,sizes,weights_ptr,weights=packed(arena,public,q,experts=4)
            n,k,topk=256,512,2
            for m in (*DECODE_M,9,16,128,5120):
                params=Params(1,C.sizeof(Params),q,n,k,4,m,topk,channels,0,k,k*channels,topk,n,arr,sizes)
                assert api.supported(C.byref(params))==1
                handle=C.c_void_p();info=Info(1,C.sizeof(Info))
                api.check(api.prepare(context,C.byref(params),*weights_ptr,C.byref(handle),C.byref(info)))
                assert info.compute_type==1 and info.config and info.route in (0,1,2)
                lanes=[];graphs=[]
                try:
                    for lane in range(2):
                        lanes.append((arena.stream(),arena.buffer(m*channels*k*4),arena.buffer(m*topk*4),
                                      arena.buffer(m*topk*n*4),arena.buffer(info.workspace_bytes)))
                    for repeat in range(3):
                        expected=[]
                        for lane,(stream,a,ids,out,scratch) in enumerate(lanes):
                            js=(np.arange(m)[:,None]*31+np.arange(channels)[None,:]*17+repeat*13)%k
                            amplitude=262144.0 if repeat==2 else (-1.0 if lane else 1.0)
                            values=np.zeros((m,channels,k),'f4')
                            values[np.arange(m)[:,None],np.arange(channels)[None,:],js]=amplitude
                            routes=((np.arange(m)[:,None]+np.arange(topk)[None,:]+repeat+lane)%4).astype('i4')
                            cols=js[:,np.arange(topk)%channels]
                            want=amplitude*weights[routes[:,:,None],np.arange(n),cols[:,:,None]].astype('f8')
                            expected.append(want)
                            a.write(values);ids.write(routes);out.poison()
                        arena.sync()
                        for lane,(stream,a,ids,out,scratch) in enumerate(lanes):
                            def run(a=a,ids=ids,out=out,scratch=scratch,stream=stream):
                                return api.run(handle,a.ptr,ids.ptr,out.ptr,scratch.ptr,scratch.size,stream)
                            if repeat==1:graphs.append(Replay(sdk,stream,run,1))
                            api.check(graphs[lane]() if graphs else run())
                        arena.sync()
                        for lane,(_,a,ids,out,scratch) in enumerate(lanes):
                            got=np.frombuffer(out.bytes(),'f4').reshape(m*topk,n)
                            want=expected[lane].reshape(m*topk,n)
                            compare(got,want,np.abs(want))
                            for b in (a,ids,scratch):b.guard()
                finally:
                    arena.sync()
                    for graph in graphs:graph.close()
    finally:api.destroy(context)


@pytest.mark.parametrize("q", QTYPES)
@pytest.mark.parametrize("tokens", DECODE_M)
@pytest.mark.parametrize("compute,channels,split", [(0, 1, 1), (1, 2, 2), (0, 2, 4), (1, 1, 8)])
def test_indexed_simt_formats_tokens_splits(package, sdk, q, tokens, compute, channels, split):
    execution = C.CDLL(str(package / "quactlize-runtime/libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL)
    query, launch = bind_simt_compute(execution)
    with Arena(sdk) as arena:
        arr, _, weight_ptrs, weights = packed(arena, Public(package), q, experts=4)
        arr = Arrangement.from_buffer_copy(arr)
        n, k, topk = 256, 512, 2
        stream = arena.stream()
        source = arena.buffer(tokens * channels * k * 4)
        ids = arena.buffer(tokens * topk * 4)
        output = arena.buffer(tokens * topk * n * 4)
        call = Call(version=1, size=C.sizeof(Call), qtype=q, n=n, k=k, experts=4,
            rows=tokens*topk, mode=2, input_type=1, channels=channels, topk=topk,
            a_row_stride=k, a_token_stride=k*channels, ids_stride=topk, out_row_stride=n,
            a=source.ptr, low=weight_ptrs[0], high=weight_ptrs[1], units=weight_ptrs[2],
            ids=ids.ptr, output=output.ptr, stream=stream.value)
        typed = SimtCallV2(call, compute)
        cfg = SimtConfig(0 if q == 8 else 3, 4, 8, 8, split)
        sizes = Sizes()
        checked(query(C.byref(typed), C.byref(cfg), C.byref(arr), C.byref(sizes)), "indexed SIMT query")
        scratch = arena.buffer(sizes.workspace_bytes)
        typed.call.workspace = scratch.ptr
        typed.call.workspace_bytes = sizes.workspace_bytes
        run = lambda: launch(C.byref(typed), C.byref(cfg), C.byref(arr))
        graph = None
        try:
            for repeat in range(3):
                a = np.random.default_rng(q + tokens * 31 + repeat).uniform(-.13, .13, (tokens, channels, k)).astype("f4")
                route = ((np.arange(tokens)[:, None] * 3 + np.arange(topk)[None, :] + repeat) % 4).astype("i4")
                source.write(a); ids.write(route); output.poison(); arena.sync()
                if repeat == 1:
                    graph = Replay(sdk, stream, run, 1)
                checked(graph() if graph else run(), "indexed SIMT run")
                arena.sync()
                got = np.frombuffer(output.bytes(), "f4").reshape(tokens, topk, n)
                for token in range(tokens):
                    for slot in range(topk):
                        check_output(got[token, slot:slot+1], a[token, slot % channels][None, :],
                                     weights[route[token, slot]], compute)
                source.guard(); ids.guard(); scratch.guard()
        finally:
            arena.sync()
            if graph:
                graph.close()


@pytest.mark.parametrize("q,n,k", [(12, 512, 2048), (13, 2048, 512)])
@pytest.mark.parametrize("route", [2, 3], ids=["fq", "sf"])
@pytest.mark.parametrize("tokens", [1, 128], ids=["decode", "prefill"])
def test_selected_grouped_tc_existing_oracle(package, sdk, tmp_path, q, n, k, route, tokens, record_property):
    # Reuse the existing independently checked eager + changed-router replay
    # gate, not a new copy of the grouped launcher/reference. Its diagnostic
    # sample is NOT exported as performance evidence by CI.
    from ci.support.weights import Weights
    from ci.support.native_gate import run
    runtime = package / "quactlize-runtime"
    jit = dict(python=sys.executable, helper=runtime / "jit/tools/kpack_jit.py",
               sdk=Path(os.environ["PPU_SDK"]),
               cache=Path(os.environ.get("QUACTLIZE_PPU_CACHE_DIR", str(tmp_path / "jit"))))
    result = run(sdk, runtime, Weights(q, n, k, 256), route, tokens, 1, jit=jit)
    assert result["status"] == "PASS" and result["graph_replays"] == 3
    record_property("selected_parent", result["selection"]["parent"])


@pytest.mark.parametrize("compute", ["fp16", "bf16"])
@pytest.mark.parametrize("tokens", [1, 2, 8])
@pytest.mark.parametrize("router", [False, True])
def test_selected_moe_chain_existing_oracle(package, sdk, tmp_path, compute, tokens, router):
    from ci.support.moe_gate import chain_case
    # The existing chain gate validates standalone gate/up/down and the fused
    # chain, including its SwiGLU range/rounding, GPU router and indexed finish.
    args = SimpleNamespace(sdk=Path(os.environ["PPU_SDK"]), bundle=package / "quactlize-runtime",
        jit_helper=package / "quactlize-runtime/jit/tools/kpack_jit.py",
        jit_cache=Path(os.environ.get("QUACTLIZE_PPU_CACHE_DIR", str(tmp_path / "jit"))),
        compute=compute, mixed=True, smallm_table=True, paired=False, samples=1)
    assert chain_case(args, sdk, Public(package).lib, True, tokens, router)["status"] == "PASS"
