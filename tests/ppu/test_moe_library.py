"""MoE accuracy using the private closure shipped with the public package.

These low-level interfaces remain separate from the public dense caller API.
No llama checkout is needed. Recipes are bounded CI witnesses, not a sweep.
"""
import ctypes as C
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quactlize.execution.native import Arrangement, Call, SimtCallV2, SimtConfig, Sizes, bind_simt_compute
from quactlize.runtime.native import checked
from tools.run_kpack_grouped_decode_probe import Replay
from .support import Arena, DECODE_M, Public, QTYPES, check_output, packed


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
    from tools.kpack_warmup_fixture import Weights
    from tools.run_kpack_native_gate import run
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
    from tools.run_kpack_moe_gate import chain_case
    # The existing chain gate validates standalone gate/up/down and the fused
    # chain, including its SwiGLU range/rounding, GPU router and indexed finish.
    args = SimpleNamespace(sdk=Path(os.environ["PPU_SDK"]), bundle=package / "quactlize-runtime",
        jit_helper=package / "quactlize-runtime/jit/tools/kpack_jit.py",
        jit_cache=Path(os.environ.get("QUACTLIZE_PPU_CACHE_DIR", str(tmp_path / "jit"))),
        compute=compute, mixed=True, smallm_table=True, paired=False, samples=1)
    assert chain_case(args, sdk, Public(package).lib, True, tokens, router)["status"] == "PASS"
