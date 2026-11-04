"""Actual NVIDIA execution through the PPU matmul record layout."""

import ctypes as C

import pytest
import torch

from quactlize.compat.native import Problem, Sizes, Buffers
from quactlize.cuda_compat import CudaPpuMatmulLibrary
from test_cuda_compat import raw_blocks, decoded, assert_gemm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA CUDA')


def buffers(source, raw, ids, output, workspace, error):
    b = Buffers(version=1, size=C.sizeof(Buffers))
    for name, t in (('input', source), ('raw', raw), ('ids', ids), ('output', output), ('workspace', workspace)):
        if t is not None:
            setattr(b, name, t.data_ptr())
            span = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride())) if t.numel() else 0
            setattr(b, name + '_bytes', span * t.element_size())
    b.device_error, b.stream = error.data_ptr(), torch.cuda.current_stream(source.device).cuda_stream
    return b


@pytest.mark.parametrize('q', [7, 8, 12])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('m', [1, 9])
@pytest.mark.parametrize('mode', ['dense', 'broadcast', 'slots'])
def test_strided_graph_replay_preserves_padding_and_sticky_status(q, dtype, m, mode):
    lib = CudaPpuMatmulLibrary()
    e, topk = (1, 1) if mode == 'dense' else (3, 2)
    n, k, pad = 37, (512 if q == 12 else 640), 7
    host_raw = raw_blocks(q, e, n, k)
    reference = decoded(host_raw, q, dtype)
    raw = torch.from_numpy(host_raw).cuda()

    def operands():
        base = torch.randn((m, topk, k + pad) if mode == 'slots' else (m, k + pad), dtype=dtype, device='cuda')
        source = base[..., :k]
        id_storage = torch.full((m, topk + pad), -2, dtype=torch.int32, device='cuda')
        ids = id_storage[:, :topk] if mode != 'dense' else None
        if ids is not None:
            ids.zero_()
        output_base = torch.full((m, topk, n + pad) if mode != 'dense' else (m, n + pad),
                                 -123, dtype=dtype, device='cuda')
        output = output_base[..., :n]
        return source, ids, output, output_base

    source, ids, output, output_base = operands()
    p = Problem(version=1, size=C.sizeof(Problem), qtype=q, layout=0, mode=int(ids is not None),
        dtype=int(dtype == torch.bfloat16), m=m, n=n, k=k, experts=e, topk=topk,
        channels=topk if mode == 'slots' else 1, device=0,
        input_row_stride=source.stride(1) if source.ndim == 3 else k,
        input_token_stride=source.stride(0), ids_stride=ids.stride(0) if ids is not None else 0,
        output_stride=output.stride(-2))
    sizes = Sizes()
    lib.check(lib.query(C.byref(p), None, C.byref(sizes)))
    assert sizes.version == 1 and sizes.size == C.sizeof(Sizes)
    assert sizes.route == 1  # This explicit NVIDIA implementation always uses GEMM.
    workspace = torch.empty(sizes.workspace_bytes, dtype=torch.uint8, device='cuda')
    error = torch.full((), 2, dtype=torch.int32, device='cuda')
    h = C.c_void_p()
    b = buffers(source, raw, ids, output, workspace, error)
    lib.check(lib.prepare(C.byref(p), None, C.byref(b), C.byref(h)))
    assert h.value and error.item() == 2  # Prepare must preserve the caller's bits.
    graph = torch.cuda.CUDAGraph()
    try:
        source, ids, output, output_base = operands()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            b = buffers(source, raw, ids, output, workspace, error)
            lib.check(lib.run(h, C.byref(b)))
        routes = ([0],) if mode == 'dense' else ([2, 2], [-1, 0], [3, 1], [0, 1])
        sticky = 2
        for route in routes:
            if ids is not None:
                ids.copy_(torch.tensor([route] * m, dtype=torch.int32, device='cuda'))
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            sticky |= int(3 in route)
            assert error.item() == sticky
            assert bool((output_base[..., n:] == -123).all())
            for slot, expert in enumerate(route):
                actual = output if mode == 'dense' else output[:, slot]
                if expert == -1 or expert >= e:
                    assert torch.count_nonzero(actual) == 0
                else:
                    x = (source[:, slot] if mode == 'slots' else source).cpu()
                    expected = x.double() @ reference[expert].double().T
                    assert_gemm(actual, expected, x, reference[expert])
        error.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert error.item() == 0
        # Rebinding must still reject aliasing and short physical extents.
        b = buffers(source, raw, ids, output, workspace, error)
        b.input_bytes -= 1
        assert lib.run(h, C.byref(b)) != 0
        b = buffers(source, raw, ids, output, workspace, error)
        b.output = b.input
        assert lib.run(h, C.byref(b)) != 0
    finally:
        torch.cuda.synchronize()
        graph.reset()
        lib.destroy(h)


@pytest.mark.parametrize('field,value', [('reserved', 1), ('m', 0), ('qtype', 20),
    ('input_token_stride', 1), ('output_stride', 36), ('topk', 33), ('experts', 1025)])
def test_new_query_rejects_unimplemented_or_malformed_contract(field, value):
    lib = CudaPpuMatmulLibrary()
    p = Problem(version=1, size=C.sizeof(Problem), qtype=7, layout=0, mode=0, dtype=1,
                m=1, n=37, k=640, experts=1, topk=1, channels=1, device=0,
                input_row_stride=640, input_token_stride=640, output_stride=37)
    setattr(p, field, value)
    assert lib.query(C.byref(p), None, C.byref(Sizes())) != 0


def test_native_handles_cannot_cross_legacy_and_new_entry_points():
    from quactlize.cuda_compat import PreparedMatmul, _buffers, _library

    lib = CudaPpuMatmulLibrary()
    source = torch.zeros((1, 640), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((1, 37), dtype=source.dtype, device=source.device)
    raw = torch.from_numpy(raw_blocks(8, 1, 37, 640)).cuda()
    legacy = PreparedMatmul(qtype=8, n=37, k=640, input=source, output=output, raw=raw)
    h = C.c_void_p()
    try:
        b = buffers(source, raw, None, output, legacy.workspace, legacy.device_error)
        assert lib.run(legacy.handle, C.byref(b)) != 0
        p = Problem(version=1, size=C.sizeof(Problem), qtype=8, layout=0, mode=0, dtype=1,
            m=1, n=37, k=640, experts=1, topk=1, channels=1, device=0,
            input_row_stride=640, input_token_stride=640, output_stride=37)
        lib.check(lib.prepare(C.byref(p), None, C.byref(b), C.byref(h)))
        old = _buffers(*legacy.args)
        assert _library().quactlize_cuda_run_v1(h, C.byref(old)) != 0
    finally:
        torch.cuda.synchronize()
        if h.value:
            lib.destroy(h)
        legacy.close()
