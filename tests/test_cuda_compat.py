"""Real NVIDIA format and graph qualification, independent GGUF/FP64 oracles."""
import ctypes as C
from dataclasses import asdict

import gguf
import numpy as np
import pytest
import torch

from quactlize.cuda_compat import PreparedMatmul, Problem, Sizes, _library, canonical_arrangement
from reference import gguf_kpack as ref
from tools.run_kpack_pack_gate import Arrangement


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
FORMATS = [7, 8, 10, 11, 12, 13, 14, 20]


@pytest.mark.parametrize('q', FORMATS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_direct_embedding_rows_bit_exact_graph(q, dtype):
    from quactlize.cuda_compat import gather_rows

    k = 2560 if q == 8 else 160 if q == 20 else 512 if q in range(10, 15) else 640
    raw = raw_blocks(q, 19, 1, k)
    expected_table = decoded(raw, q, dtype).squeeze(1)
    device_raw = torch.from_numpy(raw).cuda()
    ids = torch.zeros(17, dtype=torch.int64, device='cuda')
    live = torch.full((1,), 17, dtype=torch.int32, device='cuda')
    error = torch.zeros((), dtype=torch.int32, device='cuda')
    output = torch.empty((17, k), dtype=dtype, device='cuda')
    def run():
        gather_rows(device_raw, ids, output, live, error, q, 19)
    compiled = torch.compile(run, fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    try:
        for count in (17, 13, 1, 0, -1, 18, 17):
            host_ids = torch.tensor([18, 0, 18, -1, 19, -2, 1 << 40, 3, 8, 1, 7, 11, 16, 0, 1, 2, 3])
            ids.copy_(host_ids)
            live.fill_(count)
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            expected = torch.zeros_like(output, device='cpu')
            status = 0
            if not 0 <= count <= 17:
                status = 2
            else:
                for row, selected in enumerate(host_ids[:count].tolist()):
                    if 0 <= selected < 19:
                        expected[row] = expected_table[selected]
                    elif selected != -1:
                        status = 1
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            assert error.item() == status
    finally:
        stream.synchronize()
        graph.reset()


def test_direct_embedding_rows_reject_bad_buffers_and_overflow():
    from quactlize.cuda_compat import gather_rows

    raw = torch.from_numpy(raw_blocks(8, 32, 1, 256)).cuda()
    ids = torch.zeros(3, dtype=torch.int64, device='cuda')
    output = torch.empty((3, 256), dtype=torch.bfloat16, device='cuda')
    error = torch.zeros((), dtype=torch.int32, device='cuda')
    with pytest.raises(RuntimeError, match='capacity'):
        gather_rows(raw[:1], ids, output, None, error, 8, 32)
    alias = raw.flatten()[:output.nbytes].view(torch.bfloat16).view_as(output)
    with pytest.raises(RuntimeError, match='disjoint'):
        gather_rows(raw, ids, alias, None, error, 8, 32)
    with pytest.raises(ValueError, match='native ABI'):
        gather_rows(raw, ids, output, None, error, (1 << 32) + 8, 32)
    # Torch's SymInt schema rejects this before the Python/C ABI boundary.
    with pytest.raises(RuntimeError, match="Expected a value of type 'int'"):
        gather_rows(raw, ids, output, None, error, 8, (1 << 64) + 32)


def raw_blocks(q, e, n, k):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(q)]
    rng = np.random.default_rng(7100 + q)
    raw = rng.integers(0, 256, (e, n, k // block, size), dtype=np.uint8)
    offsets = {7: (0, 2), 8: (0,), 10: (80, 82), 11: (108,),
               12: (0, 2), 13: (0, 2), 14: (208,), 20: (0,)}[q]
    for offset in offsets:
        scales = rng.uniform(-0.005, 0.005, raw.shape[:-1]).astype('<f2')
        raw[..., offset:offset + 2] = scales[..., None].view(np.uint8)
    return raw


def decoded(raw, q, dtype):
    arr = gguf.dequantize(raw.reshape(-1, raw.shape[-1]), gguf.GGMLQuantizationType(q))
    shape = (*raw.shape[:2], -1)
    return torch.from_numpy(arr.reshape(shape)).to(dtype)


def assert_gemm(actual, expected, a, w):
    # Compare against FP64 multiplication of the already rounded 16-bit values.
    # Bound includes the final output rounding plus the FP32 dot accumulation.
    # abs(A)@abs(W) conditions near-zero outputs without weakening the format test.
    magnitude = (a.double().abs() @ w.double().transpose(-1, -2).abs())
    bound = expected.abs() * torch.finfo(actual.dtype).eps + magnitude * (2e-6 + a.shape[-1] * 2**-24)
    assert torch.all((actual.cpu().double() - expected).abs() <= bound + 1e-7)


@pytest.mark.parametrize("q", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 4, 64])
def test_raw_dense(q, dtype, m):
    n, k = 32, 512 if q in range(10, 15) else 640
    raw = raw_blocks(q, 1, n, k)
    expected_w = decoded(raw, q, dtype)[0]
    generator = torch.Generator().manual_seed(19)
    a = torch.randn((m, k), generator=generator, dtype=dtype)
    output = torch.empty((m, n), device='cuda', dtype=dtype)
    call = PreparedMatmul(qtype=q, n=n, k=k, input=a.cuda(), output=output, raw=torch.from_numpy(raw).cuda())
    expanded = call.dequantize().cpu()[0]
    assert torch.equal(expanded.view(torch.int16), expected_w.view(torch.int16))
    call()
    expected = a.double() @ expected_w.double().T
    assert_gemm(output, expected, a, expected_w)
    assert call.device_error.item() == 0
    torch.cuda.synchronize()
    call.close()


@pytest.mark.parametrize("q", [8, 10, 11, 12, 13, 14])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_canonical_packing_bytes_and_gemm(q, dtype):
    n, k, e = 256, 512, 3
    raw = raw_blocks(q, e, n, k)
    arr = canonical_arrangement(q)
    if q != 8:
        assert bytes(arr) == bytes(Arrangement(**asdict(ref.canonical_arrangement(q))))
    problem = Problem(1, C.sizeof(Problem), q, 1, 1, int(dtype == torch.bfloat16), 2, n, k, e, 2, 1, 0)
    sizes = Sizes(); lib = _library()
    assert lib.quactlize_cuda_query_v1(C.byref(problem), C.byref(arr), C.byref(sizes)) == 0
    planes = [torch.empty(getattr(sizes, name + '_bytes'), dtype=torch.uint8, device='cuda')
              for name in ('low', 'high', 'units')]
    device_raw = torch.from_numpy(raw).cuda()
    pack = lib.quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2
    pack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_void_p] * 2
    assert pack(device_raw.data_ptr(), planes[0].data_ptr(), planes[1].data_ptr() if planes[1].numel() else None,
                planes[2].data_ptr(), n, k, e, q, C.byref(arr), torch.cuda.current_stream().cuda_stream) == 0
    if q == 8:
        # Independent logical-byte scatter from the Q8 producer contract.
        codes = raw[..., 2:].reshape(e, n, k)
        expected_low = np.empty((e, k // 2, n, 2), dtype=np.uint8)
        kk = np.arange(k)
        expected_low[:, (kk // 16) * 8 + kk % 8, :, kk % 16 // 8] = (codes ^ 128).transpose(2, 0, 1)
        expected_planes = [torch.from_numpy(expected_low), torch.empty(0, dtype=torch.uint8),
                           torch.from_numpy(raw[..., :2].transpose(0, 2, 1, 3).copy())]
    else:
        expected_pack = ref.prepare_grouped(torch.from_numpy(raw.reshape(-1, raw.shape[-1])), n, k, q, e)
        expected_planes = (expected_pack.low, expected_pack.high, expected_pack.units)
    for actual, expected in zip(planes, expected_planes):
        assert torch.equal(actual.cpu().flatten(), expected.flatten())
    a = torch.randn((2, k), dtype=dtype)
    ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    output = torch.empty((2, 2, n), dtype=dtype, device='cuda')
    call = PreparedMatmul(qtype=q, n=n, k=k, experts=e, input=a.cuda(), output=output, ids=ids.cuda(),
        low=planes[0], high=planes[1] if planes[1].numel() else None, units=planes[2], arrangement=arr)
    w = decoded(raw, q, dtype)[ids.long()]
    expanded = call.dequantize().cpu().reshape_as(w)
    assert torch.equal(expanded.view(torch.int16), w.view(torch.int16))
    call()
    x = a[:, None, None, :].expand(2, 2, 1, k)
    expected = (x.double() @ w.double().transpose(-1, -2))
    assert_gemm(output.unsqueeze(-2), expected, x, w)
    torch.cuda.synchronize()
    call.close()


@pytest.mark.parametrize("q,n,k,m", [(7, 32, 640, 3), (12, 640, 2560, 1), (8, 2560, 640, 4)])
@pytest.mark.parametrize("channels", [1, 10])
def test_graph_dynamic_512_experts_top10(q, n, k, m, channels):
    dtype = torch.bfloat16
    raw = raw_blocks(q, 512, n, k)
    a = torch.randn((m, channels, k), dtype=dtype)
    ids = torch.zeros((m, 10), dtype=torch.int32, device='cuda')
    output = torch.empty((m, 10, n), dtype=dtype, device='cuda')
    call = PreparedMatmul(qtype=q, n=n, k=k, experts=512, input=a.cuda(), output=output,
                          ids=ids, raw=torch.from_numpy(raw).cuda())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for shift in (0, 13, 67):
        selection = (torch.arange(m * 10).reshape(m, 10) + shift) % 512
        selection[:, :4] = torch.tensor([511, 511, 0, -1])
        ids.copy_(selection.to(torch.int32))
        graph.replay()
        selected = raw[selection.clamp(min=0).numpy()].reshape(m * 10, n, *raw.shape[2:])
        w = decoded(selected, q, dtype).reshape(m, 10, n, k)
        w[selection < 0] = 0
        expanded = call.expanded.cpu().reshape_as(w)
        assert torch.equal(expanded.view(torch.int16), w.view(torch.int16))
        x = a.expand(m, 10, k).unsqueeze(-2).double()
        expected = x @ w.double().transpose(-1, -2)
        assert_gemm(output.unsqueeze(-2), expected, x, w)
        assert call.device_error.item() == 0
    ids.fill_(512)
    graph.replay()
    assert call.device_error.item() == 1
    assert torch.count_nonzero(output).item() == 0
    ids.zero_()
    graph.replay()
    assert call.device_error.item() == 0
    torch.cuda.synchronize()
    del graph
    call.close()


def test_geometry_and_buffer_rejection():
    lib = _library()
    arr = canonical_arrangement(8)
    p = Problem(1, C.sizeof(Problem), 8, 1, 0, 0, 1, 256, 640, 1, 1, 1, 0)
    assert lib.quactlize_cuda_query_v1(C.byref(p), C.byref(arr), C.byref(Sizes())) != 0
    p.layout = 0
    assert lib.quactlize_cuda_query_v1(C.byref(p), None, C.byref(Sizes())) == 0
    x = torch.ones((1, 640), device='cuda', dtype=torch.float16)
    out = torch.empty((1, 256), device='cuda', dtype=torch.float16)
    raw = torch.from_numpy(raw_blocks(8, 1, 256, 640)).cuda()
    with pytest.raises(ValueError, match='capacity'):
        PreparedMatmul(qtype=8, n=256, k=640, input=x, output=out, raw=raw,
                       workspace=torch.empty(1, device='cuda', dtype=torch.uint8))
    call = PreparedMatmul(qtype=8, n=256, k=640, input=x, output=out, raw=raw)
    call.close()
    with pytest.raises(RuntimeError, match='closed'):
        call()


@pytest.mark.parametrize("n,k", [(4, 10240), (10240, 320), (2560, 640)])
def test_q8_hyperconnection_and_short_k(n, k):
    raw = raw_blocks(8, 1, n, k)
    a = torch.randn((1, k), dtype=torch.bfloat16)
    output = torch.empty((1, n), dtype=a.dtype, device='cuda')
    call = PreparedMatmul(qtype=8, n=n, k=k, input=a.cuda(), output=output, raw=torch.from_numpy(raw).cuda())
    call()
    w = decoded(raw, 8, a.dtype)[0]
    assert_gemm(output, a.double() @ w.double().T, a, w)
    torch.cuda.synchronize()
    call.close()


def test_torch_compile_fullgraph_and_replay():
    n, k = 64, 640
    raw = raw_blocks(7, 1, n, k)
    a = torch.randn((4, k), dtype=torch.float16, device='cuda')
    output = torch.empty((4, n), dtype=a.dtype, device='cuda')
    call = PreparedMatmul(qtype=7, n=n, k=k, input=a, output=output, raw=torch.from_numpy(raw).cuda())
    compiled = torch.compile(call, fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    w = decoded(raw, 7, a.dtype)[0]
    for _ in range(2):
        a.normal_()
        graph.replay()
        x = a.cpu()
        assert_gemm(output, x.double() @ w.double().T, x, w)
    torch.cuda.synchronize()
    del graph, compiled
    call.close()
