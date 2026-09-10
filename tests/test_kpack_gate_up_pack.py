"""Paired raw sources must equal a single canonical [E,gate+up,K] producer."""
import ctypes as C
import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from tests.test_kpack_device_pack import library


@pytest.mark.parametrize("q", [8, 10, 11, 12, 13, 14])
@pytest.mark.parametrize("experts,n,k", [(1, 256, 512), (3, 256, 1024), (2, 512, 512)])
def test_pair_is_gate_then_up_within_each_expert(library, q, experts, n, k):
    block, width = (32, 34) if q == 8 else (256, ref.SPECS[q].raw_bytes)
    rng = np.random.default_rng(24680 + q + experts)
    gate, up = [rng.integers(0, 256, (experts, n, k//block, width), dtype=np.uint8) for _ in range(2)]
    joined = np.concatenate([gate, up], axis=1)
    if q == 8:
        codes = joined[...,2:].copy().reshape(experts,2*n,k)
        kk = np.arange(k)
        low = np.empty((experts,k//2,2*n,2),dtype=np.uint8)
        low[:,(kk//16)*8+kk%8,:,kk%16//8] = (codes^128).transpose(2,0,1)
        units = joined[...,:2].transpose(0,2,1,3).copy()
        expected = (low, np.empty(0, dtype=np.uint8), units)
    else:
        art = ref.prepare_grouped(torch.from_numpy(joined.reshape(-1,width)), 2*n, k, q, experts)
        expected = tuple(t.numpy().view(np.uint8).reshape(-1) for t in (art.low,art.high,art.units))
    expected = tuple(np.ascontiguousarray(x).view(np.uint8).reshape(-1) for x in expected)
    storage = [np.full(x.size+32,0xA5,dtype=np.uint8) for x in expected]
    outputs = [x[16:-16] for x in storage]
    fn = library.host_pack_pair
    fn.argtypes = [C.c_int, *([C.c_void_p]*5), C.c_int, C.c_int, C.c_int]
    fn.restype = C.c_int
    assert fn(q,gate.ctypes.data,up.ctypes.data,outputs[0].ctypes.data,
              outputs[1].ctypes.data if outputs[1].size else None,outputs[2].ctypes.data,n,k,experts) == 0
    for guard, got, want in zip(storage,outputs,expected):
        np.testing.assert_array_equal(got,want)
        assert np.all(guard[:16] == 0xA5) and np.all(guard[-16:] == 0xA5)
    # A whole-plane append is not the merged N layout, even for E=1.
    single = []
    for raw in (gate,up):
        out = [np.empty(x.size//2,dtype=np.uint8) for x in expected]
        assert library.host_pack(q,raw.ctypes.data,out[0].ctypes.data,
                                 out[1].ctypes.data if out[1].size else None,out[2].ctypes.data,n,k,experts) == 0
        single.append(out)
    assert not np.array_equal(np.concatenate([single[0][0],single[1][0]]),expected[0])
    # With one expert and one metadata unit along K, N is the only outer
    # metadata axis and appending happens to be correct. It is not general.
    one_unit = q in (11,14) and k == 512 and experts == 1
    assert np.array_equal(np.concatenate([single[0][2],single[1][2]]),expected[2]) == one_unit
