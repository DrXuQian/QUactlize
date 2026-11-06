"""ID42 byte/ABI proofs on the host; never presented as PPU execution evidence."""
import ctypes as C

import numpy as np
import pytest

from ci.support.compat import raw_blocks, canonical_planes, encode, weight
from ci.support.q2_0 import dequantize, recover
from quactlize.compat.formats import GGML_TYPE_Q2_0, RAW_BLOCK_SIZES
from quactlize.compat.native import problem, Sizes as ComputeSizes
from quactlize.compat.serving_native import Rows
from tests.test_kpack_device_pack import library, Arrangement, Sizes
from tests.test_ppu_compat import host
from tests.test_q5_1_kpack import descriptor
from tests.test_serving import host as serving_host


def test_q2_0_is_not_q2_k():
    assert GGML_TYPE_Q2_0 == 42 and RAW_BLOCK_SIZES[42] == (64, 18)
    assert RAW_BLOCK_SIZES[10] == (256, 84)
    raw = np.zeros((6, 18), 'u1')
    scales = np.array([1, -2, 0., -0., 2**-24, -2**-24], '<f2')
    raw[:, :2] = scales[:, None].view('u1')
    raw[:, 2:] = 0xe4  # 00,01,10,11, in consecutive element order.
    expected = np.tile(np.array([-1, 0, 1, 2], 'f4'), 16) * scales.astype('f4')[:, None]
    np.testing.assert_array_equal(dequantize(raw).view('u4'), expected.view('u4'))


@pytest.mark.parametrize('dtype',[0,1])
def test_q2_0_all_finite_scales_and_codes(host,dtype):
    bits=np.arange(65536,dtype='<u2')
    bits=bits[np.isfinite(bits.view('<f2'))]
    raw=np.empty((bits.size,18),'u1');raw[:,:2]=bits[:,None].view('u1');raw[:,2:]=0xe4
    with np.errstate(over='ignore'):
        gold=encode(dequantize(raw),dtype)
    got=np.empty_like(gold)
    assert host.compat_decode(42,raw.ctypes.data,got.ctypes.data,got.size,dtype)==0
    np.testing.assert_array_equal(got,gold)  # Includes signed zero and target subnormal bits.


@pytest.mark.parametrize('n,k', [(4,64), (37,192), (640,2560), (1280,2560), (2560,640)])
@pytest.mark.parametrize('dtype', [0,1])
def test_q2_0_pack_and_decode(library, host, n, k, dtype):
    raw = raw_blocks(42, 3, n, k)
    expected = canonical_planes(raw, 42)
    guards = {name: np.full(value.size+32, 0xa5, 'u1') for name, value in expected.items()}
    planes = {name: value[16:-16] for name, value in guards.items()}
    ptrs = [planes[name].ctypes.data if planes[name].size else None for name in planes]
    assert library.host_pack(42, raw.ctypes.data, *ptrs, n, k, 3) == 0
    for name in planes:
        np.testing.assert_array_equal(planes[name], expected[name])
        assert (guards[name][:16] == 0xa5).all() and (guards[name][-16:] == 0xa5).all()
    np.testing.assert_array_equal(recover(planes['low'], planes['units'], 3, n, k), raw)
    gold = encode(weight(raw, 42, dtype), dtype)
    got = np.empty_like(gold)
    assert host.compat_decode(42, raw.ctypes.data, got.ctypes.data, got.size, dtype) == 0
    np.testing.assert_array_equal(got, gold)
    assert host.compat_decode_canonical(42, *ptrs, got.ctypes.data, n, k, 3, dtype) == 0
    np.testing.assert_array_equal(got, gold)


def test_q2_0_all_scale_bits_preserved(library):
    raw = np.random.default_rng(42).integers(0,256,(1,1024,64,18),dtype='u1')
    raw[...,:2] = np.arange(65536,dtype='<u2').reshape(1,1024,64,1).view('u1')
    want = canonical_planes(raw,42)
    got = {name: np.empty_like(value) for name,value in want.items()}
    assert library.host_pack(42,raw.ctypes.data,got['low'].ctypes.data,None,
                             got['units'].ctypes.data,1024,4096,1) == 0
    for name in got: np.testing.assert_array_equal(got[name],want[name])
    np.testing.assert_array_equal(recover(got['low'],got['units'],1,1024,4096),raw)


def test_q2_0_identity_and_fail_closed(library):
    arr = descriptor(library,42)
    assert tuple(getattr(arr,name) for name,_ in arr._fields_) == (2,6,2,0,0,128,64,0,0x5132304b50000001)
    query = library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    s = Sizes()
    assert query(2560,640,512,42,C.byref(arr),C.byref(s)) == 0
    assert s.raw_bytes == s.low_bytes+s.units_bytes == 512*2560*10*18 and s.high_bytes == 0
    for field,_ in arr._fields_:
        bad = Arrangement.from_buffer_copy(arr); setattr(bad,field,getattr(bad,field)^1)
        s = Sizes(1,2,3,4)
        assert query(2560,640,512,42,C.byref(bad),C.byref(s)) == 38
        assert bytes(s) == bytes(Sizes(1,2,3,4))
    assert query(256,512,1,42,C.byref(descriptor(library,10)),C.byref(s)) == 38
    assert library.quactlize_gate_up_layout_v1(42,C.create_string_buffer(128)) == 22
    for n,k,e in ((0,640,1),(2560,608,512),(2560,640,0),(2147483647,2147483584,2147483647)):
        assert query(n,k,e,42,C.byref(arr),C.byref(s)) != 0


@pytest.mark.parametrize('n,k,c',[(640,2560,1),(1280,2560,1),(2560,640,10)])
@pytest.mark.parametrize('m',[0,1,2,4,6,7,8,9,16,128,2048])
@pytest.mark.parametrize('dtype',[0,1])
def test_q2_0_true_workspace_bound(library,host,n,k,c,m,dtype):
    arr = descriptor(library,42)
    for layout in (0,1):
        p = problem(42,m,n,k,experts=512,topk=10,channels=c,dtype=dtype,layout=layout,padding=3)
        s = ComputeSizes()
        assert host.compat_query(C.byref(p),C.byref(arr) if layout else None,C.byref(s)) == 0
        assert s.raw_bytes == (0 if layout else 512*n*k//64*18)
        assert s.route == (0 if m<=8 else 1)
        if m<=8: assert s.workspace_bytes == 0
        else:
            assert s.expanded_weight_bytes == 2*4*512*k
            assert s.grouped_input_bytes == 2*(m*10+32)*k
            assert s.grouped_output_bytes == 2*m*10*n
            assert s.workspace_bytes < (s.expanded_weight_bytes+s.grouped_input_bytes+
                                        s.grouped_output_bytes+8*m*10+8*513+8192)


def test_q2_0_serving_row_extents(serving_host):
    p = Rows(1,C.sizeof(Rows),42,1,4,640,7,7*180,32,4*640*2,
             0x10000,0x20000,0x30000,None,0x40000,1)
    assert serving_host.serving_rows_check(C.byref(p)) == 0
    for field,value in [('columns',608),('raw_bytes',7*180-1),('qtype',10),('dtype',2)]:
        bad=Rows.from_buffer_copy(p);setattr(bad,field,value)
        assert serving_host.serving_rows_check(C.byref(bad)) != 0
