import ctypes as C
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from quactlize.compat.native import Problem, Sizes, Buffers, Router, SwiGLU, Merge, problem
from ci.support.compat import FORMATS, DENSE_Q8, EXPERT_SHAPES, raw_blocks, weight, encode
from ci.support.packing import Arrangement

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    sdk = Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK'))
    if not (sdk/'include/hggc_runtime.h').is_file():
        pytest.skip('host checks require PPU SDK headers, not a device')
    out = tmp_path_factory.mktemp('compat-host')/'host.so'
    cmd = ['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-ffp-contract=off',
           '-I'+str(ROOT), '-I'+str(ROOT/'quactlize/include'),
           '-I'+str(ROOT/'third_party/actlize/include'), '-I'+str(sdk/'include'),
           str(ROOT/'tests/compat_host.cpp'), str(ROOT/'quactlize/packing/sizes.cpp'), '-o', str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lib = C.CDLL(str(out))
    lib.compat_query.argtypes = [C.POINTER(Problem), C.c_void_p, C.POINTER(Sizes)]
    lib.compat_validate.argtypes = [C.POINTER(Problem), C.POINTER(Buffers)]
    lib.compat_decode.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int]
    lib.compat_decode_canonical.argtypes = [C.c_int] + [C.c_void_p]*4 + [C.c_int]*4
    return lib


def test_native_abi(host):
    for i, cls in enumerate((Problem, Sizes, Buffers, Router, SwiGLU, Merge)):
        assert host.compat_abi_size(i) == C.sizeof(cls)


@pytest.mark.parametrize('q', FORMATS)
@pytest.mark.parametrize('dtype', [0, 1])
def test_raw_fields_order_and_exact_target_rounding(host, q, dtype):
    raw = raw_blocks(q, 2, 37, 512)
    golden = encode(weight(raw, q, dtype), dtype)
    got = np.empty_like(golden)
    assert host.compat_decode(q, raw.ctypes.data, got.ctypes.data, got.size, dtype) == 0
    np.testing.assert_array_equal(got, golden)
    broken = raw.copy()
    broken[..., -1] ^= 0xff
    assert host.compat_decode(q, broken.ctypes.data, got.ctypes.data, got.size, dtype) == 0
    assert not np.array_equal(got, golden), 'wrong field/byte control went undetected'


@pytest.mark.parametrize('q', (8,10,11,12,13,14))
@pytest.mark.parametrize('dtype', [0,1])
def test_canonical_reader_matches_independent_gguf(host,q,dtype):
    from ci.support.fixture import planes
    raw=raw_blocks(q,2,256,512)
    prepared=[planes(e,q) for e in raw]
    p={key:np.concatenate([e[key].ravel().view('u1') for e in prepared]) for key in ('low','high','units')}
    golden=encode(weight(raw,q,dtype),dtype);got=np.empty_like(golden)
    args=[p[key].ctypes.data if p[key].size else None for key in ('low','high','units')]
    assert host.compat_decode_canonical(q,*args,got.ctypes.data,256,512,2,dtype)==0
    np.testing.assert_array_equal(got,golden)


@pytest.mark.parametrize('m', [0, 1, 2, 4, 8, 9, 128, 2048, 5120])
@pytest.mark.parametrize('q,n,k,e,t', [(8,n,k,1,1) for n,k in DENSE_Q8] +
                         [(q,n,k,512,10) for q,n,k in EXPERT_SHAPES])
def test_true_model_shapes_and_linear_workspace(host, m, q, n, k, e, t):
    p = problem(q,m,n,k,experts=e,topk=t,channels=t)
    s = Sizes()
    # The vocabulary projection can exceed the explicit int32 matrix domain
    # at very large M; callers chunk that projection rather than overflow.
    if m*t*max(n,k) > 2**31-1:
        assert host.compat_query(C.byref(p), None, C.byref(s)) != 0
        return
    assert host.compat_query(C.byref(p), None, C.byref(s)) == 0
    if m <= 8:
        assert s.workspace_bytes == 0
    else:
        assert s.expanded_weight_bytes <= 2*4*512*k
        assert s.grouped_input_bytes == 2*(m*t+32)*k
        assert s.grouped_output_bytes == 2*m*t*((n+7)//8*8)
        assert s.workspace_bytes < s.expanded_weight_bytes + s.grouped_input_bytes + s.grouped_output_bytes + 8*m*t + 8*(e+1) + 8192


def test_canonical_still_rejects_special_shapes_and_wrong_mapping(host):
    arr = Arrangement()
    fn = host.quactlize_ppu_kpack_canonical_arrangement_v1
    fn.argtypes = [C.c_int, C.POINTER(Arrangement)]
    assert fn(8, C.byref(arr)) == 0
    s = Sizes()
    for n,k in ((320,10240),(10240,320),(4,10240),(2560,640)):
        p = problem(8,1,n,k,layout=1)
        assert host.compat_query(C.byref(p), C.byref(arr), C.byref(s)) != 0
    p = problem(8,1,256,512,layout=1)
    assert host.compat_query(C.byref(p), C.byref(arr), C.byref(s)) == 0
    arr.mapping_id ^= 1
    assert host.compat_query(C.byref(p), C.byref(arr), C.byref(s)) != 0
    p.layout = 0
    assert host.compat_query(C.byref(p), C.byref(arr), C.byref(s)) != 0
    p.qtype = 20
    assert host.compat_query(C.byref(p), None, C.byref(s)) != 0


@pytest.mark.parametrize('field,value', [('k', 641), ('channels', 2), ('experts', 1025),
    ('topk', 33), ('m', -1), ('input_token_stride', 1), ('output_stride', 1), ('dtype', 2),
    ('layout', 2), ('qtype', 20), ('input_token_stride', 2**63-1), ('reserved',1)])
def test_fail_closed_shape_and_overflow(host, field, value):
    p = problem(7,64,2560,640,experts=512,topk=10)
    setattr(p, field, value)
    assert host.compat_query(C.byref(p),None,C.byref(Sizes())) != 0


def test_capacity_alignment_alias_and_zero_m(host):
    p = problem(7,1,37,640)
    s = Sizes(); assert host.compat_query(C.byref(p),None,C.byref(s)) == 0
    b = Buffers(); b.version=1; b.size=C.sizeof(b)
    b.raw=0x100000; b.raw_bytes=s.raw_bytes
    b.input=0x200000; b.input_bytes=s.input_bytes
    b.output=0x300000; b.output_bytes=s.output_bytes; b.device_error=0x400000
    assert host.compat_validate(C.byref(p),C.byref(b)) == 0
    b.output_bytes-=1
    assert host.compat_validate(C.byref(p),C.byref(b)) != 0
    b.output_bytes+=1; b.output=b.input
    assert host.compat_validate(C.byref(p),C.byref(b)) != 0
    b.output=0x300001
    assert host.compat_validate(C.byref(p),C.byref(b)) != 0
    p.m=0; b.output=None; b.input=None
    assert host.compat_validate(C.byref(p),C.byref(b)) == 0
