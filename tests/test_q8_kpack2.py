"""Q8 producer ownership, signed bytes, exact scale bits and initial policy."""
import ctypes as C
import numpy as np
import pytest

from tests.test_kpack_device_pack import library, Arrangement, Sizes
from tests.test_kpack_native_dispatch import probe, query
from quactlize.runtime.compiler import validate_parent
from tools.build_kpack_dispatch import plan


def descriptor():
    return Arrangement(2,4,8,0,0,32,32,0,0x51384B5032540001)


@pytest.mark.parametrize("experts,n,k", [(1,256,256),(3,256,512),(2,512,1024)])
def test_raw_bytes_to_q8_kpack2(library,experts,n,k):
    raw=np.random.default_rng(82026).integers(0,256,(experts,n,k//32,34),dtype=np.uint8)
    codes=raw[...,2:].reshape(experts,n,k)
    # Independent logical scatter: write each source byte to its documented
    # physical b16 slot, not by replaying the GPU word gather.
    want=np.empty((experts,k//2,n,2),dtype=np.uint8)
    kk=np.arange(k)
    want[:,(kk//16)*8+kk%8,:,kk%16//8]=(codes^128).transpose(2,0,1)
    scale=raw[...,:2].transpose(0,2,1,3).copy().reshape(-1)
    sizes=Sizes()
    assert library.quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,experts,8,C.byref(descriptor()),C.byref(sizes))==0
    assert (sizes.raw_bytes,sizes.low_bytes,sizes.high_bytes,sizes.units_bytes)==(raw.nbytes,codes.size,0,scale.size)
    low_guard=np.full(codes.size+32,0xA5,dtype=np.uint8)
    scale_guard=np.full(scale.size+32,0xA5,dtype=np.uint8)
    low,meta=low_guard[16:-16],scale_guard[16:-16]
    assert library.host_pack(8,raw.ctypes.data,low.ctypes.data,None,meta.ctypes.data,n,k,experts)==0
    assert np.array_equal(low,want.reshape(-1))
    assert np.array_equal(meta,scale)
    for p in (low_guard,scale_guard):
        assert np.all(p[:16]==0xA5) and np.all(p[-16:]==0xA5)
    got=low.reshape(experts,k//2,n,2)[:,(kk//16)*8+kk%8,:,kk%16//8].transpose(1,2,0)^128
    assert np.array_equal(got,codes)
    # Wrong signedness and row-major byte reading must both be detected.
    assert np.any((low^128)!=want.reshape(-1))
    assert np.any(low!=(codes^128).reshape(-1))


def test_q8_descriptor_is_producer_owned_and_fail_closed(library):
    canonical=library.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes=[C.c_int,C.POINTER(Arrangement)]
    good=Arrangement()
    assert canonical(8,C.byref(good))==0 and bytes(good)==bytes(descriptor())
    for name,_ in Arrangement._fields_:
        wrong=descriptor()
        setattr(wrong,name,getattr(wrong,name)^1)
        sizes=Sizes(1,2,3,4)
        assert library.quactlize_ppu_kpack_sizes_for_arrangement_v1(256,512,1,8,C.byref(wrong),C.byref(sizes))==38
        assert bytes(sizes)==bytes(Sizes(1,2,3,4))
    assert canonical(9,C.byref(good))!=0 and bytes(good)==bytes(Arrangement())


def test_q8_policy_has_no_m_holes_and_is_not_a_measured_claim(probe):
    rows=[]
    for r,e in ((1,1),(3,8)):
        for m in (1,7,8,9,16,17,63,64,65,128,512,1024,4096):
            rows.append((8,r,m,1024,5120,e,m))
    for row,result in zip(rows,query(probe,rows)):
        p=result.split()
        assert p[0].startswith("sf") and int(p[1])==8 and int(p[2])==row[1]
        assert int(p[-1])==6
        split=int(p[12])
        assert row[4]%(64*split)==0 and row[4]//(64*split)>=2
    assert query(probe,[(8,r,8,1024,5120,1,8) for r in (0,2)])==["MISS","MISS"]


def test_q8_jit_disallows_packed_units_and_packed_activation(tmp_path):
    parents,_=plan(tmp_path,[(8,1,1,1024,5120,1,1)])
    p=parents[0]
    validate_parent(p)
    with pytest.raises(ValueError): validate_parent(p|dict(route="fq-dense"))
    with pytest.raises(ValueError): validate_parent(p|dict(ap=1))
