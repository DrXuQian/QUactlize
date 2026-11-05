"""Independent byte ownership and typed weight reconstruction, not GPU timing."""
import ctypes as C
import numpy as np
import pytest

from tests.test_kpack_device_pack import library, Arrangement, Sizes
from tests.test_ppu_compat import host
from ci.support.compat import raw_blocks, canonical_planes, weight, encode


def descriptor(library,q):
    fn=library.quactlize_ppu_kpack_canonical_arrangement_v1
    fn.argtypes=[C.c_int,C.POINTER(Arrangement)]
    arr=Arrangement();assert fn(q,C.byref(arr))==0
    return arr


@pytest.mark.parametrize('q,n,k',[(7,16,128),(7,2560,640),(7,640,2560),
                                 (8,2560,640),(8,10240,320)])
@pytest.mark.parametrize('dtype',[0,1])
def test_packed_native_weight_bits(library,host,q,n,k,dtype):
    raw=raw_blocks(q,3,n,k)
    expected=canonical_planes(raw,q)
    guarded={key:np.full(value.size+32,0xA5,'u1') for key,value in expected.items()}
    planes={key:buf[16:-16] for key,buf in guarded.items()}
    pointers=[planes[key].ctypes.data if planes[key].size else None for key in ('low','high','units')]
    arr=descriptor(library,q);s=Sizes()
    assert library.quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,3,q,C.byref(arr),C.byref(s))==0
    assert s.raw_bytes==s.low_bytes+s.high_bytes+s.units_bytes==raw.nbytes
    assert library.host_pack(q,raw.ctypes.data,*pointers,n,k,3)==0
    for key in planes:
        np.testing.assert_array_equal(planes[key],expected[key])
        assert np.all(guarded[key][:16]==0xA5) and np.all(guarded[key][-16:]==0xA5)
    got=np.empty((3,n,k),'<u2')
    assert host.compat_decode_canonical(q,*pointers,got.ctypes.data,n,k,3,dtype)==0
    np.testing.assert_array_equal(got,encode(weight(raw,q,dtype),dtype))
    planes['units'][:] ^= 0x80
    assert host.compat_decode_canonical(q,*pointers,got.ctypes.data,n,k,3,dtype)==0
    assert not np.array_equal(got,encode(weight(raw,q,dtype),dtype))


def test_q5_1_identity_shape_overflow_and_preserved_metadata(library):
    arr=descriptor(library,7)
    assert tuple(getattr(arr,name) for name,_ in Arrangement._fields_) == (
        2,5,4,1,0,256,32,0,0x5135314b50000001)
    layout=C.create_string_buffer(128)
    assert library.quactlize_gate_up_layout_v1(7,C.byref(layout))==22
    query=library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    assert query(2560,640,512,7,C.byref(arr),C.byref(Sizes()))==0
    for field,_ in Arrangement._fields_:
        bad=Arrangement.from_buffer_copy(arr);setattr(bad,field,getattr(bad,field)^1)
        out=Sizes(1,2,3,4)
        assert query(2560,640,512,7,C.byref(bad),C.byref(out))==38
        assert bytes(out)==bytes(Sizes(1,2,3,4))
    q5k=descriptor(library,13)
    assert query(2560,640,512,7,C.byref(q5k),C.byref(Sizes()))==38
    for n,k,e in ((15,640,512),(2560,608,512),(2560,640,0),(2147483632,2147483520,2147483647)):
        out=Sizes(1,2,3,4)
        assert query(n,k,e,7,C.byref(arr),C.byref(out))!=0
        assert bytes(out)==bytes(Sizes(1,2,3,4))
    raw=np.random.default_rng(51).integers(0,256,(3,16,20,24),dtype='u1')
    # Metadata is copied as bits, including signed zeros, subnormals and NaNs.
    raw[...,:4]=np.resize(np.array([0,128,255,127,1,0,0,124],'u1'),raw[...,:4].shape)
    expected=canonical_planes(raw,7);got={key:np.empty_like(v) for key,v in expected.items()}
    assert library.host_pack(7,raw.ctypes.data,*[got[k].ctypes.data for k in got],16,640,3)==0
    for key in got:np.testing.assert_array_equal(got[key],expected[key])


@pytest.mark.parametrize('q',[7,8,12,13])
def test_pair_640_is_per_expert_concatenation(library,q):
    n,k,e=640,2560,3
    gate=raw_blocks(q,e,n,k,1);up=raw_blocks(q,e,n,k,2)
    expected=canonical_planes(np.concatenate((gate,up),axis=1),q)
    got={key:np.empty_like(v) for key,v in expected.items()}
    pair=library.host_pack_pair
    pair.argtypes=[C.c_int]+[C.c_void_p]*5+[C.c_int]*3
    assert pair(q,gate.ctypes.data,up.ctypes.data,
                *[got[key].ctypes.data if got[key].size else None for key in got],n,k,e)==0
    for key in got:np.testing.assert_array_equal(got[key],expected[key])
