"""Host contracts and regression boundaries; not PPU execution admission."""
from pathlib import Path
import subprocess
import ctypes as C

import pytest

from test_kpack_native_dispatch import build_stub,probe,Choice
from quactlize.decode.native_compute import GroupedDeviceCall,GroupedMetadataCall
from quactlize.runtime.native import Call

ROOT=Path(__file__).resolve().parents[1]


def test_grouped_dynamic_host_contract(tmp_path):
    exe=tmp_path/'grouped-dynamic'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT),
                    str(ROOT/'tests/kpack_grouped_dynamic_host.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_grouped_dynamic_does_not_change_legacy_device_bodies():
    # The new endpoint calls these exact legacy device functions and kernel
    # types; no replacement kernel, small-M cap lift or new arithmetic here.
    text=(ROOT/'quactlize/runtime/grouped_dynamic.cuh').read_text()
    for marker in ('grouped_splitk_device_metadata<<<','grouped_device_metadata<<<',
                   'launch_build<tm>','G::run(params,stream)','reduction.run(stream)',
                   'typename Bound::Reduction','using Bound=GroupedHandle'):
        assert marker in text
    assert '__global__' not in text
    run=text.split('int run_grouped_dynamic(',1)[1].split('extern "C"',1)[0]
    for forbidden in ('make_unique','std::vector','Memcpy','Synchronize','maximum_active_blocks',
                      'GetDevice','FuncSetAttribute','gemm.initialize','rows_host['):
        assert forbidden not in run
    assert 'mode==QK_GROUPED_COMPLETE && reduction.run' in run
    assert 'auto params=K::to_underlying_arguments(args,base)' in run


def test_result_abi_is_c_compatible(tmp_path):
    subprocess.run(['cc','-std=c11','-Werror','-I'+str(ROOT),'-x','c','-fsyntax-only','-'],
        input='#include "quactlize/runtime/abi.h"\n_Static_assert(sizeof(qk_grouped_result_v1)==48,"ABI");\n',
        text=True,check=True)


class Result(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('data',C.c_void_p)]+[
        (n,C.c_int32) for n in ('dtype','partial_count','compute_type','reserved')]+[
        (n,C.c_int64) for n in ('row_stride','partial_stride')]


@pytest.mark.parametrize('present',[False,True])
def test_dispatch_grouped_ticket_and_lifetime(tmp_path,probe,present):
    f,runtime,request=build_stub(tmp_path,probe)
    if present:
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-DQK_TEST_GROUPED_DYNAMIC',
            '-I'+str(ROOT),'-I'+str(tmp_path),str(ROOT/'tests/kpack_dispatch_stub.cpp'),
            '-o',str(tmp_path/'modules'/('a'*64)/'kernel.so')],check=True)
    lib=C.CDLL(str(tmp_path/'dispatch.so'))
    prepare=lib.quactlize_kpack_dispatch_prepare_grouped_dynamic_v1
    prepare.argtypes=[C.c_void_p,C.POINTER(Choice),C.POINTER(C.c_void_p)];prepare.restype=C.c_int
    run=lib.quactlize_kpack_dispatch_run_grouped_dynamic_v1
    run.argtypes=[C.c_void_p,C.POINTER(GroupedMetadataCall),C.c_int];run.restype=C.c_int
    output=lib.quactlize_kpack_dispatch_grouped_output_v1
    output.argtypes=run.argtypes+[C.POINTER(Result)];output.restype=C.c_int
    choice,handle=Choice(),C.c_void_p()
    assert f['query'](runtime,C.byref(request),C.byref(choice))==0
    good=bytes(choice);choice.split=99
    assert prepare(runtime,C.byref(choice),C.byref(handle))==2 and not handle.value
    choice=Choice.from_buffer_copy(good)
    rc=prepare(runtime,C.byref(choice),C.byref(handle))
    f['close'](runtime)
    if not present:
        assert rc==3 and not handle.value and b'rebuild' in f['error']()
        return
    assert rc==0 and handle.value
    try:
        # Fake, disjoint addresses for the non-dereferencing host descriptor;
        # this test does not pretend the stub launches a device kernel.
        c=Call(version=1,size=C.sizeof(Call),m=request.m,n=request.n,k=request.k,experts=request.experts,
               group_size=32,device=choice.device,compute_units=choice.compute_units,mapping_id=request.mapping_id,
               a=1<<40,low=2<<40,metadata=3<<40,output=4<<40,offsets_device=5<<40,
               workspace=6<<40,workspace_bytes=256,stream=1)
        d=GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),
            GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),c,request.max_rows,0),0,0)
        for i in (0,1):
            d.device_call.call.output=(4+i*8)<<40
            d.device_call.call.stream=123+i
            result=Result();assert output(handle,C.byref(d),1,C.byref(result))==0
            assert result.partial_count==choice.split and result.compute_type==0
            assert f['run'](handle,None)==2
        d.compute_type=1
        assert output(handle,C.byref(d),1,C.byref(result))==2 and not result.data
    finally:f['destroy'](handle)
