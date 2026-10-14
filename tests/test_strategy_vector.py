"""Exact experimental vector carrier: no device/performance claims."""
import ctypes as C
import os
from pathlib import Path
import subprocess

import pytest

from dev.strategy.vector import spec, source
from quactlize.dispatch.table import Recipe, Workload


def q4():
    return Recipe.make('q4','fq',reader=2,variant=7,columns=8,warps=8,values=4)


def test_q4_explicit_source_does_not_consult_selected_policy():
    w=Workload(12,512,2048,1)
    item=spec(w,q4()); text=source(item)
    assert 'q4_decode::launch<2,7,8,4,8,512,2048,0>' in text
    assert 'select_v' not in text and 'decode_policy' not in text
    assert 'q4_s1::buffers_v2' in text and 'execution::query' in text
    assert 'c.n!=512 || c.k!=2048' in text


def test_typed_q4_and_simt_keep_exact_shipping_body_and_real_reducer():
    w=Workload(12,1024,2048,1,'indexed','bf16',experts=256,topk=8)
    assert 'launch<2,7,8,4,8,1024,2048,1>' in source(spec(w,q4()))
    r=Recipe.make('simt','fq',variant=5,columns=8,warps=4,values=4,split=8)
    text=source(spec(Workload(8,2048,4096,1),r))
    assert 'measured_decode_launch<8>' in text
    assert 'simt::q8_vector::launch_v2<8,5,8,4,4>(*d,split)' in text
    assert 'simt::buffers_v2' in text and 'simt::query_v2' in text


def test_split_k_is_runtime_axis_not_four_new_compile_units():
    w=Workload(8,2048,4096,1)
    values=[spec(w,Recipe.make('simt','fq',variant=5,columns=8,warps=4,values=4,split=s)) for s in (1,2,4,8)]
    assert all(value==values[0] for value in values)


def test_source_rejects_undeclared_or_injected_config():
    item=spec(Workload(12,512,2048,1),q4())
    with pytest.raises((ValueError,TypeError)):
        source(item | dict(config=item['config'] | dict(warps='8;bad')))
    with pytest.raises(ValueError): spec(Workload(12,512,2048,9),q4())
    with pytest.raises(ValueError): spec(Workload(14,512,2048,1),q4())


@pytest.mark.parametrize(('q','compute','kind'), [(12,'f16','q4'),(12,'bf16','q4'),(8,'f16','simt'),(13,'bf16','simt')])
def test_native_compile_only(tmp_path,q,compute,kind):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk: pytest.skip('opt-in real HGCC compile-only; no runtime/device')
    from quactlize.runtime.compiler import FLAGS
    root=Path(__file__).resolve().parents[1]
    r=q4() if kind=='q4' else Recipe.make('simt','fq',variant=5 if q==8 else 3,columns=4,warps=4,values=4,split=2)
    item=spec(Workload(q,512,2048,1,compute=compute),r)
    path=tmp_path/'kernel.cu'; path.write_text(source(item))
    args=[str(Path(sdk)/'bin/hgcc'),*FLAGS,
          *['-I'+str(root/p) for p in ('quactlize/execution','quactlize/include','third_party/actlize/include',
                                      'third_party/actlize/tools/util/include')],'-c',str(path),'-o',str(tmp_path/'kernel.o')]
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('kind',['q4','simt'])
def test_generated_query_host_abi_and_negative_contract(tmp_path,kind):
    pytest.importorskip('torch')
    from quactlize.execution.native import Call,SimtCallV2,Sizes,Arrangement,arrangement
    q=12 if kind=='q4' else 8
    r=q4() if kind=='q4' else Recipe.make('simt','fq',variant=5,columns=8,warps=4,values=4,split=8)
    text=source(spec(Workload(q,512,2048,1),r))
    # Test the emitted query verbatim with the real host validation headers;
    # omit only the GPU launch function. This is not a device substitute.
    text=text.split('extern "C" int strategy_vector_run')[0]
    text=text.replace('#include "q4_decode_kernel.cuh"','#include "q4_s1_validation.hpp"')
    text=text.replace('#include "measured_decode.cuh"','')
    for py,cpp in [(Call,'qkg_call_v1'),(SimtCallV2,'qkg_simt_call_v2'),
                   (Sizes,'qkg_sizes_v1'),(Arrangement,'quactlize_ppu_placed_arrangement_v2')]:
        text+=f'static_assert(sizeof({cpp})=={C.sizeof(py)},"ABI size");\n'
        for field,_ in py._fields_:
            text+=f'static_assert(offsetof({cpp},{field})=={getattr(py,field).offset},"ABI field");\n'
    root=Path(__file__).resolve().parents[1]
    src=tmp_path/'query.cpp';src.write_text('#include <cstddef>\n'+text)
    so=tmp_path/'query.so'
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I'+str(root/'quactlize/execution'),
                    '-I'+str(root/'quactlize/include'),str(src),'-o',str(so)],check=True)
    lib=C.CDLL(str(so));query=lib.strategy_vector_query
    query.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.POINTER(Sizes)]
    query.restype=C.c_int
    c=Call(version=1,size=C.sizeof(Call),qtype=q,n=512,k=2048,experts=1,rows=1,
        mode=0,input_type=1,channels=1,topk=1,a_row_stride=2048,a_token_stride=2048,
        ids_stride=1,out_row_stride=512)
    request=SimtCallV2(c,0);a=arrangement(q);out=Sizes();split=1 if kind=='q4' else 8
    assert query(C.byref(request),C.byref(a),split,C.byref(out))==0
    assert out.low_bytes and out.units_bytes
    assert out.workspace_bytes==(0 if kind=='q4' else 8*512*4)
    assert query(C.byref(request),C.byref(a),3,C.byref(out))!=0
    request.compute_type=1
    assert query(C.byref(request),C.byref(a),split,C.byref(out))!=0
    request.compute_type=0;a.mapping_id=0
    assert query(C.byref(request),C.byref(a),split,C.byref(out))!=0


def test_expansion_native_compile_only(tmp_path):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:pytest.skip('opt-in real HGCC object/ISA check; no library/runtime')
    from quactlize.runtime.compiler import FLAGS
    from tools.build_kpack_dequant import inspect_image
    root=Path(__file__).resolve().parents[1];sdk=Path(sdk);obj=tmp_path/'dequant.o'
    command=[str(sdk/'bin/hgcc'),*FLAGS,*['-I'+str(root/p) for p in
        ('quactlize/dequant','quactlize/execution','quactlize/include','third_party/actlize/include')],
        '-c',str(root/'quactlize/dequant/kernels.cu'),'-o',str(obj)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    records,inventory=inspect_image(sdk,obj)
    assert len(records)==73
    assert inventory['sf_bfloat_units']==10
    assert any('prepass_unit_kernel' in name for name in records)
