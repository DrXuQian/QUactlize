import ctypes as C
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from quactlize import serving
from quactlize.compat.serving_native import Router, SwiGLU, Rows
from ci.support.serving import row_blocks, dequant_rows, router_reference

ROOT=Path(__file__).resolve().parents[1]


def test_public_header_is_pure_c():
    text='#include "quactlize/compat/serving.h"\n_Static_assert(sizeof(qpm_router_v2)==160,"ABI");\n'
    subprocess.run(['gcc','-std=c11','-fsyntax-only','-x','c','-I'+str(ROOT),
                    '-I'+str(ROOT/'quactlize/include'),'-'],input=text,text=True,check=True)


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    out=tmp_path_factory.mktemp('serving-host')/'host.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT),
        '-I'+str(ROOT/'quactlize/include'),str(ROOT/'tests/serving_host.cpp'),'-o',str(out)],check=True)
    lib=C.CDLL(str(out))
    for name,cls in (('router',Router),('swiglu',SwiGLU),('rows',Rows)):
        getattr(lib,'serving_'+name+'_check').argtypes=[C.POINTER(cls)]
    return lib


def row_call():
    # Fake addresses are host validation ONLY, never dereferenced.
    return Rows(1,C.sizeof(Rows),20,1,4,160,7,7*90,4*8,4*160*2,
        0x10000,0x20000,0x30000,0x40000,0x50000,1)


def test_abi_and_64bit_row_extent(host):
    for i,cls in enumerate((Router,SwiGLU,Rows)):assert host.serving_size(i)==C.sizeof(cls)
    p=row_call();assert host.serving_rows_check(C.byref(p))==0
    p.table_rows=100_000_000;p.raw_bytes=p.table_rows*90
    p.ids=0x10000000000;p.output=0x20000000000;p.num_rows=0x30000000000;p.device_error=0x40000000000
    assert p.raw_bytes>2**32 and host.serving_rows_check(C.byref(p))==0
    p.raw_bytes=2**32-1
    assert host.serving_rows_check(C.byref(p))!=0
    p.table_rows=2**63-1;p.raw_bytes=2**64-1
    assert host.serving_rows_check(C.byref(p))!=0


@pytest.mark.parametrize('field,value',[('qtype',7),('qtype',12),('columns',161),('table_rows',-1),
    ('capacity',-1),('raw_bytes',629),('ids_bytes',31),('output_bytes',1279),('ids',0x20001),
    ('num_rows',0x30000),('device_error',0x30000),('raw',0x30000),('stream',None),('dtype',2)])
def test_invalid_rows_rejected(host,field,value):
    p=row_call();setattr(p,field,value)
    assert host.serving_rows_check(C.byref(p))!=0


def test_zero_capacity_and_count_still_required_status(host):
    p=row_call();p.capacity=0;p.ids=None;p.output=None
    assert host.serving_rows_check(C.byref(p))==0
    p.device_error=None
    assert host.serving_rows_check(C.byref(p))!=0


def test_router_and_swiglu_host_contract(host):
    p=Router(2,C.sizeof(Router),4,512,10,1,0,1,1.,0,520,12,12,12,
        4*520*2,4*12*4,4*12*4,4*12*4,0x10000,0x20000,0x30000,0x40000,0x50000,0x60000,1)
    assert host.serving_router_check(C.byref(p))==0
    for field,value in [('score_func',1),('renormalize',0),('selected_bytes',4),('weights',p.logits),
                        ('device_error',p.num_valid_tokens),('dtype',0),('logits_stride',511),('multiplier',float('nan'))]:
        bad=Router.from_buffer_copy(p);setattr(bad,field,value)
        assert host.serving_router_check(C.byref(bad))!=0
    s=SwiGLU(2,C.sizeof(SwiGLU),4,160,1,1,1,0,322,0,162,4*322*2,0,4*162*2,0x10000,None,0x20000,1)
    assert host.serving_swiglu_check(C.byref(s))==0
    for field,value in [('round_silu',2),('up',0x30000),('gate_stride',319),('output',s.gate),('output_bytes',1)]:
        bad=SwiGLU.from_buffer_copy(s);setattr(bad,field,value)
        assert host.serving_swiglu_check(C.byref(bad))!=0


def test_reference_ties_underflow_and_inactive():
    x=np.zeros((4,512),'f4');x[1,:]=-10000;x[1,0]=0;x[1,1]=-90;x[1,2]=-91;x[2:]=np.nan
    selected,ids,w,status=router_reference(x,10,2)
    np.testing.assert_array_equal(ids[0],np.arange(511,501,-1))
    assert list(ids[1,:3])==[0,1,2] and status==0
    assert (ids[2:]==-1).all() and (w[2:]==0).all() and (selected[2:]==0).all()
    assert router_reference(x,10,3)[3]==2
    assert router_reference(x,10,5)[3]==4


def test_iq4_nibbles_and_q8_signed_extremes():
    iq=row_blocks(20,1,160);dq=dequant_rows(iq,20)
    table=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],'f4')
    np.testing.assert_array_equal(dq[0,:16],table*.03125)
    np.testing.assert_array_equal(dq[0,16:32],table[::-1]*.03125)
    assert iq.shape==(1,90)
    q8=row_blocks(8,1,2560);dq=dequant_rows(q8,8)
    np.testing.assert_array_equal(dq[0,:32],np.arange(-128,-96,dtype='f4')*.03125)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_round_silu_has_observable_witness(dtype):
    gate=torch.linspace(-4,4,4096).to(dtype);up=torch.full_like(gate,1.5703125)
    a=gate.float()*torch.sigmoid(gate.float())
    once=(a*up.float()).to(dtype);twice=(a.to(dtype).float()*up.float()).to(dtype)
    assert torch.count_nonzero(once!=twice)>100


def test_fake_ops_and_complete_mutation_schemas():
    expected={'router':('selected','ids','weights','error'),'swiglu':('output',),
              'gather_rows':('output','error'),'merge':('output',),'matmul':('output','workspace','error')}
    for name,writes in expected.items():
        schema=getattr(torch.ops.quactlize_serving,name).default._schema
        assert {a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write}==set(writes)
    with FakeTensorMode():
        error=torch.empty(1,dtype=torch.int32);count=torch.empty_like(error)
        ids=torch.empty((4,10),dtype=torch.int32);weights=torch.empty((4,10));sel=torch.empty_like(weights)
        serving.router(torch.empty((4,512),dtype=torch.bfloat16),sel,ids,weights,error,count,1.,9999)
        out=torch.empty((40,640),dtype=torch.bfloat16)
        serving.swiglu(torch.empty((40,1280),dtype=torch.bfloat16),None,out,True,9999)
        serving.gather_rows(torch.empty(7*90,dtype=torch.uint8),torch.empty(4,dtype=torch.int64),
            torch.empty((4,160),dtype=torch.bfloat16),count,error,20,7,9999)
        serving.gather_rows(torch.empty(7*180,dtype=torch.uint8),torch.empty(4,dtype=torch.int64),
            torch.empty((4,640),dtype=torch.bfloat16),count,error,42,7,9999)
        serving.merge(torch.empty((4,10,640),dtype=torch.bfloat16),weights,torch.empty((4,640)),9999)
        serving.matmul(torch.empty((4,1,640),dtype=torch.bfloat16),torch.empty(100,dtype=torch.uint8),
            None,None,None,ids,torch.empty((4,10,640),dtype=torch.bfloat16),torch.empty(256,dtype=torch.uint8),error,9999)
    assert not serving._libraries and not serving._plans


def test_compile_export_does_not_lazy_load():
    class Chain(torch.nn.Module):
        def forward(self,gate,output):
            serving.swiglu(gate,None,output,True,1234)
            return output
    ep=torch.export.export(Chain(),(torch.empty((4,1280),dtype=torch.bfloat16),
                                   torch.empty((4,640),dtype=torch.bfloat16)))
    assert any('quactlize_serving.swiglu' in str(n.target) for n in ep.graph.nodes)
    assert not serving._libraries
