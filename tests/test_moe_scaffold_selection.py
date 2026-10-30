"""Actual public C++ queries on host; no simulated GPU or performance claim."""
import ctypes as C
from collections import Counter
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from dev.moe_scaffold.protocol import Case,Plan,Prepare,Finish,PreparePlanV2,plan_calls
from dev.moe_scaffold.models import model_plan

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    path=tmp_path_factory.mktemp('moe-query')/'selection.so'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    str(ROOT/'quactlize/moe/selection.cpp'),'-o',str(path)],check=True)
    lib=C.CDLL(str(path))
    for name,typ in (('prepare',Prepare),('finish',Finish)):
        fn=getattr(lib,'quactlize_ppu_moe_'+name+'_select_v1')
        fn.argtypes=[C.POINTER(typ),C.c_int32,C.POINTER(Plan)];fn.restype=C.c_int
        setattr(lib,name,fn)
    lib.prepare_v2=lib.quactlize_ppu_moe_prepare_select_v2
    lib.prepare_v2.argtypes=[C.POINTER(Prepare),C.POINTER(PreparePlanV2)];lib.prepare_v2.restype=C.c_int
    return lib


def query(lib,call,choice=0):
    plan=Plan();fn=lib.prepare if isinstance(call,Prepare) else lib.finish
    assert fn(C.byref(call),choice,C.byref(plan))==0
    assert plan.version==1 and plan.size==C.sizeof(Plan)
    return plan


def test_model_query_inventory_uses_host_keys_not_route_names(library):
    counts=Counter();keys={};prepare=Counter()
    for case in model_plan()[0]:
        p,f=plan_calls(case)
        if case.phase=='prepare':
            plan=query(library,p);prepare[plan.implementation]+=1
            assert plan.stages==(4 if case.tokens>=32 else 3)
        if case.phase!='finish':continue
        plan=query(library,f);counts[plan.implementation]+=1
        key=bytes(f)
        if key in keys:assert keys[key]==bytes(plan)
        keys[key]=bytes(plan)
        if plan.implementation==19:
            assert case.down_order==0 and case.down_splits in (4,8)
            assert plan.stages==3 and plan.workspace_bytes==2*case.tokens*8*case.output*4
        else:assert plan.stages==1 and plan.workspace_bytes==0
    assert len(keys)==216
    assert prepare==Counter({1:82,2:90})
    assert counts==Counter({16:424,17:402,19:70})


@pytest.mark.parametrize('m',[0,9,17,31,33,65,129,257,513,1025,2049,4097,5121,65537])
def test_unknown_m_remains_supported_without_unmeasured_promotion(library,m):
    p,f=plan_calls(Case(m,columns=2048,output=3072,padding=0,down_splits=8))
    assert query(library,p).implementation==1
    plan=query(library,f)
    assert plan.implementation==16 and not plan.workspace_bytes and plan.stages==bool(m)
    assert query(library,f,19).stages==(3 if m else 0)


@pytest.mark.parametrize('change',[dict(columns=4096),dict(dtype=1),dict(padding=3),dict(experts=128),dict(topk=6)])
def test_prepare_scope_and_explicit_route_control(library,change):
    c=replace(Case(2048,columns=2048,output=3072,padding=0),**change)
    p,_=plan_calls(c)
    assert query(library,p).implementation==1
    for tile in (256,512,1024):
        p.config.tile_rows=tile
        assert query(library,p,3).stages==5
        assert query(library,p,2).stages==4
    p.config.algorithm=1
    assert query(library,p,3).stages==3
    assert query(library,p,3).workspace_bytes==0
    out=Plan();assert library.prepare(C.byref(p),2,C.byref(out))==20


@pytest.mark.parametrize('change',[dict(output=4096),dict(dtype=1),dict(padding=3),dict(topk=6),
                                     dict(down_order=1),dict(down_splits=3)])
def test_finish_unknown_storage_keeps_serial_and_explicit_arms(library,change):
    c=replace(Case(2048,output=3072,padding=0,down_splits=8),**change)
    _,f=plan_calls(c)
    assert query(library,f).implementation==16
    for implementation in (16,17,18,19):
        plan=query(library,f,implementation)
        assert plan.implementation==implementation


def test_actual_final_storage_not_original_split_drives_finish(library):
    _,f=plan_calls(Case(2048,output=3072,padding=0,down_splits=8))
    assert query(library,f).implementation==19
    f.down.state=0;f.down.partial_count=1
    assert query(library,f).implementation==16
    f.tokens=8
    assert query(library,f).implementation==17
    f.down.round_type=0
    assert query(library,f).implementation==16


def test_query_is_dynamic_binding_independent_and_rejects_bad_abi(library):
    p,f=plan_calls(Case(1024,columns=2048,output=3072,padding=0,down_splits=8))
    before=bytes(query(library,f))
    f.down.data=123;f.inverse_rows=456;f.weights=789;f.output=987;f.stream=654
    assert bytes(query(library,f))==before # Query never dereferences or validates addresses.
    out=Plan()
    for bad in (-1,1,2,3,20):assert library.finish(C.byref(f),bad,C.byref(out))==20
    f.version=2;assert library.finish(C.byref(f),0,C.byref(out))==20
    f.version=1;f.size=0;assert library.finish(C.byref(f),0,C.byref(out))==20
    p.config.algorithm=0;assert library.prepare(C.byref(p),0,C.byref(out))==20
    assert library.finish(None,0,C.byref(out))==20
    assert library.finish(C.byref(f),0,None)==20


def query_v2(lib,call):
    result=PreparePlanV2()
    assert lib.prepare_v2(C.byref(call),C.byref(result))==0
    assert (result.version,result.size,C.sizeof(result))==(2,48,48)
    return result


@pytest.mark.parametrize('columns',[2048,3072])
@pytest.mark.parametrize('m',list(range(1,10))+[16,32,64,128,256,512,1024,2048,4096,5120])
def test_combined_prepare_uses_measured_route_and_helper(library,columns,m):
    p,_=plan_calls(Case(m,columns=columns,padding=0))
    old=query(library,p)
    p.config.algorithm=p.config.tile_rows=0
    plan=query_v2(library,p)
    scalar=m<=9 or m in (16,32)
    if scalar:
        assert plan.config.algorithm==3
        assert (plan.helper.implementation,plan.helper.stages,plan.helper.workspace_bytes)==(3,3,0)
    else:
        assert (plan.config.algorithm,plan.config.tile_rows)==(2,512)
        assert bytes(plan.helper)==bytes(old)
    # The plan is immutable and does not capture live pointer or stream values.
    p.input=0x1234;p.route.ids=0x5678;p.output=0x9000;p.route.stream=0x1111
    assert bytes(query_v2(library,p))==bytes(plan)


@pytest.mark.parametrize('m',[0,10,15,17,31,33,65,129,257,513,1025,2049,4097,5121,65537])
def test_combined_prepare_unknown_m_keeps_general_path(library,m):
    p,_=plan_calls(Case(m,columns=2048,padding=0))
    old=query(library,p);p.config.algorithm=p.config.tile_rows=0
    plan=query_v2(library,p)
    assert (plan.config.algorithm,plan.config.tile_rows)==(2,512)
    assert bytes(plan.helper)==bytes(old)


@pytest.mark.parametrize('change',[dict(columns=4096),dict(dtype=1),dict(padding=3),dict(experts=128),dict(topk=6)])
def test_combined_prepare_scope_and_explicit_overrides(library,change):
    p,_=plan_calls(replace(Case(8,columns=2048,padding=0),**change))
    old=query(library,p);p.config.algorithm=p.config.tile_rows=0
    plan=query_v2(library,p)
    assert (plan.config.algorithm,plan.config.tile_rows)==(2,512)
    assert bytes(plan.helper)==bytes(old)
    for algorithm,tile in ((1,512),(2,256),(2,512),(2,1024),(3,512)):
        p.config.algorithm,p.config.tile_rows=algorithm,tile
        explicit=query_v2(library,p)
        assert bytes(explicit.config)==bytes(p.config)
        assert bytes(explicit.helper)==bytes(query(library,p))


def test_combined_prepare_invalid_call_is_not_masked(library):
    p,_=plan_calls(Case(8,columns=2048,padding=0));out=PreparePlanV2()
    p.config.algorithm=0  # AUTO tile must also be zero, not silently ignored.
    assert library.prepare_v2(C.byref(p),C.byref(out))==20
    p.config.tile_rows=0;p.config.version=0
    assert library.prepare_v2(C.byref(p),C.byref(out))==20
    p.config.version=1;p.route.tokens=-1
    assert library.prepare_v2(C.byref(p),C.byref(out))==20
    p.route.tokens=2**31-1
    assert library.prepare_v2(C.byref(p),C.byref(out))==26
    assert library.prepare_v2(None,C.byref(out))==20
    assert library.prepare_v2(C.byref(p),None)==20


@pytest.mark.parametrize('n',[2048,3072])
@pytest.mark.parametrize('m',[1,4,5,8,16,32,128,512,1024,2048,4096,5120])
@pytest.mark.parametrize('splits',[1,2,4,8])
def test_simple_finish_keeps_material_partial_exceptions(library,n,m,splits):
    _,f=plan_calls(Case(m,output=n,padding=0,down_splits=splits))
    separate=(n==2048 and (splits==8 and m>=1024 or splits==4 and m==4096) or
              n==3072 and (splits==8 and m>=512 or splits==4 and m>=1024))
    expected=17 if m*n<=32768 else 19 if separate else 16
    assert query(library,f).implementation==expected
    assert query(library,f,18).implementation==18  # explicit SLOTS8 remains ABI-compatible
