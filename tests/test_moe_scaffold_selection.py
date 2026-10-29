"""Actual public C++ queries on host; no simulated GPU or performance claim."""
import ctypes as C
from collections import Counter
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from dev.moe_scaffold.protocol import Case,Plan,Prepare,Finish,plan_calls
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
    assert counts==Counter({16:424,17:244,18:158,19:70})


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
