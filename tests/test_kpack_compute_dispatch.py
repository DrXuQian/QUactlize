import ctypes as C
import importlib
import os
from pathlib import Path
import subprocess

import pytest

from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.dispatch.native import DenseCompute, GroupedCompute, MoeEndpointV4
from tools.build_kpack_dispatch import catalog
from tools.kpack_jit import module_source

ROOT=Path(__file__).resolve().parents[1]


def test_compute_dispatch_protocol(tmp_path):
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    binary=tmp_path/'probe'
    subprocess.run(['g++','-O1','-std=c++17',f'-I{ROOT}',f'-I{tmp_path}',
        str(ROOT/'tests/kpack_compute_dispatch_host.cpp'),'-ldl','-pthread','-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'six formats, separate tickets' in result.stdout


def test_catalog_separates_compute_and_storage():
    p=dict(symbol='same',qtype=14,route='fq-grouped',tm=8,tn=64,tk=128,
           wm=8,wn=16,stages=2,ap=0,dn=16,persistent=0)
    records=[dict(parent=p,key=str(i)*64,identity=dict(compute_type=t)) for i,t in enumerate(('f16','bf16'))]
    text=catalog(records)
    assert text.count('"same"')==2 and ',{},false,1}' in text


def test_compute_ctypes_match_c(tmp_path):
    source=tmp_path/'abi.cpp';binary=tmp_path/'abi'
    source.write_text('#include "quactlize/dispatch/api.h"\n#include <cstdio>\nint main(){'
        'printf("%zu %zu %zu",sizeof(qkd_dense_call_v2),sizeof(qk_compute_device_call_v3),sizeof(qks_moe_endpoint_v4));}')
    subprocess.run(['g++','-std=c++17',f'-I{ROOT}',str(source),'-o',str(binary)],check=True)
    assert list(map(int,subprocess.check_output([str(binary)],text=True).split()))==[
        C.sizeof(DenseCompute),C.sizeof(GroupedCompute),C.sizeof(MoeEndpointV4)]


def test_module_source_matches_bf16_compilers():
    for route,cls,endpoint in [('fq-grouped',GroupedComputeCompiler,'grouped-explicit-compute-v3'),
                               ('fq-dense',DecodeCompiler,'decode-m1-8-f32-bf16-v1')]:
        p=dict(symbol='fqg_q14_l2_tm8_tn64_tk128_wm8_wn16_s2_ap0_dn16_nonpersistent' if route.endswith('grouped') else
            'fqk_tc_q14_l2_a0_tm8_tn64_tk128_wm8_wn16_s2_bc0_ap0_dn16',qtype=14,route=route,
            tm=8,tn=64,tk=128,wm=8,wn=16,stages=2,ap=0,dn=16,persistent=0 if route.endswith('grouped') else -1)
        record=dict(parent=p,identity=dict(compute_type='bf16',endpoints=endpoint))
        # A real compiler source method, independent of JIT source reconstruction.
        obj=cls.__new__(cls);obj.compute_type='bf16'
        assert module_source(record)==cls.source(obj,p,'')


def test_caller_bf16_receipts_and_symbol_precision(monkeypatch):
    caller=Path(os.environ.get('LLAMA_CI_DIR','/root/autodl-tmp/llama-v0.3.0'))
    if not (caller/'tests/quactlize_native.py').exists():pytest.skip('private caller not installed')
    monkeypatch.syspath_prepend(str(caller/'tests'))
    native=importlib.import_module('quactlize_native')
    old='void quactlize::execution::simt::register_reuse<14, 1, 3, 4, 4, 4>(qkg_call_v1, int)'
    assert native.simt_symbol_recipe(old)==(14,1,3,4,4,4,0)
    assert native.simt_symbol_recipe(old.replace('4>','4, 1>'))==(14,1,3,4,4,4,1)
    line='[quactlize-plan] tensor=test op=grouped route=gemv reader=simt-reuse q=14 rows=8 n=512 k=2048 variant=3 columns=4 warps=4 values=4 split=1 policy=11 activation=BF16'
    manifest=dict(modules=[],smallm_policy=True,compute_contract=True,execution_receipt=dict(
        simt_compute_v2=dict(compute=['f16','bf16']),simt_configs={'14':[
            dict(variant=3,columns=4,warps=4,values=4,split=1)]}))
    assert native.selection(line,manifest,['grouped'])['fully_selected']
    with pytest.raises(ValueError):native.selection(line.replace('policy=11','policy=9'),manifest,['grouped'])
    with pytest.raises(ValueError):native.selection(line,manifest|dict(compute_contract=None),['grouped'])
    module=dict(key='b'*64,parent=dict(symbol='tc',qtype=14,route='fq-grouped'),identity=dict(compute_type='bf16'))
    tc='[quactlize-plan] tensor=test op=grouped route=fq q=14 rows=8 n=512 k=2048 parent=tc build='+module['key']+' split=1 grid=0 policy=11 activation=BF16'
    assert native.selection(tc,manifest|dict(modules=[module]),['grouped'])['fully_selected']
    with pytest.raises(ValueError):native.selection(tc.replace('activation=BF16','activation=FP16'),manifest|dict(modules=[module]),['grouped'])
