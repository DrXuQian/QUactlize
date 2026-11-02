"""Public projection/selection ABI; device accuracy is in ppu/test_moe_library.py."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source',['moe_prefill_policy_host','moe_route_host','moe_scaffold_host','moe_matmul_host','ffn_host'])
def test_host_contract(tmp_path,source):
    extra=[]
    if source=='moe_scaffold_host':extra=[str(ROOT/'quactlize/moe/selection.cpp')]
    if source in ('moe_matmul_host','ffn_host'):
        (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages{};\nstatic char const kJitSource[]="";\n')
        extra=[str(ROOT/p) for p in ('quactlize/packing/sizes.cpp','quactlize/execution/simt.cpp',
               'quactlize/execution/q4_decode.cpp','quactlize/moe/selection.cpp')]
    sdk=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
    exe=tmp_path/source
    subprocess.run(['g++','-std=c++17','-O1','-pthread','-rdynamic','-I'+str(ROOT),'-I'+str(tmp_path),
        '-I'+str(sdk/'include'),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
        str(ROOT/'tests'/f'{source}.cpp'),*extra,'-ldl','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_measured_table_generated_and_fresh_evidence():
    from quactlize.dispatch.moe_prefill_policy import load,header,HEADER,validate
    from quactlize.runtime.tuning import digest
    data=load()
    assert HEADER.read_text()==header(data)
    fresh=[r for r in data['rows'] if 'evidence' in r]
    assert len(fresh)==2 and all(r['key'][0]==13 and r['key'][-1]==5120 for r in fresh)
    assert all(len(p['selected_samples_us'])==12 for r in fresh for p in r['profiles'])
    bad=copy.deepcopy(data);bad['rows'][-1]['profiles'][0]['selected_samples_us'][0][0]*=2
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='evidence differs'):validate(bad)


def test_ffn_ctypes_layout():
    import ctypes as C
    from ci.support.ffn import Weight,Params,Info
    assert C.sizeof(Weight)==112 and C.sizeof(Params)==408 and C.sizeof(Info)==656
    assert Params.gate.offset==72 and Info.projections.offset==32
