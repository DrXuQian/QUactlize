from pathlib import Path
import subprocess

import pytest

from dev.gemv_simt import spec

ROOT = Path(__file__).resolve().parents[1]


def test_all_formats_runtime_inventory():
    assert spec.QTYPES == (8, 10, 11, 12, 13, 14)
    for q in spec.QTYPES:
        candidates = spec.runtime_inventory(q)
        assert len(candidates) == (120 if q==8 else 240)
        assert len({c.key for c in candidates}) == len(candidates)
        assert {c.split for c in candidates} == {1,2,4,8}
        assert {c.values for c in candidates} == {2,4,8}
        assert {c.warps for c in candidates} == {2,4,8}
        assert all(c.tile_n<=32 for c in candidates)
        assert all(not(c.variant&2) for c in candidates) if q==8 else True


def test_codegen_query_and_launch_have_same_inventory():
    from quactlize.execution import simt_codegen
    assert spec.source is simt_codegen.source
    assert spec.runtime_inventory is simt_codegen.runtime_inventory
    for profile in ('full','smoke'):
        for q in spec.QTYPES:
            source = spec.source(q,profile)
            for c in spec.inventory(q,profile):
                condition = f'f->variant=={c.variant} && f->columns=={c.columns} && f->warps=={c.warps} && f->values=={c.values}'
                assert source.count(condition)==3
                assert source.count(f'simt::launch<{q},{c.variant},{c.columns},{c.warps},{c.values}>')==1
                assert source.count(f'simt::launch_v2<{q},{c.variant},{c.columns},{c.warps},{c.values}>')==1


def test_execution_build_includes_the_shared_candidates():
    from tools import build_kpack_execution
    import inspect
    src=inspect.getsource(build_kpack_execution.build)
    assert 'simt_codegen.source(q)' in src
    assert 'source / "simt.cpp"' in src
    assert 'simt_selection="EXPLICIT_CALLER_CONFIG_NO_POLICY_CHANGE"' in src


def test_ctypes_simt_config_and_moe_v3_sizes(tmp_path):
    import ctypes as C
    from quactlize.execution.native import SimtConfig
    from dev.gemv_simt.native import NativeConfig
    from quactlize.dispatch.native import MoeEndpoint, MoeEndpointV3
    c=NativeConfig(spec.Config(3,8,4,4,2))
    assert C.sizeof(SimtConfig)==C.sizeof(c)==c.size==28
    assert c.variant==3 and c.columns==8 and c.values==4 and c.split==2
    src=tmp_path/'abi.cpp';exe=tmp_path/'abi'
    src.write_text('#include "quactlize/dispatch/api.h"\n#include <cstdio>\n'
                   'int main(){std::printf("%zu %zu %zu\\n",sizeof(qkg_simt_config_v1),'
                   'sizeof(qks_moe_endpoint_v2),sizeof(qks_moe_endpoint_v3));}\n')
    subprocess.run(['g++','-std=c++17',f'-I{ROOT}',str(src),'-o',str(exe)],check=True)
    assert subprocess.check_output([str(exe)],text=True).strip()==f'{C.sizeof(c)} {C.sizeof(MoeEndpoint)} {C.sizeof(MoeEndpointV3)}'


@pytest.mark.parametrize('q',spec.QTYPES)
def test_group_and_warp_ownership(q):
    group = 32 if q in (8,12,13) else 16
    unit_groups = {8:1,10:16,11:32,12:8,13:8,14:32}[q]
    for k in (512,2048,3072,5120,8192,25600):
        for c in spec.runtime_inventory(q):
            workers = c.warps*32//c.columns
            owners = []
            for s in range(c.split):
                for worker in range(workers):
                    owners += list(range(s*workers+worker,k//group,c.split*workers))
            assert sorted(owners)==list(range(k//group))
            for begin in range(0,k//group,32//c.columns):
                assert begin+32//c.columns<=k//group
                if c.variant&2:
                    assert len({g//unit_groups for g in range(begin,begin+32//c.columns)})==1


def test_full_shape_registry_is_additive():
    from dev.gemv_ppu.decode_sweep import workloads
    old = workloads()
    new = spec.sweep_workloads()
    assert len(new)==6*len(old)==2232
    assert len({w['id'] for w in new})==len(new)
    for q in spec.QTYPES:
        assert [w | {'id':w['id'].removeprefix(f'q{q}-')} for w in new if w['qtype']==q] == [w|{'qtype':q} for w in old]


def test_q5_high_plane_and_actual_load_footprint():
    from dev.gemv_simt.access import pattern, plane_word
    for q in spec.QTYPES:
        c=spec.Config(1 if q==8 else 3,8,4,4)
        m=pattern(q,c,256,512,bases=dict(A=16,low=32,high=64,units=80))
        assert all(s['footprint']['32']['sectors']>0 for s in m['streams'])
        assert any(s['name'].startswith('high-') for s in m['streams']) == (q in (11,13,14))
        for p in (2,4,8):
            for col in range(0,256,p):
                for kk in range(512):
                    high=q in (11,13,14)
                    base=plane_word(q,high,col,kk,256)
                    assert all(plane_word(q,high,col+i,kk,256)==base+i for i in range(p))
    assert plane_word(13,True,0,128,256)==8
    assert plane_word(13,True,8,0,256)==0


def test_host_query_and_failure_contract(tmp_path):
    source = ROOT/'tests/simt_register_reuse_host.cpp'
    binary = tmp_path/'check'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',f'-I{ROOT}/quactlize/include',
                    str(source),'-o',str(binary)],check=True)
    result = subprocess.run([str(binary)],check=True,text=True,capture_output=True)
    assert 'SIMT_HOST PASS' in result.stdout


def test_fixture_upload_is_ordered_on_consumer_stream():
    import numpy as np
    from dev.gemv_simt.native import Runtime
    rt=Runtime.__new__(Runtime)
    rt.stream=object();calls=[]
    rt.MemcpyAsync=lambda dst,src,n,kind,stream: calls.append(('copy',dst,n,kind,stream)) or 0
    rt.sync=lambda: calls.append(('synchronize',))
    data=np.arange(16,dtype='<f4')
    rt.copy(4096,data)
    assert calls==[('copy',4096,64,1,rt.stream),('synchronize',)]
