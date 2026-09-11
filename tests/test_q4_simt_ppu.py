"""Host gates for the PPU comparison; not PPU numeric/performance admission."""
import json
import ctypes as C
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.gemv_ppu.build import ppu_api, q4_dispatch, candidate_validation
from dev.gemv_ppu.campaign import parse_cells, validate_fq_refresh, cached_batch_matches, FQ_REFRESH_CAMPAIGNS
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.run import configs, check_reduction, verify_bundle, query_l2_attribute, resolve_l2

ROOT=Path(__file__).resolve().parents[1]


def refresh_authority():
    return dict(native="old",campaign=next(iter(FQ_REFRESH_CAMPAIGNS)),runner="runner",
                bundle="simt",device={"pci":"one"},runtime={"lib":"same"},
                fixtures={"512x2048":"same"},samples=15,confirm_rounds=6,l2_override=67108864)


def test_native_only_refresh_is_explicit_and_keeps_simt_authority():
    previous=refresh_authority()
    current=previous|dict(native="new",campaign="new-campaign")
    assert validate_fq_refresh(previous,previous,False) is False
    with pytest.raises(ValueError):validate_fq_refresh(previous,current,False)
    assert validate_fq_refresh(previous,current,True) is True
    assert validate_fq_refresh(current,current|dict(native="next"),True) is True


@pytest.mark.parametrize("key",["runner","bundle","device","runtime","fixtures","samples","confirm_rounds","l2_override"])
def test_fq_refresh_cannot_excuse_any_simt_measurement_change(key):
    previous=refresh_authority()
    current=previous|dict(native="new",campaign="new-campaign")
    current[key]="changed"
    with pytest.raises(ValueError):validate_fq_refresh(previous,current,True)


def test_refresh_rejects_unknown_or_incomplete_authority():
    previous=refresh_authority()
    current=previous|dict(native="new",campaign="new-campaign")
    with pytest.raises(ValueError):validate_fq_refresh(previous|dict(campaign="unknown"),current,True)
    del previous["samples"]
    with pytest.raises(ValueError):validate_fq_refresh(previous,current,True)


def test_fq_cache_is_bound_to_native_manifest_but_simt_is_independent(tmp_path):
    log=tmp_path/"batch.log";log.write_text("record")
    cmd=["python","run.py","--child"]
    saved=dict(command=cmd,log_sha256=digest(log),rc=0)
    for arm in ("xplane","old","new"):
        assert cached_batch_matches(saved,cmd,log,arm,"new-native")
    assert not cached_batch_matches(saved,cmd,log,"fq","new-native")
    saved["native_manifest_sha256"]="old-native"
    assert not cached_batch_matches(saved,cmd,log,"fq","new-native")
    assert cached_batch_matches(saved,cmd,log,"fq","old-native")
    assert not cached_batch_matches(saved,cmd+["--profile"],log,"fq","old-native")
    assert not cached_batch_matches(saved|dict(rc=1),cmd,log,"fq","old-native")
    log.write_text("modified")
    assert not cached_batch_matches(saved,cmd,log,"new","new-native")


class AttributeLibrary:
    """Optional-query behavior only; never masquerades as a device launch."""
    def __init__(self,*,value=64*1024**2,rc=0,deferred=0):
        self.calls=[]
        def query(out,attribute,device):
            self.calls.append((attribute,device))
            if value is not None:C.cast(out,C.POINTER(C.c_int))[0]=value
            return rc
        def clear():self.calls.append("clear");return deferred
        self.hggcDeviceGetAttribute=query
        self.hggcGetLastError=clear


def test_zero_properties_uses_separate_attribute_not_a_guessed_capacity():
    lib=AttributeLibrary()
    observed=query_l2_attribute(lib)
    r=resolve_l2(0,0,observed)
    assert lib.calls==[(38,0)]
    assert r['l2_bytes']==64*1024**2 and r['l2_source']=='DEVICE_ATTRIBUTE_38'
    assert r['reported_l2_bytes']==0 and r['l2_override'] is False


@pytest.mark.parametrize('rc',[1,801,998])
def test_optional_query_rejection_is_recorded_without_poisoning_launch(rc):
    lib=AttributeLibrary(rc=rc,deferred=rc)
    observed=query_l2_attribute(lib)
    assert observed==dict(status=rc,bytes=0) and lib.calls[-1]=='clear'
    with pytest.raises(ValueError,match='confirmed capacity'):resolve_l2(0,0,observed)
    r=resolve_l2(0,64*1024**2,observed)
    assert r['l2_source']=='EXPLICIT_OVERRIDE'


@pytest.mark.parametrize('value,status',[(None,'UNWRITTEN'),(0,0)])
def test_success_without_capacity_does_not_invent_l2(value,status):
    observed=query_l2_attribute(AttributeLibrary(value=value))
    assert observed==dict(status=status,bytes=0)
    with pytest.raises(ValueError):resolve_l2(0,0,observed)


def test_missing_attribute_and_real_runtime_failures_are_distinct():
    assert query_l2_attribute(object())==dict(status='NOT_EXPORTED',bytes=0)
    with pytest.raises(RuntimeError,match='status=200'):query_l2_attribute(AttributeLibrary(rc=200))
    with pytest.raises(RuntimeError,match='status=700'):query_l2_attribute(AttributeLibrary(rc=1,deferred=700))
    with pytest.raises(ValueError,match='negative'):query_l2_attribute(AttributeLibrary(value=-2))


def test_explicit_capacity_keeps_both_sdk_observations():
    r=resolve_l2(0,64*1024**2,dict(status=0,bytes=32*1024**2))
    assert r['l2_source']=='EXPLICIT_OVERRIDE' and r['l2_override'] is True
    assert r['l2_attribute']['bytes']==32*1024**2 and r['reported_l2_bytes']==0


def test_native_runtime_translation_preserves_device_body():
    source='#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n__global__ void k(){__syncthreads();}\ncudaGetLastError();'
    native=ppu_api(source)
    assert '#include <hggc_runtime.h>' in native
    assert '#include <hggc_fp16.h>' in native
    assert '__global__ void k(){__syncthreads();}' in native
    assert 'hggcGetLastError();' in native
    assert 'cuda' not in native


def test_q4_only_query_and_launch():
    original=(ROOT/'quactlize/execution/dispatch.cpp').read_text()
    src=q4_dispatch(original)
    assert src.count('if(c->qtype!=12) return QKG_FORMAT;')==2
    assert 'return (pair ? qkg_pair_launch_12 : qkg_launch_12)(*c,*f);' in src
    assert 'static Launch const' not in src
    with pytest.raises(ValueError):q4_dispatch(original.replace('if (!c || !f || !out)', 'if (!c)'))


def test_candidate_inventory_includes_cuda_winners_without_expanding_production():
    assert [len(configs(a)) for a in ('old','new','xplane','fq')]==[24,108,12,1]
    assert len(set(configs('new')))==108
    assert (4,5,1) in configs('new') and (8,10,1) in configs('new')
    assert (4,5,2) not in configs('new')
    old=(ROOT/'quactlize/execution/validation.hpp').read_text()
    new=candidate_validation(old)
    assert '(c.qtype == 12 && (f.warps == 16' in new
    assert 'f.split == 1 && (f.warps == 5 || f.warps == 10)' in new
    assert 'f.warps == 10' not in old


def cell(mode='warm'):
    copies=1 if mode=='warm' else 128
    return dict(status='PASS',arm='new',recipe=[4,8,2],shape=[1,512,2048],mode=mode,
                zero_code_negative='PASS',reducer_check='ORDERED_FP32',error=.0001,
                samples_us=[1.,2.,3.],median_us=2.,copies=copies,weight_bytes=589824,
                calls_per_graph=max(2,(32+copies-1)//copies)*copies,
                device=dict(l2_bytes=32*1024*1024))


def parse(row,**kwargs):
    return parse_cells('Q4_PPU_CELL '+json.dumps(row),'new',[(4,8,2)],
                       [1,512,2048],kwargs.get('mode','warm'),kwargs.get('samples',3))


def test_parser_accepts_complete_ring_and_finite_confirmations():
    assert parse(cell())[0]['median_us']==2.
    assert parse(cell('rotating'),mode='rotating')[0]['copies']==128
    r=cell();r.update(samples_us=[],median_us=None)
    assert parse(r,samples=0)


@pytest.mark.parametrize('field,value',[
    ('status','FAIL'),('arm','old'),('recipe',[4,5,2]),('shape',[1,512,4096]),
    ('mode','rotating'),('zero_code_negative','SKIP'),('error',float('nan')),
    ('error',.005),('samples_us',[1.,2.]),('samples_us',[1.,float('inf'),3.]),
    ('samples_us',[1.,0.,3.]),('median_us',3.),('copies',0),('copies',2),
    ('calls_per_graph',33),('weight_bytes',1024),('reducer_check','SKIP'),
])
def test_parser_rejects_planted_invalid_cells(field,value):
    r=cell();r[field]=value
    with pytest.raises(ValueError):parse(r)


def test_rotating_rejects_incomplete_ring_or_too_small_working_set():
    r=cell('rotating');r['calls_per_graph']-=1
    with pytest.raises(ValueError):parse(r,mode='rotating')
    r=cell('rotating');r.update(copies=2,calls_per_graph=32)
    with pytest.raises(ValueError):parse(r,mode='rotating')


def test_missing_duplicate_and_wrong_sample_denominator_rejected():
    line='Q4_PPU_CELL '+json.dumps(cell())
    for text in ('',line+'\n'+line):
        with pytest.raises(ValueError):parse_cells(text,'new',[(4,8,2)],[1,512,2048],'warm',3)


def test_ordered_fp32_reducer_rejects_missing_or_bad_partials():
    parts=np.array([[1.,2.],[3.,4.],[5.,6.]],dtype='<f4')
    check_reduction(parts,np.array([9.,12.],dtype='<f4'))
    with pytest.raises(ValueError):check_reduction(parts,np.array([9.,13.],dtype='<f4'))
    parts[1,0]=np.nan
    with pytest.raises(ValueError):check_reduction(parts,np.array([9.,12.],dtype='<f4'))


def test_shell_preserves_docker_and_does_not_recompile_simt():
    path=ROOT/'tools/run_q4_simt_ppu_box.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    src=path.read_text()
    assert '\n(\n' in src and src.rstrip().endswith(')')
    assert 'dev/gemv_ppu/build.py' not in src
    assert 'PRODUCTION_SELECTOR_JIT_MISS_ONLY' in src
    assert 'source "$SDK/envsetup.sh"' in src


def test_shipped_bundle_is_real_and_source_closed():
    bundle=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1'
    if not (bundle/'manifest.json').exists():pytest.skip('bundle not packaged')
    # LFS pointer detection is deliberate: a checkout must fetch its payloads.
    m=verify_bundle(bundle)
    assert m['device_validated'] is False and m['production_changed'] is False
    for r in m['payloads'].values():
        with (bundle/r['file']).open('rb') as f:assert f.read(4)==b'\x7fELF'
