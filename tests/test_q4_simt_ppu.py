"""Host gates for the PPU comparison; not PPU numeric/performance admission."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.gemv_ppu.build import ppu_api, q4_dispatch, candidate_validation
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import configs, check_reduction, verify_bundle

ROOT=Path(__file__).resolve().parents[1]


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
