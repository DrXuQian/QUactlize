"""Host and planner gates; actual GPU conversion admission runs separately."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dev.dense_helpers.io_gate import inputs, expected, check_oracle, compare_bits, GateFailure
from dev.prefill_asys.finalists import case_identity, load, partition
from dev.prefill_asys.io_refine import make_plan, validate, converted
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


def test_actual_cpp_eligibility(tmp_path):
    exe=tmp_path/'guard'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),
                    str(ROOT/'tests/dense_cont_host.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize('storage',[0,1])
@pytest.mark.parametrize('input',[0,1])
def test_device_oracle_vectors_cover_all_words_rounding_and_faults(storage,input):
    data=inputs(storage,input)
    if not input:assert np.array_equal(data,np.arange(65536,dtype='<u2'))
    else:
        assert np.isnan(data).any() and np.isinf(data).any()
        assert np.any((data!=0)&(np.abs(data)<np.finfo('f4').tiny))
        assert np.any(data==65504) and np.any(data==65520)
    want=expected(data,storage,input)
    check_oracle(want,want,storage,input)
    bad=want.copy();bad.view('<u2' if input else '<u4')[0]^=1
    with pytest.raises(ValueError,match='IEEE'):check_oracle(bad,want,storage,input)


def test_io_plan_only_changes_cast_and_preserves_every_route_control():
    from test_q8_tp_refine import review
    p=make_plan(review())
    assert load(p)==p and p['counts']['points']==12 and p['counts']['configs']==48
    for point in p['points']:
        by_id={c['id']:c for c in point['cases']}
        for cid in point['prior_route_winners'].values():
            old=case_identity(by_id[cid]);new=converted(old)
            assert digest(new) in by_id
            if old['arm']=='sf':assert new['recipe']==old['recipe']
            else:assert new['helper_options']['dequant_variant']==old['helper_options']['dequant_variant']
    assert sorted(x['id'] for i in range(4) for x in partition(p['points'],i,4))==sorted(x['id'] for x in p['points'])


@pytest.mark.parametrize('fault',['gemm','dq','control','bf16','duplicate','incomplete'])
def test_io_plan_rejects_changed_semantics_even_when_resealed(fault):
    from test_q8_tp_refine import review
    r=review()
    if fault=='incomplete':
        r['incomplete']=['point'];r['review_sha256']=digest({k:v for k,v in r.items() if k!='review_sha256'})
        with pytest.raises(ValueError):make_plan(r)
        return
    p=make_plan(r);point=p['points'][0];c=point['cases'][1]
    if fault=='gemm':c['recipe']['parameters'][0][1]='changed'
    elif fault=='dq':c=point['cases'][3];c['helper_options']['dequant_variant']=22
    elif fault=='control':point['cases'][0]['roles']=[]
    elif fault=='bf16':point['workload']['compute']='bf16'
    elif fault=='duplicate':point['cases'][1]=copy.deepcopy(point['cases'][0])
    c['id']=digest(case_identity(c));p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    with pytest.raises((ValueError,KeyError,TypeError)):validate(p)


def test_io_phase_uses_existing_coordinator_without_dequant_search(tmp_path,monkeypatch):
    from test_q8_tp_refine import review
    from tools import run_q8_tp_refine as runner
    path=tmp_path/'review.json';path.write_text(json.dumps(review()))
    calls=[]
    monkeypatch.setattr(runner,'run_coordinator',lambda *a:calls.append(a) or 0)
    monkeypatch.setattr(sys,'argv',[runner.__file__,'--review',str(path),'--phase','io',
        '--output',str(tmp_path/'run'),'--run','--confirm-idle','--gpus','1,2,3,4,5,7'])
    assert runner.main()==0 and len(calls)==1
    p=json.loads((tmp_path/'run/plan.json').read_text())
    assert p==make_plan(review())
    assert json.loads((tmp_path/'run/status.json').read_text())['phases']=={'io':0}


def test_io_plan_reaches_existing_asys_shards_without_a_device(tmp_path):
    from test_q8_tp_refine import review
    p=make_plan(review());path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(path),'--output',str(tmp_path/'dry'),'--plan-only',
        '--shard','1','--shards','4','--sdk','/no-sdk','--llama-source','/no-llama'],
        check=True,capture_output=True)
    actual=json.loads((tmp_path/'dry/plan.json').read_text())
    assert actual['counts']['points']==3 and actual['counts']['configs']==12
    assert not (tmp_path/'dry/build').exists()


@pytest.mark.parametrize('fault',[None,'write','error','nan-old','nan-new','nan-class','guard'])
def test_device_gate_harness_with_explicit_host_double(monkeypatch,fault):
    # This checks only the gate orchestration and its negative controls, not
    # CUDA math. The real library/device test remains mandatory on Box.
    from types import SimpleNamespace
    from dev.dense_helpers import io_gate as gate
    class Sdk:
        def __init__(self,*_):pass
        def synchronize(self,*_):pass
        def download(self,ptr,size):return C.string_at(ptr,size)
    class Buffers:
        def __init__(self,sdk):self.sdk=sdk;self.stream=None;self.keep=[]
        def alloc(self,size):
            x=C.create_string_buffer(size+255);self.keep.append(x)
            return (C.addressof(x)+255)&~255
        def upload(self,x):
            p=self.alloc(x.nbytes);C.memmove(p,x.ctypes.data,x.nbytes);return p
        def fill(self,p,byte,size):C.memset(p,byte,size)
        def close(self):pass
    class Fn:
        def __call__(self,storage,input,vector,src,dst,rows,width,stride,error,stream):
            if src==dst:return 1
            src_count=(rows-1)*stride+width if input else rows*width
            source=np.frombuffer(C.string_at(src,src_count*(4 if input else 2)),
                                 dtype='<f4' if input else '<u2')
            source=np.stack([source[r*(stride if input else width):r*(stride if input else width)+width]
                             for r in range(rows)])
            value=expected(source,storage,input)
            if not input and error and C.c_int.from_address(error).value:
                if not (vector==4 and fault=='error'):value.view('<u4')[:]=0x7fc00000
            if vector==4 and fault=='write':value.view('<u2' if input else '<u4')[0,0]^=1
            if not input and not error:
                raw=value.view('<u4')
                if (vector in (1,4) and fault=='nan-old') or (vector==4 and fault=='nan-new'):
                    raw[0,0]^=1
                if vector==4 and fault=='nan-class':raw[0,0]=0
            for r in range(rows):C.memmove(dst+r*(width if input else stride)*value.itemsize,
                                          value[r].ctypes.data,width*value.itemsize)
            if vector==4 and fault=='guard':C.c_uint8.from_address(dst-1).value=0
            return 0
    original=gate.inputs
    def vectors(s,i):
        data=original(s,i)[:33]
        if fault and fault.startswith('nan-') and not i:
            data[0]=0x7fc1 if s else 0x7e01
        return data
    monkeypatch.setattr(gate,'inputs',vectors)
    monkeypatch.setattr(gate,'SDK',Sdk);monkeypatch.setattr(gate,'Resources',Buffers)
    monkeypatch.setattr(gate.C,'CDLL',lambda *a,**kw:SimpleNamespace(quactlize_dense_io_v1=Fn()))
    if fault:
        with pytest.raises(GateFailure) as failed:gate.run('/host-double','/not-a-device-library')
        diagnostic=failed.value.report['first_failure']
        if fault.startswith('nan-'):
            assert failed.value.report['cases']==3  # finish this cell only, then stop
            changes=diagnostic['comparisons']['4-vs-0']
            assert changes['raw_bad']==1
            assert changes['nan_class_bad']==int(fault=='nan-class')
            assert changes['nan_encoding_bad']==int(fault!='nan-class')
            assert diagnostic['comparisons']['4-vs-1']['raw_bad']==int(fault!='nan-old')
            assert diagnostic['source_unchanged']
        elif fault=='guard':assert diagnostic['vector']==4 and diagnostic['first_byte']==255
    else:
        result=gate.run('/host-double','/not-a-device-library')
        assert result['status']=='PASS' and result['cases']==216


@pytest.mark.parametrize('storage',[0,1])
@pytest.mark.parametrize('input',[0,1])
@pytest.mark.parametrize('kind',['finite','zero-sign','infinity-sign','nan-class','nan-payload'])
def test_bit_diagnostics_distinguish_classes_without_relaxing_raw_equality(storage,input,kind):
    words=([0,0x8000,0x3f80,0x7f80,0x7fc1] if storage else [0,0x8000,0x3c00,0x7c00,0x7e01]) if input else [
        0,0x80000000,0x3f800000,0x7f800000,0x7fc00001]
    want=np.array(words,dtype='<u2' if input else '<u4')
    if not input:want=want.view('<f4')
    got=want.copy();raw=got.view('<u2' if input else '<u4')
    sign=0x8000 if input else 0x80000000
    if kind=='finite':raw[2]^=1
    elif kind=='zero-sign':raw[0]^=sign
    elif kind=='infinity-sign':raw[3]^=sign
    elif kind=='nan-class':raw[4]=0
    else:raw[4]^=1
    source=np.zeros(5,dtype='<f4' if input else '<u2')
    d=compare_bits(got,want,storage,input,source)
    assert d['raw_bad']==1
    assert d['nan_encoding_bad']==int(kind=='nan-payload')
    assert d['nan_class_bad']==int(kind=='nan-class')
    assert d['non_nan_bad']==int(kind in ('finite','zero-sign','infinity-sign'))
    assert d['first'][0]['got_bits']!=d['first'][0]['want_bits']
    json.dumps(d,allow_nan=False)


@pytest.mark.parametrize('kind',['bits','runtime'])
def test_cli_saves_failure_receipt_and_nonzero_exit(tmp_path,monkeypatch,kind):
    from tools import check_dense_io as cli
    library=tmp_path/'library.so';library.write_bytes(b'host-only input, not loaded')
    output=tmp_path/'failure.json'
    def fail(*_):
        if kind=='bits':raise GateFailure(dict(status='FAIL',cases=3,error='raw bits differ',
            first_failure=dict(storage=0,input=0,comparisons={'4-vs-1':dict(raw_bad=1)}),records=[]))
        raise RuntimeError('device test double')
    monkeypatch.setattr(cli,'run',fail)
    monkeypatch.setattr(sys,'argv',[cli.__file__,'--sdk','/host-double','--library',str(library),
                                    '--output',str(output)])
    assert cli.main()==1
    data=json.loads(output.read_text())
    assert data['status']=='FAIL' and data['library_sha256']==cli.sha(library)
    assert data['gate_sha256']==cli.sha(ROOT/'dev/dense_helpers/io_gate.py')
    if kind=='bits':assert data['first_failure']['comparisons']['4-vs-1']['raw_bad']==1
    else:assert data['error']=='RuntimeError: device test double'
