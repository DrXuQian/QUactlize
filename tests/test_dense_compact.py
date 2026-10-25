"""Compact reuse is opt-in; these host checks are not PPU performance evidence."""
import copy
import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dev.dense_helpers.reducer_gate import partials, expected, cases
from dev.prefill_asys.finalists import load, case_identity
from dev.prefill_asys.io_refine import make_plan, compact_case, validate
from dev.strategy.prescreen import parent, rejection
from dev.strategy.resources import inventory
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import Compiler, validate_parent
from quactlize.decode.compiler import DecodeCompiler
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


def split_review():
    from test_q8_tp_refine import review, seal
    r=review()
    for point in r['points'].values():
        row=point['measurements'][0];old=r['candidates'][row['candidate']]
        case=copy.deepcopy(old);recipe=Recipe.from_dict(case['recipe'])
        p=dict(recipe.parameters);p['split']=4
        case['recipe']=json.loads(json.dumps(asdict(Recipe.make(recipe.kind,recipe.route,**p))))
        cid=digest(case);r['candidates'][cid]=case;row['candidate']=cid
    return seal(r,'review_sha256')


def test_factorial_keeps_mainloop_and_is_separately_identified():
    p=make_plan(split_review(),compact=True)
    assert load(p)==p and p['counts']['configs']==72
    for point in p['points']:
        w=Workload(**point['workload']);controls={c['id']:c for c in point['cases']}
        old=case_identity(controls[point['prior_route_winners']['sf']]);candidate=compact_case(old)
        assert candidate['helper_options']==old['helper_options']
        r=Recipe.from_dict(candidate['recipe']);newp=dict(r.parameters);oldp=dict(old['recipe']['parameters'])
        assert newp.pop('dense_reducer')==2
        assert newp.pop('symbol')==oldp.pop('symbol')+'_rc2' and newp==oldp
        assert rejection(r,w) is None
        oldparent=parent(Recipe.from_dict(old['recipe']),w);newparent=parent(r,w)
        before=Compiler.source(None,oldparent,'key');after=Compiler.source(None,newparent,'key')
        assert '#define QK_DENSE_REDUCER' not in before
        assert after.replace('#define QK_DENSE_REDUCER 2\n','').replace('_rc2"','"')==before
        with pytest.raises(ValueError,match='native F16'):DecodeCompiler.source(None,newparent,'key')
        for changes in ({'dense_reducer':True},{'dense_reducer':1},{'symbol':oldparent['symbol']},
                        {'route':'sf-grouped'}):
            with pytest.raises(ValueError):validate_parent(newparent|changes)
    assert len(inventory(p)['modules'])==2  # synthetic cohort uses one old parent
    assert len(cases(p['points']))==19


@pytest.mark.parametrize('fault',['split','precision','symbol','control','extra','option'])
def test_resealed_factorial_cannot_modify_or_drop_controls(fault):
    from test_q8_tp_refine import seal
    p=make_plan(split_review(),compact=True);point=p['points'][0];c=point['cases'][2]
    if fault=='precision':point['workload']['compute']='bf16'
    elif fault=='control':point['cases'].pop(0)
    elif fault=='extra':point['cases'].append(copy.deepcopy(c))
    else:
        params=dict(c['recipe']['parameters'])
        params[{'split':'split','symbol':'symbol','option':'dense_reducer'}[fault]]=2 if fault=='split' else 'wrong'
        c['recipe']['parameters']=sorted(params.items());c['id']=digest(case_identity(c))
    with pytest.raises((ValueError,KeyError,TypeError)):validate(seal(p))


def test_helpers_plan_reaches_existing_worker_and_coordinator(tmp_path,monkeypatch):
    from tools import run_q8_tp_refine as runner
    review=tmp_path/'review.json';review.write_text(json.dumps(split_review()))
    calls=[];monkeypatch.setattr(runner,'run_coordinator',lambda *a:calls.append(a) or 0)
    monkeypatch.setattr(sys,'argv',[runner.__file__,'--review',str(review),'--phase','helpers',
        '--output',str(tmp_path/'run'),'--run','--confirm-idle','--gpus','1,2,3,4,5,7'])
    assert runner.main()==0 and len(calls)==1
    assert json.loads((tmp_path/'run/status.json').read_text())['phases']==dict(helpers=0)
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(tmp_path/'run/plan.json'),'--output',str(tmp_path/'worker'),
        '--plan-only','--shard','1','--shards','4','--sdk','/no-sdk','--llama-source','/no-llama'],
        check=True,capture_output=True)
    assert json.loads((tmp_path/'worker/plan.json').read_text())['counts']['configs']==18


def test_plan_only_runs_reducer_contract_before_build(tmp_path,monkeypatch):
    from dev.dense_helpers import reducer_gate
    from tools import run_prefill_model_asys as runner
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(make_plan(split_review(),compact=True)))
    output=tmp_path/'worker'
    def reject(points):raise ValueError('host reducer preflight sentinel')
    monkeypatch.setattr(reducer_gate,'cases',reject)
    monkeypatch.setattr(sys,'argv',[runner.__file__,'--finalist-plan',str(plan),'--output',str(output),
        '--plan-only','--sdk','/no-sdk','--llama-source','/no-llama'])
    with pytest.raises(ValueError,match='host reducer preflight sentinel'):runner.main()
    assert not (output/'build').exists() and not (output/'authority.json').exists()


@pytest.mark.parametrize('fault',[None,'generic','split','grid','block','shared'])
def test_trace_must_prove_compact_body_and_actual_geometry(fault):
    from test_prefill_asys import PrefillAsysTests
    from dev.prefill_asys.trace import parse_rows
    rows,c=PrefillAsysTests().rows(split=4)
    c['recipe']['parameters'].append(['dense_reducer',2]);c['workload']=dict(m=128,n=256)
    for r in rows:
        if 'ReductionKernel' not in r['name']:continue
        r.update(name='cutlass::Kernel<cutlass::gemm::device::splitk_parallel::'
                      'PpuMixedInputSplitKParallelM1FastReductionKernel<2, 4>>(...)',
                 grid=[512,1,1],block=[32,1,1])
        if fault=='generic':r['name']='cutlass::Kernel<ReductionKernel>(...)'
        elif fault=='split':r['name']=r['name'].replace('<2, 4>','<2, 8>')
        elif fault in ('grid','block'):r[fault][0]+=1
        elif fault=='shared':r['shared_bytes']=16
    if fault:
        with pytest.raises(ValueError,match='compact reducer'):parse_rows(rows,[c])
    else:assert parse_rows(rows,[c])[0]['phases_us']['reducer']==1


def test_independent_oracle_detects_changed_addition_order():
    x=partials(3,64,4);correct=expected(x);wrong=expected(x[[0,2,1,3]])
    assert correct[0]==np.float16(1).view('u2') and correct[1]==0
    assert np.any(correct!=wrong)
    assert correct[2]==np.float16(1).view('u2') and correct[5]==0x7c00


@pytest.mark.parametrize('m',[3072,4096,5120,8192])
@pytest.mark.parametrize('width',[2,4,8])
def test_reducer_gate_covers_extended_m_span(m,width):
    point=dict(workload=dict(operation='dense',compute='f16',m=m,n=256),
               cases=[dict(arm='sf',recipe=dict(parameters=[['split',4],['dense_reducer',width]]))])
    assert (m,256,256,4,0,0) in cases([point])


@pytest.mark.parametrize('change',[dict(m=8193),dict(m=8),dict(n=65537),dict(n=0),
    dict(m=9,n=7),dict(compute='bf16'),dict(operation='grouped'),dict(split=3),
    dict(width=3),dict(width=True),dict(arm='fq')])
def test_reducer_gate_keeps_non_m_contracts(change):
    w=dict(operation='dense',compute='f16',m=5120,n=256)
    w.update({k:v for k,v in change.items() if k in w})
    point=dict(workload=w,cases=[dict(arm=change.get('arm','sf'),recipe=dict(parameters=[
        ['split',change.get('split',2)],['dense_reducer',change.get('width',2)]]))])
    with pytest.raises(ValueError,match='M=.*N=.*split=.*width='):cases([point])


@pytest.mark.parametrize('fault',[None,'oracle','guard','fast','invalid'])
def test_gate_orchestration_with_explicit_host_double(monkeypatch,fault):
    # Only test the gate's positive/negative logic. This does not emulate or
    # certify any CUDA/PPU instruction; Box still runs the compiled carrier.
    from dev.dense_helpers import reducer_gate as gate
    class Sdk:
        def __init__(self,*_):pass
        def synchronize(self,*_):pass
        def download(self,p,size):return C.string_at(p,size)
    class Buffers:
        def __init__(self,sdk):self.stream=None;self.keep=[]
        def alloc(self,size):
            x=C.create_string_buffer(size+255);self.keep.append(x)
            return (C.addressof(x)+255)&~255
        def upload(self,x):
            p=self.alloc(x.nbytes);C.memmove(p,x.ctypes.data,x.nbytes);return p
        def fill(self,p,value,size):C.memset(p,value,size)
        def close(self):pass
    handles={}
    def prepare(compact,m,n,s,src,bytes_,dst,stride,out):
        out._obj.value=None
        valid=bytes_>=s*m*n*4 and src%16==0 and stride>=n and not src<=dst<src+s*m*n*4
        if not valid:return 0 if fault=='invalid' else 2
        key=len(handles)+1;handles[key]=(compact,m,n,s,src,dst,stride);out._obj.value=key;return 0
    def fast(h):
        compact,m,n,s,src,dst,stride=handles[h.value]
        return int((compact or m==1) and stride==n and (m*n if compact else n)%64==0
                   and s in (2,4,8) and src%(16 if compact else 128)==0 and dst%16==0)^int(fault=='fast')
    def launch(h,stream):
        compact,m,n,s,src,dst,stride=handles[h.value]
        x=np.frombuffer(C.string_at(src,s*m*n*4),dtype='<f4').reshape(s,m*n)
        value=expected(x).reshape(m,n)
        if compact and fault=='oracle':value[0,0]^=1
        for i in range(m):C.memmove(dst+i*stride*2,value[i].ctypes.data,n*2)
        if compact and fault=='guard':C.c_uint8.from_address(dst-1).value=0
        return 0
    monkeypatch.setattr(gate,'bind',lambda _: (prepare,launch,fast,lambda h:handles.pop(h.value)))
    monkeypatch.setattr(gate,'SDK',Sdk);monkeypatch.setattr(gate,'Resources',Buffers)
    monkeypatch.setattr(gate,'cases',lambda _: [(1,512,512,2,0,0),(1,512,512,4,16,0),
        (3,21,21,4,0,0),(3,64,69,8,0,0),(4,128,128,4,0,2),(3,64,64,8,0,0),
        (3,64,64,1,0,0),(8192,256,256,2,0,0)])
    if fault:
        with pytest.raises(ValueError):gate.run('/host-double','/not-a-device-library',[])
    else:
        value=gate.run('/host-double','/not-a-device-library',[])
        assert value['status']=='PASS' and value['cases']==32 and not handles
