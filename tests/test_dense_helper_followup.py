"""Host-only bounded planning/trace tests; numbers below are explicit test fixtures."""
import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.prefill_asys.helper_followup import make_plan, validate, grain, reordered
from dev.prefill_asys.finalists import load, case_identity, full_case
from quactlize.dispatch.dense_prefill import EVIDENCE, execution
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    from test_dense_compact import split_review
    from dev.prefill_asys.io_refine import make_plan as factorial
    from test_q8_tp_refine import seal
    source=split_review();p=factorial(source,compact=True)
    r=dict(source,points={},candidates={})
    for point in p['points']:
        w=point['workload'];rows=[]
        for c in point['cases']:
            cid=c['id'];r['candidates'][cid]=case_identity(c)
            reducer=1.5 if w['n']>=4096 else .5
            rows.append(dict(candidate=cid,arm=c['arm'],roles=c['roles'],status='PASS',
                gpu_sum_us=(200 if c['arm']=='sf' else 120) if w['m']==2048 else (60 if c['arm']=='sf' else 120),
                confirmation_captures=[{},{}],
                phases_us=dict(reducer=reducer if 'compact' in c['roles'][0] else 1.)))
        rows.append(dict(candidate='native',arm='native-fp16',roles=['native'],status='PASS',
                         gpu_sum_us=110,confirmation_captures=[{},{}]))
        r['points'][point['id']]=dict(workload=w,consumers=[],measurements=rows)
    seal(r,'review_sha256')
    catalog=json.loads(EVIDENCE.read_text())
    catalog['helper_updates']=[dict(review_sha256=r['review_sha256'],retained_pending=[dict(key=p['points'][0]['id'])])]
    seal(catalog)
    return r,catalog


def test_grain_and_order_budget_keep_incumbents(tmp_path):
    r,c=fixture();p=make_plan(r,c)
    assert load(p)==p and p['counts']['points']<=16
    assert all(len(x['cases'])<=8 for x in p['points'])
    assert all(x['production_control'] in {c['id'] for c in x['cases']} for x in p['points'])
    path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(path),'--output',str(tmp_path/'worker'),'--plan-only',
        '--sdk','/no-device','--llama-source','/no-llama'],check=True,capture_output=True)
    assert not (tmp_path/'worker/build').exists()


@pytest.mark.parametrize('fault',['incumbent','grain','order','missing','precision'])
def test_resealed_plan_cannot_drop_controls_or_change_math(fault):
    from test_q8_tp_refine import seal
    r,c=fixture();p=make_plan(r,c)
    if fault=='order':
        point=next(x for x in p['points'] if 'full-order-residual' in x['reasons'])
        case=next(x for x in point['cases'] if 'dequant-before-input' in x['roles'])
        case['helper_options']['dequant_variant']=99;case['id']=digest(case_identity(case))
    else:
        point=next(x for x in p['points'] if 'large-output-reducer' in x['reasons'])
        case=next(x for x in point['cases'] if 'grain:8' in x['roles'])
        if fault=='incumbent':point['production_control']='missing'
        elif fault=='precision':point['workload']['compute']='bf16'
        elif fault=='missing':point['cases'].remove(case)
        else:case['recipe']['parameters'].append(['bad',1]);case['id']=digest(case_identity(case))
    with pytest.raises((ValueError,KeyError)):validate(seal(p))


@pytest.mark.parametrize('width',[2,4,8])
def test_grain_launch_is_measured_not_assumed(width):
    from test_prefill_asys import PrefillAsysTests
    from dev.prefill_asys.trace import parse_rows
    rows,c=PrefillAsysTests().rows(split=2)
    c['recipe']['parameters'].append(['dense_reducer',width]);c['workload']=dict(m=128,n=4096)
    for r in rows:
        if 'ReductionKernel' in r['name']:
            r.update(name=f'cutlass::Kernel<PpuMixedInputSplitKParallelM1FastReductionKernel<{width}, 2>>(...)',
                     grid=[128*4096//(32*width),1,1],block=[32,1,1])
    assert parse_rows(rows,[c])[0]['phases_us']['reducer']==1
    next(r for r in rows if 'ReductionKernel' in r['name'])['grid'][0]+=1
    with pytest.raises(ValueError,match='compact reducer'):parse_rows(rows,[c])


def test_full_order_uses_existing_stages_and_keeps_error_reset(monkeypatch):
    from dev.prefill_point import typed
    core=typed.TypedFull.__new__(typed.TypedFull)
    core.library=SimpleNamespace(run='existing-public-API');core.handle=None
    core.calls=(typed.Call*2)()
    monkeypatch.setattr(typed,'call',lambda fn,kind,ptrs,args:args)
    assert core.descriptors((4,1,0,2,3))==[[s] for _ in range(2) for s in (4,1,0,2,3)]
    with pytest.raises(ValueError):core.descriptors((1,0,2,3))
    # Order is a sealed execution axis, never silently serialized as the old
    # Full recipe. The actual captured phase order must agree.
    candidate=reordered(full_case(41,4))
    assert candidate==full_case(41,4,dequant_first=True)
    phases=['error_reset','expansion','input','main','output']
    definitions={str(i):dict(phase=p) for i,p in enumerate(phases)}
    row=dict(components=list(definitions))
    actual=execution(candidate,row,definitions)
    assert actual==dict(route=2,io=4,dequant=41,tc=None,dequant_first=True)
    from dev.prefill_asys.supplement import execution_case
    assert execution_case(actual)==candidate
    with pytest.raises(ValueError,match='stage order'):
        execution(full_case(41,4),row,definitions)
    # A cuBLAS split reduction belongs to the provider, not the input stage.
    definitions['3']['phase']='reducer'
    with pytest.raises(ValueError,match='stage order'): execution(candidate,row,definitions)
    definitions['3']['phase']='main'; definitions['reduce']=dict(phase='reducer')
    assert execution(candidate,dict(components=['0','1','2','3','reduce','4']),definitions)==actual
    for order in (1,'true',None):
        with pytest.raises(ValueError): full_case(41,4,dequant_first=order)


def test_ordered_full_trace_still_requires_every_measured_reset():
    from test_prefill_asys import PrefillAsysTests
    from dev.prefill_asys.trace import parse_rows
    rows,c,apis=PrefillAsysTests().primer()
    for offset in range(0,len(rows),5):
        for field in ('name','symbol'):
            rows[offset+1][field],rows[offset+2][field]=rows[offset+2][field],rows[offset+1][field]
    c['stages']=['error_reset','expansion','input','main','output']
    result=parse_rows(rows,[c])[0]
    assert result['gpu_sum_us']==5
    absent=parse_rows(rows[1:],[c],apis=apis)[0]
    assert absent['gpu_sum_us']==5 and len(absent['primer_exceptions'])==1
    cut=5*c['trace']['discarded_calls']
    with pytest.raises(ValueError):parse_rows(rows[:cut]+rows[cut+1:],[c],apis=apis)


def test_followup_coordinator_really_runs(monkeypatch,tmp_path):
    from tools import run_q8_tp_refine as runner
    from dev.prefill_asys import helper_followup
    r,c=fixture();p=make_plan(r,c)
    path=tmp_path/'review.json';path.write_text(json.dumps(r));calls=[]
    monkeypatch.setattr(helper_followup,'make_plan',lambda _:p)
    monkeypatch.setattr(runner,'run_coordinator',lambda *args:calls.append(args) or 0)
    monkeypatch.setattr(sys,'argv',[runner.__file__,'--review',str(path),'--phase','followup',
        '--run','--confirm-idle','--gpus','1,2,3','--output',str(tmp_path/'run')])
    assert runner.main()==0 and len(calls)==1
    status=json.loads((tmp_path/'run/status.json').read_text())
    assert status['phases']==dict(followup=0) and not status['not_measured_in_this_invocation']
