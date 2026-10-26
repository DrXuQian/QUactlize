"""Real model inventory and C++ carrier parity; no device timings."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy.smallm_asys import workloads,make_plan,model_coverage,assignments,check_producers
from dev.strategy.token_cta import q8_control_entry
from dev.strategy.m_coverage import families
from dev.strategy.refresh import request
from dev.strategy.vector import execution_contract
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Workload,Recipe

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model_plan(tmp_path_factory):
    ws=workloads('token-cta-model-gaps')
    old=plan_smallm(tmp_path_factory.mktemp('model-queries'),[request(w) for w in ws],dense_fallbacks=True)
    return make_plan(old['requests'],'token-cta-model-gaps')


def test_registry_is_not_a_format_cartesian_product():
    all_ws=workloads('token-cta-models');gaps=workloads('token-cta-model-gaps')
    assert len(all_ws)==360 and len(gaps)==328
    assert len({w.key for w in gaps})==328
    assert {(w.qtype,w.n,w.k) for w in all_ws}==set(families())
    skipped=[w for w in all_ws if w.key not in {g.key for g in gaps}]
    assert len(skipped)==32 and {w.qtype for w in skipped}=={8,12,13,14}
    assert all((w.n,w.k)==(1024,5120) for w in skipped)
    assert Counter(w.qtype for w in gaps)=={8:136,11:16,12:80,13:48,14:48}
    assert all(w.qtype!=10 and w.storage=='f32' and w.compute=='f16' and w.operation=='dense' for w in all_ws)
    c=model_coverage('token-cta-model-gaps')
    assert c['total_families']==45 and c['total_points']==360 and c['excluded_previous_points']==32
    assert not c['prior_timings_reused'] and not c['prior_passes_added']


def test_model_plan_keeps_current_control_and_all_new_bodies(model_plan):
    p=model_plan
    assert p['counts']==dict(points=328,arms=9337,tc_modules=23,vector_modules=149,families=41)
    assert not p['native_comparison'] and not p['production_changed'] and not p['historical_times_reused']
    assert p['coverage']['excluded_previous_points']==32
    assert 'policies/dense_prefill_v1.json' in p['sources']
    assert max(len(point['arms']) for point in p['points'])==37
    assert sum(len(point['arms']) for point in p['points'])<10000
    for point in p['points']:
        w=Workload(**point['workload'])
        original,=[a for a in point['arms'] if a['id']=='original']
        assert original['required'] and 'CURRENT_CPP_SELECTED_RECIPE' in original['reasons']
        assert original['recipe']==point['original_recipe']
        assert point['selection']['request']==request(w) and point['selection']['status']==0
        assert len(point['sf_configs'])<=1
        new=[a for a in point['arms'] if 'cta_tokens' in a['spec']]
        assert len(new)==(0 if w.m==1 else 16 if w.qtype==8 else 12)
        for a in new:
            assert a['required'] and a['spec']['cta_tokens']==w.m
            assert dict(a['recipe']['parameters'])['split'] in (1,2,4,8)
        for a in point['arms']:
            if a['recipe']['kind']=='tc':
                assert a['required'] or 'EXACT_OLD_WINNER_OR_WITHIN_FIVE_PERCENT' in a['reasons']


@pytest.mark.parametrize('workers',[1,4,6,7])
def test_assignments_are_complete_deterministic_whole_pairs(model_plan,workers):
    a=assignments(model_plan,workers)
    assert a==assignments(model_plan,workers)
    assert Counter(x for shard in a for x in shard)==Counter(p['id'] for p in model_plan['points'])
    points={p['id']:p for p in model_plan['points']}
    def weight(pid):
        p=points[pid];w=p['workload']
        return (w['n']*w['k']+4*1024*1024)*len(p['arms'])*(8+w['m'])
    loads=[sum(map(weight,shard)) for shard in a]
    assert max(loads)-min(loads)<=max(map(weight,points))
    old=dict(model_plan,cohort='token-cta-formats')
    assert assignments(old,workers)==[[p['id'] for j,p in enumerate(old['points']) if j%workers==i] for i in range(workers)]


def test_q8_entry_predictions_match_actual_cpp_carrier_predicates(tmp_path,model_plan):
    # Mirror the actual carrier order, which intentionally does not route an
    # explicit old-body arm through the new exact token/Q8-address overrides.
    src=tmp_path/'entries.cpp';exe=tmp_path/'entries'
    src.write_text(r'''
#include "quactlize/execution/simt_strategy.hpp"
#include <iostream>
using namespace quactlize::execution::simt;
int main() {
    qkg_call_v1 c{1,sizeof(c)};c.qtype=8;c.mode=0;c.input_type=1;c.experts=c.topk=c.channels=1;
    qkg_simt_config_v1 f{1,sizeof(f)};
    while(std::cin>>c.n>>c.k>>c.rows>>f.variant>>f.columns>>f.warps>>f.values>>f.split) {
        qkg_simt_call_v2 d{2,sizeof(d),c,QKG_COMPUTE_F16};
        const char* entry="q8_vector::kernel";
        if(measured_decode(d,f)!=MeasuredDecode::None)entry="measured_decode_kernel";
        else switch(q8_strategy(d,f.variant,f.columns,f.warps,f.values,f.split)) {
            case Q8Strategy::FixedCold:case Q8Strategy::FixedHoisted:entry="q8_vector::kernel_model";break;
            case Q8Strategy::S1Hoisted:case Q8Strategy::S1Narrow:entry="q8_vector::kernel_s1";break;
            default:break;
        }
        std::cout<<"quactlize::execution::simt::"<<entry<<'\n';
    }
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(src),'-o',str(exe)],check=True)
    calls=[]
    for point in model_plan['points']:
        w=Workload(**point['workload'])
        if w.qtype!=8:continue
        for arm in point['arms']:
            if arm['recipe']['kind']!='simt' or any(f in arm['spec'] for f in ('cta_tokens','q8_body')):continue
            c=arm['spec']['config'];split=dict(arm['recipe']['parameters'])['split']
            calls.append((w,c,split))
    # Include the legal narrow specialization even if not currently selected.
    calls.extend((Workload(8,512,2048,m),dict(variant=5,columns=4,warps=2,values=4),s)
                 for m in (1,8) for s in (1,2))
    inputs=''.join(' '.join(map(str,[w.n,w.k,w.m,*[c[k] for k in ('variant','columns','warps','values')],s]))+'\n'
                   for w,c,s in calls)
    actual=subprocess.check_output([str(exe)],input=inputs,text=True).splitlines()
    assert actual==[q8_control_entry(w,c,s) for w,c,s in calls]
    assert {x.rsplit('::',1)[-1] for x in actual}=={'kernel','kernel_model','kernel_s1','measured_decode_kernel'}


@pytest.mark.parametrize('w,c,split',[
    (Workload(8,6144,3072,1),dict(variant=5,columns=8,warps=4,values=4),8),
    (Workload(8,2048,4096,1),dict(variant=5,columns=8,warps=4,values=4),8),
    (Workload(8,4096,2048,1),dict(variant=5,columns=4,warps=8,values=4),1),
    (Workload(8,4096,2048,2),dict(variant=5,columns=4,warps=8,values=4),1),
])
def test_capture_requires_the_correct_fixed_or_generic_q8_entry(w,c,split):
    contract=execution_contract(w);r=Recipe.make('simt','fq',**c,split=split)
    arm=dict(id='original',spec=dict(q=8,config=c),recipe=asdict(r))
    point=dict(arms=[arm],workload=asdict(w),execution_contract=contract)
    name=q8_control_entry(w,c,split)
    symbol='_ZN'+''.join(str(len(part))+part for part in name.split('::'))+'IiEv'
    data=dict(arms=dict(original=dict(receipt=dict(execution_contract=contract))),arm_components=dict(original=['p']),
        component_records=dict(p=dict(kind='producer',identity=dict(node=dict(name=symbol,symbol=symbol)))))
    check_producers(point,data)
    bad=deepcopy(data)
    bad['component_records']['p']['identity']['node']=dict(name='void wrong::kernel<int>(call)')
    with pytest.raises(ValueError,match='wrong production'):check_producers(point,bad)


def test_model_cohort_cli_and_resume_without_gpu(tmp_path):
    args=[sys.executable,str(ROOT/'tools/run_smallm_regressions_asys.py'),'--output',str(tmp_path/'plan'),
          '--cohort','token-cta-model-gaps','--plan-only']
    for extra in ([],['--resume']):
        result=subprocess.run(args+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert '"points": 328' in result.stdout and '"families": 41' in result.stdout
    assert not (tmp_path/'plan/run.json').exists()
    p=json.loads((tmp_path/'plan/plan.json').read_bytes())
    assert p['coverage']['excluded_previous_points']==32 and len(p['points'])==328
