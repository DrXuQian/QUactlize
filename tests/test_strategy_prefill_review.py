"""Host-only receipt/selector tests. No synthetic timings are device evidence."""
import copy
import json
from pathlib import Path
import statistics
import subprocess

import pytest

from dev.strategy.prefill_review import choose, audit_point
from tools.fit_prefill_confirmation import EVIDENCE, HEADER, header

ROOT = Path(__file__).resolve().parents[1]


def fixture(values):
    point = dict(original='original', arms={
        a: dict(original=a=='original', roles=['control'] if a=='original' else ['rule']) for a in values})
    measured = dict(statistics={a: dict(rounds_us=v, median_us=statistics.median(v),
        spread_pct=100*(max(v)/min(v)-1)) for a,v in values.items()})
    return point, measured


def test_old_control_helper_is_eligible_only_when_generated():
    p, m = fixture(dict(original=[10.]*6, candidate=[11.]*6))
    assert choose(p,m,{'candidate','original'})['winner']=='original'
    with pytest.raises(ValueError,match='no generated'):
        choose(p,m,{'candidate'})
    assert p['arms']['original']['roles']==['control']  # raw provenance unchanged


def test_feasible_first_instead_of_rejecting_all_for_minimum_median():
    p,m=fixture(dict(original=[100.]*6,candidate=[99.,99.,99.,100.5,100.5,100.5]))
    assert m['statistics']['candidate']['median_us']<100
    assert choose(p,m,{'original','candidate'})['winner']=='original'


def test_noise_requires_strict_range_dominance_not_median_only():
    p,m=fixture(dict(original=[10.,11.,10.,11.,10.,11.],candidate=[9.]*6))
    selected=choose(p,m,{'candidate'})
    assert selected['comparisons']['original']['stability_basis']=='SEPARATED_OBSERVED_RANGES'
    for values in ([10.]*6,[9.,10.,9.,10.,9.,10.]):
        bad=fixture(dict(original=[10.,11.,10.,11.,10.,11.],candidate=values))
        with pytest.raises(ValueError):choose(*bad,{'candidate'})


def test_generated_policy_keeps_original_failures_and_all_430_keys():
    data=json.loads(EVIDENCE.read_text())
    assert HEADER.read_text()==header(data)
    assert len(data['points'])==len({p['point'] for p in data['points']})==430
    assert data['summary']['raw_status']==dict(PASS=412,RULE_REGRESSION=16,UNSTABLE=2)
    assert data['summary']['reviewed_decisions_changed']==18
    assert not data['legacy_fallback'] and not data['model_performance_admitted']
    assert data['rule_proof']=='R27_LEAVE_ENTIRE_NK_OUT_NO_CHAMPION_INJECTION'
    for p in data['points']:
        w=p['decision']['winner'];s=p['statistics'][w]
        assert w in p['decision']['qualified'] and w in p['decision']['eligible']
        assert s['spread_pct']<=5
        assert s['median_us']<=p['statistics'][p['original']]['median_us']*(1+1e-12)
        assert all(c['paired_wins']>=4 for c in p['decision']['comparisons'].values())
        assert p['accuracy']['checks']>0 and p['accuracy']['relative_l1_error_max']<.005
        assert p['workload']['channels']==1 and len(p['round_receipts'])==6


@pytest.mark.parametrize('fault',[None,'accuracy','old-round','order','cost','missing','omit-stage'])
def test_compact_review_rejects_coverage_or_cost_mutation(fault):
    from test_strategy_prefill_closure import selected, measurements
    from dev.strategy.prefill_confirmation import verdict
    from tools.pack_kpack_strategy_review import accuracy
    p=selected();raw=measurements(p);measured=verdict(p,raw)
    components={cid:dict(c,authority=row['authority']) for row in raw for cid,c in row['components'].items()}
    rows=[dict(row,jobs={jid:dict(job,accuracy=accuracy(job)) for jid,job in row['jobs'].items()}) for row in raw]
    if fault=='accuracy':next(iter(rows[0]['jobs'].values()))['accuracy']['bad_max']=1
    if fault=='old-round':rows[1]['authority']=rows[0]['authority']
    if fault=='order':rows[1]['order']=rows[0]['order']
    if fault=='cost':next(iter(measured['statistics'].values()))['median_us']+=1
    if fault=='missing':rows.pop()
    if fault=='omit-stage':next(iter(measured['components'].values()))[0]=[]
    if fault:
        with pytest.raises(ValueError):audit_point(p,measured,rows,components,'host-fixture')
    else:
        assert audit_point(p,measured,rows,components,'host-fixture')['checks']>0


def test_prefill_prewarm_uses_native_dense_and_bf16_grouped_only(tmp_path):
    from quactlize.dispatch.planning import plan_prefill,prewarm_groups
    plan=plan_prefill(tmp_path)
    assert len(plan['requests'])==430
    groups=prewarm_groups(plan)
    assert {(compute,typed) for compute,typed,_ in groups}=={('f16',False),('bf16',False)}
    for compute,typed,parents in groups:
        assert parents and all(p['route'].endswith('grouped' if compute=='bf16' else 'dense') for p in parents)


def test_all_host_selections_preserve_measured_recipe_precision_and_scope(tmp_path):
    source=tmp_path/'prefill.cpp';exe=tmp_path/'prefill'
    source.write_text(r'''
#include "quactlize/dispatch/selection.hpp"
#include <cassert>
#include <cstring>
int main() {
 using namespace quactlize::dispatch;
 int count=0,tc=0,full=0;
 for(auto const& k:prefill::data::kKnots) {
   bool grouped=k.experts>1;
   qks_request_v1 r{1,sizeof(r),k.q,grouped?2:0,k.tokens*(grouped?8:1),k.n,k.k,k.experts,k.tokens,
     k.q==12?UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001)};
   qks_prefill_choice_v2 out{};
   assert(prefill::query(r,7,k.compute,1,out)==QKS_OK);
   assert(out.policy==19 && out.compute_type==k.compute && !out.base.predicted);
   assert(out.base.route==k.route && out.base.dequant_config==k.dequant && out.base.measured_tokens==k.tokens);
   for(unsigned mask:{1U,3U,7U})
     assert(prefill::query(r,mask,k.compute,1,out)==((mask&(1U<<k.route))?QKS_OK:QKS_BINDING));
   assert(!prefill::knot(r,1-k.compute,1));
   assert(!prefill::knot(r,k.compute,8));
   auto bad=r;bad.m++;assert(!prefill::knot(bad,k.compute,1));
   bad=r;bad.max_rows++;assert(!prefill::knot(bad,k.compute,1));
   bad=r;bad.mapping_id=0;assert(!prefill::knot(bad,k.compute,1));
   bad=r;bad.n++;assert(!prefill::knot(bad,k.compute,1));
   bad=r;bad.qtype=8;assert(!prefill::knot(bad,k.compute,1));
   if(k.config>=0) {
     auto const& c=prefill::data::kConfigs[k.config];r.route=c.route;
     auto selected=select_tc(r,false,k.compute,{&c,QKS_PREFILL_COMPONENT_MEASURED});
     assert(selected && selected.policy==19 && selected.symbol==c.symbol);
     auto actual=selected.config();
     assert(actual.tm==c.tm && actual.tn==c.tn && actual.tk==c.tk && actual.wm==c.wm && actual.wn==c.wn);
     assert(actual.dn==c.dn && actual.stages==c.stages && actual.split==c.split && actual.grid_mode==c.grid_mode && actual.grid_b==c.grid_b);
     auto launch=recipe(actual,r,4,72);
     assert(launch.split==c.split && launch.grid>=0);
     tc++;
   } else { assert(k.route==2);full++; }
   count++;
 }
 assert(count==430 && tc==257 && full==173);
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(ROOT),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
