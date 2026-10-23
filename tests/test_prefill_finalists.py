"""Frozen bounded route handoff, fixed resource queries and lossless review."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from dev.prefill_asys.finalists import (load,validate,snapshot,partition,bind_cases,
                                       case_identity,counts,make_plan)
from dev.strategy.resources import inventory,resolve_point
from quactlize.runtime.tuning import digest
from tools.run_prefill_cohort_multi import worker_environment

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'dev/prefill_asys/finalists_20261023.json'


def plan():return load(json.loads(FROZEN.read_text()))


def seal(p):
    p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    return p


def test_scope_reuses_prior_data_and_preserves_all_finalists():
    p=plan()
    assert p['counts']==dict(points=47,families=14,configs=214,captures=188,timed_arms=522,
                            modules=14,queries=64,retained_points=118,original_points=165)
    assert sum('CURRENT_FULL_HELPER_NOT_IN_ROUTE_COHORT' in v['reasons'] for v in p['points'])==40
    assert sum('PRIOR_ROUND_SPREAD' in v['reasons'] for v in p['points'])==8
    assert sum(v['workload']['qtype']==8 for v in p['points'])==30
    for point in p['points']:
        assert 3<=len(point['cases'])<=6
        assert point['prior_winner'] in {c['id'] for c in point['cases']}
        assert all('gpu_sum_us' not in c and 'median_us' not in c for c in point['cases'])
        assert point['fixed_schedules'] and point['confirmation_captures']==2
    assert not p['production_changed'] and not p['performance_admitted']
    assert load(json.loads(json.dumps(snapshot(p))))==p


@pytest.mark.parametrize('kind',['confirmation','winner','helper','precision','resources','reuse','q8'])
def test_resealed_plan_cannot_drop_controls_or_change_precision(kind):
    p=plan();v=next(v for v in p['points'] if v['workload']['qtype']==8)
    if kind=='confirmation':v['confirmation_captures']=1
    elif kind=='winner':v['prior_winner']='dropped-control'
    elif kind=='helper':
        for c in v['cases']:c['roles']=[r for r in c['roles'] if r!='current-full-helper']
    elif kind=='precision':v['workload']['compute']='bf16'
    elif kind=='resources':v['candidates']=[]
    elif kind=='reuse':p['reuse'].append(dict(id=v['id']))
    elif kind=='q8':
        c=next(c for c in v['cases'] if c.get('helper_options',{}).get('dequant_variant')==41)
        v['cases'].remove(c)
    with pytest.raises(ValueError):validate(seal(p))


def test_unavailable_control_fails_instead_of_shrinking_timed_inventory():
    p=plan();inv=inventory(p)
    receipts={k:dict(status='READY',request=v['request'],device=dict(compute_units=72),occupancy=2)
              for k,v in inv['queries'].items()}
    for point in p['points']:
        resolved=resolve_point(point,receipts)
        assert len(resolved['candidates'])<=len(point['candidates'])  # only launch aliases, no expansion
        bound=bind_cases(point,receipts)
        assert len(bound)==len(point['cases'])
        assert all(digest(case_identity(c))==c['id'] for c in bound)
    first=next(iter(receipts));receipts[first]['status']='UNAVAILABLE'
    affected=next(p for p in p['points'] if any(u['point']==p['id'] for u in inv['queries'][first]['uses']))
    with pytest.raises(ValueError,match='unavailable'):bind_cases(affected,receipts)


def test_family_assignment_is_exclusive_and_deterministic():
    p=plan();parts=[partition(p['points'],i,7) for i in range(7)]
    assert sorted(v['id'] for part in parts for v in part)==sorted(v['id'] for v in p['points'])
    for part in parts:
        for point in part:
            w=point['workload'];family=(w['qtype'],w['n'],w['k'])
            assert sum(any((v['workload']['qtype'],v['workload']['n'],v['workload']['k'])==family
                           for v in x) for x in parts)==1
    assert partition(list(reversed(p['points'])),3,7)==parts[3]
    with pytest.raises(ValueError):partition(p['points'],0,15)


def test_finalist_worker_drops_stale_cohort_and_model_environment():
    with patch.dict('os.environ',dict(COHORT='/old',MODEL='/old-model',POINT_ID='old',FINALIST_PLAN='/old-finalist')):
        env=worker_environment(Path('/run'),'3',0,7,'sha',finalists=True)
    assert env['FINALIST_PLAN']=='/run/cohort.json'
    assert not any(k in env for k in ('COHORT','MODEL','POINT_ID'))


def test_plan_only_needs_no_sdk_model_or_old_box_results(tmp_path):
    output=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),'--finalist-plan',str(FROZEN),
        '--output',str(output),'--sdk','/no-sdk','--llama-source','/no-llama','--plan-only',
        '--shard','0','--shards','7'],check=True,capture_output=True)
    actual=json.loads((output/'plan.json').read_text())
    assert actual['finalist_plan_sha256']==plan()['sha256']
    assert actual['counts']==counts(actual['points'])
    assert not (output/'build').exists() and not (output/'probe').exists()


def test_snapshot_rejects_modified_shared_recipe():
    value=json.loads(FROZEN.read_text())
    next(iter(value['recipes'].values()))['helper_mode']='legacy'
    with pytest.raises(ValueError):load(value)


def test_planner_requires_original_complete_sealed_review():
    with pytest.raises(ValueError,match='schema/digest'):make_plan({},dict(rows=[]))
    r=dict(schema='quactlize.prefill-component-review.v1',incomplete=['missing'])
    r['review_sha256']=digest(r)
    with pytest.raises(ValueError,match='incomplete'):make_plan(r,dict(rows=[]))


def test_planner_prunes_stable_points_but_not_a_route_winner():
    original=plan()
    chosen=[next(v for v in original['points'] if v['workload']['qtype']==q) for q in (8,12)]
    review=dict(schema='quactlize.prefill-component-review.v1',points={},candidates={},
                incomplete=[],pending_points=[])
    helpers=dict(rows=[])
    for p in chosen:
        w=p['workload'];rows=[]
        current=next(c for c in p['cases'] if 'current-full-helper' in c['roles'])
        helpers['rows'].append(dict(q=w['qtype'],n=w['n'],k=w['k'],m=w['m'],storage=0,
            dequant=current['helper_options']['dequant_variant'],io=current['helper_options']['vector_io']))
        for c in p['cases']:
            identity=case_identity(c);review['candidates'][c['id']]=identity
            rows.append(dict(candidate=c['id'],arm=c['arm'],helper_mode=c['helper_mode'],
                             status='PASS',gpu_sum_us=10.,rounds_us=[10.]*3))
        rows.append(dict(arm='native-fp16',status='PASS',gpu_sum_us=11.,rounds_us=[11.]*3))
        review['points'][p['id']]=dict(workload=w,consumers=p['consumers'],measurements=rows)
    review['review_sha256']=digest(review)
    result=make_plan(review,helpers)
    assert result['counts']['points']==1 and result['counts']['retained_points']==1
    assert result['points'][0]['workload']['qtype']==8
    # The same stable Q4 point becomes a confirmation, not an inferred winner,
    # when its native timing has noisy rounds.
    review['points'][chosen[1]['id']]['measurements'][-1]['rounds_us']=[8.,11.,14.]
    review['review_sha256']=digest({k:v for k,v in review.items() if k!='review_sha256'})
    assert make_plan(review,helpers)['counts']['points']==2
