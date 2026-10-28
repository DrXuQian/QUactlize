"""Host-only sweep planning, resource replay and Asys short-screen contracts."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from dev.strategy import moe_sweep as sweep
from dev.strategy import moe_sweep_plan as plans
from dev.strategy.asys_components import parse_rows
from dev.strategy.smallm_asys import validate, check_producers
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from test_moe_model_plan import synthetic_inventory
from test_smallm_asys import capture_fixture, provenance


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return plans.make_plan(tmp_path_factory.mktemp('moe-configs'), synthetic_inventory())


def resources(point, cu=72, occupancy=2):
    images = dict(tc={}, vectors={})
    for a in point['arms']:
        images['tc' if a['recipe']['kind']=='tc' else 'vectors'][a['module']] = dict(sha256=a['module'], key='build-'+a['module'])
    receipts = {}
    for key, req in sweep.requests(point).items():
        image = images['tc'][req['module']]
        receipts[key] = dict(status='READY', rc=0, runtime_status=0, cutlass_status=0,
            request=req['request'], build_key=image['key'], module_sha256=image['sha256'],
            device=dict(name='PPU-HOST-TEST-NOT-EVIDENCE', ordinal=0, compute_units=cu),
            occupancy=occupancy, workspace_bytes=100, shared_bytes=128)
    return receipts, images


def test_model_shapes_all_tokens_formats_tc_simt_and_recall(plan):
    assert plan['cohort']==plans.COHORT and plan['counts']['points']==216
    assert not plan['historical_times_reused'] and not plan['production_changed']
    assert plan['coverage']['fused_gate_up_swiglu']=='BACKUP_NOT_EXECUTED'
    assert plan['protocol']['screen']==[4,3] and plan['protocol']['confirm']==[6,5]
    for p in plan['points']:
        assert {'tc','simt'} <= {a['recipe']['kind'] for a in p['arms']}
        assert p['config_sweep'] and p['consumers'] and not p['rule_recall']['missing']
        assert p['arms'][0]['id']=='original'
        assert all('median_us' not in ref and 'us' not in ref for ref in p['historical_references'])
    q5 = next(p for p in plan['points'] if p['workload']['qtype']==13 and p['workload']['m']==5)
    r = Recipe.make('simt','fq',variant=3,columns=8,warps=4,values=4,split=1)
    assert any(a['recipe']==json.loads(json.dumps(asdict(r))) and 'rule' in a['roles'] for a in q5['arms'])


def test_resources_expand_persistent_grids_dedup_aliases_not_ordinary(plan):
    p = next(p for p in plan['points'] if p['workload']['qtype']==12 and p['workload']['m']==8)
    receipts, images = resources(p)
    resolved = sweep.resolve(p,receipts,images)
    assert resolved['arms'][0]['id']=='original'
    assert len({a['id'] for a in resolved['arms']})==len(resolved['arms'])
    tc=[a for a in resolved['arms'] if a['recipe']['kind']=='tc']
    assert {a['binding']['launch']['algorithm'] for a in tc}=={0,1}
    assert all(a['binding']['launch']['grid']<=144 for a in tc if a['binding']['launch']['algorithm'])
    assert all(a['binding']['launch']['grid']==0 for a in tc if not a['binding']['launch']['algorithm'])
    r2,i2=resources(p,cu=64,occupancy=1)
    other=sweep.resolve(p,r2,i2)
    assert all(a['binding']['launch']['grid']<=64 for a in other['arms'] if a['recipe']['kind']=='tc')
    assert resolved['resolved_sha256']!=other['resolved_sha256']
    assert resolved['resource_resolution']['status']=='COMPLETE'
    assert any(len(v['aliases'])>2 for v in resolved['resource_resolution']['bindings'].values())


@pytest.mark.parametrize('fault',['missing','image','runtime','device','request'])
def test_bad_resource_proof_not_silently_skipped(plan,fault):
    p=plan['points'][0];receipts,images=resources(p)
    row=next(iter(receipts.values()))
    if fault=='missing':receipts.pop(next(iter(receipts)))
    elif fault=='image':row['module_sha256']='wrong'
    elif fault=='runtime':row['runtime_status']=999
    elif fault=='device':row['device']['compute_units']=0
    else:row['request']['n']+=1
    with pytest.raises(ValueError):sweep.resolve(p,receipts,images)


def test_unavailable_rule_visible_but_unavailable_control_blocks(plan):
    p=next(p for p in plan['points'] if p['arms'][0]['recipe']['kind']=='tc')
    receipts,images=resources(p)
    for row in receipts.values():row.update(status='UNAVAILABLE',rc=1,occupancy=0)
    with pytest.raises(ValueError,match='current MoE control'):sweep.resolve(p,receipts,images)
    p=next(p for p in plan['points'] if p['arms'][0]['recipe']['kind']=='simt')
    receipts,images=resources(p)
    for row in receipts.values():row.update(status='UNAVAILABLE',rc=1,occupancy=0)
    got=sweep.resolve(p,receipts,images)
    assert got['resource_resolution']['rejected']
    assert all(a['recipe']['kind']!='tc' for a in got['arms'])


def test_frozen_resource_and_execution_replay_survives_json_not_inventory_edits(plan):
    p=plan['points'][0];receipts,images=resources(p)
    resolved=sweep.resolve(p,receipts,images)
    captured=json.loads(json.dumps(dict(execution_point=resolved,resource_queries=receipts,
        data=dict(measurement_phase='screen'),screen_sha256=None)))
    assert sweep.verify_capture(p,captured,images)==resolved
    changed=deepcopy(captured);changed['execution_point']['arms'].pop()
    with pytest.raises(ValueError,match='inventory'):sweep.verify_capture(p,changed,images)
    changed=deepcopy(captured);changed['data']['measurement_phase']='confirm'
    with pytest.raises(ValueError,match='phase'):sweep.verify_capture(p,changed,images)


def short_capture():
    captured,rows=capture_fixture()
    pieces=[];schedule=[];cursor=0
    for item in captured['trace_schedule']:
        block=rows[cursor:cursor+item['kernels']];cursor+=item['kernels']
        if item['round']<4 and item['sample']<3:pieces+=block;schedule.append(item)
    captured.update(measurement_phase='screen',trace_schedule=schedule)
    captured['protocol'].update(rounds=4,samples=3)
    return captured,pieces


def test_short_screen_is_not_confirmation_and_keeps_raw_samples():
    cap,rows=short_capture()
    data=parse_rows(cap,rows,provenance());validate(data)
    assert data['measurement_phase']=='screen' and not data['production_changed']
    assert len(next(iter(data['component_records'].values()))['trace_slices'])==12
    for fault in ('relabel','bad_numeric','bad_sample'):
        changed=deepcopy(data)
        if fault=='relabel':changed.pop('measurement_phase')
        elif fault=='bad_numeric':changed['arms']['fast']['numerical'][0]['bad']=1
        else:next(iter(changed['component_records'].values()))['trace_slices'].pop()
        with pytest.raises(ValueError):validate(changed)


def test_finalists_use_means_and_rounds_not_min_timestamp_and_keep_tc():
    cap,rows=short_capture();data=parse_rows(cap,rows,provenance())
    point=dict(arms=[dict(id='original',recipe=dict(kind='simt',route='fq')),
                     dict(id='fast',recipe=dict(kind='tc',route='fq'))])
    assert sweep.finalists(point,data)==['original','fast']
    with pytest.raises(ValueError,match='bounded'):sweep.finalists(point,data,limit=1)
    data['measurement_phase']='confirm'
    with pytest.raises(ValueError,match='fresh Asys'):sweep.finalists(point,data)


def test_screen_batches_bound_live_handles_and_dedup_helper_launches(monkeypatch):
    import dev.strategy.confirm as confirm
    from dev.strategy.asys_components import Collector
    calls=[];live=[];peak=[];clock=[0];activities=[]
    w=Workload(13,1024,5120,1)
    point=dict(id=w.key,workload=asdict(w),sf_configs=[],arms=[])
    for i in range(25):
        r=Recipe.make('simt','fq',variant=3,columns=4,warps=2,values=8,split=i+1)
        point['arms'].append(dict(id='original' if i==0 else str(i),recipe=asdict(r),roles=['original' if i==0 else 'rule']))

    class Graph:
        count=16
        def __init__(self,node):self.node=node;self.rt=SimpleNamespace(sync=lambda:None)
        def close(self):pass
        def run(self):
            for _ in range(16):
                activities.append(dict(self.node,kind='kernel',start=clock[0],end=clock[0]+1000));clock[0]+=1100
        def sample(self):pytest.fail('no event timer')

    class Capture:
        groups=[[0],[1]]
        def __init__(self,aid):
            self.descriptions=[dict(name='simt::register_reuse<'+aid+'>',symbol=aid,grid=[8,1,1],block=[64,1,1],shared_bytes=0),
                               dict(name='weighted_finish',symbol='finish',grid=[8,1,1],block=[64,1,1],shared_bytes=0)]
        def isolate(self,index,**kw):return Graph(self.descriptions[index])

    @contextmanager
    def prepared(b,w,arm,*args,**kwargs):
        live.append(arm['id']);peak.append(len(live))
        proof=deepcopy(capture_fixture()[0]['arms']['original'])
        proof['pointer_negatives']=['low','high','units']
        try:yield SimpleNamespace(capture=Capture(arm['id']),data=proof)
        finally:live.remove(arm['id'])

    monkeypatch.setattr(confirm,'prepared_arm',prepared)
    monkeypatch.setattr(Collector,'start',lambda self,rt:(setattr(self,'rt',rt),calls.append('start')))
    monkeypatch.setattr(Collector,'stop',lambda self:(setattr(self,'rt',None),calls.append('stop')))
    data=sweep.measure(None,None,None,None,point,{a['id']:(None,None) for a in point['arms']},{},'test',phase='screen')
    assert max(peak)==12 and not live and calls==['start','stop']*3
    assert len(data['arms'])==25 and len(data['component_records'])==26
    assert len(data['trace_schedule'])==26*12  # same helper not rerun per batch/arm
    parsed=parse_rows(data,activities,provenance());validate(parsed)
    assert len(parsed['decision']['statistics'])==25
