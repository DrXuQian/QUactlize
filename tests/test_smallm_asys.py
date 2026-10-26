"""Host-only evidence/scheduling tests. Synthetic samples are not PPU results."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.strategy.smallm_asys import workloads,make_plan,mean_decision,revision,validate,check_producers,METHOD
from dev.strategy.asys_components import Collector,parse_rows
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest


def provenance():
    return dict(quactlize_commit='a'*40,quactlize_tree='b'*40,actlize_commit='c'*40,
        **{k:'d'*64 for k in ('trace_sha256','sqlite_sha256','harness_sha256','images_sha256',
                             'source_sha256','asys_sha256')},
        sdk={'release':'test'},device={'pci':'synthetic-not-device-evidence'})


def capture_fixture():
    w=Workload(13,1024,5120,1)
    proof=dict(correctness='PASS',numerical=[dict(cells=w.n,bad=0,nonfinite=0,tolerance=.005,
        relative_l1_error=.001)]*5,zero_input='PASS',omitted_producer='DETECTED',
        pointer_negatives=['low','high','units'],fixture=dict(l2_bytes=64,copies=4,
        minimum_read_bytes=64,active_ring_bytes=256))
    records={};ids={}
    for name in ('original','fast','reduce'):
        node=dict(name=name,symbol='_Z'+name,grid=[8,1,1],block=[256,1,1],shared_bytes=16)
        identity=dict(node=node,implementation='host-test')
        cid=digest(identity);ids[name]=cid
        records[cid]=dict(identity=identity,kind='reducer' if name=='reduce' else 'producer',
            calls_per_graph=16,timed_kernel_types=1,first_use_excluded=True,host_launch_cost_added=False)
    schedule=[];rows=[];clock=0
    for r in range(6):
        order=('original','reduce','fast') if r%2==0 else ('fast','reduce','original')
        for name in order:
            for s in range(5):
                schedule.append(dict(component=ids[name],round=r,sample=s,kernels=16))
                for _ in range(16):
                    ns={'original':8000,'fast':7000,'reduce':2000}[name]
                    rows.append(dict(records[ids[name]]['identity']['node'],kind='kernel',start=clock,end=clock+ns))
                    clock+=ns+100
    data=dict(status='CAPTURED_NOT_TIMED',correctness='PASS',timing_method=METHOD,production_changed=False,
        workload=asdict(w),arms={a:deepcopy(proof) for a in ('original','fast')},
        roles=dict(original=['original'],fast=['rule']),components=list(records),component_records=records,
        arm_components={a:[ids[a],ids['reduce']] for a in ('original','fast')},trace_schedule=schedule,
        protocol=dict(rounds=6,samples=5,order='FORWARD_REVERSE_ALTERNATING',
                      cache='SEPARATE_METADATA_AND_PRODUCER_RINGS_AT_LEAST_2P25_L2'))
    return data,rows


def test_current_cpp_bounded_inventory_and_common_candidates(tmp_path):
    from quactlize.dispatch.planning import plan_smallm
    from dev.strategy.refresh import request
    from dev.strategy.vector import source
    ws=workloads()
    assert [(w.qtype,w.m) for w in ws]==[(13,m) for m in range(1,5)]+[(14,m) for m in range(2,8)]
    assert all(w.n==1024 and w.k==5120 and w.operation=='dense' and w.storage=='f32' and w.compute=='f16' for w in ws)
    original=plan_smallm(tmp_path,[request(w) for w in ws],dense_fallbacks=True)
    plan=make_plan(original['requests'])
    assert plan['counts']==dict(points=10,arms=199,tc_modules=5,vector_modules=11,families=2)
    assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
    assert not plan['historical_times_reused'] and not plan['native_comparison'] and not plan['production_changed']
    assert all(sum(a['id']=='original' for a in p['arms'])==1 for p in plan['points'])
    assert all(p['execution_contract']['mode']==0 and not p['execution_contract']['ids_present'] for p in plan['points'])
    for spec in plan['vectors'].values():
        text=source(spec)
        assert text.index('common_reader_scope')<text.index('measured_decode(*d,f)')
    assert set(c for p in plan['points'] for c in p['sf_configs'])=={4}


def test_means_not_minima_or_variance_veto_choose_winner():
    costs=dict(original=[[10.]*5]*6,noisy=[[7.]*5,[9.]*5]*3,min_only=[[5.,12.,12.,12.,12.]]*6)
    result=mean_decision(costs,{a:['rule'] for a in costs})
    assert result['winner']=='noisy' and result['delta_pct']==pytest.approx(-20)
    assert result['statistics']['noisy']['noise_advisory']
    assert result['statistics']['min_only']['min_us']==5
    assert not result['native_no_regression_admitted'] and not result['model_tpot_admitted']


def test_asys_raw_gpu_durations_sum_helpers_once_and_keep_proofs():
    captured,rows=capture_fixture()
    result=parse_rows(captured,rows,provenance())
    assert result['decision']['winner']=='fast'
    assert result['decision']['statistics']['original']['mean_us']==10
    assert result['decision']['statistics']['fast']['mean_us']==9
    assert captured['status']=='CAPTURED_NOT_TIMED'
    assert len(result['component_records'])==3  # shared reducer measured once, included for both arms
    validate(result)
    for change in ('raw','mean','commit','gap','negative','nonfinite'):
        bad=deepcopy(result);r=next(iter(bad['component_records'].values()))
        if change=='raw':r['trace_slices'][0]['durations_ns'][0]+=1
        elif change=='mean':r['samples_us'][0][0]+=1
        elif change=='commit':r['latency_provenance']['quactlize_commit']='unknown'
        elif change=='gap':r['trace_slices'][0]['first_activity']+=1
        elif change=='negative':bad['arms']['fast']['pointer_negatives']=[]
        else:bad['arms']['fast']['numerical'][0]['nonfinite']=1
        with pytest.raises(ValueError):validate(bad)


def test_adjacent_timestamp_overlap_does_not_reject_or_clip_gpu_durations():
    data,rows=capture_fixture()
    rows[1]['start']=rows[0]['start']
    result=parse_rows(data,rows,provenance())
    record=next(r for r in result['component_records'].values() if r['identity']['node']['name']=='original')
    assert record['trace_slices'][0]['durations_ns'][:2]==[8000,16100]
    assert record['samples_us'][0][0]==pytest.approx((8000*15+16100)/16000)
    assert result['interval_policy']=='PER_RECORD_END_MINUS_START_NO_ADJACENCY_REQUIREMENT'
    validate(result)


@pytest.mark.parametrize('change',['missing','extra','zero','kernel','grid','block','shared','schedule','event'])
def test_asys_parse_fails_closed_without_event_fallback(change):
    data,rows=capture_fixture()
    if change=='missing':rows.pop()
    elif change=='extra':rows.append(dict(rows[-1],start=rows[-1]['end']+100,end=rows[-1]['end']+200))
    elif change=='zero':rows[0]['end']=rows[0]['start']
    elif change=='kernel':rows[0]['symbol']='other'
    elif change=='grid':rows[0]['grid']=[9,1,1]
    elif change=='block':rows[0]['block']=[128,1,1]
    elif change=='shared':rows[0]['shared_bytes']=32
    elif change=='schedule':data['trace_schedule'][1]=deepcopy(data['trace_schedule'][0])
    else:data['timing_method']='CUDA_EVENT'
    with pytest.raises(ValueError):parse_rows(data,rows,provenance())


def test_collector_calls_only_graph_run_and_closes_profiler_after_sync_error():
    calls=[]
    start=lambda:calls.append('start') or 0
    stop=lambda:calls.append('stop') or 0
    rt=SimpleNamespace(lib=SimpleNamespace(hggcProfilerStart=start,hggcProfilerStop=stop),
                       sync=lambda:calls.append('sync'))
    graph=SimpleNamespace(rt=rt,count=16,run=lambda:calls.append('run'),
        sample=lambda:pytest.fail('event sample is forbidden'))
    c=Collector()
    with pytest.raises(ValueError):c.sample('id',graph,0,0)
    c.start(rt);assert c.sample('id',graph,0,0) is None;c.stop()
    assert c.schedule==[dict(component='id',round=0,sample=0,kernels=16)]
    assert calls==['sync','start','sync','run','sync','sync','stop']
    c.start(rt)
    def broken():raise RuntimeError('sync failure')
    rt.sync=broken
    with pytest.raises(RuntimeError,match='sync failure'):c.stop()
    assert c.rt is None and calls[-1]=='stop'


def test_asys_shared_memory_matches_launch_dynamic_not_resource_total(tmp_path):
    from dev.prefill_asys.trace import activities
    path=tmp_path/'trace.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
        db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'kernel'),(2,'_Zkernel')])
        fields='start end demangledName mangledName gridX gridY gridZ blockX blockY blockZ staticSharedMemory dynamicSharedMemory deviceId contextId streamId globalPid'
        db.execute('CREATE TABLE HGPTI_ACTIVITY_KIND_KERNEL ('+','.join(f+' INTEGER' for f in fields.split())+')')
        db.execute('INSERT INTO HGPTI_ACTIVITY_KIND_KERNEL VALUES ('+','.join(['?']*16)+')',
            [1,2,1,2,1,1,1,256,1,1,1024,16,0,1,2,99])
    assert activities(path)[0]['shared_bytes']==1040
    assert activities(path,launch_shared=True)[0]['shared_bytes']==16


def test_frozen_source_requires_full_commit_and_lite_repack_recognizes_schema(tmp_path):
    from tools.pack_simt_split_review import result_root,exclusion
    p=provenance();value={k:p[k] for k in ('quactlize_commit','quactlize_tree','actlize_commit')}
    value['snapshot']='COMMITTED_ARCHIVE'
    path=tmp_path/'.quactlize-source.json';path.write_text(json.dumps(value))
    assert revision(tmp_path)==value
    value['quactlize_commit']='a1234';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='full commit'):revision(tmp_path)
    (tmp_path/'plan.json').write_text(json.dumps(dict(schema='quactlize.smallm-asys.v1')))
    assert result_root(tmp_path)==tmp_path
    assert exclusion(Path('trace.asysrep')) and exclusion(Path('trace.sqlite'))
    assert exclusion(Path('point.json')) is None


def test_wrong_lookalike_reader_and_missing_config_rejected():
    w=Workload(13,1024,5120,2)
    from dev.strategy.vector import execution_contract
    contract=execution_contract(w)
    arm=dict(id='original',spec=dict(config=dict(variant=3,columns=4,warps=8,values=8)),recipe=dict(kind='simt'))
    point=dict(arms=[arm],execution_contract=contract)
    data=dict(arms=dict(original=dict(receipt=dict(execution_contract=contract))),arm_components=dict(original=['c']),
        component_records=dict(c=dict(kind='producer',identity=dict(node=dict(name='quactlize::execution::simt::common_reader<13>',
                                                    symbol='quactlize::execution::simt::common_reader<13>')))))
    check_producers(point,data)
    data['component_records']['c']['identity']['node']={k:'quactlize::execution::simt::register_reuse<13>' for k in ('name','symbol')}
    with pytest.raises(ValueError,match='wrong production'):check_producers(point,data)
    data['arms']={}
    with pytest.raises(ValueError,match='inventory'):check_producers(point,data)


@pytest.mark.parametrize('baseline',['quactlize','native'])
def test_box_freezes_committed_sources_and_pin_without_network(tmp_path,baseline):
    source=tmp_path/'checkout';source.mkdir();(source/'tools').mkdir()
    actual=Path(__file__).resolve().parents[1]/'tools/run_smallm_regressions_asys_box.sh'
    wrapper=source/'tools/run_smallm_regressions_asys_box.sh';wrapper.write_bytes(actual.read_bytes())
    # A host-only fake planner tests shell checkout/resume, not GPU measurements.
    planner=source/'tools/run_smallm_regressions_asys.py'
    planner.write_text('''import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
assert json.loads((root/'.quactlize-source.json').read_text())['snapshot']=='COMMITTED_ARCHIVE'
assert (root/'third_party/actlize/header.h').read_text()=='pinned'
assert '--plan-only' in sys.argv
p=Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(exist_ok='--resume' in sys.argv)
(p/'plan.json').write_text('{}')
''')
    native_planner=source/'tools/run_smallm_native_asys.py'
    native_planner.write_bytes(planner.read_bytes())
    act=source/'third_party/actlize';act.mkdir(parents=True)
    def git(directory,*args):
        return subprocess.check_output(['git','-C',str(directory),*args],text=True,stderr=subprocess.PIPE).strip()
    for root in (act,source):
        git(root,'init','-q');git(root,'config','user.name','Host Test');git(root,'config','user.email','test@example.invalid')
    (act/'header.h').write_text('pinned');git(act,'add','header.h');git(act,'commit','-qm','dependency fixture')
    pin=git(act,'rev-parse','HEAD')
    git(source,'add','tools');git(source,'update-index','--add','--cacheinfo','160000,'+pin+',third_party/actlize')
    git(source,'commit','-qm','harness fixture');rev=git(source,'rev-parse','HEAD')
    # Dirty source and dirty dependency must not get measured/relabelled as HEAD.
    planner.write_text('raise RuntimeError("uncommitted must not run")')
    native_planner.write_text('raise RuntimeError("uncommitted must not run")')
    (act/'header.h').write_text('dirty')
    env={k:v for k,v in os.environ.items() if k not in ('RUN','RESUME_RUN','PPU_SDK','ASYS','PPU_SMI','COHORT')}
    env.update(PLAN_ONLY='1',OUTPUT_ROOT=str(tmp_path),PYTHON=sys.executable,GPUS='0,1,2,3',BASELINE=baseline)
    got=subprocess.run(['bash',str(wrapper)],env=env,text=True,capture_output=True)
    assert got.returncode==0,got.stdout+got.stderr
    run,=tmp_path.glob('kpack-smallm-asys.*')
    assert json.loads((run/'source/.quactlize-source.json').read_text())['quactlize_commit']==rev
    env['RESUME_RUN']=str(run)
    got=subprocess.run(['bash',str(wrapper)],env=env,text=True,capture_output=True)
    assert got.returncode==0,got.stdout+got.stderr
