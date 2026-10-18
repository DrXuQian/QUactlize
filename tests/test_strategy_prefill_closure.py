"""Host contract tests only; these fixtures are not PPU performance evidence."""
import copy
from dataclasses import asdict
import io
import json
import lzma
from pathlib import Path

import pytest

from dev.strategy.components import statistics_checked
from dev.strategy.prefill_closure import point_finalists, read_compact
from dev.strategy.prefill_confirmation import bind_point, ordered_jobs, verdict
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest


def tc(dn=16):
    return Recipe.make('tc','fq',qtype=12,symbol='host_fixture',tm=32,tn=64,tk=256,
                       wm=32,wn=16,stages=2,ap=0,dn=dn,persistent=-1,
                       split=1,grid_mode=0,grid_b=0,dequant_config=-1)


def selected():
    w=Workload(12,512,2048,128)
    arms={a:dict(recipe=tc(dn),roles=roles,core_components=[a],helpers={})
          for a,dn,roles in [('original',16,['original']),('candidate',32,['rule'])]}
    records={a:dict(median_us=us,spread_pct=0) for a,us in [('original',10),('candidate',9)]}
    p=point_finalists(w,arms,records,{},[tc(32)])
    old=dict(id=w.key,workload=asdict(w),arms=[dict(id=a,recipe=asdict(v['recipe']),
        module=a,spec={},binding=dict(launch=dict(split=1,grid=0,algorithm=0),query=a)) for a,v in arms.items()])
    return bind_point(p,old)


def measurements(p, candidate=9., original=10.):
    rounds=[]
    original_jobs=set(p['arms'][p['original']]['jobs'])
    for i in range(6):
        components,jobs={},{}
        authority=dict(sha256='host-fixture',round=i,point=p['id'])
        for j in ordered_jobs(p,i):
            cost=original if j['id'] in original_jobs else candidate
            identity=dict(authority=authority.copy(),job=j['id'],node=dict(name='host_fixture'),
                          contract=p['workload'])
            cid=digest(identity)
            row=dict(statistics_checked([[cost]*3]*3),identity=identity,kind='producer',
                     first_use_excluded=True,host_launch_cost_added=False,timed_kernel_types=1)
            row['record_sha256']=digest(row);components[cid]=row
            jobs[j['id']]=dict(correctness='PASS',components=[cid],
                              numerical=[dict(nonfinite=0,relative_l1_error=.001)])
        rounds.append(dict(point=p['id'],round=i,status='PASS',jobs=jobs,components=components,
                           authority=authority,order=[j['id'] for j in ordered_jobs(p,i)]))
    return rounds


def test_real_dn32_rule_generated_without_target_champion():
    from tools.export_kpack_strategy import load_sources
    from dev.strategy.prescreen import Prescreen
    from dev.strategy.space import scheduled_key
    w=Workload(13,4096,3072,128)
    table,_=load_sources()
    out,report=Prescreen(table).select(w,blind=True,per_stratum=2,stencil='warps')
    matching=[r for r in out if r.kind=='tc' and r.route=='sf' and
              [dict(r.parameters)[k] for k in ('tm','tn','tk','wm','wn','stages','split')]==[32,64,256,32,16,2,1]]
    assert {dict(r.parameters)['dn'] for r in matching}>={16,32}
    assert report['champion_injection'] is False and report['rule_revision']==27


def test_bind_is_bounded_and_alternates_without_expanding_configs():
    p=selected()
    assert len(p['arms'])==len(p['jobs'])==2
    assert ordered_jobs(p,0)==list(reversed(ordered_jobs(p,1)))
    assert ordered_jobs(p,0)==ordered_jobs(p,2)
    with pytest.raises(ValueError):ordered_jobs(p,6)


def test_confirmation_cannot_fall_back_to_original_on_regression():
    p=selected()
    ok=verdict(p,measurements(p))
    assert ok['status']=='PASS' and ok['winner']==p['best_rule']
    assert not ok['legacy_fallback'] and not ok['performance_admitted']
    bad=verdict(p,measurements(p,candidate=11))
    assert bad['status']=='RULE_REGRESSION' and bad['winner'] is None


@pytest.mark.parametrize('fault',['missing','same-order','numerical','warmup','old-round','samples'])
def test_confirmation_negatives(fault):
    p=selected();rows=measurements(p)
    row=rows[1];j=next(iter(row['jobs'].values()));c=next(iter(row['components'].values()))
    if fault=='missing':rows.pop()
    if fault=='same-order':row['order'].reverse()
    if fault=='numerical':j['numerical'][0]['relative_l1_error']=float('nan')
    if fault=='warmup':c['first_use_excluded']=False
    if fault=='old-round':c['identity']['authority']['round']=0
    if fault=='samples':c['samples_us'][0][0]=0
    if fault in ('warmup','old-round','samples'):
        c['record_sha256']=digest({k:v for k,v in c.items() if k!='record_sha256'})
    with pytest.raises(ValueError):verdict(p,rows)


def test_incomplete_compact_is_not_read_as_finished():
    value=lzma.compress((json.dumps(dict(type='point',id='host-fixture'))+'\n').encode())
    with pytest.raises(ValueError,match='incomplete'):read_compact(io.BytesIO(value))


def test_no_graph_or_profiler_for_dense_provider_and_no_new_tc_build():
    root=Path(__file__).resolve().parents[1]
    worker=(root/'dev/strategy/prefill_confirm_worker.py').read_text()
    parent=(root/'tools/confirm_kpack_prefill.py').read_text()
    dense=worker.split("elif b.w.operation == 'dense':",1)[1].split('\n                            else:',1)[0]
    assert 'measure_ordinary(b, full)' in dense and 'measure_core' not in dense and '.capture' not in dense
    assert 'current_module(record, spec' in worker and 'compiler.build(' not in worker+parent
    assert 'measure-{index}-{uuid.uuid4().hex}' in worker


def test_small_package_keeps_accuracy_and_raw_samples_without_binary(tmp_path):
    import tarfile
    from tools.confirm_kpack_prefill import package
    p=selected(); rows=measurements(p)
    shard=tmp_path/'shard-0';shard.mkdir()
    (shard/'authority.json').write_text(json.dumps(dict(host_test=True)))
    (tmp_path/'finalists.json').write_text(json.dumps(p))
    (shard/'fake.so').write_bytes(b'DO_NOT_EXPORT')
    for row in rows:
        row['record_sha256']=digest(row)
        folder=shard/'points'/p['id'];folder.mkdir(parents=True,exist_ok=True)
        (folder/f'round-{row["round"]}.json').write_text(json.dumps(row))
    package(tmp_path)
    archive,=tmp_path.glob('confirmation-*.tgz')
    with tarfile.open(archive) as tar:
        assert not any(n.endswith('.so') for n in tar.getnames())
        compact,=[m for m in tar if m.name.endswith('.xz')]
        tables,groups=read_compact(tar.extractfile(compact))
    assert len(groups['prefill_confirmation_round'])==6
    assert len(groups['component'])==12
    assert groups['component'][0]['samples_us']==[[9.]*3]*3 or groups['component'][0]['samples_us']==[[10.]*3]*3
    assert all(j['accuracy']['correctness']=='PASS' for r in groups['prefill_confirmation_round']
               for j in r['jobs'].values())
