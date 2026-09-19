"""Handoff failures must be visible before a user launches a PPU run."""
from dataclasses import asdict
from pathlib import Path
import copy
import csv
import io
import json
import pytest

from dev.tp2_decode import fallback_plan as plan
from dev.tp2_decode import plan as prior
from dev.tp2_decode.build_fallback import source
from dev.tp2_decode.run_fallback import profile_check,profile_status,checkpoint,verify,candidate_entry


def test_bounded_scope_uses_previous_minimum_not_always_new_candidate():
    assert len(plan.POINTS)==18 and len(set(p.name for p in plan.POINTS))==18
    assert len(plan.BEST)==11 and len(plan.REDUCERS)==5
    assert plan.BEST['q5-routed-down']==plan.BEST['q8-ssm-out']=='incumbent'
    for p in plan.MODEL_POINTS:
        if plan.BEST[p.name]!='incumbent':
            cfg=prior.candidates(p)[int(plan.BEST[p.name])]
            assert cfg.name not in ('clone','generic')
    assert {p.q for p in plan.REDUCERS}=={10,11,12,13,14}
    assert all(p.compute==1 for p in plan.REDUCERS)


def test_candidate_calls_production_launch_not_a_copied_kernel():
    for q in (8,10,11,12,13,14):
        v=5 if q==8 else 3
        text=source(q,{(v,4,4,4)})
        assert f'launch_v2<{q},{v},4,4,4>(d,f.split)' in text
        assert 'kernel_body' not in text and 'register_reuse_body' not in text
        assert 'c.n=' not in text and 'c.k=' not in text


def test_production_entry_is_explicit_and_cannot_substitute_old_execution():
    entry=dict(library='libquactlize_ppu_candidate.so',symbol='quactlize_kpack_simt_run_v2',scope='PRODUCTION_C_ABI')
    assert candidate_entry(dict(entry=entry))==entry
    assert candidate_entry({})['symbol']=='fallback_run'
    for key,value in (('library','libquactlize_ppu_execution.so'),('symbol','fallback_run'),
                      ('library','../candidate.so'),('scope','MEASURED')):
        with pytest.raises(ValueError):candidate_entry(dict(entry=entry|{key:value}))


def raw_csv(names,grids,blocks):
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['ID','Kernel Name','Grid Size','Block Size'],quoting=csv.QUOTE_ALL)
    w.writeheader()
    for i,(name,grid,block) in enumerate(zip(names,grids,blocks)):
        w.writerow({'ID':i,'Kernel Name':name,'Grid Size':f'({grid},1,1)','Block Size':f'({block},1,1)'})
    return buf.getvalue()


def test_acu_checks_reducer_and_exact_launch_identity():
    p=next(p for p in plan.MODEL_POINTS if p.name=='tp2-q8-out')
    cfg=dict(variant=5,columns=8,warps=4,values=4,split=8)
    r=dict(point=asdict(p),candidate=cfg)
    names=list(plan.kernel_names(p,cfg))
    good=raw_csv(names,[p.n//32*8,p.n//64],[128,32])
    assert len(profile_check(good,r,'fallback'))==2
    for bad in (raw_csv(names[:1],[p.n//32*8],[128]),
                good.replace('reduce_decode_rows<8>','reduce_decode_rows<4>'),
                good.replace('(128,1,1)','(256,1,1)'),
                good.replace('q8_vector::kernel','wrong::kernel')):
        with pytest.raises(ValueError):profile_check(bad,r,'fallback')


def test_reference_incumbent_uses_exact_old_qtype_recipe_reducer_geometry():
    p=plan.REDUCERS[0]
    cfg=dict(variant=3,columns=4,warps=4,values=4,split=4)
    r=dict(point=asdict(p),reference=dict(arm='incumbent',record=dict(candidates=[cfg])))
    good=raw_csv(plan.frozen_kernel_names(p,cfg),[p.n//16*4,p.n//128],[128,128])
    assert len(profile_check(good,r,'reference'))==2
    for bad in (good.replace('register_reuse<10,','register_reuse<11,'),
                good.replace('register_reuse_reduce<10>','register_reuse_reduce<12>'),
                good.replace('(128,1,1)','(256,1,1)')):
        with pytest.raises(ValueError):profile_check(bad,r,'reference')


def test_receipt_flags_match_actual_m1_implementation():
    q8=next(p for p in plan.POINTS if p.name=='tp2-q8-qkv')
    a=plan.production_config(q8,dict(variant=5,columns=4,warps=8,values=4,split=1))
    assert a['hoist'] and not a['fixed']
    q5=next(p for p in plan.POINTS if p.name=='q5-routed-down')
    b=plan.production_config(q5,dict(variant=3,columns=4,warps=2,values=8,split=1))
    assert b['fixed'] and b['changes']==3 and not b['hoist']
    assert plan.frozen_kernel_names(q5,b)[0]=='register_reuse_model<13,1,3,4,2,8,1,3>'


def test_verify_rejects_modified_payload_reference_and_source(tmp_path,monkeypatch):
    from dev.tp2_decode import run_fallback as gate
    (tmp_path/'source.hpp').write_text('source')
    (tmp_path/'reference-manifest.json').write_text('{}')
    (tmp_path/'candidate.so').write_bytes(b'not-an-ELF-host-check')
    pin=gate.sha(tmp_path/'reference-manifest.json');monkeypatch.setattr(gate,'REFERENCE_MANIFEST',pin)
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    m=dict(schema=plan.SCHEMA,records=[dict(point=asdict(p)) for p in plan.POINTS],
           payloads={'candidate.so':gate.sha(tmp_path/'candidate.so')},
           source_hashes={'source.hpp':gate.sha(tmp_path/'source.hpp')})
    def write(data): (tmp_path/'manifest.json').write_text(json.dumps(data))
    write(m);assert verify(tmp_path)['schema']==plan.SCHEMA
    for key in ('payloads','source_hashes'):
        bad=copy.deepcopy(m);bad[key][next(iter(bad[key]))]='bad';write(bad)
        with pytest.raises(ValueError):verify(tmp_path)
    write(m);(tmp_path/'reference-manifest.json').write_text('{"wrong":1}')
    with pytest.raises(ValueError,match='reference|authority'):verify(tmp_path)


def test_prepare_gate_calls_real_dispatch_and_checks_its_mapping():
    root=Path(__file__).resolve().parents[1]
    text=(root/'dev/moe_prepare/bench.cu').read_text()
    assert 'prepare_detail::launch<Shape,Stride>(plan,stream)' in text
    assert 'direct=arm==1 && prepare_detail::all_simt_supported(plan)' in text
    assert text.count('#ifdef QK_PREPARE_PRODUCTION')==2


def test_profile_denominator_cannot_turn_partial_into_pass():
    row=dict(profiles=[dict(arm='reference',status='PASS')])
    assert profile_status(row,True)=='INCOMPLETE'
    assert profile_status(row,False)=='NOT_REQUESTED'
    row['profiles'].append(dict(arm='fallback',status='PASS'))
    assert profile_status(row,True)=='PASS'
    row['profiles'][1]['arm']='reference'
    assert profile_status(row,True)=='INCOMPLETE'


def test_interrupted_resume_preserves_later_points_and_each_profile(tmp_path):
    records=[dict(point=dict(name=n)) for n in ('first','later')]
    later=dict(point='later',status='PASS',profiles=[dict(arm='reference',status='PASS')])
    states={'later':later}
    first=dict(point='first',status='PARTIAL',profiles=[dict(arm='reference',status='PASS')])
    checkpoint(tmp_path,records,states,first)
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert not saved['complete'] and saved['records']==[first,later]
    assert saved['records'][0]['profiles'][0]['status']=='PASS'
