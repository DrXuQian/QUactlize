"""Data-only evidence selection; no initialization tuner or GPU calls."""
from dataclasses import replace
import json

import pytest

from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload
from tools.export_kpack_strategy import load_sources


def simt(warps=4):
    return Recipe.make('simt','fq',variant=1,columns=4,warps=warps,values=4,split=1)


def observation(w,r,priority=10,scope='COMPLETE_CALL'):
    return Observation(w,r,'test',scope,priority,'TEST_ONLY','host-fixture')


def test_old_box_data_is_imported_not_replaced_with_an_empty_table():
    table,sources=load_sources()
    assert len(sources)==11
    assert len(table.observations)==7474
    assert len(table.exact)==4059
    assert {r.workload.qtype for r in table.observations}=={8,10,11,12,13,14}
    assert sum(r.source=='effective-decode' for r in table.observations)==2089
    assert sum(r.source=='historical-resident' for r in table.observations)==2982
    assert sum(r.scope=='COMPONENT_SUM' for r in table.observations)==1720
    assert all(r.workload.compute=='f16' and r.details['external_adapters_included'] is False
               for r in table.observations if r.scope=='COMPONENT_SUM')
    assert all(r.workload.storage=='f16' for r in table.observations if r.scope=='RESIDENT_GEMM')
    assert StrategyTable.from_dict(table.as_dict()).identity==table.identity


def test_effective_decode_promotions_keep_their_exact_recipe():
    from tools.generate_decode_selection import effective,EVIDENCE
    from tools.export_kpack_strategy import selected_recipe,small_workload
    board=effective(json.loads(EVIDENCE.read_text()))
    table,_=load_sources()
    for row in board['exact']:
        r,w=selected_recipe(row),small_workload(row['key'])
        records=[x for x in table.exact[w.key] if x.source=='effective-decode']
        assert len(records)==1 and records[0].recipe==r
        if r.eligible(w):
            assert table.lookup(w).recipe==r
        if row.get('implementation',{}).get('kind')=='q8-address':
            assert dict(r.parameters)['q8_body']==row['implementation']['body']
        if row.get('dequant_config',-1)>=0:
            assert dict(r.parameters)['dequant_config']==row['dequant_config']


def test_all_exact_winners_survive_control_inventory_and_small_budget_is_rejected():
    w=Workload(12,1024,5120,1)
    table=StrategyTable([observation(w,simt(i),i) for i in (2,4,8)])
    assert set(r.key for r in table.shortlist(w,simt(4),3))=={simt(i).key for i in (2,4,8)}
    with pytest.raises(ValueError,match='drop an existing exact winner'):
        table.shortlist(w,simt(4),2)
    assert table.lookup(replace(w,n=2048,output_stride=0)) is None
    assert len(table.shortlist(replace(w,n=2048,output_stride=0),simt(4),8))==3
    assert table.shortlist(replace(w,n=4096,output_stride=0),simt(4),8)==(simt(4),)
    assert table.lookup(replace(w,compute='bf16')) is None
    assert table.shortlist(replace(w,compute='bf16'),simt(4))==(simt(4),)


def test_precision_stride_and_proposal_only_evidence_remain_distinct():
    w=Workload(12,1024,5120,1)
    assert w.key==replace(w,input_stride=5120,output_stride=1024).key
    for other in (replace(w,input_stride=5136),replace(w,compute='bf16'),replace(w,storage='f16'),
                  replace(w,m=2),replace(w,fusion='swiglu')):
        assert other.key!=w.key
    table=StrategyTable([observation(w,simt(8),scope='PROPOSAL')])
    assert table.lookup(w) is None
    tc=Recipe.make('tc','fq',tm=8,tn=64,tk=256,wm=8,wn=16,stages=3,split=4,ap=1)
    assert tc.eligible(w)
    assert not tc.eligible(replace(w,m=2))
    assert not tc.eligible(replace(w,k=256))
    assert not Recipe.make('full','full',dequant_config=4).eligible(w)


def test_table_creation_lookup_and_shortlist_do_not_write_or_invoke_a_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w=Workload(12,1024,5120,1)
    table=StrategyTable([observation(w,simt())])
    assert table.lookup(w).recipe==simt()
    assert table.shortlist(w,simt())==(simt(),)
    assert not list(tmp_path.iterdir())
