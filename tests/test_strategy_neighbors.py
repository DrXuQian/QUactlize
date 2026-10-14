from dataclasses import replace

import pytest

from dev.strategy.neighbors import one_axis
from dev.strategy.prescreen import Prescreen, rejection
from dev.strategy.space import producer_key
from quactlize.dispatch.table import Recipe, StrategyTable, Workload
from test_strategy_prescreen import tc, simt, observation


def test_stencil_changes_one_axis_and_never_crosses_format_or_precision():
    for seed in (tc(),simt()):
        before=dict(seed.parameters)
        for r,axis in one_axis(seed,'local'):
            after=dict(r.parameters)
            changed={k for k in before if k!='symbol' and before[k]!=after[k]}
            assert changed=={axis}
            assert r.route==seed.route and r.kind==seed.kind
            assert axis not in ('qtype','persistent','ap','split','grid_mode','dequant_config')
            if r.kind=='tc':assert before['symbol']!=after['symbol']


def test_warp_rule_generates_unseen_config_without_answer_injection():
    w=Workload(8,2048,512,1)
    donor=replace(w,n=512,output_stride=0)
    seed=Recipe.make('simt','fq',variant=1,columns=4,warps=8,values=2,split=1)
    rows=[observation(donor,seed)]
    screen=Prescreen(StrategyTable(rows))
    selected,report=screen.select(w,blind=True,stencil='warps')
    assert {dict(r.parameters)['warps'] for r in selected}=={2,4,8}
    assert report['seed_prototype_count']==1
    assert report['prototype_count']==3
    # An arbitrary rewrite of the target family does not change proposals,
    # origin labels, ranks, rejection reasons or candidate counts.
    hidden=observation(replace(w,m=8),simt(5),priority=999)
    assert (selected,report)==Prescreen(StrategyTable(rows+[hidden])).select(w,blind=True,stencil='warps')


def test_no_recursive_cartesian_growth_and_invalid_neighbors_are_rejected():
    w=Workload(13,1024,512,1)
    donor=replace(w,n=2048,output_stride=0)
    base=Recipe.make('tc','fq',**(dict(tc().parameters)|dict(qtype=13,tk=256)))
    rows=[observation(donor,base)]
    selected,report=Prescreen(StrategyTable(rows)).select(w,blind=True,stencil='local')
    assert all(not rejection(r,w) for r in selected)
    assert all(dict(r.parameters)['tk']==256 for r in selected)
    assert len({producer_key(r) for r in selected})==len(selected)
    assert report['seed_prototype_count']==1
    assert report['prototype_count']<=1+len(one_axis(base,'local'))
    q4=Recipe.make('q4','fq',reader=2,variant=7,warps=8,values=8,columns=4)
    assert one_axis(q4,'local')==()
    with pytest.raises(ValueError,match='blind factorized'):
        Prescreen(StrategyTable(rows)).select(w,stencil='local')
