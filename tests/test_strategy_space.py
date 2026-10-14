"""Factored candidate coverage is not a timing or new-device admission."""
from dataclasses import replace
import subprocess

import pytest

from dev.strategy.prescreen import Prescreen, rejection, semantic_config
from dev.strategy.space import (expansion_configs, factored_recall, factor_candidates,
                                helpers, producer_key, prototype, split_variants, rebind_template,
                                grid_variants, resolved_grid, tile_work)
from quactlize.dispatch.table import Recipe, StrategyTable, Workload
from test_strategy_prescreen import tc, observation
from test_kpack_native_dispatch import recipe_probe


def with_values(r, **fields):
    return Recipe.make(r.kind, r.route, **(dict(r.parameters) | fields))


def test_helper_space_is_api_defined_and_not_small_m_full_dequant():
    w = Workload(12, 1024, 2048, 128)
    assert expansion_configs(w, 'fq') == ()
    assert expansion_configs(replace(w, m=8), 'full') == ()
    assert expansion_configs(replace(w, qtype=8), 'sf') == ()
    assert expansion_configs(w, 'sf') == (-1, 0, 1, 2, 3, 4, 5)
    assert expansion_configs(w, 'full') == (0, 1, 2, 3, 4, 5, 10, 11, 12)
    grouped = Workload(12, 1024, 2048, 128, 'indexed', 'bf16', experts=256, topk=8)
    assert expansion_configs(grouped, 'sf') == (-1, 0, 4, 5)
    assert expansion_configs(replace(grouped, qtype=14), 'sf') == (-1, 0)
    assert expansion_configs(replace(w, qtype=14), 'full') == tuple(range(6))


def test_recall_requires_both_producer_and_expansion_not_champion_injection():
    w = Workload(12, 1024, 2048, 128)
    r = tc()
    r = Recipe.make('tc', 'sf', **dict(r.parameters))
    canonical = prototype(r)
    control = with_values(r, dequant_config=5)
    assert semantic_config(canonical) != semantic_config(control)
    assert factored_recall(w, [canonical], [control])['status'] == 'PASS'
    assert factored_recall(w, [canonical], [with_values(control, split=2)])['status'] == 'RULE_MISS'
    # This axis is explicitly generated from the API namespace, not ignored.
    assert factored_recall(w, [canonical], [with_values(control, grid_mode=2, grid_b=4)])['status'] == 'PASS'
    assert factored_recall(w, [canonical], [with_values(control, grid_mode=2, grid_b=33)])['status'] == 'RULE_MISS'
    assert factored_recall(w, [canonical], [with_values(control, grid_mode=2, grid_b=4, split=2)])['status'] == 'RULE_MISS'
    assert factored_recall(w, [canonical], [with_values(control, dequant_config=12)])['status'] == 'RULE_MISS'
    assert len(factor_candidates([canonical, control])) == 1
    assert helpers(w, [canonical]) == helpers(w, [control])


def test_split_is_runtime_but_not_free_or_a_helper_only_axis():
    w = Workload(12, 1024, 512, 1)
    r = tc()
    variants = split_variants(w, prototype(r))
    assert [dict(v.parameters)['split'] for v in variants] == [1, 2, 4, 8]
    assert len({producer_key(v) for v in variants}) == 4
    shallow = with_values(r, tk=256, stages=3)
    assert [dict(v.parameters)['split'] for v in split_variants(w, shallow)
            if not rejection(v, w)] == [1]
    persistent = Recipe.make('tc', 'sf', **(dict(r.parameters) | dict(grid_mode=2, grid_b=4)))
    variants = split_variants(w, persistent)
    assert [dict(v.parameters)['split'] for v in variants] == [1, 2, 4, 8]
    assert all(dict(v.parameters)['grid_mode'] == 0 for v in variants)
    assert rejection(with_values(persistent, split=2), w) == 'DENSE_PERSISTENT_REQUIRES_SF_S1'


def test_cross_channel_transfer_is_proposal_not_same_nk_leakage():
    w = Workload(14, 1024, 2048, 2, 'indexed', 'bf16', experts=256, topk=8)
    donor = replace(w, n=2048, output_stride=0, channels=8)
    r = Recipe.make('simt', 'fq', variant=3, columns=4, warps=4, values=4, split=1)
    rows = [observation(donor, r)]
    selected, report = Prescreen(StrategyTable(rows)).select(w, blind=True, factorized=True)
    assert selected and not report['measured_latency_reused']
    leaked = observation(replace(w, channels=8, m=8), with_values(r, warps=8))
    assert (selected, report) == Prescreen(StrategyTable(rows + [leaked])).select(w, blind=True, factorized=True)


def test_aliases_and_helper_configs_cannot_consume_geometry_slots():
    w=Workload(12,1024,2048,1)
    donor=replace(w,n=2048,output_stride=0)
    a,b=tc(),tc(tn=128)
    rows=[observation(donor,a),observation(donor,b)]
    rows += [observation(donor,with_values(a,symbol=f'alias_{i}')) for i in range(20)]
    selected,_=Prescreen(StrategyTable(rows)).select(w,blind=True,per_stratum=2)
    bases={producer_key(prototype(r)) for r in selected if r.kind=='tc'}
    assert bases=={producer_key(prototype(a)),producer_key(prototype(b))}


def test_cross_format_proposal_uses_target_transport_and_withholds_all_qtypes():
    w=Workload(13,1024,2048,1)
    donor=Workload(12,2048,2048,1)
    illegal=tc()  # TK64 cannot contain a whole Q5 K-pack transport tile.
    legal=with_values(illegal,tk=256,symbol='generic_tk256')
    target=rebind_template(legal,13)
    assert dict(target.parameters)['qtype']==13 and not rejection(target,w)
    assert rejection(rebind_template(illegal,13),w).startswith('PARENT_CONTRACT')
    rows=[observation(donor,legal)]
    screen=Prescreen(StrategyTable(rows))
    selected,report=screen.select(w,blind=True,cross_format=True)
    assert selected and not report['measured_latency_reused']
    leaked=observation(Workload(12,w.n,w.k,8),with_values(legal,tn=128))
    assert (selected,report)==Prescreen(StrategyTable(rows+[leaked])).select(w,blind=True,cross_format=True)


def test_cross_format_transfer_is_not_the_default():
    w=Workload(13,1024,2048,1)
    donor=Workload(12,2048,2048,1)
    r=with_values(tc(),tk=256)
    selected,report=Prescreen(StrategyTable([observation(donor,r)])).select(w,blind=True)
    assert not selected and not report['cross_format_templates']


@pytest.mark.parametrize('compute_units', [64,72,96])
@pytest.mark.parametrize('operation', ['dense','indexed'])
def test_resolved_grid_matches_cpp_and_sparse_directory(recipe_probe, compute_units, operation):
    w=(Workload(12,3072,2048,9) if operation=='dense' else
       Workload(12,3072,2048,9,'indexed','bf16',experts=256,topk=8))
    for tm in (8,16,64):
        for split in ((1,) if operation=='dense' else (1,2,4,8)):
            for mode in (0,2,3):
                for b,occupancy in ((0,1),(1,12),(4,3),(32,6)):
                    r=Recipe.make('tc','sf',**(dict(tc(tm=tm).parameters) |
                        dict(split=split,grid_mode=mode,grid_b=b)))
                    rows=w.m*w.topk
                    fields=(1 if operation=='dense' else 3,rows,w.n,w.experts,w.m,
                            tm,64,split,mode,b,occupancy,compute_units)
                    line=subprocess.check_output([str(recipe_probe),*map(str,fields)],text=True)
                    expected={k:int(v) for k,v in (item.split('=') for item in line.split())}
                    assert resolved_grid(w,r,compute_units,occupancy)==expected


def test_grid_search_deduplicates_equal_launches_not_schedules():
    w=Workload(12,256,2048,1)
    r=Recipe.make('tc','sf',**dict(tc().parameters))
    result=grid_variants(w,r,compute_units=64,occupancy=12)
    assert len(result)==2  # one ordinary body, one persistent body
    assert {x['launch']['algorithm'] for x in result}=={0,1}
    persistent=next(x for x in result if x['launch']['algorithm']==1)
    assert persistent['launch']['grid']==4
    assert len(persistent['aliases'])==64  # B1..32 x capacity/balanced, timed ONCE
    larger=replace(w,m=4096,n=4096,output_stride=0)
    result=grid_variants(larger,r,compute_units=64,occupancy=3)
    assert len(result)<=7  # not 64 arms, nor 32*64 hardcoded capacity
    assert max(x['launch']['grid'] for x in result)<=64*3
    assert len({(x['launch']['algorithm'],x['launch']['split'],x['launch']['grid']) for x in result})==len(result)


@pytest.mark.parametrize('cu,occ',[(0,1),(64,0),(-1,2),(72,-1),(True,2),(64,1.5)])
def test_grid_space_never_guesses_missing_device_resources(cu,occ):
    w=Workload(12,1024,2048,128)
    r=Recipe.make('tc','sf',**dict(tc().parameters))
    with pytest.raises(ValueError,match='probed compute_units'):
        grid_variants(w,r,compute_units=cu,occupancy=occ)


def test_grid_cannot_silently_use_indexed_tokens_as_ragged_grouped_rows():
    w=Workload(12,1024,2048,128,'grouped',experts=256)
    with pytest.raises(ValueError,match='dense or indexed'):
        tile_work(w,tc())


def test_sf_scheduler_is_runtime_but_fq_grouped_schedule_is_a_different_parent():
    w=Workload(12,1024,2048,128)
    sf=Recipe.make('tc','sf',**(dict(tc().parameters) | dict(grid_mode=2,grid_b=4)))
    variants=split_variants(w,prototype(sf))
    control=with_values(sf,grid_mode=0,grid_b=0,split=4)
    assert factored_recall(w,variants,[control,sf])['status']=='PASS'
    # Persistent S4 is still not a dense execution entry.
    assert factored_recall(w,variants,[with_values(sf,split=4)])['status']=='RULE_MISS'
    group=Workload(12,1024,2048,1,'indexed','bf16',experts=256,topk=8)
    fq=with_values(tc(),persistent=1,grid_mode=2,grid_b=4)
    rows=grid_variants(group,fq,compute_units=64,occupancy=3)
    assert all(row['launch']['algorithm']==1 for row in rows)
    with pytest.raises(ValueError,match='compiled FQ grouped parent'):
        resolved_grid(group,with_values(fq,grid_mode=0),64,3)
