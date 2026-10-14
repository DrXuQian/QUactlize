"""Rule proposals cannot inspect their target's historical champions."""
from dataclasses import replace

import pytest

from dev.strategy.prescreen import Prescreen, compile_key, recall, rejection
from dev.strategy.plan import component_champions
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload


def tc(tm=16, tn=64, split=1, stages=2):
    return Recipe.make('tc', 'fq', symbol=f'parent_tm{tm}_tn{tn}_st{stages}',
        qtype=12, tm=tm, tn=tn, tk=64, wm=min(tm,16), wn=16, stages=stages,
        ap=0, dn=16, persistent=-1, split=split, grid_mode=0, grid_b=0)


def simt(variant=0):
    return Recipe.make('simt', 'fq', variant=variant, warps=4, columns=4, values=4, split=1)


def observation(w, r, scope='COMPLETE_CALL', priority=50):
    return Observation(w, r, 'test', scope, priority, 'MEASURED', 'immutable-evidence')


def test_small_pool_is_not_filled_to_quota():
    w=Workload(12,1024,2048,1)
    rows=[observation(w,tc())]
    rows += [observation(replace(w,n=2048,output_stride=0),tc(tm,tn))
             for tm in (8,16,32,64,128) for tn in (32,64,128,256)]
    rows.append(observation(replace(w,n=2048,output_stride=0),simt()))
    selected, report=Prescreen(StrategyTable(rows)).select(w,limit=6,optional=0)
    assert len(selected)==2
    assert any(r.kind=='simt' for r in selected)
    assert report['historical_selected']==1 and report['coverage_selected']==1
    assert report['optional_selected']==0
    assert report['total_with_original']==3
    assert report['unselected_optional']


def test_former_winner_never_pruned_for_tile_waste_or_budget():
    w=Workload(12,1024,2048,1)
    winners=[tc(8),tc(16),tc(32),tc(64)]
    screen=Prescreen(StrategyTable(observation(w,r) for r in winners))
    selected, report=screen.select(w,limit=3)
    assert set(selected)==set(winners)
    assert report['exceeds_limit_to_preserve_winners']
    assert report['optional_selected']==0
    assert all(report['selected_reasons'][r.key] for r in winners)


def test_no_measured_status_transfer_from_precision_or_endpoint():
    w=Workload(12,1024,2048,1,compute='bf16')
    rows=[observation(replace(w,compute='f16'),tc()),
          observation(replace(w,storage='bf16'),simt())]
    selected, report=Prescreen(StrategyTable(rows)).select(w)
    assert len(selected)==2 and report['historical_selected']==0
    assert not report['measured_latency_reused']


def test_real_contract_pruning_retains_negative_receipt_for_required():
    w=Workload(12,1024,512,1)
    bad=tc(split=8,stages=4)
    selected, report=Prescreen(StrategyTable()).select(w,mandatory=[(bad,'HISTORICAL_WINNER')])
    assert not selected
    assert report['rejected'][0]['mandatory']
    assert report['rejected'][0]['reason']=='SHAPE_PRECISION_OR_PIPELINE_CONTRACT'
    assert rejection(tc(256,256),replace(w,m=128),max_threads=512)=='CTA_THREAD_LIMIT'
    assert rejection(Recipe.make('simt','fq',variant=2,warps=4,values=4,columns=4,split=1),
                     replace(w,qtype=8))=='Q8_SIMT_VARIANT'
    assert rejection(Recipe.make('full','full',dequant_config=4),w)


def test_old_candidate_labels_do_not_make_new_mandatory_winners():
    w=Workload(12,1024,2048,1)
    selected, report=Prescreen(StrategyTable([observation(w,tc(),scope='PROPOSAL')])).select(w,optional=0)
    assert len(selected)==1 and report['historical_selected']==0
    assert report['coverage_selected']==1 and not report['measured_latency_reused']


def test_shape_coverage_is_not_cut_to_three_or_four_candidates():
    w=Workload(12,1024,2048,1)
    donor=replace(w,n=2048,output_stride=0)
    recipes=[tc(split=s) for s in (1,2,4,8)]
    recipes += [Recipe.make('simt','fq',variant=0,warps=4,values=4,columns=4,split=s)
                for s in (1,2,4,8)]
    selected, report=Prescreen(StrategyTable(observation(donor,r) for r in recipes)).select(w,limit=4)
    assert set(selected)==set(recipes)
    assert report['total_with_original']==9
    assert report['coverage_selected']==8 and report['exceeds_limit_to_preserve_winners']
    assert not report['candidate_limit_is_hard'] and not report['search_complete']


def test_compile_inventory_deduplicates_runtime_axes_not_endpoints():
    w=Workload(12,1024,2048,1)
    assert compile_key(w,tc(split=2))==compile_key(replace(w,m=8,n=2048,output_stride=0),tc(split=8))
    assert compile_key(w,tc())!=compile_key(replace(w,m=128),tc())
    assert compile_key(w,tc())!=compile_key(replace(w,compute='bf16'),tc())
    assert compile_key(w,tc())!=compile_key(w,tc(stages=3))
    assert compile_key(w,simt()) is None
    assert compile_key(replace(w,m=128),Recipe.make('full','full',dequant_config=4)) is None


def test_prefill_keeps_route_champions_not_all_stable_losers():
    # Do not compare across different profiles. A route losing globally is
    # still retained because its dequant/reducer cost may change.
    candidates=[]
    for route in (0,1):
        for i,cost in enumerate((10.,20.,30.)):
            candidates.append(dict(route=route,stable=True,cost_us=cost,dequant_config=-1,
                tactic=dict(tc(tm=(16,32,64)[i]).parameters)))
    candidates.append(dict(route=2,stable=True,cost_us=40.,dequant_config=4))
    point=dict(q=12,n=1024,k=2048,experts=1,tokens=128,profile='real',candidates=candidates)
    rows=component_champions([point])
    recipes=list(rows[12,1024,2048,1,128,''].values())
    assert len(recipes)==3 and {r.route for r in recipes}=={'fq','sf','full'}
    assert all(r.kind=='full' or dict(r.parameters)['tm']==16 for r in recipes)


def test_blind_pool_cannot_read_any_m_or_precision_of_held_out_nk():
    w = Workload(12, 1024, 2048, 1)
    donor = replace(w, n=2048, output_stride=0)
    available = [observation(donor, tc()), observation(donor, simt())]
    # An adversarial replacement of the answer must not change proposals,
    # ranks, exclusions or counts -- not even through another M or dtype.
    answer = [observation(replace(w, m=m, compute=dtype), tc(8, 128, split=4), priority=999)
              for m in (1, 2, 8, 128) for dtype in ('f16', 'bf16')]
    clean = Prescreen(StrategyTable(available)).select(w, blind=True)
    polluted = Prescreen(StrategyTable(available + answer)).select(w, blind=True)
    assert clean == polluted
    assert not polluted[1]['champion_injection']
    assert polluted[1]['historical_selected'] == 0


def test_blind_pool_rejects_mandatory_champions_and_preserves_a_miss():
    w = Workload(12, 1024, 2048, 1)
    donor = replace(w, n=2048, output_stride=0)
    screen = Prescreen(StrategyTable([observation(donor, tc())]))
    with pytest.raises(ValueError, match='champion injection'):
        screen.select(w, mandatory=[(simt(), 'HISTORICAL_WINNER')], blind=True)
    selected, report = screen.select(w, blind=True)
    original = tuple(selected)
    verdict = recall(selected, [simt()])
    assert verdict['status'] == 'RULE_MISS'
    assert verdict['missing'] == [simt().key]
    assert tuple(selected) == original
    assert not report['measured_latency_reused']


def test_empty_blind_catalog_does_not_recover_held_out_champion():
    w = Workload(12, 1024, 2048, 1)
    selected, report = Prescreen(StrategyTable([observation(w, tc())])).select(w, blind=True)
    assert selected and all(r.kind == 'q4' for r in selected)  # API-only floor, not this TC answer
    assert report['nearby_observations'] == 0
    assert recall(selected, [tc()])['status'] == 'RULE_MISS'


def test_expanded_warps_do_not_exhaust_seed_slots():
    w = Workload(8, 1024, 2048, 1)
    donor = replace(w, n=2048, output_stride=0)
    near = [Recipe.make('simt', 'fq', variant=0, columns=4, values=4, warps=b, split=1)
            for b in (2, 4, 8)]
    farther = Recipe.make('simt', 'fq', variant=1, columns=8, values=4, warps=4, split=1)
    rows = [observation(donor, r) for r in near]
    rows.append(observation(replace(donor, n=4096, output_stride=0), farther))
    selected, report = Prescreen(StrategyTable(rows)).select(
        w, blind=True, stencil='warps', per_stratum=2, optional=0)
    assert report['seed_prototype_count'] == 2
    assert report['seed_aliases_covered_by_expansion'] == 2
    assert all(any(dict(r.parameters) == dict(a.parameters) for r in selected) for a in near)
    assert any(dict(r.parameters)['columns'] == 8 for r in selected)


def test_grid_variants_do_not_exhaust_seed_slots_or_drop_parent_bodies():
    w = Workload(12, 1024, 2048, 128)
    donor = replace(w, n=2048, output_stride=0)
    near = [Recipe.make('tc', 'sf', **(dict(tc().parameters) | dict(grid_mode=2, grid_b=b)))
            for b in (1, 2, 4)]
    farther = Recipe.make('tc', 'sf', **(dict(tc(tn=128).parameters) | dict(grid_mode=2, grid_b=1)))
    rows = [observation(donor, r) for r in near]
    rows.append(observation(replace(donor, n=4096, output_stride=0), farther))
    selected, report = Prescreen(StrategyTable(rows)).select(w, blind=True, per_stratum=2, optional=0)
    assert report['seed_prototype_count'] == 3  # two SF parents plus the generic full route
    assert {dict(r.parameters)['tn'] for r in selected if r.kind == 'tc'} == {64, 128}
    from dev.strategy.space import factored_recall
    assert factored_recall(w, selected, near + [farther])['status'] == 'PASS'


def test_rank_ties_are_not_decided_by_symbol_hash():
    w = Workload(12, 1024, 2048, 1)
    donor = replace(w, n=2048, output_stride=0)
    recipes = [tc(tn=tn) for tn in (32, 64, 128)]
    rows = [observation(donor, r) for r in recipes]
    selected, report = Prescreen(StrategyTable(rows)).select(w, blind=True, per_stratum=2, optional=0)
    assert {dict(r.parameters)['tn'] for r in selected if r.kind == 'tc'} == {32, 64, 128}
    renamed = [Recipe.make(r.kind, r.route, **(dict(r.parameters) | dict(symbol='renamed_'+str(i))))
               for i, r in enumerate(recipes)]
    after, _ = Prescreen(StrategyTable(observation(donor, r) for r in renamed)).select(
        w, blind=True, per_stratum=2, optional=0)
    from dev.strategy.space import scheduled_key
    assert {scheduled_key(r) for r in after} == {scheduled_key(r) for r in selected}
    assert report['stratum_boundary_ties'] == 'KEEP_EQUAL_RANK_NOT_HASH_ORDER'


def test_prefill_retains_narrow_and_throughput_tile_bands():
    w = Workload(12, 1024, 2048, 128)
    near = replace(w, n=2048, output_stride=0)
    far = replace(w, n=4096, output_stride=0)
    rows = [observation(near, tc(tm=16)), observation(near, tc(tm=32)),
            observation(far, tc(tm=64)), observation(far, tc(tm=128))]
    selected, _ = Prescreen(StrategyTable(rows)).select(w, blind=True, per_stratum=2, optional=0)
    assert {dict(r.parameters)['tm'] for r in selected if r.kind == 'tc'} == {16, 32, 64, 128}


def test_decode_retains_cta_count_and_wide_output_regimes():
    w=Workload(12,1024,2048,1)
    near=replace(w,n=2048,output_stride=0)
    far=replace(w,n=4096,output_stride=0)
    rows=[observation(near,tc(tn=32)),observation(near,tc(tn=64)),observation(far,tc(tn=128))]
    selected,_=Prescreen(StrategyTable(rows)).select(w,blind=True,per_stratum=2,optional=0)
    assert {dict(r.parameters)['tn'] for r in selected if r.kind=='tc'}=={32,64,128}


def test_cross_m_only_fills_missing_geometry_not_existing_stratum_or_route():
    w=Workload(12,1024,2048,1)
    near=replace(w,n=2048,output_stride=0)
    far=replace(near,m=128)
    rows=[observation(near,tc(tn=32)),observation(near,tc(tn=64)),
          observation(far,tc(tn=128)),observation(far,tc(tn=64,stages=4)),
          observation(far,Recipe.make('tc','sf',**dict(tc(tn=128).parameters)))]
    selected,report=Prescreen(StrategyTable(rows)).select(w,blind=True,optional=0,per_stratum=2)
    actual=[r for r in selected if r.kind=='tc']
    assert {dict(r.parameters)['tn'] for r in actual}=={32,64,128}
    assert all(r.route=='fq' and dict(r.parameters)['stages']==2 for r in actual)
    assert any('VACANT_STRATUM_CROSS_M' in reason for reasons in report['selected_reasons'].values() for reason in reasons)
