"""MoE rule recall is not winner injection, latency reuse or admission."""
from dataclasses import asdict, replace
import json
import subprocess

import pytest

from dev.strategy.moe_plan import (
    MS, RULES, fresh_measurement_contract, make_plan, proposal_only, source_stamp,
)
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload
from quactlize.runtime.tuning import digest


def workload(**kwargs):
    return replace(Workload(13, 1024, 2048, 4, 'indexed', 'bf16',
                            experts=256, topk=8, router='real'), **kwargs)


def simt(**kwargs):
    return Recipe.make('simt', 'fq', **(dict(variant=3, columns=4, warps=4, values=4, split=1) | kwargs))


def tc(**kwargs):
    return Recipe.make('tc', 'fq', **(dict(symbol='host_test_parent', qtype=13,
        tm=16, tn=64, tk=128, wm=16, wn=16, stages=2, ap=0, dn=16,
        persistent=0, split=1, grid_mode=0, grid_b=0, dequant_config=-1) | kwargs))


def observation(w, r, **kwargs):
    return Observation(w, r, 'fixture', 'COMPLETE_CALL', kwargs.get('priority', 99),
                       'OLD_WINNER', 'fixture-hash', kwargs.get('us', 100.),
                       dict(old_samples=[1, 2, 3], winner=True))


def reference(w, r):
    value = dict(workload=asdict(w), recipe=asdict(r), source='fixture', role='HISTORICAL_ROUTER_WINNER')
    return dict(value, id=digest(value))


def test_times_labels_priorities_are_removed_before_candidate_generation():
    w = workload()
    first = observation(w, simt())
    second = replace(first, us=.001, priority=1000, status='OTHER', details={'winner': False})
    a, b = proposal_only([first]), proposal_only([second])
    assert a.as_dict() == b.as_dict()
    row = a.observations[0]
    assert row.us is None and row.details == {} and row.priority == 0 and row.scope == 'PROPOSAL'


def test_target_family_poison_cannot_influence_rule_candidates():
    w = workload()
    donor = replace(w, n=2048, output_stride=0)
    available = [observation(donor, simt())]
    poisons = [observation(replace(w, m=m, compute=compute, router=router), tc(tn=128))
               for m in (1, 8, 128) for compute in ('f16', 'bf16') for router in ('real', 'cluster')]
    refs = [reference(w, simt())]
    before = make_plan(StrategyTable(available), refs)
    after = make_plan(StrategyTable(available + poisons), refs)
    assert before['points'] == after['points']
    assert before['status'] == 'PASS'


def test_revealing_missing_champion_fails_without_adding_it():
    w = workload()
    table = StrategyTable([observation(replace(w, n=2048, output_stride=0), simt())])
    first = make_plan(table, [reference(w, simt())])
    missed = make_plan(table, [reference(w, simt()), reference(w, tc())])
    assert first['points'][0]['candidates'] == missed['points'][0]['candidates']
    assert missed['status'] == 'RULE_GAPS'
    assert tc().key not in missed['points'][0]['candidates']
    assert missed['points'][0]['recall']['reasons'][tc().key] == 'PRODUCER_RECIPE_NOT_GENERATED'
    assert not missed['sweep_ready'] and not missed['production_changed']


def test_router_profiles_are_audit_axes_not_host_id_readback():
    w = workload()
    table = StrategyTable([observation(replace(w, n=2048, output_stride=0), simt())])
    refs = [reference(replace(w, router=r), simt()) for r in ('real', 'spread', 'cluster')]
    plan = make_plan(table, refs)
    assert plan['counts']['points'] == 3 and plan['status'] == 'PASS'
    assert len({tuple(p['candidates']) for p in plan['points']}) == 1
    assert 'SYNTHETIC' in plan['future_inventory']['old_real_router']


def test_split_axes_are_generated_not_removed_from_recall():
    w = workload()
    table = StrategyTable([observation(replace(w, n=2048, output_stride=0), simt())])
    plan = make_plan(table, [reference(w, simt(split=s)) for s in (1, 2, 4, 8)])
    assert plan['status'] == 'PASS'
    recipes = [Recipe.from_dict(plan['configurations'][k]) for k in plan['points'][0]['candidates']]
    assert {dict(r.parameters)['split'] for r in recipes} == {1, 2, 4, 8}


def test_indexed_reader_and_column_peers_are_rules_not_shape_exceptions():
    for n,k in ((3072,512),(4096,1024),(6144,2048)):
        w=workload(n=n,k=k,input_stride=0,output_stride=0)
        donor=replace(w,n=8192,output_stride=0)
        table=StrategyTable([observation(donor,simt(warps=2))])
        refs=[reference(w,simt(warps=2,columns=8)), reference(w,simt(warps=2,variant=2))]
        assert make_plan(table,refs)['status']=='PASS'


def test_impl_identity_is_not_recalled_by_geometry_alone():
    w = workload()
    body = simt(implementation_revision='different-body')
    table = StrategyTable([observation(replace(w, n=2048, output_stride=0), simt())])
    plan = make_plan(table, [reference(w, body)])
    assert plan['status'] == 'RULE_GAPS'
    assert body.key in plan['points'][0]['recall']['missing']


def test_no_implicit_dense_or_fp16_moe_contract():
    with pytest.raises(ValueError, match='indexed F32/BF16'):
        make_plan(StrategyTable(), [reference(workload(compute='f16'), simt())])
    with pytest.raises(ValueError, match='blind'):
        make_plan(StrategyTable(), [], rules=RULES | {'blind': False})


def test_full_m_inventory_and_fresh_asys_contract():
    plan = make_plan(StrategyTable(), [reference(workload(), simt())])
    assert MS == (*range(1, 9), 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
    assert plan['future_inventory']['shape_token_points'] == 18
    assert not plan['future_inventory']['complete_model_coverage']
    contract = fresh_measurement_contract()
    assert contract['timing_method'] == 'ASYS_GPU_ACTIVITY'
    assert not contract['historical_timing_reuse']
    assert {'quactlize_commit', 'source_sha256', 'recipe_and_body', 'asys_trace_and_range'} <= set(contract['tags'])
    assert plan['old_latency_rows_imported'] == plan['kernel_launches'] == 0
    assert not plan['performance_admitted']


def test_source_tag_records_dirty_content_without_requiring_commit(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'quactlize').mkdir()
    source = tmp_path / 'quactlize' / 'kernel.cu'
    source.write_text('original\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    before = source_stamp(tmp_path)
    source.write_text('modified implementation\n')
    after = source_stamp(tmp_path)
    assert before['quactlize_commit'] == after['quactlize_commit']
    assert before['source_sha256'] != after['source_sha256'] and before['tag'] != after['tag']
    assert not before['dirty'] and after['dirty']
