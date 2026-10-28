"""Host gates for the Q5 completion cohort; no device timings are fabricated."""
import ctypes as C
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy import moe_single_row_plan as single
from dev.strategy.refresh import original_recipe, request
from dev.strategy.smallm_asys import check_producers
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def planned(tmp_path_factory):
    output = tmp_path_factory.mktemp('moe-single-plan')
    original = plan_smallm(output, [request(w) for w in single.workloads()])['requests']
    return single.make_plan(original), original


def test_all_m_and_router_profiles_with_both_incumbents(planned):
    plan, original = planned
    assert plan['counts'] == dict(points=36, arms=442, tc_modules=0, vector_modules=5, families=1)
    assert plan['timing_method'] == 'ASYS_GPU_ACTIVITY'
    assert not plan['historical_times_reused'] and not plan['production_changed']
    assert 'SIMT_VS_TC_CONFIG_SWEEP_AFTER_SIMT_OPTIMIZATION' in plan['remaining']
    for m in range(1, 9):
        points = [p for p in plan['points'] if p['workload']['m'] == m]
        assert {p['workload']['router'] for p in points} == ({'spread'} if m == 1 else set(single.ROUTERS))
        # Router labels never change the candidate pool or shipping choice.
        assert len({tuple(a['id'] for a in p['arms']) for p in points}) == 1
    for p, old in zip(plan['points'], original):
        a = p['arms'][0]
        assert a['id'] == 'original' and Recipe.from_dict(a['recipe']) == original_recipe(old)
        assert p['indexed_single_row'] and p['execution_contract']['mode'] == 2
        assert p['execution_contract']['rows'] == p['workload']['m'] * 8
        for geometry in ((4, 2, 8), (4, 8, 8), (8, 4, 4)):
            configs = [dict(a['recipe']['parameters']) for a in p['arms']
                       if a['recipe']['kind'] == 'simt' and dict(a['recipe']['parameters'])['variant'] == 3]
            assert {c['split'] for c in configs if tuple(c[k] for k in ('columns', 'warps', 'values')) == geometry} == {1, 2, 4, 8}
        assert all('moe_pair_rows' not in dict(a['recipe']['parameters']) for a in p['arms'])
        if p['workload']['m'] == 8:
            prior, = [a for a in p['arms'] if 'PRE_ADMISSION_CONTROL_REMEASURED_NOT_OLD_LATENCY' in a['reasons'] and
                      dict(a['recipe']['parameters'])['variant']==2]
            assert dict(prior['recipe']['parameters'])['variant'] == 2
            assert prior['id'] != 'original'


def test_prior_times_and_production_latency_labels_do_not_rank_candidates(planned, monkeypatch):
    plan, original = planned
    rows = single.load()
    for r in rows['rows']:
        for p in r['profiles']:
            for c in p['candidates'].values():c['mean_us'] = 1e30
            p['candidates'][p['original']]['mean_us'] = 1e-10
    monkeypatch.setattr(single, 'load', lambda: rows)
    changed = single.make_plan([dict(o, median_us=1e-12) for o in original])
    assert [p['arms'] for p in plan['points']] == [p['arms'] for p in changed['points']]


def test_expected_entries_match_actual_cpp_and_split_workspace(planned, tmp_path):
    from quactlize.execution.native import Call, SimtCallV2, SimtConfig, Sizes, Arrangement, arrangement
    cpp, lib = tmp_path/'entry.cpp', tmp_path/'entry.so'
    cpp.write_text('''#include "quactlize/execution/simt_strategy.hpp"
#include "quactlize/execution/simt_validation.hpp"
extern "C" char const* entry(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f) {
    return quactlize::execution::simt::implementation(*d,*f).producer;
}
extern "C" int query(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f,
 quactlize_ppu_placed_arrangement_v2 const* a,qkg_sizes_v1* out) {
    return quactlize::execution::simt::query_v2(*d,*f,a,*out);
}
''')
    subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-I'+str(ROOT), cpp, '-o', lib], check=True)
    module = C.CDLL(str(lib));module.entry.argtypes = [C.POINTER(SimtCallV2), C.POINTER(SimtConfig)]
    module.entry.restype = C.c_char_p
    module.query.argtypes = [C.POINTER(SimtCallV2), C.POINTER(SimtConfig), C.POINTER(Arrangement), C.POINTER(Sizes)]
    module.query.restype = C.c_int
    names = {'common-packet-3d': 'common_reader', 'register-reuse-fixed': 'register_reuse_model',
             'register-reuse': 'register_reuse'}
    arr = arrangement(13)
    for p in planned[0]['points']:
        m = p['workload']['m']
        d = SimtCallV2(Call(version=1, size=C.sizeof(Call), qtype=13, mode=2, n=2048, k=512,
            experts=256, topk=8, channels=8, rows=m*8, input_type=1, a_row_stride=512,
            a_token_stride=4096, ids_stride=8, out_row_stride=2048), 1)
        for arm in p['arms']:
            cfg = dict(arm['recipe']['parameters'])
            f = SimtConfig(*[cfg[k] for k in ('variant', 'columns', 'warps', 'values', 'split')])
            name = module.entry(C.byref(d), C.byref(f)).decode()
            assert arm['expected_producer'] == 'quactlize::execution::simt::' + names[name]
            size = Sizes()
            assert module.query(C.byref(d), C.byref(f), C.byref(arr), C.byref(size)) == 0
            assert size.workspace_bytes == (0 if cfg['split'] == 1 else m*8*2048*cfg['split']*4)


def test_asys_checks_single_row_producer_grid_and_precision(planned):
    p = planned[0]['points'][0]
    components, arm_components, arms = {}, {}, {}
    for a in p['arms']:
        f = dict(a['recipe']['parameters'])
        node = dict(name=a['expected_producer'], symbol=a['expected_producer'],
            grid=[2048//(f['columns']*f['values']), 8, f['split']], block=[f['warps']*32, 1, 1])
        if not a['expected_producer'].endswith('::common_reader'):
            node['grid']=[node['grid'][0]*node['grid'][1]*node['grid'][2],1,1]
        components[a['id']] = dict(kind='producer', identity=dict(node=node))
        arm_components[a['id']] = [a['id']]
        arms[a['id']] = dict(receipt=dict(execution_contract=p['execution_contract']))
    data = dict(arms=arms, component_records=components, arm_components=arm_components)
    check_producers(p, data)
    for field in ('entry', 'grid', 'block', 'compute'):
        bad = deepcopy(data)
        node = bad['component_records']['original']['identity']['node']
        if field == 'entry':node.update(name='wrong', symbol='wrong')
        elif field == 'grid':node['grid'][1] += 1
        elif field == 'block':node['block'][0] = 32
        else:bad['arms']['original']['receipt']['execution_contract']['compute_type'] = 0
        with pytest.raises(ValueError):check_producers(p, bad)


def test_m1_exception_and_unverified_extensions_are_explicit():
    w = single.workloads()[0]
    r = Recipe.make('simt', 'fq', variant=3, columns=4, warps=2, values=8, split=1)
    assert single.single_row_entry(w, r).endswith('register_reuse_model')
    assert single.single_row_entry(replace(w, m=2), r).endswith('common_reader')
    for other in (replace(w, n=3072, output_stride=0), replace(w, storage='bf16'), replace(w, compute='f16')):
        with pytest.raises(ValueError):single.single_row_entry(other, r)


def test_plan_cli_has_no_gpu_work_and_replays_same_inventory(tmp_path):
    out = tmp_path/'plan'
    command = [sys.executable, str(ROOT/'tools/run_smallm_regressions_asys.py'),
               '--cohort', single.COHORT, '--output', str(out), '--plan-only']
    for extra in ([], ['--resume']):
        r = subprocess.run(command+extra, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout+r.stderr
        assert '"points": 36' in r.stdout
    assert not (out/'run.json').exists()
    assert json.loads((out/'plan.json').read_text())['cohort'] == single.COHORT
