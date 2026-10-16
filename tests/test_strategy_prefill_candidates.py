"""Candidate coverage tests, not performance admission or fabricated timings."""
from dataclasses import replace

from dev.strategy.prefill_candidates import regime, supplement
from dev.strategy.prescreen import Prescreen
from dev.strategy.space import scheduled_key
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload


def tc(q=12, route='sf', tn=32, tk=128, stages=2, ap=0, dn=32):
    return Recipe.make('tc', route, symbol=f'p_q{q}_{route}_n{tn}_k{tk}_s{stages}_dn{dn}',
                       qtype=q, tm=16, tn=tn, tk=tk, wm=16, wn=min(32, tn),
                       stages=stages, ap=ap, dn=dn, persistent=-1, split=1, grid_mode=0, grid_b=0)


def observation(w, recipe, time=1.):
    return Observation(w, recipe, 'host-fixture', 'PROPOSAL', 1, 'MEASURED', 'test', time)


def target():
    return Workload(12, 1024, 2048, 512)


def test_all_missing_observed_regimes_not_cartesian_or_only_one_gap():
    w = target(); near = replace(w, n=2048, output_stride=0)
    far = replace(near, m=32)
    base = tc()
    geometries = [tc(tn=16, dn=16), tc(tn=128), tc(tk=256), tc(stages=4)]
    rows = [observation(far, r) for r in geometries]
    out, _ = supplement(w, [base], rows)
    assert {scheduled_key(r) for r in out} == {scheduled_key(r) for r in geometries}
    assert not any(dict(r.parameters)['tn'] == 128 and dict(r.parameters)['stages'] == 4 for r in out)


def test_entire_target_nk_excluded_across_m_operation_and_format():
    w = target()
    rows = [observation(replace(w, m=32), tc(tn=128)),
            observation(replace(w, qtype=10, m=128, operation='indexed', experts=256, topk=8), tc(10, tn=16))]
    assert supplement(w, [tc()], rows) == ([], {})


def test_only_sf_ap0_templates_can_cross_format_or_operation():
    w = target(); donor = replace(w, qtype=10, n=2048, output_stride=0,
                                  operation='indexed', experts=256, topk=8, compute='bf16')
    rows = [observation(donor, tc(10, tn=128)),
            observation(donor, tc(10, 'fq', tn=16)),
            observation(donor, tc(10, ap=1)),
            observation(replace(donor, qtype=8), tc(8, stages=3))]
    out, reasons = supplement(w, [], rows)
    assert len(out) == 1 and out[0].route == 'sf'
    assert dict(out[0].parameters)['qtype'] == 12
    assert 'REINSTANTIATE_TARGET_KQUANT_NO_TIMING_TRANSFER' in reasons[out[0].key]


def test_nearest_m_seeds_are_never_replaced():
    w = target(); donor = replace(w, n=2048, output_stride=0)
    base = tc(); faster_same_regime = Recipe.make('tc', 'sf', **(dict(base.parameters) | {'wm':8}))
    rows = [observation(donor, faster_same_regime, .001)]
    assert regime(base) == regime(faster_same_regime)
    assert supplement(w, [base], rows) == ([], {})


def test_equal_distance_aliases_do_not_hash_prune_distinct_geometries():
    w = target(); donor = replace(w, n=2048, output_stride=0)
    a = tc(tn=128)
    b = Recipe.make('tc', 'sf', **(dict(a.parameters) | {'wm':8}))
    rows = [observation(donor, a, 1), observation(donor, b, 100)]
    out, _ = supplement(w, [], rows)
    assert {scheduled_key(r) for r in out} == {scheduled_key(a), scheduled_key(b)}
    changed = [replace(row, us=1000-row.us) for row in rows]
    assert supplement(w, [], changed)[0] == out


def test_breadth_is_additive_and_does_not_change_decode():
    for m in (1, 512):
        w = replace(target(), m=m)
        donor = replace(w, n=2048, output_stride=0)
        rows = [observation(donor, tc()), observation(replace(donor, m=32), tc(tn=128))]
        s = Prescreen(StrategyTable(rows))
        before, _ = s.select(w, blind=True, optional=0, prefill_breadth=False)
        after, report = s.select(w, blind=True, optional=0)
        assert set(before) <= set(after)
        if m <= 8:
            assert after == before and report['rule_revision'] == 25
        else:
            assert report['rule_revision'] == 26


def test_supplement_only_uses_prefill_and_preserves_target_legality():
    w = replace(target(), k=512, input_stride=0)
    donor = replace(w, n=2048, output_stride=0)
    rows = [observation(replace(donor, m=1), tc(tn=128)),
            observation(donor, tc(tk=256, stages=12))]
    assert supplement(w, [], rows) == ([], {})
