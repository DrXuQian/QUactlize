"""Geometry-only fallback must not leak held-out champions or measured costs."""
from dataclasses import replace

from dev.strategy.prescreen import Prescreen, semantic_config
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload


def tc(tm=8, dn=32, route='sf', ap=0):
    return Recipe.make('tc', route, qtype=8, symbol='donor', tm=tm, tn=64, tk=64,
        wm=tm, wn=16, stages=2, ap=ap, dn=dn, split=4, grid_mode=0,
        grid_b=0, persistent=-1, dequant_config=-1)


def row(w, r, us=10.):
    return Observation(w, r, 'donor', 'COMPONENT_SUM', 1, 'PASS', 'test', us)


def select(rows, w):
    return Prescreen(StrategyTable(rows)).select(w, blind=True, factorized=True, stencil='warps')


def target():
    return Workload(8, 1024, 2048, 8, 'indexed', 'bf16', experts=256, topk=8, router='repeat4')


def test_missing_indexed_tc_family_can_use_other_nk_dense_geometry():
    w = target()
    donor = Workload(8, 8192, 2048, 8, compute='bf16')
    recipes, report = select([row(donor, tc())], w)
    assert semantic_config(tc()) in {semantic_config(r) for r in recipes}
    assert {dict(r.parameters)['split'] for r in recipes} == {1, 2, 4, 8}
    assert any('MISSING_TC_FAMILY_CROSS_OPERATION' in reason
               for reasons in report['selected_reasons'].values() for reason in reasons)
    assert not report['champion_injection'] and not report['measured_latency_reused']


def test_held_out_nk_is_excluded_even_from_other_operator_and_precision():
    w = target()
    hidden = [row(Workload(8, w.n, w.k, m, compute=compute), tc(), us)
              for m in (1, 8) for compute in ('f16', 'bf16') for us in (1., 100.)]
    assert select(hidden, w)[0] == []


def test_portable_donor_never_displaces_existing_local_stratum():
    w = target()
    local = row(replace(w, n=512, output_stride=0), tc(dn=64))
    foreign = row(Workload(8, 8192, 2048, 8, compute='bf16'), tc(), .001)
    assert select([local], w)[0] == select([local, foreign], w)[0]
    # It must not expand every missing tile band, either: preserve the old
    # bounded inventory if any local TC family is already represented.
    local = row(replace(w, n=512, output_stride=0), tc(tm=16, dn=64))
    assert select([local], w)[0] == select([local, foreign], w)[0]


def test_no_timing_or_foreign_qtype_transfer_and_no_prefill_expansion():
    w = target()
    donor = Workload(8, 8192, 2048, 8, compute='bf16')
    assert select([row(donor, tc(), 1.)], w) == select([row(donor, tc(), 999.)], w)
    assert select([row(replace(donor, qtype=12), tc())], w)[0] == []
    assert select([row(donor, tc(route='fq')), row(donor, tc(ap=1))], w)[0] == []
    assert select([row(donor, tc())], replace(w, m=128))[0] == []


def test_router_and_shape_are_not_a_special_case_whitelist():
    donor = Workload(8, 8192, 2048, 8, compute='bf16')
    for n, k, m in ((512, 2048, 1), (3072, 512, 4), (4096, 4096, 8)):
        w = replace(target(), n=n, k=k, m=m, input_stride=0, output_stride=0, router='cluster')
        recipes, _ = select([row(donor, tc())], w)
        assert any(r.kind == 'tc' and dict(r.parameters)['tm'] == 8 for r in recipes)
