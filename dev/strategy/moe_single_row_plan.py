"""Q5 single-row M1..8 completion, not the subsequent SIMT-vs-TC sweep.

Run the actual shipping reader dispatch, including its fixed M1 exception.
Router labels are fixtures, never runtime selection keys. Keep both the live
production choice and the pre-admission control, but reuse no old timings.
"""
from dataclasses import asdict
import json
from pathlib import Path

from dev.gemv_simt.general_plan import configs
from dev.strategy.moe_token_pair_plan import production_entry
from dev.strategy.refresh import original_recipe, request
from dev.strategy.resources import module_spec
from dev.strategy.vector import execution_contract, spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.execution.moe_decode_policy import load
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
COHORT = 'moe-single-row-q5'
ROUTERS = ('spread', 'cluster', 'repeat2', 'repeat4', 'real')


def workloads():
    return [Workload(13, 2048, 512, m, 'indexed', 'bf16', experts=256,
                     topk=8, channels=8, router=router)
            for m in range(1, 9) for router in (('spread',) if m == 1 else ROUTERS)]


def single_row_entry(w, r):
    """Expected symbols for this bounded public-body cohort, host-C++ tested.

    Do not use the pair experiment's common-only launcher: the public M1
    C4/W2/P8/S1 still preserves register_reuse_model. New shapes must have
    their entry contract verified before broadening this diagnostic.
    """
    if (w.qtype, w.n, w.k, w.experts, w.topk, w.channels, w.operation, w.compute, w.storage) != (
            13, 2048, 512, 256, 8, 8, 'indexed', 'bf16', 'f32') or not 1 <= w.m <= 8:
        raise ValueError('single-row symbol contract is only verified for the Q5 completion cohort')
    p = dict(r.parameters)
    if r.kind != 'simt' or r.route != 'fq' or set(p) != {'variant', 'columns', 'warps', 'values', 'split'}:
        raise ValueError('single-row completion requires public SIMT configs')
    geometry = p['variant'], p['columns'], p['warps'], p['values']
    if p['split'] not in (1, 2, 4, 8):
        raise ValueError('invalid single-row Split-K')
    if geometry in ((3, 4, 2, 8), (3, 4, 8, 8), (3, 8, 4, 4)):
        name = ('register_reuse_model' if w.m == 1 and p['split'] == 1 and
                geometry == (3, 4, 2, 8) else 'common_reader')
    elif geometry in ((2, 4, 2, 8), (3, 8, 2, 4)):
        name = 'register_reuse'  # retained old/current controls, not new kernels
    else:
        raise ValueError('unverified single-row producer geometry')
    return 'quactlize::execution::simt::' + name


def make_plan(originals):
    ws = workloads()
    if [row['request'] for row in originals] != [request(w) for w in ws]:
        raise ValueError('single-row production control inventory differs')
    # These are immutable prior recipes, not ranking data or reused samples.
    previous = {tuple(row['key']): row['original_config'] for row in load()['rows']}
    points, tc, vectors = [], {}, {}
    for w, old in zip(ws, originals):
        if old['status']:
            raise ValueError('current indexed production control unavailable')
        arms = {}

        def add(r, reason, entry, *, original=False):
            item = module_spec(w, r) if r.kind == 'tc' else spec(w, r)
            mid = digest(item)
            (tc if r.kind == 'tc' else vectors)[mid] = item
            if r.key not in arms:
                arms[r.key] = dict(id='original' if original else r.key,
                    recipe=asdict(r), roles=['original'] if original else ['rule'],
                    aliases=[], required=True, reasons=[], spec=item, module=mid,
                    expected_producer=entry)
            arm = arms[r.key]
            if arm['expected_producer'] != entry:
                raise ValueError('public reader expectation differs from production selector')
            if reason not in arm['reasons']:
                arm['reasons'].append(reason)

        r = original_recipe(old)
        add(r, 'CURRENT_CPP_SELECTED_RECIPE', production_entry(old), original=True)
        if tuple(request(w)) in previous:
            cfg = dict(previous[tuple(request(w))])
            r = Recipe.make(cfg.pop('kind'), 'fq', **cfg)
            add(r, 'PRE_ADMISSION_CONTROL_REMEASURED_NOT_OLD_LATENCY', single_row_entry(w, r))
        for cfg in configs(13):
            for split in (1, 2, 4, 8):
                r = Recipe.make('simt', 'fq', **(cfg | {'split': split}))
                add(r, 'THREE_COMMON_GEOMETRIES_WITH_LEGAL_SPLIT_NEIGHBORS', single_row_entry(w, r))
        arms = list(arms.values())
        if any(a['recipe']['route'] != 'fq' for a in arms):
            raise ValueError('single-row completion has no SF helper sweep; declare it separately')
        points.append(dict(id=w.key, workload=asdict(w), cohort='decode', selection=old,
            execution_contract=execution_contract(w), arms=arms, sf_configs=[],
            indexed_single_row=True, excluded=[], consumers=[]))
    result = dict(schema='quactlize.smallm-asys.v1', phase='regression-asys', cohort=COHORT,
        points=points, tc=tc, vectors=vectors,
        counts=dict(points=len(points), arms=sum(len(p['arms']) for p in points),
                    tc_modules=len(tc), vector_modules=len(vectors), families=1),
        timing_method='ASYS_GPU_ACTIVITY', selection_metric='MEAN_COMPONENT_SUM',
        noise_policy='DIAGNOSTIC_NOT_VETO', historical_times_reused=False,
        production_changed=False, native_comparison=False,
        scope='Q5_SINGLE_ROW_M1_TO_8_COMPLETION_NOT_TC_SELECTION_OR_MODEL_ADMISSION',
        coverage=dict(tokens=list(range(1, 9)), m1_profiles=['spread'], multi_token_profiles=list(ROUTERS),
            real_profile='SYNTHETIC_HOT_EXPERT_SAMPLING_NOT_MODEL_TRACE',
            complete_model_coverage=False, weight_shape=[13, 2048, 512, 256, 8, 8]),
        remaining=['SIMT_VS_TC_CONFIG_SWEEP_AFTER_SIMT_OPTIMIZATION', 'OTHER_MODEL_LOCAL_SHAPES',
                   'BF16_STORAGE_ENDPOINTS', 'NATIVE_AND_MODEL_AB'],
        sources={p: sha(ROOT / p) for p in (
            'dev/strategy/moe_single_row_plan.py', 'dev/gemv_simt/general_plan.py',
            'dev/strategy/vector.py', 'policies/kpack_moe_decode_exact_v1.json',
            'policies/kpack_smallm_effective_v1.hpp', 'quactlize/execution/simt_strategy.hpp',
            'quactlize/execution/simt_common_scope.hpp', 'quactlize/execution/simt_common.cuh',
            'quactlize/execution/simt_kernel.cuh', 'quactlize/execution/measured_decode.inc')})
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))
