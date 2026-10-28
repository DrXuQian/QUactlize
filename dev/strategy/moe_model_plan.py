"""Three-model, actual-format MoE SIMT completion; not a new TC/fusion sweep."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from dev.gemv_simt.general_plan import configs
from dev.strategy.moe_inventory import collect
from dev.strategy.moe_single_row_plan import ROUTERS
from dev.strategy.moe_token_pair_plan import production_entry
from dev.strategy.prescreen import rejection
from dev.strategy.refresh import original_recipe, request
from dev.strategy.resources import module_spec
from dev.strategy.vector import execution_contract, spec
from quactlize.dispatch.planning import FIELDS, plan_smallm, plan_simt_configs
from quactlize.dispatch.table import Recipe, Workload
from quactlize.execution.moe_decode_policy import load
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
COHORT = 'moe-single-row-models'
MODELS = (
    ('qwen35-35b-a3b', 1, 'qwen35_model',
     'Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf'),
    ('qwen35-122b-a10b', 2, 'qwen122_model',
     'Qwen3.5-122B-A10B-GGUF-Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf'),
    ('deepseek-v4-flash', 4, 'deepseek_model',
     'DeepSeek-V4-Flash-0731-UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf'),
)


def model_paths(args):
    """Exact first shards, not an ambiguous recursive filename search."""
    return [dict(name=name, tp=tp,
                 path=str((getattr(args, flag, None) or args.model_root/relative).resolve(strict=True)))
            for name, tp, flag, relative in MODELS]


def model_inventory(args):
    from dev.prefill_asys.tp_header import HeaderTP, read_headers
    from dev.strategy.journal import atomic_json
    if args.llama_source is None:
        raise ValueError('--llama-source is required for CPU-only TP2/TP4 shape derivation; no llama binary is run')
    splitter = HeaderTP(args.llama_source, args.output/'tp-headers', indexed=True)
    snapshots = {}

    def loader(path):
        rows, tensors = read_headers(path)
        snapshots[path] = rows
        return rows, tensors

    result = collect(model_paths(args), loader=loader, partitioner=splitter, ms=list(range(1, 9)))
    # Retain the actual headers, not weights, so model/type/TP coverage can be
    # audited from the small review archive without another Box round.
    result['headers_snapshot_sha256'] = digest(snapshots)
    atomic_json(args.output/'moe-headers.json', snapshots)
    atomic_json(args.output/'moe-inventory.json', result)
    return result


def verify_inventory(output, plan):
    if plan.get('cohort') != COHORT:
        return
    inventory = json.loads((Path(output)/'moe-inventory.json').read_bytes())
    headers = json.loads((Path(output)/'moe-headers.json').read_bytes())
    if (digest(inventory)!=plan['coverage']['inventory_sha256'] or
            digest(headers)!=inventory['headers_snapshot_sha256']):
        raise ValueError('frozen model headers/TP inventory changed')


def expanded_points(inventory):
    if (inventory.get('schema')!='quactlize.moe-model-inventory.v1' or
            inventory.get('status')!='PASS' or inventory.get('blocked') or
            inventory.get('tokens')!=list(range(1, 9))):
        raise ValueError('incomplete MoE header/TP inventory; inspect moe-inventory.json')
    expected = {(name, tp) for name, tp, _, _ in MODELS}
    if {(m['name'], m['tp']) for m in inventory['models']} != expected:
        raise ValueError('this cohort requires Qwen35 TP1, Qwen122 TP2 and DeepSeek Flash TP4')
    if any(m.get('expert_weights', 0)<=0 for m in inventory['models']):
        raise ValueError('requested model has no expert weights')
    if any(e['reason'].startswith('NATIVE_ONLY_SHAPE:') for e in inventory['excluded']):
        raise ValueError('unsupported local shape in requested formats; not silently omitted')
    points = {}
    model_ms = {name: set() for name, _ in expected}
    family_ms = {}
    for source in inventory['points']:
        w = Workload(**source['workload'])
        if (w.operation, w.compute, w.storage, w.fusion) != ('indexed', 'bf16', 'f32', 'none'):
            raise ValueError('model completion requires unfused indexed BF16/F32 endpoints')
        family = w.qtype, w.n, w.k, w.experts, w.topk, w.channels
        family_ms.setdefault(family, set()).add(w.m)
        if not source['consumers']:
            raise ValueError('model point has no actual tensor consumer')
        for consumer in source['consumers']:
            if (consumer['model'], consumer['tp']) not in expected:
                raise ValueError('unexpected model consumer')
            model_ms[consumer['model']].add(w.m)
        for router in (('spread',) if w.m == 1 else ROUTERS):
            value = replace(w, router=router)
            if value.key in points:
                raise ValueError('duplicate model workload; inventory must deduplicate consumers')
            points[value.key] = dict(id=value.key, workload=asdict(value), consumers=source['consumers'])
    if not family_ms or any(ms != set(range(1, 9)) for ms in [*model_ms.values(), *family_ms.values()]):
        raise ValueError('M1..8 coverage missing for a model or weight family')
    return sorted(points.values(), key=lambda p: (
        *(p['workload'][k] for k in ('qtype', 'n', 'k', 'experts', 'topk', 'channels', 'm')), p['id']))


def candidate_recipes(w):
    # Bounded common-reader geometries, not N/K x all GGUF types. M1..8
    # shares this rule; router concentration never changes the candidate set.
    for config in configs(w.qtype):
        config = {k: v for k, v in config.items() if k != 'hoist'}
        for split in (1, 2, 4, 8):
            r = Recipe.make('simt', 'fq', **(config | {'split': split}))
            if not rejection(r, w):
                yield r


def make_plan(output, inventory):
    from dev.strategy.journal import atomic_json
    source_points = expanded_points(inventory)
    originals = plan_smallm(output, [request(Workload(**p['workload'])) for p in source_points],
                            indexed_fallbacks=True)
    atomic_json(Path(output)/'cpp-originals.json', originals)
    previous = {tuple(row['key']): row['original_config'] for row in load()['rows']}
    pending, queries = [], {}
    for source, old in zip(source_points, originals['requests']):
        w = Workload(**source['workload'])
        if old['status']:
            raise ValueError('current indexed control unavailable: '+str(request(w)))
        recipes = [(original_recipe(old), 'CURRENT_CPP_SELECTED_RECIPE')]
        if tuple(request(w)) in previous:
            cfg = dict(previous[tuple(request(w))])
            recipes.append((Recipe.make(cfg.pop('kind'), 'fq', **cfg),
                            'PRE_ADMISSION_CONTROL_REMEASURED_NOT_OLD_LATENCY'))
        recipes.extend((r, 'COMMON_SIMT_GEOMETRIES_WITH_LEGAL_SPLIT_NEIGHBORS') for r in candidate_recipes(w))
        for r, _ in recipes:
            if r.kind == 'simt':
                key = tuple(request(w)+[dict(r.parameters)[f] for f in FIELDS])
                queries[key] = None
        pending.append((source, w, old, recipes))
    queries.update(zip(queries, plan_simt_configs(output, queries)))
    points, tc, vectors = [], {}, {}
    for source, w, old, recipes in pending:
        arms = {}
        for r, reason in recipes:
            if rejection(r, w):
                raise ValueError('required control/config violates workload contract: '+str(request(w)))
            if r.kind == 'simt':
                query = queries[tuple(request(w)+[dict(r.parameters)[f] for f in FIELDS])]
                if query['status']:
                    raise ValueError('SIMT host launch validation failed: '+str(query))
                entry = production_entry(query)
            else:
                # Only the actual current incumbent may introduce TC/Q4.
                if reason != 'CURRENT_CPP_SELECTED_RECIPE':
                    raise ValueError('no new TC or specialized Q4 search in SIMT completion')
                entry = production_entry(old)
            item = module_spec(w, r) if r.kind == 'tc' else spec(w, r)
            mid = digest(item)
            (tc if r.kind == 'tc' else vectors)[mid] = item
            original = reason == 'CURRENT_CPP_SELECTED_RECIPE'
            arm = arms.setdefault(r.key, dict(id='original' if original else r.key, recipe=asdict(r),
                roles=['original'] if original else ['rule'], aliases=[], required=True, reasons=[],
                spec=item, module=mid, expected_producer=entry))
            arm['reasons'].append(reason)
        arms = list(arms.values())
        sf = sorted({max(0, dict(a['recipe']['parameters']).get('dequant_config', -1))
                     for a in arms if a['recipe']['route'] == 'sf' and w.qtype != 8})
        points.append(dict(source, cohort='decode', selection=old, execution_contract=execution_contract(w),
                           arms=arms, sf_configs=sf, indexed_single_row=True, excluded=[]))
    model_coverage = []
    for model in inventory['models']:
        rows = [p for p in points if any(c['model']==model['name'] for c in p['consumers'])]
        families = sorted({tuple(p['workload'][k] for k in ('qtype', 'n', 'k', 'experts', 'topk', 'channels'))
                           for p in rows})
        model_coverage.append(dict(model=model['name'], tp=model['tp'], points=len(rows), families=families,
                                   formats=sorted({f[0] for f in families})))
    result = dict(schema='quactlize.smallm-asys.v1', phase='regression-asys', cohort=COHORT,
        points=points, tc=tc, vectors=vectors,
        counts=dict(points=len(points), arms=sum(len(p['arms']) for p in points), tc_modules=len(tc),
                    vector_modules=len(vectors), families=len(points)//36),
        timing_method='ASYS_GPU_ACTIVITY', selection_metric='MEAN_COMPONENT_SUM',
        noise_policy='DIAGNOSTIC_NOT_VETO', historical_times_reused=False,
        production_changed=False, native_comparison=False,
        scope='THREE_MODEL_TP_LOCAL_MOE_SIMT_M1_TO_8_NOT_TC_SEARCH_OR_MODEL_ADMISSION',
        coverage=dict(models=model_coverage, tokens=list(range(1, 9)), m1_profiles=['spread'],
            multi_token_profiles=list(ROUTERS), profiles='SYNTHETIC_IDS_NOT_CAPTURED_MODEL_ROUTES',
            inventory_sha256=digest(inventory), excluded=inventory['excluded'],
            production_kinds=dict(Counter(p['arms'][0]['recipe']['kind'] for p in points)),
            fused_gate_up_swiglu='BACKUP_NOT_EXECUTED', actual_model_weights_used=False),
        remaining=['SIMT_VS_TC_CONFIG_SWEEP_AFTER_SIMT_OPTIMIZATION', 'BF16_STORAGE_ENDPOINTS',
                   'FUSED_GATE_UP_SWIGLU', 'NATIVE_AND_MODEL_AB'],
        sources={p: sha(ROOT/p) for p in (
            'dev/strategy/moe_model_plan.py', 'dev/strategy/moe_inventory.py',
            'dev/prefill_asys/tp_header.py', 'dev/prefill_asys/cohort.py',
            'dev/gemv_simt/general_plan.py', 'dev/strategy/vector.py', 'tools/kpack_selection.cpp',
            'quactlize/dispatch/binding.cpp', 'quactlize/dispatch/planning.py',
            'quactlize/dispatch/smallm.hpp', 'quactlize/execution/q4_decode.cpp',
            'policies/kpack_moe_decode_exact_v1.json', 'policies/kpack_smallm_effective_v1.hpp',
            'quactlize/execution/simt_strategy.hpp', 'quactlize/execution/simt_common_scope.hpp',
            'quactlize/execution/simt_common.cuh', 'quactlize/execution/simt_kernel.cuh',
            'quactlize/execution/measured_decode.inc')})
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))
