"""Host-only MoE candidate recall; historical latencies never become new data.

Candidates are generated before references are revealed, with the entire
target N/K family withheld. A missed champion fails the audit: it is not
appended to the scan. Passing establishes configuration coverage, not speed.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

from dev.strategy.plan import profile_winners
from dev.strategy.prescreen import Prescreen, compile_key, rejection
from dev.strategy.space import factored_recall, factor_candidates, grid_search, helpers
from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import load_sources

ROOT = Path(__file__).resolve().parents[2]
MS = (*range(1, 9), 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
SCHEMA = 'quactlize.moe-rule-audit.v1'
RULES = dict(limit=8, optional=0, per_stratum=2, blind=True,
             factorized=True, cross_format=True, reader_strata=True, stencil='warps',
             prefill_exact_tiles=True, portable_sf_delivery=True)


def proposal_only(rows):
    """Remove timing, ranking priority and winner/admission labels at the seam."""
    return StrategyTable(Observation(r.workload, r.recipe, r.source, 'PROPOSAL', 0,
                                    'GEOMETRY_ONLY', r.evidence) for r in rows)


def load_catalog():
    table, sources = load_sources()
    references, proposals = {}, list(table.observations)
    sources['effective-decode'] = digest([dict(workload=asdict(r.workload), recipe=asdict(r.recipe),
                                             evidence=r.evidence) for r in table.observations
                                        if r.source == 'effective-decode'])

    def add(w, recipe, source, role):
        if w.operation != 'indexed' or w.compute != 'bf16' or w.storage != 'f32':
            return
        key = digest(dict(workload=asdict(w), recipe=asdict(recipe), source=source, role=role))
        references[key] = dict(id=key, workload=asdict(w), recipe=asdict(recipe),
                               source=source, role=role)
        proposals.append(Observation(w, recipe, source, 'PROPOSAL', 0,
                                    'GEOMETRY_ONLY', sources[source]))

    # Effective production choices include robust router compromises. They
    # are controls in addition to, not instead of, each profile's best recipe.
    for row in table.observations:
        if row.source != 'effective-decode' or row.workload.operation != 'indexed':
            continue
        profiles = row.details.get('profiles') or [row.workload.router or 'real']
        source = 'effective-decode'
        for router in profiles:
            add(replace(row.workload, router=router), row.recipe, source, 'PRODUCTION_REFERENCE')

    name = 'docs/measurements/smallm_matched_20260915.json.gz'
    sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for key, recipes in profile_winners().items():
        q, mode, n, k, experts, topk, channels, tokens, compute, router = key
        if not mode:
            continue
        w = Workload(q, n, k, tokens, 'indexed', compute, experts=experts,
                     topk=topk, channels=channels, router=router)
        for recipe in recipes.values():
            add(w, recipe, name, 'HISTORICAL_ROUTER_WINNER')

    name = 'policies/kpack_prefill_confirmation_v1.json'
    raw = json.loads((ROOT / name).read_text())
    sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for point in raw['points']:
        add(Workload(**point['workload']), Recipe.from_dict(point['recipe']),
            name, 'HISTORICAL_PREFILL_WINNER')
    # No old us/component samples/accuracy decisions cross this boundary.
    return proposal_only(proposals), sorted(references.values(), key=lambda r: r['id']), sources


def family(w):
    return (w.qtype, w.n, w.k, w.experts, w.topk, w.channels)


def future_inventory(references):
    """Historical families only, NOT an assertion of target-model coverage."""
    families = sorted({family(Workload(**r['workload'])) for r in references})
    return dict(families=[dict(zip(('qtype', 'n', 'k', 'experts', 'topk', 'channels'), f))
                         for f in families], tokens=list(MS),
                shape_token_points=len(families) * len(MS),
                routing_profiles='SYNTHETIC_BALANCED_SPREAD_CLUSTER_PLUS_RECORDED_MODEL_IDS',
                old_real_router='SYNTHETIC_HOT_EXPERT_SAMPLING_NOT_MODEL_TRACE',
                model_headers='PENDING_CURRENT_GGUF_INVENTORY',
                tp_local_extents='PENDING_SOURCE_BOUND_CALLER_SPLITS_NOT_NK_DIVIDED_BLINDLY',
                complete_model_coverage=False)


def fresh_measurement_contract():
    return dict(timing_method='ASYS_GPU_ACTIVITY', historical_timing_reuse=False,
                resume='ONLY_SAME_NEW_CAMPAIGN_SOURCE_BINARY_DEVICE_AND_WORKLOAD_IDENTITY',
                precision='F32_ENDPOINTS_BF16_OPERANDS_FP32_ACCUMULATION',
                components='SEPARATE_PREPARE_MAIN_REDUCER_FINISH_ACTUAL_INVOCATIONS_ONCE',
                statistic='PAIRED_ARITHMETIC_MEANS_NOT_MINIMUM',
                native='UNCHANGED_MUL_MAT_ID_DISPATCH_WITH_RECORDED_ARITHMETIC',
                warmup='JIT_PROVIDER_TUNING_AND_FIRST_USE_EXCLUDED',
                weight_cache='VERIFIED_L2_ROTATING_RING_AT_LEAST_2.25_L2',
                tags=['quactlize_commit', 'source_sha256', 'actlize_commit',
                      'sdk_compiler_runtime', 'physical_device', 'binary_sha256',
                      'recipe_and_body', 'fixture_and_router', 'asys_trace_and_range'],
                performance_admitted=False, production_changed=False)


def moe_candidates(screen, w, options):
    """Sparse one-axis neighbors of blind seeds, never references/target times.

    Indexed grids have expert/row parallelism absent from one dense vector:
    retain low-warp Q4 reuse. SIMT column cohorts change packed request width;
    bit 0 changes the existing reader implementation. Change only one axis
    per seed, never recursively form a columns/values/variant cross product.
    """
    selected, report = screen.select(w, **options)
    additions = []
    for seed in tuple(selected):
        p = dict(seed.parameters)
        axes = {}
        if seed.kind == 'q4' and p['reader'] == 2:
            axes['warps'] = (2, 4, 8)
        elif seed.kind == 'simt':
            axes = dict(columns=(4, 8), variant=(p['variant'] ^ 1,))
        for axis, values in axes.items():
            for value in values:
                if p[axis] == value:
                    continue
                r = Recipe.make(seed.kind, seed.route, **(p | {axis: value}))
                if rejection(r, w):
                    continue
                additions.append(r)
                report['selected_reasons'].setdefault(r.key, []).append('MOE_ONE_AXIS_BLIND_SEED:' + axis)
    return factor_candidates([*selected, *additions]), report


def source_stamp(root=ROOT):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    files = {}
    for directory in ('dev', 'quactlize', 'tools', 'reference'):
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh', '.cpp', '.cu', '.h', '.hpp', '.cuh', '.inl', '.inc'):
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = digest(files)
    commit = git('rev-parse', 'HEAD')
    link = git('ls-tree', 'HEAD', 'third_party/actlize').split()
    return dict(quactlize_commit=commit, source_sha256=source,
                actlize_commit=link[2] if len(link) >= 3 else None,
                dirty=bool(git('status', '--porcelain', '--untracked-files=all')),
                tag=commit[:12] + '-src-' + source[:12],
                scope='HOST_PLAN_ACTUAL_SOURCE_NOT_A_COMPILED_BINARY_RECEIPT')


def make_plan(table, references, sources=None, rules=None):
    table = proposal_only(table.observations)
    screen = Prescreen(table)
    options = dict(RULES if rules is None else rules)
    if options.get('blind') is not True:
        raise ValueError('MoE audit requires blind candidate generation')
    grouped = defaultdict(list)
    for reference in references:
        w = Workload(**reference['workload'])
        if w.operation != 'indexed' or w.compute != 'bf16' or w.storage != 'f32':
            raise ValueError('MoE audit requires indexed F32/BF16 workloads')
        grouped[w.key].append(reference)

    @lru_cache(maxsize=None)
    def generate(w):
        return moe_candidates(screen, w, options)

    definitions, points, modules, helper_jobs = {}, [], set(), set()
    for wid, refs in sorted(grouped.items()):
        w = Workload(**refs[0]['workload'])
        # Selection cannot inspect any references. Router is not read on the
        # host in production; generate once per geometry, audit each profile.
        candidates, pruning = generate(replace(w, router=''))
        candidate_ids = []
        for r in candidates:
            definitions[r.key] = asdict(r)
            candidate_ids.append(r.key)
            key = compile_key(w, r)
            if key:
                modules.add(key)
        controls = {Recipe.from_dict(r['recipe']).key: Recipe.from_dict(r['recipe']) for r in refs}
        for key, r in controls.items():
            definitions[key] = asdict(r)
        verdict = factored_recall(w, candidates, controls.values())
        rejected = {key: reason for key, r in controls.items() if (reason := rejection(r, w))}
        expansions = helpers(w, candidates)
        for helper in expansions:
            helper_jobs.add(digest(helper))
        points.append(dict(id=wid, workload=asdict(w), candidates=candidate_ids,
                           references=refs, recall=verdict, rejected_references=rejected,
                           seeds=pruning['seed_prototype_count'],
                           prototypes=pruning['prototype_count'],
                           rule_reasons={key: pruning['selected_reasons'].get(key, []) for key in candidate_ids},
                           expansion_search=expansions,
                           grid_search={r.key: domain for r in candidates if (domain := grid_search(w, r))},
                           champion_injection=False, measured_latency_reused=False))
    tally = Counter(p['recall']['status'] for p in points)
    result = dict(schema=SCHEMA, status='PASS' if points and tally['PASS'] == len(points) else 'RULE_GAPS',
                  scope='HOST_HISTORICAL_CONFIG_RECALL_NOT_MEASUREMENT_OR_FULL_MODEL_COVERAGE',
                  rule_revision='MOE_BLIND_COVERAGE_V1',
                  rules=options, catalog_sha256=table.identity, reference_sources=sources or {},
                  configurations=definitions, points=points, future_inventory=future_inventory(references),
                  measurement=fresh_measurement_contract(),
                  counts=dict(points=len(points), reference_records=len(references), statuses=dict(tally),
                              candidate_arms=sum(len(p['candidates']) for p in points),
                              tc_or_provider_compile_keys=len(modules),
                              expansion_domains=len(helper_jobs),
                              resource_expanded_grids='PENDING_COMPILED_OCCUPANCY_QUERY',
                              gpu_seconds=None),
                  old_latency_rows_imported=0, kernel_launches=0,
                  production_changed=False, performance_admitted=False, sweep_ready=False,
                  pending=['CURRENT_MODEL_AND_TP_SHAPES', 'RESOURCE_AND_BF16_ENDPOINT_GATES',
                           'FRESH_ASYS_INDEXED_RUNNER_AND_NATIVE_AB', 'REAL_ROUTER_CAPTURE'])
    result['sha256'] = digest(result)
    return result
