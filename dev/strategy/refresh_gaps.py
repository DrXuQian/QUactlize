"""Finish reviewed MoE profile gaps, without proposing another config sweep."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from dev.llama_ab.contract import inventory
from dev.strategy.components import kernel_entry
from dev.strategy.prescreen import rejection, semantic_config
from dev.strategy.refresh import SCHEMA, family, original_recipe, request
from dev.strategy.refresh_selection import identity, supplement_plan
from dev.strategy.resources import module_spec
from dev.strategy.vector import spec as vector_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize

ROOT = Path(__file__).resolve().parents[2]
TABLE = 'policies/kpack_decode_refresh_v1.json'
PHASE = 'profile-gaps'
SELECTION = '67fe29755f8ecc07b30374f4f76c1609f9ea9fccf52753c29063dc6f00997841'


def opportunities():
    table = json.loads((ROOT/TABLE).read_bytes())
    supplement = supplement_plan(table)  # Verifies the complete evidence seal.
    if (supplement['selection'] != SELECTION or supplement['counts'] !=
            dict(host_keys=14, points=43, candidate_profile_cells=57)):
        raise ValueError('reviewed profile-gap inventory changed; do not expand this run implicitly')
    keys = {tuple(c['key']) for c in supplement['cells']}
    result = []
    for row in table['rows']:
        if tuple(row['key']) not in keys:
            continue
        for alternative in row['profile_tradeoffs']:
            states = {p['status'] for p in alternative['profiles']}
            if 'NOT_MEASURED' in states and states <= {'NOT_MEASURED', 'ELIGIBLE'}:
                result.append((row, alternative))
    if len(result) != 20:
        raise ValueError('reviewed candidate inventory changed')
    return result


def query_points():
    """Host-only: inspect all 70 profiles; only 49 need device measurements."""
    ids = {p['point'] for _, alt in opportunities() for p in alt['profiles']}
    points = [p for p in inventory() if p['id'] in ids and p['cohort'] == 'decode']
    if len(points) != 70 or {p['id'] for p in points} != ids:
        raise ValueError('profile gaps differ from the workload registry')
    return points


def _recipe(candidate):
    if candidate['implementation'] != {'kind': 'shipping'}:
        raise ValueError('profile supplement cannot substitute a new implementation')
    return normalize(candidate['config'], dequant=candidate['dequant_config'])


def _producer(w, recipe):
    # This supplement is deliberately M8 only. Fixed measured SIMT paths are
    # M1; host C++ checks below independently verify every current incumbent.
    p = dict(recipe.parameters)
    if (w.m != 8 or w.operation != 'indexed' or w.compute != 'bf16' or w.storage != 'f32' or
            recipe.route != 'fq' or rejection(recipe, w)):
        raise ValueError('profile supplement left its reviewed F32/BF16 indexed M8 domain')
    if recipe.kind == 'tc':
        return 'tc'
    if recipe.kind != 'simt':
        raise ValueError('undeclared supplement kernel family')
    if (p['variant'] == 3 and (
            (p['columns'] == 4 and p['values'] == 8 and p['warps'] in (2, 8)) or
            (p['columns'] == 8 and p['values'] == 4 and p['warps'] == 4) or
            (w.qtype == 12 and p['columns'] == 4 and p['values'] == 4 and p['warps'] == 4))):
        return 'common-packet-3d'
    return 'register-reuse'


def make_plan(originals):
    points = query_points()
    if [r['request'] for r in originals] != [request(Workload(**p['workload'])) for p in points]:
        raise ValueError('C++ profile-gap incumbent queries differ')
    by_id = {p['id']: p for p in points}
    cpp = {p['id']: r for p, r in zip(points, originals)}
    entries = opportunities()
    changed_keys = set()
    for row, alt in entries:
        for profile in alt['profiles']:
            p = by_id[profile['point']]
            w, current = Workload(**p['workload']), cpp[p['id']]
            original = original_recipe(current)
            old = normalize(row['original_config'], dequant=row['original_dequant_config'])
            if request(w) != row['key'] or semantic_config(original) != semantic_config(old):
                raise ValueError('production incumbent changed; re-review the comparison scope')
            expected = _producer(w, original)
            if original.kind == 'simt' and current['implementation']['producer'] != expected:
                raise ValueError('host production implementation differs from the supplement carrier')
            if expected == 'common-packet-3d' or _producer(w, _recipe(alt)) == 'common-packet-3d':
                changed_keys.add(tuple(row['key']))

    paired, cell_records = {}, []
    for row, alt in entries:
        recheck = tuple(row['key']) in changed_keys
        for profile in alt['profiles']:
            if profile['status'] != 'NOT_MEASURED' and not recheck:
                continue
            pid = profile['point']
            candidate = {k: alt[k] for k in ('config', 'dequant_config', 'implementation')}
            paired.setdefault(pid, []).append(candidate)
            cell_records.append(dict(key=row['key'], point=pid, router=profile['router'],
                candidate=identity(candidate), previous_status=profile['status'],
                reason='REPLACE_OLD_CARRIER_COMPARISON' if recheck else 'MISSING_PROFILE_PAIR'))

    planned, modules, vectors = [], {}, {}
    for p in points:
        if p['id'] not in paired:
            continue
        w = Workload(**p['workload'])
        original = original_recipe(cpp[p['id']])
        arms = []
        for cid, r in [('original', original)] + [(identity(c), _recipe(c)) for c in paired[p['id']]]:
            spec = module_spec(w, r) if r.kind == 'tc' else vector_spec(w, r)
            mid = digest(spec)
            (modules if r.kind == 'tc' else vectors)[mid] = spec
            arms.append(dict(id=cid, roles=['original' if cid == 'original' else 'rule'],
                aliases=[], recipe=asdict(r), required=True, spec=spec, module=mid,
                reasons=['CURRENT_CPP_SELECTED_RECIPE' if cid == 'original' else 'REVIEWED_PROFILE_GAP'],
                expected_producer=_producer(w, r)))
        if len({semantic_config(Recipe.from_dict(a['recipe'])) for a in arms}) != len(arms):
            raise ValueError('duplicate or incumbent-as-candidate pairing')
        planned.append(dict(p, arms=arms, original_recipe=asdict(original), sf_configs=[], excluded=[],
                            profile_gap=True, current_cpp=cpp[p['id']]))

    counts = dict(host_keys=len({tuple(c['key']) for c in cell_records}), points=len(planned),
        full_points=1114, families=len({family(Workload(**p['workload'])) for p in planned}),
        candidate_profile_cells=len(cell_records),
        missing_profile_cells=sum(c['previous_status'] == 'NOT_MEASURED' for c in cell_records),
        recheck_profile_cells=sum(c['previous_status'] == 'ELIGIBLE' for c in cell_records),
        controls=len(planned), arms=sum(len(p['arms']) for p in planned),
        max_arms=max(len(p['arms']) for p in planned), tc_modules=len(modules), vector_modules=len(vectors),
        operations=dict(Counter(p['workload']['operation'] for p in planned)))
    expected = dict(host_keys=14, points=49, families=10, candidate_profile_cells=69,
                    missing_profile_cells=57, recheck_profile_cells=12, controls=49, arms=118)
    if {k: counts[k] for k in expected} != expected or len(changed_keys) != 2:
        raise ValueError('bounded supplement changed; review scope before launching')
    result = dict(schema=SCHEMA, phase=PHASE, points=planned, tc=modules, vectors=vectors, counts=counts,
        historical_times_reused=False, production_changed=False,
        scope='MOE_M8_MISSING_PROFILE_PAIRS_AND_CHANGED_CARRIER_CONTROLS_NOT_FULL_SWEEP',
        comparison='REVIEWED_CANDIDATES_VS_CURRENT_CPP_PRODUCTION_BODY',
        protocol=dict(rounds=6, samples=5, order='FORWARD_REVERSE_ALTERNATING',
                      producer_ring='AT_LEAST_2P25_L2', helper_timing='ONE_KERNEL_AT_A_TIME'),
        profile_gaps=dict(selection=SELECTION, cells=cell_records, body_recheck_keys=sorted(changed_keys),
            prior_evidence='REPLACE_ALL_PROFILES_FOR_BODY_RECHECK_KEYS; OTHER_KEYS_FILL_GAPS_ONLY'),
        omitted=['NEW_CONFIG_SEARCH', 'LARGE_M', 'Q8', 'SF', 'MODEL_AB', 'ACU', 'PRODUCTION_PROMOTION'],
        sources={TABLE: sha(ROOT/TABLE)})
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))


def producer_entry(node):
    """Recognize only known outer callees if c++filt hits its recursion limit.

    Deep CuTe types can remain mangled; do not recursively parse their template
    arguments or use a substring search (a helper may name a GEMM internally).
    Exact namespace/source-name lengths plus the template boundary identify
    the same outer functions accepted in the readable-name path.
    """
    display = kernel_entry(node['name'])
    symbol = node.get('symbol', node['name'])
    if not symbol.startswith('_Z'):
        return display
    for name in ('cutlass::device_kernel', 'cutlass::Kernel',
                 'quactlize::execution::simt::common_reader',
                 'quactlize::execution::simt::register_reuse'):
        *namespace, function = name.split('::')
        outer = '_ZN' + ''.join(str(len(p))+p for p in namespace)
        for internal in ('', 'L'):
            prefix = outer + internal + str(len(function)) + function + 'I'
            if symbol.startswith(prefix) and len(symbol) > len(prefix):
                if display not in (symbol, name):
                    raise ValueError('producer symbol/display name disagree')
                return name
    return symbol  # Unknown outer function: retain fail-closed validation.


def validate_producers(point, data):
    """Do not accept the old generic lookalike as a current production control."""
    if not point.get('profile_gap'):
        return
    if set(data['arms']) != {a['id'] for a in point['arms']} or data.get('resource_unavailable'):
        raise ValueError('profile supplement did not measure every required pairing')
    for arm in point['arms']:
        nodes = [data['component_records'][cid]['identity']['node']
                 for cid in data['arm_components'][arm['id']]
                 if data['component_records'][cid]['kind'] == 'producer']
        if len(nodes) != 1:
            raise ValueError('profile supplement lacks one producer per arm')
        entry = producer_entry(nodes[0])
        expected = arm['expected_producer']
        if ((expected == 'common-packet-3d' and entry != 'quactlize::execution::simt::common_reader') or
                (expected == 'register-reuse' and entry != 'quactlize::execution::simt::register_reuse') or
                (expected == 'tc' and entry not in ('cutlass::device_kernel', 'cutlass::Kernel'))):
            raise ValueError('captured producer differs from current production: '+arm['id']+' '+entry)
