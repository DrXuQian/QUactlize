"""One reviewed, cross-card retest; never import an old performance sample."""
from collections import Counter
import copy
import json
import math
from pathlib import Path

from dev.strategy.prefill_closure import require
from dev.strategy.prefill_plan import family
from dev.strategy.prefill_refresh import make_plan
from dev.strategy.resource_pilot import current_module, verified_plan
from dev.strategy.resources import compiler, module_spec, probe_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

SCOPE = Path(__file__).with_name('prefill_reconfirm_scope.json')


def load_scope():
    scope = json.loads(SCOPE.read_text())
    require(scope['sha256'] == digest({k: v for k, v in scope.items() if k != 'sha256'}),
            'reconfirmation scope receipt changed')
    return scope


def make_reconfirm_plan(selector, cards):
    scope = load_scope()
    # Recreate the OLD ownership to select its entire suspect cohort, NOT
    # today's new card assignment or only the old losing/noisy candidates.
    old = make_plan(selector, scope['prior_cards'])
    require(old['plan_sha256'] == scope['prior_plan_sha256'],
            'reviewed full shortlist/current C++ controls changed')
    require(cards and len(cards) == len(set(cards)) and all(c.isdigit() for c in cards),
            'unique physical card selectors required')
    extra = set(scope['other_unresolved'])
    points, retained = [], []
    for p in old['points']:
        suspect = old['cards'][p['shard']] in scope['suspected_cards']
        if suspect or p['id'] in extra:
            points.append(p | dict(reconfirm_reason='WHOLE_SUSPECT_CARD_COHORT' if suspect
                                   else 'UNRESOLVED_SELECTION'))
        else:
            retained.append(p['id'])
    require(extra <= {p['id'] for p in points} and len(extra) == 3, 'unresolved point inventory changed')
    # Partial indexed families would change the router's minimum active count
    # and hence the cold-weight ring. Only dense E=1 families may be partial.
    all_sizes = Counter(family(p) for p in old['points'])
    sizes = Counter(family(p) for p in points)
    require(all(sizes[family(p)] == all_sizes[family(p)] or
                p['workload']['operation'] == 'dense' and p['workload']['experts'] == 1 for p in points),
            'partial indexed family would change cold-ring semantics')
    weights = scope['scheduling_seconds']
    require(set(weights) == set(sizes) and len(cards) <= len(weights) and
            all(math.isfinite(v) and v > 0 for v in weights.values()), 'invalid retest scheduling hints')
    loads, owners, order = [0.]*len(cards), {}, []
    for key in sorted(weights, key=lambda k: (-weights[k], k)):
        owner = min(range(len(cards)), key=lambda i: (loads[i], i))
        owners[key] = owner
        loads[owner] += weights[key]
        order.append(key)
    modules, queries = {}, {}
    for p in points:
        p['shard'] = owners[family(p)]
        w = Workload(**p['workload'])
        for arm in p['arms'].values():
            r = Recipe.from_dict(arm['recipe'])
            if r.kind == 'tc':
                spec = module_spec(w, r)
                query = probe_spec(w, r, int(dict(r.parameters)['grid_mode'] >= 2))
                modules[digest(spec)] = spec
                queries[digest(query)] = query
    counts = dict(points=len(points), arms=sum(len(p['arms']) for p in points), families=len(weights),
                  modules=len(modules), retained=len(retained))
    require(counts == scope['counts'], 'reviewed retest denominator changed')
    plan = old | dict(points=points, modules=modules, queries=queries, cards=cards,
        family_order=order, family_owners=owners, scheduling_seconds=weights,
        historical_shard_seconds=loads, scheduling_scope=scope['scheduling_scope'],
        counts=counts | dict(queries=len(queries), rounds=old['counts']['rounds']),
        scope='REVIEWED_COHORT_RECONFIRMATION_ALL_ARMS_FRESH_ON_NEW_CARDS',
        reconfirm=scope, retained_point_ids=sorted(retained), retained_timings_imported=False)
    plan.pop('plan_sha256')
    plan['plan_sha256'] = digest(plan)
    return plan


def cohort(device):
    h = device['hardware']
    return dict(sdk=h['sdk'], l2_bytes=h['l2']['bytes'],
        capabilities={k: v for k, v in h['device'].items() if k not in ('pci', 'ordinal', 'host')},
        providers=device['providers'], oracle=device['oracle'], graph_abi=device['graph_abi'])


def check_cohort(plan, devices):
    require(devices and all(digest(cohort(d)) == plan['reconfirm']['cohort_sha256'] for d in devices),
            'retest SDK/device capability/provider/oracle differs from reviewed cohort')


def build_paths(output, reuse=None):
    root = Path(reuse) if reuse else Path(output)
    return dict(cache=str(root/'cache/tc'), helpers_path=str(root/'helpers'), dequant_path=str(root/'dequant'))


def reuse_images(sdk, directory, plan):
    """Read-only compiled-image reuse. No compiler.build(), old query or round import.

    New source/SDK/compiler checks still apply to each module, with new-device
    resource queries later. A mismatch is an error, never silently rebuilt.
    """
    from dev.strategy.prefill_build import verify as verify_helpers
    from tools.run_kpack_strategy_decode import dequant_checked
    directory = Path(directory).resolve(strict=True)
    old = verified_plan(json.loads((directory/'finalists.json').read_text()))
    require(old['plan_sha256'] == plan['reconfirm']['prior_plan_sha256'], 'reuse build is not the reviewed run')
    images = json.loads((directory/'images.json').read_text())
    records, compilers = {}, {}
    paths = build_paths(directory)
    cache = Path(paths['cache']).resolve(strict=True)
    for mid, spec in plan['modules'].items():
        require(mid in images['tc'], 'missing required reused TC module: '+mid)
        record = images['tc'][mid]
        path = Path(record['path'])
        require(not path.is_symlink() and path.resolve(strict=True) == cache/record['key']/'kernel.so' and
                record['parent'] == spec['parent'] and sha(path) == record['sha256'],
                'reused TC image/path/parent changed: '+mid)
        cached = json.loads((path.parent/'manifest.json').read_text())
        require(all(record[k] == cached[k] for k in ('key', 'parent', 'identity', 'sha256')),
                'reused TC cache manifest changed: '+mid)
        key = spec['endpoint'], spec['compute']
        if key not in compilers:
            compilers[key] = compiler(spec, sdk, cache)
        current_module(record, spec, compilers[key])
        records[mid] = copy.deepcopy(record)
    hp = verify_helpers(sdk, directory/'helpers')
    dq = dequant_checked(directory/'dequant', sdk)
    require(hp == images['helpers'] and dq == images['dequant'], 'reused helper/dequant receipt changed')
    return dict(tc=records, helpers=hp, dequant=dq, reuse=dict(
        directory=str(directory), prior_plan=old['plan_sha256'], prior_images_sha256=digest(images),
        modules=len(records), new_compiles=0, previous_timings_reused=False,
        paths=paths, scope='VERIFIED_IMAGES_ONLY_FRESH_DEVICE_QUERIES_AND_MEASUREMENTS'))
