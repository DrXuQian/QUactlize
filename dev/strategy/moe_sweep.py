"""Resource-resolved MoE screen/finalists; no policy writes or old times."""
from copy import deepcopy
import statistics

from dev.strategy.resources import probe_spec, resolve_point
from dev.strategy.space import grid_search, scheduled_key
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest


def requests(point):
    w = Workload(**point['workload'])
    out = {}
    for a in point['arms']:
        r = Recipe.from_dict(a['recipe'])
        if r.kind != 'tc':
            continue
        domain = grid_search(w, r) if 'rule' in a['roles'] else None
        modes = domain['modes'] if domain else [dict(r.parameters).get('grid_mode', 0)]
        modes = set(modes) | {dict(r.parameters).get('grid_mode', 0)}
        for algorithm in sorted({int(m >= 2) for m in modes}):
            query = probe_spec(w, r, algorithm)
            out[digest(query)] = dict(request=query, module=a['module'], spec=a['spec'])
    return out


def resolve(point, receipts, images):
    """Pure replay of resource admission, including each omitted candidate."""
    requested = requests(point)
    if set(requested) != set(receipts):
        raise ValueError('MoE resource inventory differs')
    devices = set()
    for key, row in receipts.items():
        req = requested[key]
        image = images['tc'][req['module']]
        if (row.get('request') != req['request'] or row.get('module_sha256') != image['sha256'] or
                row.get('build_key') != image['key'] or row.get('runtime_status') != 0 or
                (row.get('status'), row.get('rc')) not in (('READY', 0), ('UNAVAILABLE', 1)) or
                row.get('status') == 'READY' and row.get('occupancy', 0) <= 0 or
                row.get('device', {}).get('compute_units', 0) <= 0):
            raise ValueError('MoE resource proof/module/device differs: '+key)
        devices.add(digest(row['device']))
    if len(devices) > 1:
        raise ValueError('resource queries crossed physical execution contexts')
    old, = [a for a in point['arms'] if a['id'] == 'original']
    raw = resolve_point(dict(workload=point['workload'],
        candidates=[a['recipe'] for a in point['arms'] if 'rule' in a['roles']],
        champion_controls=[old['recipe']]), receipts)
    resource = raw['resource_resolution']
    if resource['missing'] or any(r['role'] == 'control' for r in resource['rejected']):
        raise ValueError('current MoE control is resource-unavailable; no replacement admission')
    seeds = {scheduled_key(Recipe.from_dict(a['recipe'])): a for a in point['arms']}
    w = Workload(**point['workload'])
    arms, signatures = [], {}
    for value, original in [(old['recipe'], True), *[(r, False) for r in raw['candidates']]]:
        r = Recipe.from_dict(value)
        arm = deepcopy(old if original else seeds[scheduled_key(r)])
        arm.update(id='original' if original else r.key, recipe=value,
                   roles=['original'] if original else ['rule'], required=original or r.kind != 'tc')
        if r.kind == 'tc':
            from dev.strategy.space import resolved_grid
            q = receipts[digest(probe_spec(w, r, int(dict(r.parameters).get('grid_mode', 0) >= 2)))]
            launch = resolved_grid(w, r, q['device']['compute_units'], q['occupancy'])
            arm['binding'] = dict(launch=launch)
            signature = digest([arm['module'], scheduled_key(r), launch])
        else:
            signature = r.key
        if signature in signatures:
            found = arms[signatures[signature]]
            found['roles'] = sorted(set(found['roles']+arm['roles']))
            found['aliases'].append(r.key)
        else:
            signatures[signature] = len(arms)
            arms.append(arm)
    out = dict(point, arms=arms, resource_unavailable={}, resource_resolution=resource)
    out['resolved_sha256'] = digest(out)
    return out


def bind(point, images):
    from dev.strategy.resources import NativeProbe
    probes, receipts = {}, {}
    for key, req in requests(point).items():
        mid = req['module']
        if mid not in probes:
            probes[mid] = NativeProbe(req['spec'], images['tc'][mid])
        receipts[key] = probes[mid].query(req['request'])
    resolved = resolve(point, receipts, images)
    w = Workload(**point['workload'])
    bindings = {}
    for a in resolved['arms']:
        r = Recipe.from_dict(a['recipe'])
        q = receipts[digest(probe_spec(w, r, int(dict(r.parameters).get('grid_mode', 0) >= 2)))] if r.kind == 'tc' else None
        bindings[a['id']] = images['tc' if r.kind == 'tc' else 'vectors'][a['module']], q
    return resolved, bindings, receipts


def finalists(point, screen, *, window_pct=5, limit=32):
    """Keep near winners, fast variable rounds and the best of every family."""
    if screen.get('measurement_phase') != 'screen':
        raise ValueError('finalists require fresh Asys screen, not historical samples')
    stats = screen['decision']['statistics']
    if set(stats) != {a['id'] for a in point['arms']}:
        raise ValueError('incomplete MoE screen')
    floor = min(r['mean_us'] for r in stats.values())
    threshold = floor*(1+window_pct/100)
    chosen = {'original', screen['decision']['winner']}
    strata = {}
    for a in point['arms']:
        aid = a['id']; r = a['recipe']
        strata.setdefault((r['kind'], r['route']), []).append(aid)
        # A noisy mean is not a veto. A fast full round merits confirmation;
        # a single minimum kernel timestamp does not decide admission.
        if stats[aid]['mean_us'] <= threshold or min(map(statistics.fmean, stats[aid]['samples_us'])) <= threshold:
            chosen.add(aid)
    for aids in strata.values():
        chosen.add(min(aids, key=lambda aid: (stats[aid]['mean_us'], aid)))
    ordered = sorted(chosen, key=lambda aid: (aid != 'original', stats[aid]['mean_us'], aid))
    if len(ordered) > limit:
        raise ValueError(f'{len(ordered)} competitive finalists exceed bounded {limit}; screen retained, no silent top-K')
    return ordered


def subset(point, ids):
    if len(ids) != len(set(ids)) or 'original' not in ids:
        raise ValueError('invalid finalist IDs')
    arms = {a['id']: a for a in point['arms']}
    if not set(ids) <= set(arms):
        raise ValueError('finalist was not resource-screened')
    return dict(point, arms=[arms[aid] for aid in ids])


def measure(b, mb, sf, msf, point, bindings, authority, source, *, phase):
    from dev.strategy.asys_components import Collector
    from dev.strategy.confirm import confirm_point
    if phase not in ('screen', 'confirm'):
        raise ValueError('invalid sweep phase')

    class SharedCollector(Collector):
        def __init__(self):
            super().__init__(); self.seen = set()

        def sample(self, cid, graph, rnd, index):
            key = cid, rnd, index
            if key not in self.seen:
                super().sample(cid, graph, rnd, index)
                self.seen.add(key)

    collector = SharedCollector()
    size = 12 if phase == 'screen' else len(point['arms'])
    results = []
    for begin in range(0, len(point['arms']), size):
        arms = point['arms'][begin:begin+size]
        value = confirm_point(b, mb, sf, msf, dict(point, arms=arms),
            dict(best_sf=point['sf_configs'][0] if sf else None),
            dict(status='READY', arms=[a['id'] for a in arms], aliases={}),
            lambda arm: bindings[arm['id']], authority, source,
            rounds=4 if phase == 'screen' else 6, samples=3 if phase == 'screen' else 5,
            screening=phase == 'screen', trace_collector=collector)
        results.append(value)
    result = results[0]
    for value in results[1:]:
        for key in ('arms', 'roles', 'arm_components'):
            if set(result[key]) & set(value[key]):
                raise ValueError('duplicate screen arm')
            result[key].update(value[key])
        for cid, record in value['component_records'].items():
            if cid in result['component_records'] and result['component_records'][cid] != record:
                raise ValueError('shared helper identity changed across screen batches')
            result['component_records'][cid] = record
        result['wall_seconds'] += value['wall_seconds']
        result['expansion_proofs'].update(value['expansion_proofs'])
        result['expansion_fixture'] = result['expansion_fixture'] or value['expansion_fixture']
    result.update(components=list(result['component_records']), trace_schedule=collector.schedule,
                  measurement_phase=phase, resource_unavailable={})
    return result


def verify_capture(point, captured, images, *, screen=None):
    if screen is not None and captured['resource_queries'] != screen['resource_queries']:
        raise ValueError('resource identity changed between screen and confirmation')
    resolved = resolve(point, captured['resource_queries'], images)
    selected = resolved if screen is None else subset(resolved, finalists(resolved, screen['data']))
    if captured['execution_point'] != selected:
        raise ValueError('captured sweep inventory differs from rules/resources/finalists')
    phase = 'screen' if screen is None else 'confirm'
    if (captured['data'].get('measurement_phase') != phase or
            captured.get('screen_sha256') != (None if screen is None else screen['sha256'])):
        raise ValueError('sweep phase/source screen binding differs')
    return selected
