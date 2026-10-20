"""Reviewed new-clock measurements -> one host-key candidate selection table.

This module never changes a runtime policy. In particular, a Q8 body ID must
not disappear when a measured recipe is converted to the old geometry ABI.
"""
from collections import Counter, defaultdict
import hashlib
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

from dev.strategy.components import component_key, stage_role
from dev.strategy.followup import paired_verdict, validate_confirmation
from dev.strategy.production import config, key, require, semantic, validate_component_decision
from dev.strategy.refresh import family
from dev.strategy.refresh_clock import ADVISORY, parse
from dev.strategy.refresh_retry import check_transition
from quactlize.dispatch.table import Recipe, Workload
from quactlize.execution.simt_codegen import Config, runtime_inventory
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.decode-refresh-selection.v1'
ARCHIVES = {
    'parent': 'c7cf6c49661643ec90456964d06b7f48f2fdb8e63c4ccdd86a4d2368276139bb',
    'retry': '7a7e02783e670c4a7674c1ee5b43296e1f1b0114274370036cddb4b865710e3e',
    'prior_pass': '960595f5c597ce7ffb84c3f3d047e48beb3eef2fd7c381a04e5cb04c4dd5b49c',
}
SCOPE = 'ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT'
FIELDS = ('variant', 'columns', 'warps', 'values', 'split')


def sealed(value):
    require(value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'invalid sealed receipt')


def timing_validation(data):
    # Reproduce the original verdict from its recorded sums. Python 3.12
    # changed float sum: only the proved rounding bound may differ on 3.10.
    validate_component_decision(data['component_records'], data['arm_components'],
                                data['roles'], data['decision'])
    checked = dict(data)
    costs = {aid: [[sum(data['component_records'][cid]['samples_us'][r][i] for cid in ids)
                    for i in range(5)] for r in range(6)]
             for aid, ids in data['arm_components'].items()}
    checked['decision'] = paired_verdict(costs, data['roles'])
    validate_confirmation(checked)


def measured_recipe(arm, workload):
    value = config(arm['recipe'], workload)
    body = value.pop('q8_body', None)
    implementation = dict(kind='shipping')
    if body is not None:
        require(workload['qtype'] == 8 and workload['operation'] == 'dense' and
                workload['storage'] == 'f32' and value['kind'] == 'simt' and
                value['variant'] == 5 and value['values'] == 4 and value['split'] == 1 and
                (value['columns'], value['warps']) in ((4, 8), (8, 4)) and
                type(body) is int and body in (0, 1, 3, 4, 5), 'invalid measured Q8 body')
        implementation = dict(kind='q8-address', body=body)
    params = dict(Recipe.from_dict(arm['recipe']).parameters)
    dequant = params.get('dequant_config', -1) if value['kind'] == 'tc' else -1
    return dict(config=value, dequant_config=dequant, implementation=implementation)


def identity(candidate):
    return digest(dict(config=semantic(candidate['config']),
                       dequant_config=candidate['dequant_config'],
                       implementation=candidate['implementation']))


def validate_point(raw, name, top):
    """Bind raw per-component proofs to the frozen recipe, image and device."""
    sealed(raw)
    plan, run = top['plan.json'], top['run.json']
    point = top['_points'][raw['point']]
    w = Workload(**point['workload'])
    require(w.key == point['id'] and w.m <= 8 and w.storage == 'f32' and
            w.compute == ('f16' if w.operation == 'dense' else 'bf16') and
            w.fusion == 'none' and w.input_stride == w.k and w.output_stride == w.n,
            'unsupported refresh workload')
    shard = top['_owners'][family(w)]
    hardware = run['devices'][shard]
    authority = dict(run=run['sha256'], shard=shard, pci=hardware['device']['pci'],
                     hardware=hardware, cu_mhz=run['cu_mhz'], memory_mhz=run['memory_mhz'])
    if 'clock_policy' in run:
        authority['clock_policy'] = run['clock_policy']
    require(raw['authority'] == authority and
            name == f'gpu-{run["cards"][shard]}/points/{w.key}.json', 'point authority differs')
    result = dict(point=w.key, workload=point['workload'], receipt_sha256=raw['sha256'],
                  source=run['sha256'], physical_device=hardware, candidates={},
                  status=raw['status'])
    if raw['status'] != 'PASS':
        require(raw['status'] == 'FAIL', 'unknown receipt status')
        return result
    data = raw['data']
    require(data['point'] == w.key and data['workload'] == point['workload'] and
            not data.get('resource_unavailable'), 'measurement/candidate coverage differs')
    timing_validation(data)
    arms = {a['id']: a for a in point['arms']}
    require(len(arms) == len(point['arms']) and set(arms) == set(data['arms']) and
            data['roles'] == {aid: arm['roles'] for aid, arm in arms.items()},
            'declared arm inventory differs')
    clocks = data['clock_observations']
    require(len(clocks) == 6 and [r['round'] for r in clocks] == list(range(6)),
            'clock observations incomplete')
    for clock in clocks:
        for edge in ('before', 'after'):
            observed = clock[edge]
            parsed = parse(observed['stdout'], authority['pci'], authority['cu_mhz'],
                           authority['memory_mhz'], advisory=authority.get('clock_policy') == ADVISORY)
            require(all(observed.get(k) == v for k, v in parsed.items()), 'clock metadata differs')
    result.update(status=data['decision']['status'], raw_winner=data['decision']['winner'])
    for aid, arm in arms.items():
        recipe = Recipe.from_dict(arm['recipe'])
        proof = data['arms'][aid]
        image = top['images.json']['tc' if recipe.kind == 'tc' else 'vectors'][arm['module']]
        require(proof['receipt']['module_sha256'] == image['sha256'] and
                proof['receipt']['build_key'] == image['key'], 'executed image differs')
        if recipe.kind != 'tc':
            require(proof['receipt']['spec'] == arm['spec'], 'executed vector body differs')
        else:
            require(proof['receipt'].get('compute', proof['receipt'].get('compute_type')) == w.compute,
                    'executed TC precision differs')
        split = dict(recipe.parameters).get('split', 1)
        require(proof['receipt'].get('split', proof['receipt'].get('launch', {}).get('split')) == split,
                'executed split differs')
        stages = []
        for cid in data['arm_components'][aid]:
            c = data['component_records'][cid]
            node = c['identity']['node']
            expected, contract = component_key(
                w, None if c['kind'] == 'expansion' else recipe, node, authority=authority,
                source=run['body_source'], cache=c['identity']['cache'])
            require(cid == expected and contract == c['identity'] and
                    stage_role(node['name'], split) == c['kind'], 'component semantics differ')
            stages.append(dict(kind=c['kind'], median_us=c['median_us'], id=cid))
        require(sum(s['kind'] == 'producer' for s in stages) == 1 and
                (split == 1 or any(s['kind'] in ('reducer', 'reducer_finish') for s in stages)),
                'incomplete producer/reducer call')
        stats = data['decision']['statistics'][aid]
        result['candidates'][aid] = measured_recipe(arm, point['workload']) | dict(
            arm=aid, median_us=stats['median_us'], rounds=stats['round_medians_us'],
            spread_pct=stats['spread_pct'], roles=arm['roles'], components=stages,
            numerical_error=max(p['relative_l1_error'] for p in proof['numerical']),
            module_sha256=image['sha256'], build_key=image['key'])
    return result


def read_archive(path, role):
    path = Path(path)
    require(sha(path) == ARCHIVES[role], 'unreviewed '+role+' archive')
    top, members, points = {}, {}, {}
    with tarfile.open(path, 'r|xz') as stream:
        for member in stream:
            name = member.name
            p = PurePosixPath(name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and name not in members,
                    'unsafe or duplicate archive member')
            body = stream.extractfile(member).read()
            members[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if '/' not in name and name.endswith('.json'):
                top[name] = json.loads(body)
                if name == 'plan.json':
                    top['_points'] = {p['id']: p for p in top[name]['points']}
                if name == 'run.json':
                    top['_owners'] = {key: i for i, keys in enumerate(top[name]['assignments']) for key in keys}
            elif re.fullmatch(r'gpu-\d+/points/[0-9a-f]{64}\.json', name):
                raw = json.loads(body)
                require(raw['point'] not in points, 'duplicate point')
                points[raw['point']] = validate_point(raw, name, top)
    transfer = top['transfer-manifest.json']
    wanted = {r['path']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in transfer['included']}
    require(len(wanted) == len(transfer['included']) and
            wanted == {k: v for k, v in members.items() if k != 'transfer-manifest.json'},
            'transfer hashes differ')
    plan, run = top['plan.json'], top['run.json']
    sealed(plan)
    sealed(run)
    require(run['plan'] == plan['sha256'] and run['body_source'] == digest(run['harness']) and
            set(points) == set(top['_points']), 'run/plan coverage differs')
    require(all(d['device']['compute_units'] == 72 and d['device']['warp_size'] == 32 for d in run['devices']),
            'not the reviewed ZW810 hardware')
    for kind in ('tc', 'vectors'):
        require(set(top['images.json'][kind]) == set(plan[kind]) and run['images'][kind] ==
                {k: v['sha256'] for k, v in top['images.json'][kind].items()}, 'image inventory differs')
    return top, points


def load(parent, retry, prior_pass):
    top, points = read_archive(parent, 'parent')
    followup, updated = read_archive(retry, 'retry')
    op, old, np, new = (top['plan.json'], top['run.json'], followup['plan.json'], followup['run.json'])
    check_transition(old['harness'], new['harness'])
    binding = np['clock_retry']
    require(binding['previous_run'] == old['sha256'] and binding['previous_plan'] == op['sha256'] and
            new['clock_policy'] == ADVISORY, 'invalid retry binding')
    for field in ('devices', 'sdk', 'assignments', 'cards'):
        require(new[field] == old[field], 'retry changed '+field)
    require(followup['cpp-originals.json'] == top['cpp-originals.json'], 'retry changed incumbent choices')
    before = {r['point']: r for r in binding['previous_points']}
    require(len(before) == len(binding['previous_points']) == 1114 and set(before) == set(points) and
            set(updated) == {pid for pid, r in before.items() if r['status'] != 'PASS'},
            'retry scope differs')
    for point in np['points']:
        require(point == top['_points'][point['id']], 'retry changed measured candidates')
    for kind in ('tc', 'vectors'):
        require(new['images'][kind] == {k: old['images'][kind][k] for k in np[kind]},
                'retry changed module images')
    path = Path(prior_pass)
    require(sha(path) == ARCHIVES['prior_pass'], 'unreviewed prior PASS upload')
    raw = json.loads(lzma.decompress(path.read_bytes()))
    pid = raw['point']
    require(points[pid]['status'] == 'FAIL' and before[pid]['status'] == 'PASS' and
            raw['status'] == 'PASS' and raw['sha256'] == before[pid]['sha256'],
            'prior PASS supplement differs')
    shard = before[pid]['shard']
    points[pid] = validate_point(raw, f'gpu-{old["cards"][shard]}/points/{pid}.json', top)
    for pid, r in before.items():
        if r['status'] == 'PASS':
            require(points[pid]['receipt_sha256'] == r['sha256'] and points[pid]['status'] != 'FAIL',
                    'previous completed result changed')
        else:
            require(updated[pid]['status'] in ('PASS', 'UNSTABLE_RETAIN_ORIGINAL'), 'retry did not pass')
            points[pid] = updated[pid]
    require(len(points) == 1114 and sum(len(p['candidates']) for p in points.values()) == 22743,
            'full measurement denominator differs')
    sources = [dict(run=r['sha256'], plan=r['plan'], body_source=r['body_source'], sdk=r['sdk'],
                    devices=r['devices'], images_sha256=digest(r['images']),
                    clock_policy=r.get('clock_policy', 'STRICT_OBSERVATIONS_NO_LOCK_REQUEST'),
                    requested_mhz=r['cu_mhz']) for r in (old, new)]
    return list(points.values()), top['cpp-originals.json']['requests'], sources


def allowed(candidate, original):
    return (candidate['median_us'] <= original['median_us'] and
            candidate['spread_pct'] <= 5 and original['spread_pct'] <= 5 and
            sum(x <= y for x, y in zip(candidate['rounds'], original['rounds'])) >= 4)


def selected_profiles(profiles):
    """Runtime does not know the router distribution: intersect eligible bodies."""
    known = []
    for p in profiles:
        old = p['candidates']['original']
        candidates = {identity(old): old}
        if p['status'] == 'PASS':
            for aid, candidate in p['candidates'].items():
                if aid != 'original' and 'rule' in candidate['roles'] and allowed(candidate, old):
                    cid = identity(candidate)
                    if cid not in candidates or candidate['median_us'] < candidates[cid]['median_us']:
                        candidates[cid] = candidate
        known.append(candidates)
    common = set.intersection(*(set(k) for k in known))
    require(bool(common), 'incumbent identity differs across router profiles')

    def score(cid):
        ratios = [k[cid]['median_us']/p['candidates']['original']['median_us']
                  for k, p in zip(known, profiles)]
        return max(ratios), math.fsum(math.log(x) for x in ratios), any(k[cid]['arm'] != 'original' for k in known), cid

    winner = min(common, key=score)
    return [k[winner] for k in known], len(common)


def profile_tradeoffs(profiles, chosen):
    """Distinguish a measured regression from an unmeasured router pairing."""
    alternatives = {}
    for p in profiles:
        c = p['candidates'][p['raw_winner']]
        cid = identity(c)
        if cid != identity(chosen):
            alternatives[cid] = c
    result = []
    for cid, candidate in sorted(alternatives.items()):
        checks = []
        for p in profiles:
            matches = [c for c in p['candidates'].values() if identity(c) == cid]
            old = p['candidates']['original']
            if not matches:
                checks.append(dict(point=p['point'], router=p['workload']['router'], status='NOT_MEASURED'))
                continue
            c = min(matches, key=lambda c: (c['median_us'], c['arm'] != 'original'))
            status = ('RETAIN_RAW_UNSTABLE' if p['status'] != 'PASS' else
                      'MEASURED_REGRESSION' if c['median_us'] > old['median_us'] else
                      'CONFIRMATION_GATE_NOT_MET' if not allowed(c, old) else 'ELIGIBLE')
            checks.append(dict(point=p['point'], router=p['workload']['router'], status=status,
                               delta_pct=100*(c['median_us']/old['median_us']-1)))
        result.append(dict(config=candidate['config'], dequant_config=candidate['dequant_config'],
                           implementation=candidate['implementation'], profiles=checks))
    return result


def fit(points, originals, sources, baseline):
    by_key = {tuple(r['key']): r for r in baseline['exact']}
    cpp = {}
    for request in originals:
        k = tuple(request['request'])
        require(k not in cpp or cpp[k] == request, 'incumbent depends on hidden router profile')
        cpp[k] = request
    groups = defaultdict(list)
    for p in points:
        groups[key(p['workload'])].append(p)
    require(set(groups) == set(cpp) and set(groups) <= set(by_key), 'runtime key inventory differs')
    rows, pending = [], []
    for k, profiles in sorted(groups.items()):
        profiles.sort(key=lambda p: p['point'])
        require(len({p['workload']['router'] for p in profiles}) == len(profiles), 'duplicate router profile')
        require(len({digest(p['physical_device']) for p in profiles}) == 1,
                'router profiles were measured on different physical cards')
        source = by_key[k]
        old = profiles[0]['candidates']['original']
        require(semantic(source['config']) == semantic(old['config']) and
                source.get('dequant_config', -1) == old['dequant_config'] and
                old['implementation'] == dict(kind='shipping'), 'production incumbent differs')
        winners, common = selected_profiles(profiles)
        candidate = winners[0]
        changed = identity(candidate) != identity(old)
        metrics = []
        for p, c in zip(profiles, winners):
            control = p['candidates']['original']
            point_winner = p['candidates'][p['raw_winner']]
            require(identity(control) == identity(old), 'profile incumbent differs')
            metrics.append(dict(
                point=p['point'], router=p['workload']['router'], raw_status=p['status'],
                raw_winner=p['raw_winner'], receipt_sha256=p['receipt_sha256'], source=p['source'],
                selected_arm=c['arm'], original_us=control['median_us'], selected_us=c['median_us'],
                delta_pct=100*(c['median_us']/control['median_us']-1),
                original_rounds_us=control['rounds'], selected_rounds_us=c['rounds'],
                original_spread_pct=control['spread_pct'], selected_spread_pct=c['spread_pct'],
                numerical_error=c['numerical_error'], components=c['components'],
                module_sha256=c['module_sha256'], build_key=c['build_key'],
                point_winner={f: point_winner[f] for f in
                              ('config', 'dequant_config', 'implementation', 'median_us')},
                selected_vs_point_winner_pct=100*(c['median_us']/point_winner['median_us']-1)))
            if p['status'] != 'PASS':
                fastest = min(p['candidates'].values(), key=lambda a: (a['median_us'], a['arm'] != 'original'))
                pending.append(dict(point=p['point'], key=list(k), router=p['workload']['router'],
                    status=p['status'], source=p['source'], receipt_sha256=p['receipt_sha256'],
                    baseline_us=control['median_us'], contender_us=fastest['median_us'],
                    potential_reduction_pct=100*(1-fastest['median_us']/control['median_us']),
                    contender={f: fastest[f] for f in ('config', 'dequant_config', 'implementation', 'arm')},
                    paired_wins=sum(x <= y for x, y in zip(fastest['rounds'], control['rounds']))))
        rows.append(dict(key=list(k), config=candidate['config'],
            dequant_config=candidate['dequant_config'], implementation=candidate['implementation'],
            changed=changed, status='CONFIRMED_REPLACEMENT' if changed else 'RETAIN_INCUMBENT',
            common_eligible=common, original_config=source['config'],
            original_dequant_config=source.get('dequant_config', -1), profiles=metrics,
            worst_profile_regret_pct=max(p['selected_vs_point_winner_pct'] for p in metrics),
            profile_tradeoffs=profile_tradeoffs(profiles, candidate) if len(profiles) > 1 else []))
    summary = dict(points=len(points), host_keys=len(rows), changed=sum(r['changed'] for r in rows),
        retained=sum(not r['changed'] for r in rows), pending_points=len(pending),
        pending_over_1pct=sum(p['potential_reduction_pct'] >= 1 for p in pending),
        q8_body_keys=sum(r['implementation']['kind'] == 'q8-address' for r in rows),
        kinds=dict(Counter(r['config']['kind'] for r in rows)),
        router_keys_not_each_profile_winner=sum(
            len(r['profiles']) > 1 and r['worst_profile_regret_pct'] > 0 for r in rows),
        router_keys_with_unmeasured_alternative=sum(
            any(p['status'] == 'NOT_MEASURED' for a in r['profile_tradeoffs'] for p in a['profiles'])
            for r in rows))
    result = dict(schema=SCHEMA, archives=ARCHIVES, sources=sources, rows=rows, pending=pending,
        baseline_sha256=digest(baseline), cpp_originals_sha256=digest(originals), scope=SCOPE,
        summary=summary, device='PPU-ZW810', compute_units=72, tokens=list(range(1, 9)),
        fallback='UNCHANGED_NO_UNMEASURED_RECIPE_PROMOTION', runtime_router_readback=False,
        active=False, production_changed=False, model_tpot_admitted=False, native_comparison=False,
        selection='ELIGIBLE_INTERSECTION_MIN_WORST_RELATIVE_COST_THEN_GEOMEAN_INCUMBENT_TIE',
        noisy_policy='RETAIN_RAW_UNSTABLE_INCUMBENT_NO_GATE_RELAXATION',
        frequency_normalization=False, big_m_changed=False)
    return result | dict(evidence_sha256=digest(result))


def verify(table):
    require(table['schema'] == SCHEMA and table['archives'] == ARCHIVES and
            table['evidence_sha256'] == digest({k: v for k, v in table.items() if k != 'evidence_sha256'}),
            'refresh table identity differs')
    require(table['scope'] == SCOPE and not table['active'] and not table['production_changed'] and
            not table['runtime_router_readback'] and not table['frequency_normalization'] and
            not table['big_m_changed'], 'refresh scope differs')
    seen, points = set(), set()
    for row in table['rows']:
        k = tuple(row['key'])
        require(len(k) == 9 and k not in seen and 1 <= k[7] <= 8 and
                k[0] in (8, 10, 11, 12, 13, 14) and
                ((k[1] == 0 and k[4:7] == (1, 1, 1) and k[-1] == 0) or
                 (k[1] == 2 and k[4:6] == (256, 8) and k[6] in (1, 8) and k[-1] == 1)),
                'invalid/duplicate runtime key')
        seen.add(k)
        changed = (semantic(row['config']) != semantic(row['original_config']) or
                   row['dequant_config'] != row['original_dequant_config'] or
                   row['implementation'] != dict(kind='shipping'))
        require(row['changed'] == changed and row['status'] ==
                ('CONFIRMED_REPLACEMENT' if changed else 'RETAIN_INCUMBENT'),
                'retained/replacement identity differs')
        body = row['implementation']
        if body['kind'] != 'shipping':
            c = row['config']
            require(body['kind'] == 'q8-address' and type(body.get('body')) is int and
                    body['body'] in (0, 1, 3, 4, 5) and k[0:2] == (8, 0) and
                    c['kind'] == 'simt' and c['variant'] == 5 and c['values'] == 4 and
                    c['split'] == 1 and (c['columns'], c['warps']) in ((4, 8), (8, 4)),
                    'unmeasured Q8 implementation domain')
        for p in row['profiles']:
            require(p['point'] not in points, 'duplicate profile point')
            points.add(p['point'])
            require(math.isfinite(p['numerical_error']) and 0 <= p['numerical_error'] < .005 and
                    all(math.isfinite(p[f]) and p[f] > 0 for f in ('original_us', 'selected_us')) and
                    len(p['original_rounds_us']) == len(p['selected_rounds_us']) == 6,
                    'invalid profile proof')
            if row['changed']:
                require(p['raw_status'] == 'PASS' and p['selected_us'] <= p['original_us'] and
                        p['original_spread_pct'] <= 5 and p['selected_spread_pct'] <= 5 and
                        sum(x <= y for x, y in zip(p['selected_rounds_us'], p['original_rounds_us'])) >= 4,
                        'replacement lacks paired evidence')
    require(len(points) == table['summary']['points'] and len(seen) == table['summary']['host_keys'],
            'table coverage differs')
    require(len(points) == 1114 and len(seen) == 930 and
            table['summary']['changed'] == sum(r['changed'] for r in table['rows']) and
            table['summary']['retained'] == sum(not r['changed'] for r in table['rows']) and
            {p['point'] for p in table['pending']} == {p['point'] for r in table['rows']
                                                     for p in r['profiles'] if p['raw_status'] != 'PASS'},
            'summary or pending scope differs')


def readiness(table, current, *, q8_bindings=()):
    """Structural capability audit, not an execution or new-device admission."""
    verify(table)
    require(table['baseline_sha256'] == digest(current), 'current runtime catalog differs from measured incumbent')
    from quactlize.execution.q4_decode_codegen import recipes
    q4 = recipes()
    q8 = {(tuple(r['key']), r['body'], tuple(r['config'][f] for f in FIELDS)) for r in q8_bindings}
    pending, requirements = [], defaultdict(set)
    for row in table['rows']:
        k, c = row['key'], row['config']
        if row['implementation']['kind'] == 'q8-address' and (
                tuple(k), row['implementation']['body'], tuple(c[f] for f in FIELDS)) not in q8:
            pending.append(dict(key=k, reason='Q8_BODY_REQUIRES_EXPLICIT_LAUNCHER_BINDING',
                                implementation=row['implementation'], config=c))
        if c['kind'] == 'simt':
            require(Config(**{f: c[f] for f in FIELDS}) in runtime_inventory(k[0]), 'SIMT geometry absent')
            requirements['simt'].add((k[0], k[-1], *(c[f] for f in FIELDS)))
        elif c['kind'] == 'q4':
            recipe = tuple(c[f] for f in ('reader', 'variant', 'warps', 'values', 'columns'))
            requirements['q4'].add((k[2], k[3], k[-1], *recipe))
            if recipe not in {tuple(r) for r in q4.get((k[2], k[3]), [])}:
                pending.append(dict(key=k, reason='Q4_SPECIALIZATION_REQUIRES_GENERATION', config=c))
        else:
            require(c['kind'] == 'tc', 'unexpected decode kind')
            requirements['tc'].add((c['symbol'], k[-1], k[1] == 0))
    return dict(status='IMPLEMENTATION_REQUIRED' if pending else 'STRUCTURALLY_READY_NOT_DEVICE_ADMITTED',
                pending=pending, requirements={k: sorted(v) for k, v in requirements.items()},
                production_changed=False, kernel_launches=0)


def supplement_plan(table):
    """Only missing pairings of existing winners with otherwise eligible profiles.

    Already-measured regressions/noisy alternatives cannot justify more GPU
    work merely because another profile is missing. Fresh candidate/control
    pairs are needed; old absolute times cannot be used as the new control.
    """
    verify(table)
    cells = []
    for row in table['rows']:
        for alternative in row['profile_tradeoffs']:
            states = {p['status'] for p in alternative['profiles']}
            if 'NOT_MEASURED' not in states or not states <= {'NOT_MEASURED', 'ELIGIBLE'}:
                continue
            for p in alternative['profiles']:
                if p['status'] != 'NOT_MEASURED':
                    continue
                cells.append(dict(key=row['key'], point=p['point'], router=p['router'],
                    candidate={k: alternative[k] for k in ('config', 'dequant_config', 'implementation')},
                    incumbent=dict(config=row['original_config'],
                                   dequant_config=row['original_dequant_config'],
                                   implementation=dict(kind='shipping'))))
    return dict(schema='quactlize.decode-refresh-profile-gaps.v1',
                selection=table['evidence_sha256'], cells=cells,
                counts=dict(host_keys=len({tuple(c['key']) for c in cells}),
                            points=len({c['point'] for c in cells}), candidate_profile_cells=len(cells)),
                scope='UNMEASURED_PAIRINGS_OF_EXISTING_CANDIDATES_NOT_A_FULL_SWEEP',
                contemporaneous_incumbent_required=True, production_changed=False,
                kernel_launches=0, executable=False)
