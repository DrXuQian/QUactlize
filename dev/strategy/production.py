"""Offline admission of reviewed confirmations, never a runtime tuner.

Router profiles are evidence, not host-visible keys. A replacement must be
confirmed and no slower than the incumbent in every tested profile of its key.
"""
from collections import Counter, defaultdict
import hashlib
import io
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

from dev.strategy.followup import paired_verdict, validate_confirmation
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest
from tools.pack_kpack_strategy_review import Review

ARCHIVES = {
    'screen': '4d78e5c015b790e96cc6a204eb86c542437a44b5e959823d5f5605f36b928dab',
    'retry': '0fce9389c80ebb4412e714e5e4f8f75dd12d5dc90ced59fe99eebdf2b6c6d782',
    'q8': '585b35284c6c0a1c09d37520dc73b5d15dda49ebacbd267e0cf24c52766af902',
}


def serialize(evidence):
    """One auditable record per line; do not turn a table into 60k diff lines."""
    metadata = {k: v for k, v in evidence.items() if k != 'rows'}
    prefix = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False)
    rows = ',\n'.join('    '+json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False)
                     for r in evidence['rows'])
    return prefix[:-2]+',\n  "rows": [\n'+rows+'\n  ]\n}\n'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(path, role):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(block)
    require(hasher.hexdigest() == ARCHIVES[role], 'unreviewed '+role+' archive')
    return path


def key(w):
    return (w['qtype'], 2 if w['operation'] == 'indexed' else 0, w['n'], w['k'],
            w['experts'], w['topk'], w['channels'], w['m'], int(w['compute'] == 'bf16'))


def config(recipe, w):
    r = Recipe.from_dict(recipe)
    require(r.eligible(Workload(**w)), 'ineligible confirmed recipe')
    p = dict(r.parameters)
    if r.kind != 'tc':
        return dict(kind=r.kind, **p)
    p.pop('dequant_config', None)
    p['parent_persistent'] = p.pop('persistent')
    p['route'] = int(r.route == 'sf') + (2 if w['operation'] == 'indexed' else 0)
    return dict(kind='tc', **p)


def semantic(c):
    # Parent names are aliases, not a reason to compare different bodies.
    # All codegen-affecting geometry/precision/operator fields remain in key.
    return digest({k: v for k, v in c.items() if k not in ('symbol', 'persistent')})


def sf_config(node):
    name, block = node['name'], node['block'][0]
    if 'prepass_unit_kernel<' in name:
        return 0
    if 'sf_unit16<' in name:
        require(block in (128, 256), 'unknown SF unit16 geometry')
        return 4 if block == 128 else 5
    if 'sf_columns<' in name:
        if ', 16>' in name:
            require(block == 256, 'unknown SF columns16 geometry')
            return 1
        require(', 32>' in name and block in (128, 256), 'unknown SF columns geometry')
        return 2 if block == 256 else 3
    if 'sf_bfloat' in name:
        require(block in (128, 256), 'unknown BF16 SF geometry')
        return 0 if block == 128 else 5
    raise ValueError('unrecognized measured SF implementation: '+name)


def validate_component_decision(components, arms, roles, decision):
    """Allow only sum roundoff between Python versions, not performance slack.

    Python 3.12 sums floats differently from 3.10. Verify each recorded sum
    against its raw positive component samples within the rounding bound,
    then reproduce the ENTIRE verdict exactly from those recorded sums.
    Never change a raw sample, winner, threshold or recorded decision.
    """
    require(set(decision['statistics']) == set(arms), 'decision arm coverage differs')
    costs = {}
    for aid, ids in arms.items():
        require(bool(ids), 'empty component call')
        rounds = decision['statistics'][aid]['samples_us']
        require(len(rounds) == 6 and all(len(r) == 5 for r in rounds), 'wrong summed sample count')
        for i, values in enumerate(rounds):
            for j, value in enumerate(values):
                terms = [components[cid]['samples_us'][i][j] for cid in ids]
                require(all(math.isfinite(t) and t > 0 for t in terms), 'invalid component summand')
                exact = math.fsum(terms)
                require(math.isfinite(value) and value > 0 and
                        abs(value-exact) <= 2*len(terms)*math.ulp(exact),
                        'decision sum differs from raw components beyond floating-point roundoff')
        costs[aid] = rounds
    require(paired_verdict(costs, roles) == decision, 'decision differs from recorded component sums')


class Capture:
    def __init__(self):
        self.tables = defaultdict(dict)
        self.points, self.confirmations, self.components = {}, {}, {}
        self.run = None

    def proof(self, proof, w):
        a = proof['accuracy']
        require(a['correctness'] == 'PASS' and a['checks'] == (6 if w['compute'] == 'bf16' else 5),
                'numerical repeat coverage differs')
        require(a['cells'] == [w['m']*w['topk']*w['n']] and a['tolerance'] == [.005],
                'numerical output denominator differs')
        require(a['bad_max'] == a['nonfinite_max'] == 0 and
                math.isfinite(a['relative_l1_error_max']) and 0 <= a['relative_l1_error_max'] < .005,
                'failed numerical proof')
        planes = ['low', 'high', 'units'] if w['qtype'] in (11, 13, 14) else ['low', 'units']
        require(a['pointer_negatives'] == planes and a['zero_input'] == 'PASS' and
                a['omitted_producer'] == 'DETECTED', 'missing numerical negative controls')
        fixture = self.tables['fixture'][proof['fixture']]
        require(fixture['l2_bytes'] == self.run['authority']['l2']['bytes'] and
                fixture['active_ring_bytes'] == fixture['copies']*fixture['minimum_read_bytes'] and
                fixture['active_ring_bytes'] >= 2.25*fixture['l2_bytes'], 'invalid cold weight ring')
        receipt = self.tables['execution'][proof['execution']]
        images = {v for group in self.run['images'].values() for v in group.values()}
        require(receipt['module_sha256'] in images, 'unbound executed image')
        return receipt

    def consume(self, r):
        t = r['type']
        if t == 'run':
            require(self.run is None, 'duplicate run')
            self.run = r
            dev = r['authority']['resource']['device']
            require(dev['compute_units'] == 72 and dev['warp_size'] == 32, 'not the reviewed ZW810 cohort')
        elif t == 'point':
            require(r['id'] not in self.points, 'duplicate point')
            w = r['workload']
            require(Workload(**w).key == r['id'] and w['storage'] == 'f32' and
                    w['compute'] == ('f16' if w['operation'] == 'dense' else 'bf16') and
                    w['m'] <= 8 and w['fusion'] == 'none', 'unexpected workload contract')
            self.points[r['id']] = w
        elif t == 'definition':
            table = self.tables[r['table']]
            require(r['index'] not in table, 'duplicate definition')
            table[r['index']] = r['value']
        elif t == 'component' and r['phase'] == 'confirmation':
            require(r['authority']['sha256'] == self.run['authority_sha256'] and
                    r['implementation'] == self.run['authority']['body_source'], 'mixed component authority')
            samples = r['samples_us']
            require(len(samples) == 6 and all(len(v) == 5 for v in samples), 'confirmation sample count differs')
            require(all(math.isfinite(x) and x > 0 for v in samples for x in v), 'invalid timing')
            medians = list(map(statistics.median, samples))
            require(r['median_us'] == statistics.median(medians) and
                    r['spread_pct'] == 100*(max(medians)/min(medians)-1) and
                    r['timed_kernel_types'] == 1 and r['first_use_excluded'] and
                    r['host_launch_cost_added'] is False and r['calls_per_graph'] >= 16,
                    'invalid isolated timing contract')
            self.components[r['point'], r['id']] = r
        elif t == 'confirmation':
            self.confirmation(r)
        elif t == 'export_issue':
            raise ValueError('incomplete export: '+repr(r))

    def confirmation(self, r):
        pid = r['point']; w = self.points[pid]
        require(pid not in self.confirmations, 'duplicate confirmation')
        require(r['authority']['sha256'] == self.run['authority_sha256'], 'mixed confirmation authority')
        result = dict(point=pid, workload=w, source=self.run['authority_sha256'],
                      record_sha256=r['record_sha256'], status=r['status'], candidates={})
        self.confirmations[pid] = result
        if r['journal_status'] != 'PASS':
            return
        if r['status'] == 'INCUMBENT_UNCHANGED':
            self.proof(r['original_proof'], w)
            require(r['winner'] == 'original', 'invalid unchanged decision')
            return
        require(r['protocol'] == dict(rounds=6, samples=5, order='FORWARD_REVERSE_ALTERNATING',
                cache='SEPARATE_METADATA_AND_PRODUCER_RINGS_AT_LEAST_2P25_L2'), 'wrong confirmation protocol')
        require(set(r['arms']) == set(r['roles']) == set(r['arm_components']), 'confirmation arm coverage differs')
        validate_component_decision({cid:c for (point,cid),c in self.components.items() if point==pid},
                                    r['arm_components'], r['roles'], r['decision'])
        result['status'] = r['decision']['status']
        result['winner'] = r['decision']['winner']
        for aid, proof in r['arms'].items():
            receipt = self.proof(proof, w)
            ids = r['arm_components'][aid]
            producer = [self.components[pid, cid] for cid in ids if self.components[pid, cid]['kind'] == 'producer']
            require(len(producer) == 1, 'wrong producer denominator')
            contract = self.tables['component_contract'][producer[0]['contract']]
            c = config(contract['recipe'], w)
            if 'kind' in receipt:
                require(c['kind'] == receipt['kind'], 'recipe and executed kind differ')
            else:
                require(c['kind'] == 'tc' and w['operation'] == 'dense' and
                        receipt['source'] == 'SEALED_RESOURCE_MODULE_NOT_DISPATCHER' and
                        receipt['compute'] == w['compute'] and receipt['storage'] == 'F32' and
                        receipt['endpoint'] == 'typed', 'wrong dense TC endpoint')
            expansion = [self.components[pid, cid] for cid in ids if self.components[pid, cid]['kind'] == 'expansion']
            c['dequant_config'] = -1
            if expansion:
                require(len(expansion) == 1 and c['kind'] == 'tc' and c['route'] % 2 == 1,
                        'unexpected expansion')
                node = dict(self.tables['kernel'][expansion[0]['node']])
                node['name'] = self.tables['symbol'][node['name']]
                c['dequant_config'] = sf_config(node)
                fixture = self.tables['fixture'][r['expansion_fixture']]
                require(fixture['active_ring_bytes'] >= 2.25*fixture['l2_bytes'] and
                        fixture['domain'] == 'ALL_EXPERT_METADATA', 'invalid metadata ring')
                require(r['expansion_proofs'], 'missing expansion correctness')
                for values in r['expansion_proofs'].values():
                    require(len(values) == 2 and all(len(pair) == 2 and all(v['bad'] == 0 for v in pair)
                                                   for pair in values), 'bad expansion oracle')
            stats = r['decision']['statistics'][aid]
            result['candidates'][aid] = dict(config=c, roles=r['roles'][aid],
                median_us=stats['median_us'], rounds=stats['round_medians_us'], spread_pct=stats['spread_pct'],
                error=proof['accuracy']['relative_l1_error_max'],
                components=[dict(kind=self.components[pid, cid]['kind'],
                                 median_us=self.components[pid, cid]['median_us']) for cid in ids],
                execution=receipt)


def read_review(path, role):
    capture = Capture(); body = hashlib.sha256(); counts = Counter(); footer = False
    with lzma.open(checked_path(path, role), 'rb') as stream:
        for line in stream:
            require(not footer, 'trailing review records')
            r = json.loads(line)
            if r['type'] == 'end':
                require(r['prior_lines_sha256'] == body.hexdigest() and r['counts'] == dict(counts) and
                        r['export_status'] == 'COMPLETE', 'review footer mismatch')
                footer = True
            else:
                body.update(line); counts[r['type']] += 1; capture.consume(r)
    require(footer and len(capture.confirmations) == len(capture.points), 'incomplete review')
    return capture


def add_retries(capture, path):
    names = set(); records = []
    with tarfile.open(checked_path(path, 'retry')) as source:
        for member in source:
            p = PurePosixPath(member.name)
            require(not p.is_absolute() and '..' not in p.parts and member.name not in names and
                    member.isfile(), 'unsafe retry member')
            names.add(member.name)
            if member.name.endswith('/current.json'):
                row = json.load(source.extractfile(member))
                require(row['record_sha256'] == digest({k: v for k, v in row.items() if k != 'record_sha256'}),
                        'retry receipt changed')
                validate_confirmation(row['data']); records.append(row)
    require(len(records) == 6, 'retry denominator differs')
    for row in records:
        pid = row['identity']['point']
        require(capture.confirmations[pid]['status'] == 'FAIL', 'retry replaces a successful point')
        # Reuse the compact exporter without extracting or altering the archive.
        stream = io.BytesIO(); writer = Review(Path('/unused'), stream)
        writer.arm(row)
        retry = Capture(); retry.run = capture.run; retry.points[pid] = capture.points[pid]
        for line in stream.getvalue().splitlines():
            retry.consume(json.loads(line))
        capture.confirmations[pid] = retry.confirmations[pid]


def load(screen, retry, q8):
    main = read_review(screen, 'screen')
    require(len(main.points) == 1114, 'full small-M denominator differs')
    add_retries(main, retry)
    new = read_review(q8, 'q8')
    require(len(new.points) == 12 and all(w['qtype'] == 8 and w['operation'] == 'indexed' and
            w['n'] == 1024 and w['k'] == 2048 for w in new.points.values()), 'wrong Q8 follow-up scope')
    require(all(main.points.get(pid) == w for pid, w in new.points.items()), 'Q8 workload differs')
    main.confirmations.update(new.confirmations)
    require(all(r['status'] in ('PASS', 'INCUMBENT_UNCHANGED', 'UNSTABLE_RETAIN_ORIGINAL')
                for r in main.confirmations.values()), 'unresolved small-M point')
    return list(main.confirmations.values()), [main.run, new.run]


def core_semantic(c):
    return semantic({k: v for k, v in c.items() if k != 'dequant_config'})


def no_regression(candidate, original):
    if candidate == original:
        return True
    return (candidate['median_us'] <= original['median_us'] and
            candidate['spread_pct'] <= 5 and original['spread_pct'] <= 5 and
            sum(a <= b for a, b in zip(candidate['rounds'], original['rounds'])) >= 4)


def fit(points, runs, baseline):
    """Produce one host-key choice without inventing a GPU-router readback."""
    original = {tuple(r['key']): r for r in baseline['exact']}
    groups = defaultdict(list)
    for row in points:
        groups[key(row['workload'])].append(row)
    rows = []
    for request, profiles in sorted(groups.items()):
        require(request in original, 'new evidence has no original production row')
        base = original[request]
        require(len({r['source'] for r in profiles}) == 1, 'mixed device/cohort within a host key')
        require(len({r['workload']['router'] for r in profiles}) == len(profiles), 'duplicate router profile')
        known = []
        for r in profiles:
            if not r['candidates']:
                known.append({}); continue
            old = r['candidates']['original']
            require(core_semantic(base['config']) == core_semantic(old['config']), 'original wiring changed')
            eligible = {}
            for aid, c in r['candidates'].items():
                if aid != 'original' and ('rule' not in c['roles'] or r['status'] != 'PASS'):
                    continue
                if no_regression(c, old):
                    ck = semantic(c['config'])
                    if ck not in eligible or c['median_us'] < eligible[ck]['median_us']:
                        eligible[ck] = dict(c, arm=aid)
            known.append(eligible)
        common = set.intersection(*(set(k) for k in known)) if all(known) else set()
        selected = None
        if common:
            def score(ck):
                ratios = [c[ck]['median_us']/r['candidates']['original']['median_us'] for c, r in zip(known, profiles)]
                # Runtime cannot see profile. Protect worst relative cost,
                # then the geometric mean. Incumbent wins exact ties.
                incumbent = all(c[ck]['arm'] == 'original' for c in known)
                return max(ratios), sum(math.log(v) for v in ratios), not incumbent, ck
            selected = min(common, key=score)
        cfg = dict(base['config']); dequant = -1
        metrics = []
        for index, r in enumerate(profiles):
            candidate = known[index][selected] if selected else r['candidates'].get('original')
            if candidate:
                if selected:
                    cfg = dict(candidate['config'])
                old = r['candidates']['original']
                metrics.append(dict(point=r['point'], router=r['workload']['router'],
                    original_us=old['median_us'], selected_us=candidate['median_us'],
                    delta_pct=100*(candidate['median_us']/old['median_us']-1),
                    selected_arm=candidate.get('arm', 'original'),
                    spread_pct=candidate['spread_pct'],
                    components=candidate['components'], error=candidate['error'],
                    record_sha256=r['record_sha256']))
            else:
                metrics.append(dict(point=r['point'], router=r['workload']['router'],
                    status=r['status'], selected_arm='original', record_sha256=r['record_sha256']))
        dequant = cfg.pop('dequant_config', -1)
        # Canonical producer dictionaries do not carry redundant alias fields.
        cfg.pop('persistent', None)
        base_cfg = dict(base['config']); base_cfg.pop('persistent', None)
        changed = core_semantic(cfg) != core_semantic(base_cfg) or dequant > 0
        # No measured replacement: preserve every legacy field/policy exactly.
        if not changed:
            cfg = base_cfg; dequant = -1
        rows.append(dict(key=list(request), config=cfg, dequant_config=dequant,
            original=base_cfg, changed=changed,
            status='CONFIRMED_COMPONENT_REPLACEMENT' if changed else 'RETAIN_ORIGINAL',
            reason=('ALL_PROFILES_CONFIRMED_NO_REGRESSION' if selected else
                    'RETAIN_UNSTABLE_OR_NO_COMMON_CONFIRMED_IMPROVEMENT'),
            source=profiles[0]['source'], profiles=metrics))
    result = dict(schema='quactlize.component-decode.v1', archives=ARCHIVES,
        baseline_sha256=digest(baseline), rows=rows,
        sources=[dict(authority_sha256=r['authority_sha256'], authority=r['authority'], images=r['images']) for r in runs],
        summary=dict(points=len(points), host_keys=len(rows), changed=sum(r['changed'] for r in rows),
                     retained=sum(not r['changed'] for r in rows)),
        scope='ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT',
        device='PPU-ZW810', model_admission='PENDING_ORIGINAL_WIRING_AB',
        runtime_router_readback=False, small_m_full_dequant=False)
    return result | dict(evidence_sha256=digest(result))


def apply(evidence, baseline):
    """Flatten admitted replacements into the one existing runtime catalog."""
    import copy
    require(evidence['schema'] == 'quactlize.component-decode.v1' and
            evidence['archives'] == ARCHIVES, 'component evidence authority differs')
    require(evidence['evidence_sha256'] == digest({k: v for k, v in evidence.items()
            if k != 'evidence_sha256'}), 'component evidence was edited; recapture the reviewed archives')
    require(evidence['baseline_sha256'] == digest(baseline), 'component incumbent catalog changed')
    require(evidence['scope'] == 'ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT' and
            not evidence['runtime_router_readback'] and not evidence['small_m_full_dequant'],
            'component admission scope differs')
    result = copy.deepcopy(baseline)
    by_key = {tuple(r['key']): r for r in result['exact']}
    seen = set()
    for r in evidence['rows']:
        k = tuple(r['key'])
        require(k not in seen and k in by_key, 'duplicate or unknown component host key')
        seen.add(k)
        row = by_key[k]
        require(core_semantic(row['config']) == core_semantic(r['original']), 'component incumbent differs')
        if not r['changed']:
            require(r['dequant_config'] == -1 and
                    core_semantic(r['config']) == core_semantic(r['original']), 'changed retained row')
            continue
        require(r['status'] == 'CONFIRMED_COMPONENT_REPLACEMENT' and r['profiles'] and
                all(p['selected_arm'] != 'original' and p['selected_us'] <= p['original_us'] and
                    p['spread_pct'] <= 5 and 0 <= p['error'] < .005 for p in r['profiles']),
                'replacement lacks confirmed no-regression evidence')
        dq = r['dequant_config']
        require(dq == -1 or (r['config']['kind'] == 'tc' and r['config']['route'] == 1 and
                            k[0] in (12, 13) and dq in range(6)), 'unsupported measured SF helper')
        row.update(config=r['config'], policy=18, dequant_config=dq, status=r['status'])
        # Buckets are proposals on unseen shapes. A Q4 S1 body bakes in N/K;
        # keep its old generic guard as bucket donor, never launch it elsewhere.
        if r['config']['kind'] != 'q4':
            row['guard'] = r['config']
    return result
