"""Review frozen confirmation receipts without rewriting their original verdict.

Qualification is replayed from the leave-N/K-out generator, not inferred from
who won. A helper retained as a control can be in that same generated domain.
Only *measured complete compositions* may be selected; no synthetic pairing.
"""
from collections import Counter
import json
import math
from pathlib import PurePosixPath
import statistics
import tarfile

from dev.strategy.components import statistics_checked
from dev.strategy.prefill_closure import read_compact, require
from dev.strategy.prescreen import Prescreen
from dev.strategy.space import factored_recall
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ARCHIVE_SHA256 = 'b43e1d28bada3d28d15aed2ebd5e92d693c63cb4429924e24ab79fbf5d5ce8fb'
PLAN_SHA256 = '7cdf365ac87435117fb6d10bfbc19eaa3cea5d6ab13d3b7fabdfa56bb7d1a103'


def choose(point, measured, qualified):
    """Fastest feasible finalist, not fastest first and then fail the point.

    Candidate stability, zero median regression and four of six paired wins
    are unchanged. A noisy control is usable only when its *best* round is
    slower than the candidate's *worst* round. Overlapping noisy ranges still
    require another device experiment. This is a component gate, not model AB.
    """
    stats = measured['statistics']
    controls = [a for a, v in point['arms'].items() if 'control' in v['roles'] or v['original']]
    require(controls and point['original'] in controls, 'missing original/control')
    best_control = min(controls, key=lambda a: (stats[a]['median_us'], a))
    against = sorted({best_control, point['original']})
    eligible = {}
    for aid in sorted(qualified):
        a = stats[aid]
        if a['spread_pct'] > 5:
            continue
        comparisons = {}
        for cid in against:
            c = stats[cid]
            paired = sum(x <= y * (1 + 1e-12) for x, y in zip(a['rounds_us'], c['rounds_us']))
            separated = max(a['rounds_us']) < min(c['rounds_us'])
            if (a['median_us'] > c['median_us'] * (1 + 1e-12) or paired < 4 or
                    (c['spread_pct'] > 5 and not separated)):
                break
            comparisons[cid] = dict(paired_wins=paired, rounds=6,
                control_spread_pct=c['spread_pct'],
                stability_basis='SEPARATED_OBSERVED_RANGES' if c['spread_pct'] > 5 else 'STABLE_CONTROL')
        else:
            eligible[aid] = comparisons
    require(eligible, 'no generated stable non-regressing finalist; device confirmation required')
    winner = min(eligible, key=lambda a: (stats[a]['median_us'], a))
    return dict(winner=winner, best_control=best_control, comparisons=eligible[winner],
                eligible=sorted(eligible), qualified=sorted(qualified))


def component(row, tables):
    require(row['status'] == 'PASS' and row['phase'] == 'confirmation' and
            row['first_use_excluded'] and row['host_launch_cost_added'] is False,
            'component status/warmup/launch contract differs')
    checked = statistics_checked(row['samples_us'])
    require(all(row[k] == checked[k] for k in ('median_us', 'spread_pct')),
            'component statistics differ from raw samples')
    node = dict(tables['kernel'][row['node']])
    node.update({k: tables['symbol'][node[k]] for k in ('name', 'symbol') if k in node})
    identity = dict(schema='quactlize.isolated-component.v1', authority=row['authority'],
        implementation=row['implementation'], node=node, kind=row['kind'], cache=row['cache'],
        contract=tables['component_contract'][row['contract']])
    if row.get('timing_method') == 'CUDA_EVENTS_ORDINARY_CALL':
        identity.update(schema='quactlize.provider-event-component.v1', cuda_graph=False, profiler=False,
                        submission_frontend='CPP_PREBOUND_CONTINUOUS',
                        phase={'adapter': 'input', 'provider': 'provider', 'postprocess': 'output'}[row['kind']])
    require(digest(identity) == row['id'], 'component identity differs')
    return row


def audit_point(point, measured, rounds, components, authority_sha):
    require(measured['point'] == point['id'] and measured['workload'] == point['workload'] and
            measured['recipes'] == {a: v['recipe'] for a, v in point['arms'].items()},
            'result workload/recipe differs')
    require(len(rounds) == 6 and {r['round'] for r in rounds} == set(range(6)), 'six fresh rounds required')
    costs = {a: [] for a in point['arms']}
    max_error, checks = 0., 0
    for row in sorted(rounds, key=lambda r: r['round']):
        idx = row['round']
        expected_authority = dict(sha256=authority_sha, round=idx, point=point['id'])
        require(row['authority'] == expected_authority, 'confirmation authority differs')
        order = sorted(row['jobs'], reverse=bool(idx % 2))
        require(row['order'] == order, 'confirmation order did not alternate')
        used = set()
        for job in row['jobs'].values():
            proof = job['accuracy']
            error = proof.get('relative_l1_error_max', 0.)
            require(proof['correctness'] == 'PASS' and proof['checks'] > 0 and
                    proof.get('bad_max', 0) == proof.get('nonfinite_max', 0) == 0 and
                    math.isfinite(error) and 0 <= error < .005 and job['components'],
                    'independent numerical proof failed')
            max_error = max(max_error, error); checks += proof['checks']
            require(len(job['components']) == len(set(job['components'])), 'component counted twice in one job')
            for cid in job['components']:
                require(components[cid]['authority'] == expected_authority,
                        'timing reused from another round')
                used.add(cid)
        composed = set()
        for aid in point['arms']:
            parts = measured['components'][aid][idx]
            ids = [c['id'] for c in parts]
            require(ids and len(ids) == len(set(ids)) and set(ids) <= used, 'unbound/double-counted composition')
            # Every selected stage must be an entire measured job, not just its
            # producer with an inconvenient reducer or adapter removed.
            for job in row['jobs'].values():
                jj = set(job['components'])
                # Identical adapters can legitimately be shared by TC jobs.
                # The entire body job must occur; shared adapters alone do not
                # make a different producer part of this composition.
                bodies = {c for c in jj if components[c]['kind'] in ('producer', 'provider')}
                require(not bodies.intersection(ids) or jj <= set(ids), 'partial job in composition')
            require(all(part == dict(id=part['id'], kind=components[part['id']]['kind'],
                        median_us=components[part['id']]['median_us']) for part in parts),
                    'composition differs from raw component')
            costs[aid].append(math.fsum(c['median_us'] for c in parts)); composed.update(ids)
        require(composed == used, 'confirmation job/composition denominator differs')
    for aid, values in costs.items():
        stats = dict(median_us=statistics.median(values), rounds_us=values,
                     spread_pct=100 * (max(values) / min(values) - 1))
        require(measured['statistics'][aid] == stats, 'point cost differs from raw component sums')
    return dict(checks=checks, relative_l1_error_max=max_error, bad=0, nonfinite=0)


def review(archive, table):
    require(sha(archive) == ARCHIVE_SHA256, 'unreviewed confirmation archive')
    selector = Prescreen(table)
    result, authorities, pci = [], [], set()
    with tarfile.open(archive, 'r:gz') as tar:
        names = set()
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts and
                    member.name not in names, 'unsafe/duplicate confirmation member')
            names.add(member.name)
        def read(name):
            return json.load(tar.extractfile(name))
        plan = read('finalists.json')
        require(plan['plan_sha256'] == PLAN_SHA256 == digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
                and plan['catalog'] == table.identity and plan['rule_revision'] == 27,
                'frozen plan/catalog differs')
        points = {p['id']: p for p in plan['points']}
        for shard in range(7):
            authority = read(f'shard-{shard}/authority.json'); auth = digest(authority)
            summary = read(f'shard-{shard}/summary.json')
            require(summary['authority'] == auth and authority['plan'] == PLAN_SHA256 and
                    authority['previous_timings_reused'] is False and authority['idle'] == 'OPERATOR_CONFIRMED' and
                    authority['protocol'] == plan['protocol'] and
                    authority['device']['compute_units'] == 72 and authority['l2']['bytes'] == 67108864,
                    'device/confirmation authority differs')
            require(authority['device']['pci'] not in pci, 'duplicate physical device')
            pci.add(authority['device']['pci'])
            compact, = [n for n in names if n.startswith(f'shard-{shard}/') and n.endswith('.xz')]
            tables, groups = read_compact(tar.extractfile(compact))
            cc = {r['id']: component(r, tables) for r in groups['component']}
            require(len(cc) == len(groups['component']), 'duplicate component')
            rows = groups['prefill_confirmation_round']
            shard_points = [p for p in points.values() if p['shard'] == shard]
            require(len(rows) == 6 * len(shard_points) and summary['expected'] == len(shard_points),
                    'shard/round denominator differs')
            for point in sorted(shard_points, key=lambda p: p['id']):
                pid = point['id']; w = Workload(**point['workload'])
                require(w.key == pid, 'point identity differs')
                measured = read(f'shard-{shard}/results/{pid}.json')
                accuracy = audit_point(point, measured, [r for r in rows if r['point'] == pid], cc, auth)
                generated, report = selector.select(w, blind=True, per_stratum=2, stencil='warps')
                require(report['champion_injection'] is False and report['rule_revision'] == 27,
                        'blind rule contract changed')
                qualified = [a for a, v in point['arms'].items() if
                    factored_recall(w, generated, [Recipe.from_dict(v['recipe'])])['status'] == 'PASS']
                decision = choose(point, measured, qualified); aid = decision['winner']
                parts = measured['components'][aid]
                expansion = [math.fsum(c['median_us'] for c in rr if c['kind'] == 'expansion') for rr in parts]
                core = [math.fsum(c['median_us'] for c in rr if c['kind'] != 'expansion') for rr in parts]
                result.append(dict(point=pid, workload=point['workload'], shard=shard,
                    raw_status=measured['status'], raw_winner=measured['winner'],
                    raw_roles={a: v['roles'] for a, v in point['arms'].items()},
                    original=point['original'], decision=decision, recipe=point['arms'][aid]['recipe'],
                    statistics=measured['statistics'], component_rounds=parts, accuracy=accuracy,
                    gemm_us=statistics.median(core), dequant_us=statistics.median(expansion),
                    result_sha256=digest(measured), round_receipts=[r['record_sha256'] for r in
                        sorted((r for r in rows if r['point'] == pid), key=lambda r: r['round'])]))
            authorities.append(dict(shard=shard, sha256=auth, device=authority['device'],
                sdk=authority['sdk'], body_source=authority['body_source'],
                dequant=authority['dequant'], helpers=authority['helpers'],
                providers_sha256=digest(authority['providers'])))
            print(f'PREFILL_REVIEW shard={shard} points={len(result)}/430', flush=True)
    require(len(result) == len({r['point'] for r in result}) == 430, 'point denominator differs')
    changed = sum(r['raw_winner'] != r['decision']['winner'] for r in result)
    return dict(schema='quactlize.prefill-confirmed.v1', archive_sha256=ARCHIVE_SHA256,
        plan_sha256=PLAN_SHA256, catalog=table.identity, rule_revision=27, authorities=authorities,
        points=sorted(result, key=lambda r: r['point']),
        summary=dict(points=430, confirmed=430, reviewed_decisions_changed=changed,
                     raw_status=dict(Counter(r['raw_status'] for r in result)),
                     routes=dict(Counter(r['recipe']['route'] for r in result)),
                     max_relative_l1_error=max(r['accuracy']['relative_l1_error_max'] for r in result)),
        rule_proof='R27_LEAVE_ENTIRE_NK_OUT_NO_CHAMPION_INJECTION', legacy_fallback=False,
        timing='SEPARATE_COMPONENTS_CUBLAS_ORDINARY_EVENTS',
        scope='CONFIRMED_COMPONENT_SELECTION_NOT_MODEL_AB', model_performance_admitted=False)
