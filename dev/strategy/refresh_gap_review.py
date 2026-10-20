"""Close the reviewed M8 profile gaps without rewriting the original scan.

Every shortlisted alternative has a fresh paired regression. This receipt
rejects universal replacements; it does not choose a router-specific kernel,
admit the other 160 proposals, or promote an experimental body to production.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import tarfile

from dev.strategy.production import require
from dev.strategy.refresh_gaps import SELECTION, make_plan, opportunities, validate_producers
from dev.strategy.refresh_selection import allowed, identity, sealed, supplement_plan, validate_point
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT/'policies/kpack_decode_profile_gaps_v1.json'
ARCHIVE = '951a6dae38e735e55d9b0d7717a5dc447f2b2e53685cab54ef984f62c1baa451'
RUN = '07ce7bffe77bfb703585c03543a934ba42b36d27e64cadd0f3b0f1b76751b9e5'
HARNESS = '7c362c0f9508abe3ac2fe73b33bd06d8f6720e6060e59937bd9a49626c1ecec8'
SCHEMA = 'quactlize.decode-profile-gap-review.v1'
RECHECK_KEYS = [[14, 2, 512, k, 256, 8, 8, 8, 1] for k in (2048, 3072)]


def read(path):
    require(sha(path) == ARCHIVE, 'unreviewed profile-gap archive')
    files = {}
    with tarfile.open(path, 'r|xz') as stream:
        for member in stream:
            p = PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and
                    member.name not in files and member.size < 64*2**20, 'unsafe archive member')
            files[member.name] = stream.extractfile(member).read()
    top = {n: json.loads(b) for n, b in files.items() if '/' not in n and n.endswith('.json')}
    transfer = top['transfer-manifest.json']
    wanted = {r['path']: (r['bytes'], r['sha256']) for r in transfer['included']}
    require(len(wanted) == len(transfer['included']) and wanted == {
        n: (len(b), hashlib.sha256(b).hexdigest())
        for n, b in files.items() if n != 'transfer-manifest.json'}, 'transfer hashes differ')
    plan, run = top['plan.json'], top['run.json']
    sealed(plan)
    sealed(run)
    require(run['sha256'] == RUN and run['body_source'] == digest(run['harness']) == HARNESS and
            run['plan'] == plan['sha256'] and
            plan == make_plan(top['cpp-originals.json']['requests']), 'frozen profile-gap run differs')
    require(len(run['devices']) == len({d['device']['pci'] for d in run['devices']}) == 7 and
            all(d['device']['compute_units'] == 72 and d['device']['warp_size'] == 32 and
                d['l2']['bytes'] == 67108864 for d in run['devices']), 'reviewed ZW810 devices differ')
    for kind in ('tc', 'vectors'):
        require(set(top['images.json'][kind]) == set(plan[kind]) and run['images'][kind] ==
                {k: v['sha256'] for k, v in top['images.json'][kind].items()}, 'image bindings differ')
    top['_points'] = {p['id']: p for p in plan['points']}
    top['_owners'] = {k: i for i, keys in enumerate(run['assignments']) for k in keys}
    points, counts, errors = {}, Counter(), []
    for name, body in files.items():
        if '/points/' not in name:
            continue
        raw = json.loads(body)
        require(raw['point'] not in points and raw['status'] == 'PASS', 'missing/failed/duplicate point')
        validate_producers(top['_points'][raw['point']], raw['data'])
        row = validate_point(raw, name, top)
        require(row['status'] == 'PASS', 'profile-gap confirmation did not pass')
        points[row['point']] = row
        data = raw['data']
        counts.update(arms=len(data['arms']), components=len(data['component_records']))
        counts['samples'] += sum(sum(map(len, c['samples_us'])) for c in data['component_records'].values())
        for arm in data['arms'].values():
            errors.extend(p['relative_l1_error'] for p in arm['numerical'])
    require(set(points) == set(top['_points']) and counts == dict(
        arms=118, components=260, samples=7800) and len(errors) == 708, 'profile-gap coverage differs')
    summary = top['summary.json']
    require(summary['status'] == 'COMPLETE' and summary['counts'] == dict(PASS=49) and
            summary['decisions'] == dict(PASS=49) and summary['returncodes'] == [0]*7 and
            summary['profile_gap_coverage']['completed'] == 69, 'summary differs from receipts')
    return top, points, dict(counts, points=len(points), files=len(files),
                             numerical_proofs=len(errors), maximum_error=max(errors))


def comparison(candidate, control):
    return ('MEASURED_REGRESSION' if candidate['median_us'] > control['median_us'] else
            'ELIGIBLE' if allowed(candidate, control) else 'CONFIRMATION_GATE_NOT_MET')


def review(path):
    top, points, counts = read(path)
    plan, run = top['plan.json'], top['run.json']
    rows = []
    for row, alternative in opportunities():
        cid = identity(alternative)
        profiles = []
        for old in alternative['profiles']:
            p = points.get(old['point'])
            if p is None or cid not in p['candidates']:
                require(old['status'] == 'ELIGIBLE' and row['key'] not in
                        plan['profile_gaps']['body_recheck_keys'], 'missing fresh profile comparison')
                profiles.append(dict(old, evidence='HISTORICAL_PAIRED_RESULT_NOT_RETIMED'))
                continue
            c, o = p['candidates'][cid], p['candidates']['original']
            metrics = lambda a: {f: a[f] for f in ('median_us', 'rounds', 'spread_pct')}
            profiles.append(dict(point=old['point'], router=old['router'], status=comparison(c, o),
                delta_pct=100*(c['median_us']/o['median_us']-1), evidence='FRESH_PAIRED_RESULT',
                previous_status=old['status'], receipt_sha256=p['receipt_sha256'],
                original=metrics(o), candidate=metrics(c)))
        # All 20 alternatives have a NEW measured loss; no historical absolute
        # time is needed to reject them. Do not average away a slow profile.
        require(any(p['evidence'] == 'FRESH_PAIRED_RESULT' and p['status'] == 'MEASURED_REGRESSION'
                    for p in profiles), 'alternative needs a separate admission review')
        rows.append(dict(key=row['key'], candidate=cid, config=alternative['config'],
            dequant_config=alternative['dequant_config'], implementation=alternative['implementation'],
            status='REJECT_UNIVERSAL_REPLACEMENT', profiles=profiles))
    result = dict(schema=SCHEMA, selection=SELECTION, archive_sha256=ARCHIVE,
        run=RUN, plan=plan['sha256'], body_source=HARNESS, sdk=run['sdk'],
        source_commit='24844b3b3697bb55bd388a6e8bbbec2fc1433794',
        images_sha256=digest(run['images']), source_directory=top['transfer-manifest.json']['source'],
        counts=counts | dict(host_keys=14, alternatives=20, fresh_pairs=69, filled_gaps=57,
                             body_rechecks=12, retained_keys=14, promoted_keys=0, remaining_gaps=0),
        body_recheck_keys=plan['profile_gaps']['body_recheck_keys'], rows=rows,
        worker_wall_seconds=top['summary.json']['elapsed_seconds'],
        clock_policy=run['clock_policy'], clock_deviations=top['summary.json']['clock_deviations'],
        scope='REVIEWED_M8_PROFILE_GAPS_ONLY_NOT_ALL_DECODE_OR_MODEL_TPOT',
        status='COMPLETE_RETAIN_INCUMBENTS', production_changed=False, runtime_router_readback=False,
        old_receipts_modified=False, frequency_normalization=False, model_tpot_admitted=False,
        binaries_transferred=False)
    return result | dict(evidence_sha256=digest(result))


def close_supplement(table, report):
    """Apply a rejection receipt to pending work, never to the runtime catalog."""
    pending = supplement_plan(table)
    require(report['schema'] == SCHEMA and report['selection'] == pending['selection'] == SELECTION and
            report['archive_sha256'] == ARCHIVE and report['run'] == RUN and
            report['body_source'] == HARNESS and report['status'] == 'COMPLETE_RETAIN_INCUMBENTS' and
            report['evidence_sha256'] == digest({k: v for k, v in report.items() if k != 'evidence_sha256'}) and
            all(report[f] is False for f in ('production_changed', 'runtime_router_readback',
                'old_receipts_modified', 'frequency_normalization', 'model_tpot_admitted')),
            'profile-gap review identity/scope differs')
    require(report['body_recheck_keys'] == RECHECK_KEYS and report['counts'] == dict(
        alternatives=20, arms=118, body_rechecks=12, components=260, files=125,
        filled_gaps=57, fresh_pairs=69, host_keys=14, maximum_error=0.003884254875700735,
        numerical_proofs=708, points=49, promoted_keys=0, remaining_gaps=0, retained_keys=14,
        samples=7800), 'reviewed completion counts differ')
    expected = {(tuple(row['key']), identity(a)): (row, a) for row, a in opportunities()}
    observed, filled, rechecks = set(), set(), set()
    for entry in report['rows']:
        pair = tuple(entry['key']), entry['candidate']
        require(pair in expected and pair not in observed, 'unexpected/duplicate reviewed alternative')
        observed.add(pair)
        row, alternative = expected[pair]
        require(identity(entry) == pair[1] and entry['status'] == 'REJECT_UNIVERSAL_REPLACEMENT',
                'reviewed candidate differs')
        before = {p['point']: p for p in alternative['profiles']}
        require(len(entry['profiles']) == len(before) and
                {p['point'] for p in entry['profiles']} == set(before), 'reviewed profile coverage differs')
        fresh_loss = False
        for p in entry['profiles']:
            old = before[p['point']]
            require(p['router'] == old['router'], 'router profile changed')
            if p['evidence'] == 'HISTORICAL_PAIRED_RESULT_NOT_RETIMED':
                require(p == dict(old, evidence=p['evidence']) and old['status'] == 'ELIGIBLE' and
                        row['key'] not in report['body_recheck_keys'], 'old comparison relabelled')
                continue
            require(p['evidence'] == 'FRESH_PAIRED_RESULT' and p['previous_status'] == old['status'],
                    'fresh comparison provenance differs')
            for a in (p['candidate'], p['original']):
                rounds = a['rounds']
                require(len(rounds) == 6 and all(type(t) in (float, int) and 0 < t < float('inf')
                                                for t in rounds) and
                        a['median_us'] == statistics.median(rounds) and
                        a['spread_pct'] == 100*(max(rounds)/min(rounds)-1), 'paired statistics differ')
            require(p['status'] == comparison(p['candidate'], p['original']) and
                    p['delta_pct'] == 100*(p['candidate']['median_us']/p['original']['median_us']-1),
                    'paired decision differs')
            fresh_loss |= p['status'] == 'MEASURED_REGRESSION'
            target = filled if old['status'] == 'NOT_MEASURED' else rechecks
            target.add((tuple(row['key']), p['point'], pair[1]))
        require(fresh_loss, 'rejection lacks a fresh measured regression')
    missing = {(tuple(c['key']), c['point'], identity(c['candidate'])) for c in pending['cells']}
    require(observed == set(expected) and filled == missing and len(rechecks) == 12 and
            len({k for k, _, _ in rechecks}) == 2, 'supplement closure differs')
    return pending | dict(cells=[], counts=dict(host_keys=0, points=0, candidate_profile_cells=0),
        status='COMPLETE_RETAIN_INCUMBENTS', closed_counts=pending['counts'],
        completion=report['evidence_sha256'], remaining_gpu_work=False,
        scope='ONLY_THE_REVIEWED_57_GAPS_CLOSED_OTHER_DECODE_WORK_UNCHANGED')
