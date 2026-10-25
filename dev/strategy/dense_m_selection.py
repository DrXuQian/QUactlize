"""Import measured dense M1-8 gaps without changing any bucket donor."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import tarfile

from dev.strategy.production import key, require
from dev.strategy.refresh_selection import allowed, identity, sealed, validate_point
from quactlize.execution.simt_codegen import Config, runtime_inventory
from quactlize.execution.q8_refresh import DENSE_EVIDENCE
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT/'policies/kpack_dense_decode_v1.json'
SCHEMA = 'quactlize.dense-decode-exact.v1'
ARCHIVE = 'd771c38d77d964276bb00087af444c9b068a4ab3276f51cc54899bf81bec26a7'


def capture(path, baseline):
    """Verify original raw stages before exporting their chosen full-call cost."""
    require(sha(Path(path)) == ARCHIVE, 'unreviewed dense M archive')
    top, raw, members = {}, [], {}
    with tarfile.open(path, 'r|xz') as stream:
        for member in stream:
            name = member.name
            p = PurePosixPath(name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and
                    name not in members, 'unsafe or duplicate archive member')
            body = stream.extractfile(member).read()
            members[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if '/' not in name and name.endswith('.json'):
                top[name] = json.loads(body)
            elif re.fullmatch(r'gpu-\d+/points/[0-9a-f]{64}\.json', name):
                raw.append((name, json.loads(body)))
    transfer = top['transfer-manifest.json']['included']
    require(len(transfer) == len(members)-1 and {r['path']: dict(bytes=r['bytes'], sha256=r['sha256'])
        for r in transfer} == {k:v for k,v in members.items() if k != 'transfer-manifest.json'},
        'transfer hashes differ')
    plan, run = top['plan.json'], top['run.json']
    sealed(plan); sealed(run)
    require(plan['phase'] == 'dense-m-gaps' and plan['counts']['points'] == 243 and
        plan['counts']['arms'] == 5064 and run['plan'] == plan['sha256'] and
        run['body_source'] == digest(run['harness']), 'dense M run/plan identity differs')
    require(all(sha(ROOT/n) == h for n,h in plan['sources'].items()),
            'measured incumbent policy sources differ')
    require(all(d['device']['compute_units'] == 72 and d['device']['warp_size'] == 32
                for d in run['devices']), 'not the reviewed ZW810 cohort')
    for kind in ('tc', 'vectors'):
        images = top['images.json'][kind]
        require(set(images) == set(plan[kind]) and run['images'][kind] ==
                {k:v['sha256'] for k,v in images.items()}, 'executed image inventory differs')
    top['_points'] = {p['id']:p for p in plan['points']}
    top['_owners'] = {k:i for i,ks in enumerate(run['assignments']) for k in ks}
    require(len(raw) == len(top['_points']) == 243 and
            {r['point'] for _,r in raw} == set(top['_points']), 'point coverage differs')
    originals = {tuple(r['request']):r for r in top['cpp-originals.json']['requests']}
    rows = []
    for name, record in raw:
        point = validate_point(record, name, top)
        require(point['status'] in ('PASS', 'UNSTABLE_RETAIN_ORIGINAL'), 'unresolved numerical point')
        k = list(key(point['workload']))
        old = point['candidates']['original']
        chosen = point['candidates'][point['raw_winner']]
        rows.append(dict(key=k, point=point['point'], status=point['status'],
            admitted=point['status'] == 'PASS', original_cpp=originals[tuple(k)],
            original=old, selected=chosen, changed=identity(chosen) != identity(old),
            receipt_sha256=point['receipt_sha256'], source=point['source']))
    rows.sort(key=lambda r:r['key'])
    data = dict(schema=SCHEMA, archive_sha256=ARCHIVE, baseline_sha256=digest(baseline),
        run_sha256=run['sha256'], plan_sha256=plan['sha256'], sdk=run['sdk'],
        devices=run['devices'], body_source=run['body_source'],
        images_sha256=digest(run['images']), rows=rows,
        summary=dict(points=len(rows), admitted=sum(r['admitted'] for r in rows),
            changed=sum(r['changed'] for r in rows),
            retained=sum(r['admitted'] and not r['changed'] for r in rows),
            pending=sum(not r['admitted'] for r in rows)),
        coverage=dict(families=45, total_keys=360, existing_exact=117, new_keys=243),
        scope='ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT',
        device='PPU-ZW810', compute_units=72, tokens=list(range(1,9)),
        bucket_changed=False, big_m_changed=False, native_comparison=False,
        model_tpot_admitted=False, frequency_normalization=False,
        noisy_policy='RETAIN_RAW_UNSTABLE_INCUMBENT_NO_GATE_RELAXATION')
    data['evidence_sha256'] = digest(data)
    verify(data)
    return data


def verify(data):
    require(data['schema'] == SCHEMA and data['archive_sha256'] == ARCHIVE and
            data['evidence_sha256'] == DENSE_EVIDENCE and
            data['evidence_sha256'] == digest({k:v for k,v in data.items() if k != 'evidence_sha256'}),
            'dense M table identity differs')
    require(not any(data[f] for f in ('bucket_changed', 'big_m_changed', 'native_comparison',
                                     'model_tpot_admitted', 'frequency_normalization')),
            'dense M admission scope differs')
    keys, points = set(), set()
    for r in data['rows']:
        k = tuple(r['key']); c, old = r['selected'], r['original']
        require(len(k) == 9 and k not in keys and r['point'] not in points and
                k[0] in (8,11,12,13,14) and k[1] == 0 and k[4:7] == (1,1,1) and
                k[8] == 0 and 1 <= k[7] <= 8 and k[2] > 0 and k[2]%256 == 0 and
                k[3] > 0 and k[3]%(512 if k[0] in (11,14) else 256) == 0,
                'invalid or duplicate dense M key')
        keys.add(k); points.add(r['point'])
        require(r['status'] in ('PASS', 'UNSTABLE_RETAIN_ORIGINAL') and
                r['admitted'] == (r['status'] == 'PASS') and r['source'] == data['run_sha256'] and
                r['changed'] == (identity(c) != identity(old)) and
                r['original_cpp']['request'] == r['key'], 'dense M selected scope differs')
        if not r['admitted']:
            require(c == old and not r['changed'], 'unstable point cannot replace its incumbent')
        elif r['changed']:
            require(c['arm'] != 'original' and 'rule' in c['roles'] and allowed(c, old),
                    'dense M replacement lacks paired evidence')
        else:
            require(c == old, 'retained row differs from incumbent')
        for v in (old,c):
            require(math.isfinite(v['numerical_error']) and 0 <= v['numerical_error'] < .005 and
                    len(v['rounds']) == 6 and all(math.isfinite(x) and x > 0 for x in v['rounds']) and
                    math.isclose(statistics.median(v['rounds']), v['median_us'], rel_tol=1e-14) and
                    math.isclose(100*(max(v['rounds'])/min(v['rounds'])-1), v['spread_pct'], abs_tol=1e-12),
                    'invalid dense M timing/numerical evidence')
            f = v['config']; split = f['split']
            kinds = Counter(s['kind'] for s in v['components'])
            require(kinds['producer'] == 1 and (split == 1 or kinds['reducer'] == 1) and
                    all(math.isfinite(s['median_us']) and s['median_us'] > 0 for s in v['components']),
                    'dense M call lost a producer or reducer')
            if f['kind'] == 'simt':
                require(Config(**{n:f[n] for n in ('variant','columns','warps','values','split')})
                        in runtime_inventory(k[0]), 'SIMT config absent from execution')
            else:
                require(f['kind'] == 'tc' and f['qtype'] == k[0] and f['route'] in (0,1) and
                        k[3]%(f['tk']*split) == 0 and k[3]//(f['tk']*split) >= f['stages']-1 and
                        (not f['ap'] or k[7] == 1), 'invalid dense TC geometry')
            impl = v['implementation']
            require(impl == dict(kind='shipping') or (k[0] == 8 and f['kind'] == 'simt' and
                impl.get('kind') == 'q8-address' and impl.get('body') in (1,3,4,5) and
                f['variant'] == 5 and f['values'] == 4 and split == 1 and
                (f['columns'],f['warps']) in ((4,8),(8,4))), 'unmeasured Q8 body scope')
        require(not r['admitted'] or not c['config'].get('legacy_adapters'),
                'legacy adapters cannot become a typed exact row')
    require(data['summary'] == dict(points=243, admitted=237, changed=160, retained=77, pending=6) and
        len(keys) == 243 and sum(r['admitted'] for r in data['rows']) == 237 and
        sum(r['changed'] for r in data['rows']) == 160, 'dense M table denominator differs')


def load():
    data = json.loads(TABLE.read_bytes())
    verify(data)
    return data


def apply(baseline):
    data = load()
    require(digest(baseline) == data['baseline_sha256'], 'dense M incumbent catalog differs')
    result = copy.deepcopy(baseline)
    keys = {tuple(r['key']) for r in result['exact']}
    for row in data['rows']:
        require(tuple(row['key']) not in keys, 'dense gap cannot overwrite existing exact evidence')
        if not row['admitted']:
            continue
        c = row['selected']
        result['exact'].append(dict(key=row['key'], config=c['config'], guard=c['config'],
            dequant_config=c['dequant_config'], implementation=c['implementation'], policy=18,
            status='MEASURED_MINIMUM', worst_regret_pct=0, source='dense-m-exact-v1'))
    # Appending exact rows preserves every old bucket row index and donor.
    return result
