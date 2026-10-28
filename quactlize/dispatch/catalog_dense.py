"""Sealed production catalog validation/generation; no device experiments."""
from collections import Counter


import copy


import json


import math


from pathlib import Path, PurePosixPath


import statistics


from quactlize.dispatch.catalog_components import key, require


from quactlize.dispatch.catalog_refresh import allowed, identity


from quactlize.execution.simt_codegen import Config, runtime_inventory


from quactlize.execution.q8_refresh import DENSE_EVIDENCE


from quactlize.runtime.tuning import digest


ROOT = Path(__file__).resolve().parents[2]


TABLE = ROOT/'policies/kpack_dense_decode_v1.json'


SCHEMA = 'quactlize.dense-decode-exact.v1'


ARCHIVE = 'd771c38d77d964276bb00087af444c9b068a4ab3276f51cc54899bf81bec26a7'


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
