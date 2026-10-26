"""Build-time exact token-reuse admission; no runtime tuner or bucket override."""
import copy
import json
import math
from pathlib import Path

from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT/'policies/kpack_token_cta_v1.json'
CATALOG = ROOT/'quactlize/execution/token_cta.inc'
ARCHIVE = '28a8d7dbaea294311ac1d2b61c329b39068c246f8bf3a585e0a146069a3c1631'
EVIDENCE = 'fe131fec811ca2b066efc842312570e52fc351313b003523a5f558cdbf032bbb'
FIELDS = ('variant', 'columns', 'warps', 'values', 'split')


def verify(data):
    if (data['schema'] != 'quactlize.token-cta-exact.v2' or data['archive_sha256'] != ARCHIVE or
            data['evidence_sha256'] != EVIDENCE or digest({k:v for k,v in data.items()
                if k != 'evidence_sha256'}) != EVIDENCE):
        raise ValueError('token CTA evidence identity differs')
    keys = {tuple(r['key']) for r in data['rows']}
    if len(data['rows']) != 376 or len(keys) != 376 or any(
            q not in (8,10,11,12,13,14) or mode or (e,t,ch,compute) != (1,1,1,0) or
            not 1 <= m <= 8 or n <= 0 or k <= 0 for q,mode,n,k,e,t,ch,m,compute in keys):
        raise ValueError('token CTA exact scope differs')
    if not {'quactlize/execution/simt_token_cta.cuh', 'quactlize/execution/simt_q8_token_cta.cuh'} <= data['body_hashes'].keys():
        raise ValueError('token CTA measured source closure missing')
    for row in data['rows']:
        chosen = row['selected']
        if (chosen['mean_us'] != min(row['candidate_means_us'].values()) or
                chosen['mean_us'] != row['candidate_means_us'][chosen['id']] or
                not all(math.isfinite(t) and t > 0 for t in row['candidate_means_us'].values()) or
                not 0 <= chosen['numerical_error'] < .005):
            raise ValueError('token CTA selected mean/numerics differ')
        impl, config = chosen['implementation'], chosen['config']
        if row['latency_provenance']['quactlize_commit'] != data['cohorts'][row['cohort']]['source_revision']['quactlize_commit']:
            raise ValueError('token CTA source provenance differs')
        if impl['kind'] == 'token-cta':
            q,_,n,k,_,_,_,m,_ = row['key']
            if (m < 2 or config['kind'] != 'simt' or config['split'] not in (1,2,4,8) or
                    config['variant'] != (5 if q == 8 else 3) or
                    impl['tokens'] != m or impl['window'] not in ((1,2) if q == 8 else (0,))):
                raise ValueError('unmeasured token CTA implementation')
        elif impl['kind'] == 'q8-address':
            if (row['key'][0] != 8 or config['kind'] != 'simt' or config['split'] != 1 or
                    config['variant'] != 5 or config['values'] != 4 or impl['body'] not in (1,3,4,5)):
                raise ValueError('unmeasured Q8 address implementation')
        elif impl['kind'] != 'shipping':
            raise ValueError('unknown measured implementation')
    if sum(r['selected']['implementation']['kind']=='token-cta' for r in data['rows']) != 121:
        raise ValueError('token CTA binding denominator differs')


def load():
    data = json.loads(TABLE.read_bytes())
    verify(data)
    return data


def rows():
    return [r for r in load()['rows'] if r['selected']['implementation']['kind']=='token-cta']


def apply(baseline):
    data = load()
    if digest(baseline) != data['baseline_sha256']:
        raise ValueError('token CTA incumbent catalog differs')
    result = copy.deepcopy(baseline)
    exact = {tuple(r['key']): r for r in result['exact']}
    for measured in data['rows']:
        row = exact.get(tuple(measured['key']))
        if not measured['changed'] and row is not None:
            continue
        chosen = measured['selected']
        if row is None:
            # These previously open cells have new complete measurements.
            # Append exact entries only; never add them as bucket donors.
            row = dict(key=measured['key'], config=chosen['config'], guard=chosen['config'],
                       source='dense-m-exact-v1')
            result['exact'].append(row)
            exact[tuple(measured['key'])] = row
        # Preserve the complete old donor, including Q4's generic TC guard.
        row['bucket_config'] = copy.deepcopy(row.get('bucket_config',
            row['guard'] if row['config']['kind']=='q4' else row['config']))
        row['bucket_dequant_config'] = row.get('bucket_dequant_config',
            -1 if row['config']['kind']=='q4' else row.get('dequant_config',-1))
        row['previous_config'] = copy.deepcopy(row['config'])
        # kGuards also controls donor eligibility at unseen K. Keep that old
        # guard: making a SIMT exact choice must not legalize its old TC donor.
        row.update(config=chosen['config'],
            implementation=chosen['implementation'], dequant_config=chosen['dequant_config'],
            policy=18, status='MEASURED_MINIMUM', worst_regret_pct=0,
            timing_source='token-cta-asys-mean-v1')
    return result


def catalog():
    text = '// Generated by tools/generate_token_cta.py; exact dense F32/F16 only.\n'
    text += '// Q, N, K, M, variant, columns, warps, values, split, half-window\n'
    for row in rows():
        q,_,n,k,_,_,_,m,_ = row['key']
        c, impl = row['selected']['config'], row['selected']['implementation']
        text += 'QTC('+','.join(map(str,[q,n,k,m,*[c[f] for f in FIELDS],impl['window']]))+')\n'
    return text


def capability():
    data = load()
    if CATALOG.read_text() != catalog():
        raise ValueError('stale token CTA catalog; regenerate before building')
    for path, expected in data['body_hashes'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('token CTA measured body changed; new device evidence required')
    return dict(evidence_sha256=EVIDENCE, catalog_sha256=sha(CATALOG),
        body_hashes=data['body_hashes'], keys=[dict(key=r['key'],
            config={f:r['selected']['config'][f] for f in FIELDS},
            window=r['selected']['implementation']['window']) for r in rows()])
