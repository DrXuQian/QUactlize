"""Build-time, router-independent MoE admission into the existing flat catalog.

Only complete confirmations common to every tested router may replace a row.
The table refresh does not promote router-specific TC wins or bucket donors.
"""
import copy
import json
import math
from pathlib import Path
import statistics

from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT/'policies/kpack_moe_decode_exact_v1.json'
ARCHIVE = '55e8bd5bdc175a1c1396bcf43ac65f2f1e89ff6c4c9ba569583ea7a3d8b38022'
COMMIT = '9c4fe3b6f3a342e2073661b9427bfc173d1bc511'
EVIDENCE = 'cbfb0f2a1a05c96f5f37d0ddff12794c83d4870aae910a45fe65449328e2bada'
FAMILIES = ((12, 512, 2048, 1), (12, 512, 3072, 1),
            (13, 2048, 512, 8), (13, 3072, 512, 8))
ROUTERS = ('cluster', 'real', 'repeat2', 'repeat4', 'spread')


def configuration(recipe):
    """Shipping public configurations; never alias an experimental reader body."""
    kind = recipe['kind']
    p = dict(recipe['parameters'])
    if kind == 'tc':
        fields = {'qtype','symbol','tm','tn','tk','wm','wn','stages','ap','dn',
                  'persistent','split','grid_mode','grid_b','dequant_config'}
        if set(p) != fields or recipe['route'] != 'fq' or p['dequant_config'] != -1:
            raise ValueError('MoE TC admission requires the measured FQ endpoint and helper contract')
        p.pop('dequant_config')
        persistent = p.pop('persistent')
        return dict(kind='tc', route=2, parent_persistent=persistent, **p)
    fields = {'reader', 'variant', 'columns', 'warps', 'values'} if kind == 'q4' else {
        'variant', 'columns', 'warps', 'values', 'split'}
    if kind not in ('q4', 'simt') or recipe['route'] != 'fq' or set(p) != fields:
        raise ValueError('MoE selected recipe is not a shipping single-row reader')
    if any(type(v) != int or v <= 0 for v in p.values()):
        raise ValueError('invalid MoE reader geometry')
    return dict(kind=kind, **p)


def choose(profiles):
    """Minimax regret, constrained to no measured router regressing the control."""
    common = set.intersection(*(set(p['candidates']) for p in profiles))
    originals = {p['original'] for p in profiles}
    if len(originals) != 1 or not originals <= common:
        raise ValueError('MoE profiles lack the same confirmed current control')
    original, = originals
    eligible = [cid for cid in common if all(
        p['candidates'][cid]['mean_us'] <= p['candidates'][p['original']]['mean_us']
        for p in profiles)]
    def score(cid):
        costs = [p['candidates'][cid]['mean_us'] for p in profiles]
        floors = [min(c['mean_us'] for c in p['candidates'].values()) for p in profiles]
        return max(c/f for c, f in zip(costs, floors)), math.fsum(costs), cid != original, cid
    return min(eligible, key=score)


def worst_regret(profiles, selected):
    return 100 * (max(p['candidates'][selected]['mean_us'] /
                     min(c['mean_us'] for c in p['candidates'].values()) for p in profiles) - 1)


def validate(data):
    """Semantic replay used by the importer; the reviewed digest is checked by load."""
    if (data.get('schema') != 'quactlize.moe-decode-exact.v1' or
            data.get('archive_sha256') != ARCHIVE or
            data.get('evidence_sha256') != digest({k:v for k,v in data.items() if k != 'evidence_sha256'}) or
            data['source_revision']['quactlize_commit'] != COMMIT or
            data['timing_method'] != 'ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM' or
            data['selection'] != 'CONFIRMED_COMMON_MINIMAX_NO_PROFILE_REGRESSION' or
            data['runtime_router_readback'] or data['bucket_changes'] or data['model_tpot_admitted'] or
            data['native_no_regression_admitted'] or data['bf16_storage_admitted']):
        raise ValueError('MoE exact evidence identity/scope differs')
    keys = [[q,2,n,k,256,8,ch,m,1] for q,n,k,ch in FAMILIES for m in range(1,9)]
    if [r['key'] for r in data['rows']] != keys:
        raise ValueError('MoE exact measured denominator differs')
    if len(data['body_hashes']) < 5 or any(sha(ROOT/p) != h for p,h in data['body_hashes'].items()):
        raise ValueError('MoE measured kernel body changed; renew device evidence')
    points = set()
    for row in data['rows']:
        profiles = row['profiles']
        expected = ['spread'] if row['key'][7] == 1 else list(ROUTERS)
        if [p['router'] for p in profiles] != expected:
            raise ValueError('MoE exact router coverage differs')
        if (row['changed'] != (row['config'] != row['base_config']) or
                row['measurement_changed'] != (row['config'] != row['original_config'])):
            raise ValueError('MoE transition flags differ from the compared catalogs')
        for p in profiles:
            if p['point'] in points:
                raise ValueError('duplicate measured MoE point')
            points.add(p['point'])
            prov = p['latency_provenance']
            if (any(prov.get(k) != v for k,v in data['source_revision'].items()) or
                    prov.get('timing_method') != 'ASYS_GPU_ACTIVITY' or
                    prov['device']['device']['compute_units'] != 72 or
                    prov['device']['l2']['bytes'] != 67108864):
                raise ValueError('MoE profile provenance differs')
            for cid, value in p['candidates'].items():
                samples = value['samples_us']
                parts = value['components']
                if (cid != digest(value['recipe']) or len(samples) != 6 or any(len(r) != 5 for r in samples) or
                        any(not math.isfinite(t) or t <= 0 for r in samples for t in r) or
                        not math.isclose(statistics.fmean(t for r in samples for t in r),
                                         value['mean_us'], rel_tol=1e-12) or
                        not parts or sum(s['kind'] == 'producer' for s in parts) != 1 or
                        not math.isclose(math.fsum(s['mean_us'] for s in parts), value['mean_us'], rel_tol=1e-12) or
                        any(not math.isfinite(s['mean_us']) or s['mean_us'] <= 0 for s in parts) or
                        not 0 <= value['max_relative_l1'] < .005 or not value['module_sha256']):
                    raise ValueError('MoE complete-call samples/numerics differ')
            if configuration(p['candidates'][p['original']]['recipe']) != row['original_config']:
                raise ValueError('MoE measured original differs')
        selected = choose(profiles)
        if (row['selected'] != selected or
                configuration(profiles[0]['candidates'][selected]['recipe']) != row['config'] or
                not math.isclose(row['worst_regret_pct'], worst_regret(profiles, selected), abs_tol=1e-12)):
            raise ValueError('MoE static choice differs from common full-call confirmations')
    if len(points) != 144:
        raise ValueError('MoE model/profile denominator differs')


def verify(data):
    if data.get('evidence_sha256') != EVIDENCE:
        raise ValueError('MoE exact evidence is not the reviewed capture')
    validate(data)


def load():
    data = json.loads(TABLE.read_bytes())
    verify(data)
    return data


def apply(baseline):
    data = load()
    result = copy.deepcopy(baseline)
    exact = {tuple(r['key']):r for r in result['exact']}
    for measured in data['rows']:
        row = exact.get(tuple(measured['key']))
        if (row is None or row['config'] != measured['base_config'] or
                row.get('dequant_config', -1) != measured['base_dequant_config']):
            raise ValueError('MoE base catalog changed; review the measured transition')
        # Bucket prediction has its own evidence. Refresh the exact key only;
        # Q4's shape-bound readers must never become generic bucket donors.
        row['bucket_config'] = copy.deepcopy(row.get('bucket_config',
            row['guard'] if row['config']['kind'] == 'q4' else row['config']))
        row['bucket_dequant_config'] = row.get('bucket_dequant_config',
            -1 if row['config']['kind'] == 'q4' else row.get('dequant_config', -1))
        row.setdefault('previous_config', copy.deepcopy(measured['original_config']))
        row.update(config=copy.deepcopy(measured['config']), policy=18, dequant_config=-1,
                   status=('ROUTER_SENSITIVE_PARETO' if measured['worst_regret_pct'] > 5 else
                           'MEASURED_MOE_COMMON_CONFIRMED'), worst_regret_pct=measured['worst_regret_pct'],
                   profiles=[p['router'] for p in measured['profiles']],
                   moe_evidence_sha256=data['evidence_sha256'])
        if row['config']['kind'] != 'q4':
            row['guard'] = copy.deepcopy(row['config'])
    return result
