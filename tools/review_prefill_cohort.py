#!/usr/bin/env python3
"""Compact component evidence, exact screening table and empirical bucket proposals.

Merges complete measured points, never substituted component estimates. The
output is development data, not an installed production selector.
"""
from collections import defaultdict
import argparse
import json
import lzma
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_asys.cohort import bucket
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.prefill-component-review.v1'


def read(path):
    return json.loads(Path(path).read_text())


def eligible(row):
    return (row['status']=='PASS' and row['arm'] in ('fq','sf','full-fp16')
            and row.get('helper_mode') in ('production','cohort-current','experimental'))


def tables(points):
    exact, groups, bounds = {}, defaultdict(list), {}
    for key, point in points.items():
        w = Workload(**point['workload'])
        bid = digest(bucket(w)); bounds[bid] = bucket(w)
        valid = [r for r in point['measurements'] if eligible(r)]
        native = next((r for r in point['measurements'] if r['arm']=='native-fp16'), None)
        if not valid or not native or native['status']!='PASS':
            exact[key] = dict(status='REMEASURE_REQUIRED', bucket=bucket(w))
            groups[bid].append((key, {}))
            continue
        winner = min(valid, key=lambda r:r['gpu_sum_us'])
        exact[key] = dict(status='SCREEN_WINNER_NOT_ADMITTED', candidate=winner['candidate'],
            us=winner['gpu_sum_us'], native_us=native['gpu_sum_us'],
            delta_pct=100*(winner['gpu_sum_us']/native['gpu_sum_us']-1),
            authority=point['authority'], components=winner['components'], bucket=bucket(w))
        # Ratios within each measured point, not absolute microseconds across
        # shapes. A bucket recipe must have been measured at EVERY member.
        groups[bid].append((key, {r['candidate']:r['gpu_sum_us']/winner['gpu_sum_us'] for r in valid}))
    buckets = {}
    for key, members in groups.items():
        common = set.intersection(*(set(r) for _,r in members))
        if not common:
            buckets[key] = dict(status='NO_COMMON_MEASURED_CONFIG', bounds=bounds[key], points=[p for p,_ in members])
            continue
        config = min(common, key=lambda c:(max(r[c] for _,r in members), statistics.mean(r[c] for _,r in members), c))
        buckets[key] = dict(status='IN_SAMPLE_PROPOSAL_NOT_FALLBACK_ADMISSION', candidate=config, bounds=bounds[key],
            points=[p for p,_ in members], worst_regret_pct=100*(max(r[config] for _,r in members)-1),
            criterion='MINIMAX_PER_POINT_NORMALIZED_COMPLETE_CALL', holdout_validated=False)
    return exact, buckets


def ingest(root):
    root = Path(root).resolve(strict=True)
    if (root/'results/authority.json').is_file(): root = root/'results'
    authority = read(root/'authority.json')
    plan = read(root/'execution-plan.json')
    aid = digest(authority)
    if plan['authority'] != aid: raise ValueError('execution plan/authority differs')
    specs = {p['id']:p for p in plan['points']}
    found, definitions, candidates = defaultdict(dict), {}, {}
    for path in sorted((root/'batches').glob('*/pass.json')):
        data = read(path)
        if data['record_sha256'] != digest({k:v for k,v in data.items() if k!='record_sha256'}):
            raise ValueError('component receipt changed: '+str(path))
        if data['seal']['authority'] != aid or data['parsed']['status']!='PASS':
            raise ValueError('batch authority/status differs')
        w = Workload(**data['seal']['workload'])
        if w.key not in specs: raise ValueError('unknown measured point')
        proof_path=Path(data['cases_path'])
        if not proof_path.is_file() and proof_path.parent.name.startswith('attempt-'):
            # Uploaded compact archives retain this batch-relative path. The
            # immutable receipt/hash still binds the original numerical bytes.
            proof_path=path.parent/proof_path.parent.name/proof_path.name
        if sha(proof_path) != data['cases_sha256']: raise ValueError('numerical evidence changed')
        proof = read(proof_path)
        recipes = {c['id']:c for c in data['seal']['cases']}
        expected = {c['id']:c for c in specs[w.key]['cases']}
        expected['native-fp16'] = dict(id='native-fp16',arm='native-fp16')
        if len(recipes)!=len(data['seal']['cases']) or any(expected.get(k)!=v for k,v in recipes.items()):
            raise ValueError('measured recipe differs from execution plan')
        if ([r['case_id'] for r in proof] != list(recipes) or
                [r['case_id'] for r in data['parsed']['cases']] != list(recipes)):
            raise ValueError('case order/coverage differs')
        for p,r in zip(proof,data['parsed']['cases']):
            if p['status']!='PASS' or r['status']!='PASS': raise ValueError('unpassed numerical/timing row')
            case=recipes[r['case_id']]
            if p['arm']!=case['arm'] or r['arm']!=case['arm'] or r.get('helper_mode')!=case.get('helper_mode'):
                raise ValueError('measured arm/helper differs from planned recipe')
            checks=p['correctness']
            if checks!='PASS' and (not checks or any(c.get('nonfinite')!=0 or
                    not 0<=c.get('relative_l1_error',1)<.005 for c in checks)):
                raise ValueError('independent numerical evidence failed')
            if any(c.get('bad')!=0 for c in p.get('weight_bits',[])):
                raise ValueError('weight bit validation failed')
            samples=r['samples']
            if len(samples)<3 or not r['first_ring_excluded'] or r['cuda_graph'] or r['timing_method']!='ASYS_GPU_ACTIVITY':
                raise ValueError('changed measurement contract')
            if any(not math.isfinite(s['gpu_sum_us']) or s['gpu_sum_us']<=0 for s in samples):
                raise ValueError('invalid timing sample')
            if not math.isfinite(r['gpu_sum_us']) or abs(statistics.median(s['gpu_sum_us'] for s in samples)-r['gpu_sum_us'])>1e-7:
                raise ValueError('summary median differs')
            identity = dict(recipes[r['case_id']])
            # Role/point-local launch IDs are provenance, not recipe identity.
            for field in ('id','roles','launch','module'): identity.pop(field,None)
            candidate = digest(identity); candidates[candidate] = identity
            components, kernel_samples = [], []
            for k in r['kernel_inventory']:
                definition = {a:b for a,b in k.items() if a not in ('samples_us','median_us')}
                definition.update(workload=specs[w.key]['workload'],authority=aid)
                cid = digest(definition); definitions[cid] = definition
                if len(k['samples_us']) != len(samples): raise ValueError('component coverage differs')
                if (any(not math.isfinite(v) or v<=0 for v in k['samples_us']) or
                        not math.isfinite(k['median_us']) or
                        abs(statistics.median(k['samples_us'])-k['median_us'])>1e-7):
                    raise ValueError('component median differs')
                components.append(cid); kernel_samples.append(k['samples_us'])
            for i,s in enumerate(samples):
                if abs(sum(v[i] for v in kernel_samples)-s['gpu_sum_us'])>1e-6:
                    raise ValueError('component sum differs from complete call')
            row = dict(candidate=candidate, case_id=r['case_id'], arm=r['arm'], status='PASS',
                roles=recipes[r['case_id']].get('roles',['native']),
                helper_mode=r.get('helper_mode'), arithmetic=r.get('arithmetic'),
                gpu_sum_us=r['gpu_sum_us'], phases_us=r['phases_us'], components=components,
                component_samples_us=kernel_samples, complete_samples_us=[s['gpu_sum_us'] for s in samples],
                rounds_us=r['measured_rounds_us'], numerical=checks,
                weight_bits=p.get('weight_bits'), omitted_producer=p.get('omitted_producer'),
                raw_receipt=str(path), raw_receipt_sha256=data['record_sha256'],
                asysrep=data['asysrep'],sqlite=data['sqlite'])
            if r['case_id'] in found[w.key]: raise ValueError('duplicate point/config evidence')
            found[w.key][r['case_id']] = row
    points, incomplete, partial = {}, [], {}
    for key, spec in specs.items():
        point = dict(workload=spec['workload'], consumers=spec.get('consumers',spec.get('tensors')),
                     authority=aid, measurements=list(found[key].values()))
        if set(found[key]) != {c['id'] for c in spec['cases']} | {'native-fp16'}:
            incomplete.append(key)
            partial[digest([aid,key])] = dict(point, point_id=key,
                missing=sorted(({c['id'] for c in spec['cases']} | {'native-fp16'})-set(found[key])))
            continue
        points[key] = point
    diagnostics = [dict(path=str(p),data=read(p)) for pattern in ('failed-*.json','error-*.json')
                   for p in sorted(root.glob(pattern))]
    return dict(authorities={aid:authority}, points=points, components=definitions,
                candidates=candidates, incomplete=incomplete,diagnostics=diagnostics,partial_points=partial)


def merge(result, value, seen):
    """Replace a complete point atomically. Partial optimizations stay evidence."""
    domains={digest(domain(a)) for a in (*result['authorities'].values(),*value['authorities'].values())}
    if len(domains)>1:raise ValueError('SDK/device/clock domain changed; keep a separate review/table')
    for key,point in value['points'].items():
        if key in seen:raise ValueError('multiple new runs for one point: select the intended revision explicitly')
        seen.add(key)
        if key in result['points']:
            result['history'].append(dict(point=key,previous=result['points'][key],replacement=point['authority']))
    for field in ('authorities','points','components','candidates','partial_points'):result[field].update(value[field])
    result['diagnostics'].extend(value['diagnostics'])
    # A resumed completion resolves only its own authority's partial receipt.
    # Retain other revisions' partials for debugging, without relabelling them.
    for key,point in value['points'].items():
        result['partial_points'].pop(digest([point['authority'],key]),None)


def domain(authority):
    d=authority['device'];card=d['device']
    return dict(sdk=d['sdk'],l2_bytes=d['l2']['bytes'],requested_clock_mhz=1700,
        device={k:card[k] for k in ('compute_units','warp_size','max_threads','DriverGetVersion','RuntimeGetVersion')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',default=[])
    p.add_argument('--previous',type=Path,help='previous compact review, unchanged points retained')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--invalidate-component',action='append',default=[])
    a=p.parse_args()
    if a.output.exists():p.error('output must be new')
    if not a.run and not a.previous:p.error('supply --run and/or --previous')
    result=dict(schema=SCHEMA,authorities={},points={},components={},candidates={},history=[],diagnostics=[],partial_points={})
    if a.previous:
        with lzma.open(a.previous,'rt') as f: result=json.load(f)
        if result.get('schema')!=SCHEMA:raise ValueError('previous review schema differs')
        if result.get('review_sha256')!=digest({k:v for k,v in result.items() if k!='review_sha256'}):
            raise ValueError('previous compact review changed')
    incomplete=[];seen=set()
    for run in a.run:
        value=ingest(run);incomplete+=value['incomplete']
        merge(result,value,seen)
    for cid in a.invalidate_component:
        if cid not in result['components']:raise ValueError('unknown component ID: '+cid)
        for point in result['points'].values():
            for row in point['measurements']:
                if cid in row['components']:row['status']='STALE_IMPLEMENTATION_REMEASURE_FULL_CALL'
    coverage=dict(result['points'])
    for point in result['partial_points'].values():
        coverage.setdefault(point['point_id'],dict(point,measurements=[]))
    result['exact'],result['buckets']=tables(coverage)
    result.update(production_changed=False,performance_admitted=False,incomplete=incomplete,
        scope='COMPLETE_ASYS_SCREEN_POINTS_AND_COMPONENTS_NOT_PRODUCTION_TABLE_OR_MODEL_TPOT')
    result['pending_points']=sorted({p['point_id'] for p in result['partial_points'].values()}-set(result['points']))
    result['review_sha256']=digest({k:v for k,v in result.items() if k!='review_sha256'})
    with lzma.open(a.output,'xt',preset=6) as f:json.dump(result,f,separators=(',',':'))
    print('PREFILL_REVIEW '+json.dumps(dict(points=len(result['points']),incomplete=len(incomplete),
        bytes=a.output.stat().st_size,output=str(a.output),production_changed=False)))
    return 1 if incomplete or result['pending_points'] else 0


if __name__=='__main__':raise SystemExit(main())
