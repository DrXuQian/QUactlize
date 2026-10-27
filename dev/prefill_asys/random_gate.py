"""Seeded off-table M spot checks of production buckets, never a config sweep."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess

from quactlize.dispatch.dense_prefill import EVIDENCE, HEADER, audit_row, validate as validate_policy
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from .finalists import case_identity, counts
from .supplement import execution_case, policy_lookup

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='quactlize.dense-random-native.v1'
BANDS=((9,63),(65,255),(257,2047),(2049,8191))
ORDER=('native','quactlize','quactlize','native')
SCOPE='SEEDED_OFF_TABLE_M_COMPONENT_AB_NOT_ALL_SHAPES_OR_MODEL_TPOT'
METRIC='ALL_SAMPLE_ASYS_MEAN_WITH_EVERY_HELPER_AND_REDUCER'
PRECISION='F32_IO_F16_COMPUTE_FP32_ACC'
NATIVE='LINKED_NATIVE_UNCHANGED; CUBLAS_F16_F32; Q8_1_IF_NATIVE_SELECTS_IT'
EXCLUDED=[dict(qtype=10,reason='NO_REAL_MODEL_DENSE_FAMILY_IN_CURRENT_TABLE')]


def input_identity(policy):
    return dict(policy_sha256=policy['sha256'],selector_header_sha256=sha(HEADER),
                helper_policy_sha256=digest(json.loads((ROOT/'policies/dense_full_helpers_20261023.json').read_text())))


def draws(policy, seed):
    """Geometry/type sampling is independent of latency and winning recipes."""
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('seed must be a uint32')
    families={}
    for p in policy['points']:
        q,_,n,k=p['shape'];families.setdefault(q,set()).add((n,k))
    anchors={p['shape'][1] for p in policy['points']}
    rng=random.Random(seed);result=[]
    for q in sorted(families):
        shapes=sorted(families[q]);rng.shuffle(shapes)
        for i,(lo,hi) in enumerate(BANDS):
            available=[m for m in range(lo,hi+1) if m not in anchors]
            # Half the draws probe one-token boundaries; the other half are
            # uniform within the predeclared stratum. Never redraw a loser.
            edges=sorted({m for a in anchors for m in (a-1,a+1)
                          if lo<=m<=hi and m not in anchors})
            m=rng.choice(edges if i%2==0 and edges else available)
            n,k=shapes[i%len(shapes)]
            result.append(Workload(q,n,k,m))
    rng.shuffle(result)
    return result


def cpp_choices(workloads, policy, output):
    source=ROOT/'dev/prefill_asys/random_lookup.cpp';binary=output/'selector-query'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(source),'-o',str(binary)],check=True)
    raw=subprocess.check_output([str(binary)],text=True,
        input=''.join(f'{w.qtype} {w.m} {w.n} {w.k}\n' for w in workloads))
    rows=[tuple(map(int,line.split())) for line in raw.splitlines()]
    used=sorted({p['winner'] for p in policy['points']}|{b['recipe'] for b in policy['buckets']})
    if len(rows)!=len(workloads):raise ValueError('C++ selection coverage differs')
    result=[]
    for w,(index,kind) in zip(workloads,rows):
        coverage,rid=policy_lookup(policy,w)
        if coverage!='BUCKET_PREDICTION' or not 0<=index<len(used) or kind!=21 or used[index]!=rid:
            raise ValueError(f'random point must hit the actual production bucket, not an exact or generic fallback: {w}')
        result.append(dict(coverage=coverage,policy=kind,recipe=rid))
    return result


def make_plan(output, seed=20261027):
    policy=json.loads(EVIDENCE.read_text());validate_policy(policy)
    workload=draws(policy,seed);selected=cpp_choices(workload,policy,output)
    points=[]
    for w,s in zip(workload,selected):
        case=execution_case(policy['recipes'][s['recipe']]);cid=digest(case)
        case.update(id=cid,roles=['production-selected-bucket','model-config','required-control'])
        points.append(dict(id=w.key,workload=asdict(w),cases=[case],consumers=[],tensors=[],
            candidates=[case['recipe']] if case['recipe']['kind']=='tc' else [],champion_controls=[],
            confirmation_captures=2,capture_batch_limit=1,fixed_schedules=True,
            native_comparison=True,random_dense_probe=True,selection=s))
    result=dict(schema=SCHEMA,seed=seed,m_bands=[list(b) for b in BANDS],points=points,reuse=[],
        qtypes=sorted({w.qtype for w in workload}),excluded_qtypes=EXCLUDED,
        inputs=input_identity(policy),confirmation_captures=2,precision=PRECISION,
        native_arithmetic=NATIVE,order=list(ORDER),metric=METRIC,
        timing_method='ASYS_GPU_ACTIVITY',config_sweep=False,allowed_slowdown_pct=0,
        scope=SCOPE,production_changed=False,performance_admitted=False)
    result['counts']=counts(points);result['sha256']=digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'}) or
            plan.get('production_changed') is not False or plan.get('performance_admitted') is not False or
            plan.get('config_sweep') is not False or plan.get('allowed_slowdown_pct')!=0 or
            plan.get('order')!=list(ORDER) or plan.get('confirmation_captures')!=2 or
            plan.get('metric')!=METRIC or plan.get('precision')!=PRECISION or
            plan.get('native_arithmetic')!=NATIVE or plan.get('scope')!=SCOPE or
            plan.get('timing_method')!='ASYS_GPU_ACTIVITY' or plan.get('reuse')!=[] or
            plan.get('m_bands')!=[list(b) for b in BANDS] or plan.get('excluded_qtypes')!=EXCLUDED):
        raise ValueError('random native plan contract/hash differs')
    policy=json.loads(EVIDENCE.read_text())
    if plan['inputs']!=input_identity(policy):
        raise ValueError('random gate production table changed; use its frozen source or a new run')
    expected=draws(policy,plan['seed'])
    if plan.get('qtypes')!=sorted({w.qtype for w in expected}):
        raise ValueError('random gate format denominator changed')
    if [p['workload'] for p in plan['points']]!=[asdict(w) for w in expected]:
        raise ValueError('random workload denominator/seed changed')
    for p,w in zip(plan['points'],expected):
        coverage,rid=policy_lookup(policy,w)
        if (p['id']!=w.key or p['selection']!=dict(coverage=coverage,policy=21,recipe=rid) or
                coverage!='BUCKET_PREDICTION' or not p['native_comparison'] or not p['random_dense_probe'] or
                p['confirmation_captures']!=2 or p['capture_batch_limit']!=1 or not p['fixed_schedules'] or
                len(p['cases'])!=1 or p['champion_controls']):
            raise ValueError('random gate requires one current bucket recipe and native at every point')
        c,=p['cases'];want=execution_case(policy['recipes'][rid]);r=Recipe.from_dict(c['recipe'])
        from dev.strategy.prescreen import rejection
        if (case_identity(c)!=want or c['id']!=digest(want) or rejection(r,w) or
                c['roles']!=['production-selected-bucket','model-config','required-control'] or
                p['candidates']!=([c['recipe']] if r.kind=='tc' else [])):
            raise ValueError('random gate candidate differs from production selection')
    if plan['counts']!=counts(plan['points']):raise ValueError('random gate config denominator changed')
    return plan


def compare(plan, review):
    validate(plan)
    if (review.get('schema')!='quactlize.prefill-component-review.v1' or
            review.get('review_sha256')!=digest({k:v for k,v in review.items() if k!='review_sha256'})):
        raise ValueError('random review hash differs')
    expected={p['id'] for p in plan['points']}
    if set(review['points'])-expected:raise ValueError('unknown random point')
    result=[]
    for spec in plan['points']:
        key=spec['id'];point=review['points'].get(key)
        item=dict(point=key,workload=spec['workload'],selection=spec['selection'])
        if point is None:
            partial=[p for p in review.get('partial_points',{}).values() if p.get('point_id')==key]
            result.append(dict(item,status='INCOMPLETE',partial=partial));continue
        if point['workload']!=spec['workload']:raise ValueError('random review workload differs')
        authority=review['authorities'][point['authority']]
        if digest(authority)!=point['authority']:
            raise ValueError('random comparison authority differs')
        revision=authority.get('source_revision',{})
        if (revision.get('snapshot')!='COMMITTED_ARCHIVE' or any(
                not re.fullmatch('[0-9a-f]{40}',revision.get(k,''))
                for k in ('quactlize_commit','quactlize_tree','actlize_commit'))):
            raise ValueError('random comparison lacks a clean measurement commit')
        rows={r['case_id']:r for r in point['measurements']}
        cid=spec['cases'][0]['id']
        if len(rows)!=2 or set(rows)!={'native-fp16',cid}:
            raise ValueError('random comparison is not exactly selected vs native')
        costs={};components={};fixtures=[];contracts={}
        for arm,case in (('native','native-fp16'),('quactlize',cid)):
            row=rows[case];audit_row(row,review,point)
            identity=dict(arm='native-fp16') if arm=='native' else case_identity(spec['cases'][0])
            if row['candidate']!=digest(identity):
                raise ValueError('random comparison candidate identity differs')
            caps=row.get('confirmation_captures',[])
            if len(caps)!=2 or any(c.get('omitted_producer')!='DETECTED' for c in caps):
                raise ValueError('random comparison lacks both confirmations/negative controls')
            for cap in caps:
                evidence=cap.get('random_gate_evidence')
                if not evidence:raise ValueError('random comparison fixture proof missing')
                fixtures.append(evidence['raw_sha256'])
                arithmetic=evidence.get('arithmetic')
                if not arithmetic or (arm in contracts and contracts[arm]!=arithmetic):
                    raise ValueError('random arithmetic proof missing or changed')
                contracts[arm]=arithmetic
            if arm=='quactlize' and contracts[arm]!=dict(input='F32',output='F32',compute='F16',accumulator='FP32'):
                raise ValueError('random candidate arithmetic changed')
            costs[arm]=statistics.fmean(t for cap in caps for t in cap['complete_samples_us'])
            components[arm]=[dict(name=review['components'][k]['name'],phase=review['components'][k]['phase'],
                mean_us=statistics.fmean(t for cap in caps for t in cap['component_samples_us'][i]))
                for i,k in enumerate(row['components'])]
            if not math.isclose(costs[arm],math.fsum(c['mean_us'] for c in components[arm]),rel_tol=1e-12,abs_tol=1e-6):
                raise ValueError('random component sum differs')
        if not fixtures[0] or any(f!=fixtures[0] for f in fixtures):
            raise ValueError('random native/candidate raw GGUF fixtures differ')
        delta=costs['quactlize']-costs['native']
        result.append(dict(item,status='PASS' if delta<=0 else 'REGRESSION',native_us=costs['native'],
            quactlize_us=costs['quactlize'],delta_us=delta,delta_pct=100*delta/costs['native'],
            components=components,arithmetic=contracts,authority=point['authority'],
            quactlize_commit=revision['quactlize_commit']))
    tally=Counter(p['status'] for p in result)
    status='INCOMPLETE' if tally['INCOMPLETE'] else 'REGRESSION' if tally['REGRESSION'] else 'PASS'
    return dict(status=status,counts=dict(tally),points=result,seed=plan['seed'],expected=len(expected),
        metric=plan['metric'],scope=SCOPE,all_shapes_no_regression_proven=False,
        production_changed=False,model_tpot_admitted=False)
