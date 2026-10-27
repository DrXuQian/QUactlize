"""Model-derived dense prefill/native A/B, with one production recipe per point."""
from collections import Counter
from dataclasses import asdict
import json

from dev.prefill_asys.cohort import headers
from dev.prefill_asys.finalists import case_identity, counts
from dev.prefill_asys.random_gate import (
    METRIC, NATIVE, ORDER, PRECISION, compare_selected, cpp_choices, input_identity,
)
from dev.prefill_asys.supplement import execution_case, policy_lookup
from quactlize.dispatch.dense_prefill import EVIDENCE, validate as validate_policy
from quactlize.dispatch.table import QTYPES, Workload
from quactlize.runtime.tuning import digest

SCHEMA='quactlize.dense-model-native.v1'
SCOPE='MODEL_DENSE_PREFILL_SELECTED_VS_NATIVE_F32ACC_COMPONENT_AB_NOT_MODEL_LATENCY'
MODEL_MS={'qwen35-35b-a3b':(512,1024,2048,4096,5120),'qwen3-32b':(5120,)}
ROLE='production-selected-model-prefill'


def inventory(models):
    """Reconstruct the denominator from saved headers, without reading weights."""
    if [m['name'] for m in models]!=list(MODEL_MS):
        raise ValueError('model gate requires both requested models in fixed order')
    groups={};excluded=[]
    for model in models:
        if model['ms']!=list(MODEL_MS[model['name']]) or not model['files']:
            raise ValueError('model prefill M coverage changed')
        seen=set();supported=0
        for file in model['files']:
            for t in file['header']['tensors']:
                name=t['name'];dims=t['dims_gguf'];q=t['qtype']
                if name in seen:raise ValueError('duplicate tensor in model headers: '+name)
                seen.add(name)
                ref=dict(model=model['name'],tensor=name,qtype=q,dims=dims,tp=1,rank=0)
                if len(dims)!=2 or not name.startswith('blk.') or not name.endswith('.weight'):
                    reason='MOE_NOT_IN_DENSE_INTEGRATION' if len(dims)==3 and 'exps' in name else 'NOT_LAYER_DENSE_MATRIX'
                elif q not in QTYPES:
                    reason='NATIVE_FLOAT_OR_UNSUPPORTED_QUANT_TYPE'
                else:
                    reason=None
                if reason:
                    excluded.append(dict(ref,reason=reason));continue
                k,n=dims
                for m in model['ms']:
                    try:
                        w=Workload(q,n,k,m)
                    except ValueError as exc:
                        excluded.append(dict(ref,m=m,reason='NATIVE_ONLY_KPACK_SHAPE_CONTRACT: '+str(exc)))
                        continue
                    point=groups.setdefault(w.key,dict(id=w.key,workload=asdict(w),consumers=[],tensors=[]))
                    point['consumers'].append(ref)
                    point['tensors'].append(model['name']+':'+name)
                    supported+=1
        if not supported:raise ValueError('model has no supported quantized dense matrices: '+model['name'])
    return sorted(groups.values(),key=lambda p:p['id']),excluded


def make_plan(paths, output, *, loader=headers):
    if set(paths)!=set(MODEL_MS):raise ValueError('supply exactly the two requested GGUF models')
    models=[]
    for name,ms in MODEL_MS.items():
        files,_=loader(paths[name])
        models.append(dict(name=name,ms=list(ms),files=files,weights_hashed=False,
                           identity_scope='GGUF_HEADERS_NOT_MODEL_VALUES'))
    rows,excluded=inventory(models)
    policy=json.loads(EVIDENCE.read_text());validate_policy(policy)
    selections=cpp_choices([Workload(**p['workload']) for p in rows],policy,output,bucket_only=False)
    for p,selection in zip(rows,selections):
        case=execution_case(policy['recipes'][selection['recipe']])
        case.update(id=digest(case),roles=[ROLE,'model-config','required-control'])
        p.update(selection=selection,cases=[case],native_comparison=True,random_dense_probe=True,
                 candidates=[case['recipe']] if case['recipe']['kind']=='tc' else [],champion_controls=[],
                 confirmation_captures=2,capture_batch_limit=1,fixed_schedules=True)
    result=dict(schema=SCHEMA,models=models,points=rows,excluded=excluded,reuse=[],inputs=input_identity(policy),
        confirmation_captures=2,precision=PRECISION,native_arithmetic=NATIVE,order=list(ORDER),metric=METRIC,
        timing_method='ASYS_GPU_ACTIVITY',config_sweep=False,allowed_slowdown_pct=0,scope=SCOPE,
        fixture='SYNTHETIC_SHARED_RAW_GGUF_BYTES_MATCHING_MODEL_FORMAT_AND_SHAPE',
        production_changed=False,performance_admitted=False)
    result['counts']=counts(rows);result['sha256']=digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'}) or
            plan.get('scope')!=SCOPE or plan.get('order')!=list(ORDER) or plan.get('precision')!=PRECISION or
            plan.get('native_arithmetic')!=NATIVE or plan.get('metric')!=METRIC or
            plan.get('timing_method')!='ASYS_GPU_ACTIVITY' or plan.get('allowed_slowdown_pct')!=0 or
            plan.get('config_sweep') is not False or plan.get('production_changed') is not False or
            plan.get('performance_admitted') is not False or plan.get('reuse')!=[] or
            plan.get('confirmation_captures')!=2 or
            plan.get('fixture')!='SYNTHETIC_SHARED_RAW_GGUF_BYTES_MATCHING_MODEL_FORMAT_AND_SHAPE'):
        raise ValueError('model/native gate contract changed')
    policy=json.loads(EVIDENCE.read_text());validate_policy(policy)
    if plan['inputs']!=input_identity(policy):raise ValueError('production selector/helpers changed; regenerate plan')
    rows,excluded=inventory(plan['models'])
    if plan['excluded']!=excluded or [p['id'] for p in plan['points']]!=[p['id'] for p in rows]:
        raise ValueError('model/header workload coverage changed')
    for p,expected in zip(plan['points'],rows):
        if any(p[k]!=v for k,v in expected.items()):raise ValueError('model workload/consumer identity changed')
        w=Workload(**p['workload']);coverage,rid=policy_lookup(policy,w)
        kind={'EXACT':20,'BUCKET_PREDICTION':21}.get(coverage)
        if kind is None or p['selection']!=dict(coverage=coverage,policy=kind,recipe=rid):
            raise ValueError('model workload does not match current production selection')
        if (len(p['cases'])!=1 or not p['native_comparison'] or not p['random_dense_probe'] or
                not p['fixed_schedules'] or p['confirmation_captures']!=2 or p['capture_batch_limit']!=1 or
                p['champion_controls']):raise ValueError('model gate must measure exactly selected and native')
        case=p['cases'][0];want=execution_case(policy['recipes'][rid])
        if (case_identity(case)!=want or case['id']!=digest(want) or case['arm']=='fq' or
                case['roles']!=[ROLE,'model-config','required-control'] or
                p['candidates']!=([case['recipe']] if case['recipe']['kind']=='tc' else [])):
            raise ValueError('model gate recipe/helper differs from production or reintroduces prefill FQ')
    if plan['counts']!=counts(plan['points']):raise ValueError('model gate budget/denominator changed')
    return plan


def compare(plan, review):
    validate(plan)
    summary=compare_selected(plan,review)
    by_id={p['id']:p for p in plan['points']}
    for row in summary['points']:
        row['consumers']=by_id[row['point']]['consumers']
    model_summaries=[]
    for name,ms in MODEL_MS.items():
        for m in ms:
            selected=[r for r in summary['points'] if r['workload']['m']==m and
                      any(c['model']==name for c in r['consumers'])]
            tally=Counter(r['status'] for r in selected)
            complete=all(r['status']!='INCOMPLETE' for r in selected)
            def weighted(field):
                return sum(r[field]*sum(c['model']==name for c in r['consumers']) for r in selected)/1000 if complete else None
            native,quactlize=weighted('native_us'),weighted('quactlize_us')
            model_summaries.append(dict(model=name,m=m,expected=len(selected),counts=dict(tally),
                native_dense_matrix_ms=native,quactlize_dense_matrix_ms=quactlize,
                delta_ms=quactlize-native if complete else None,
                scope='STATIC_ONE_CALL_PER_LAYER_WEIGHT_NOT_ACTUAL_GRAPH_COUNTS_OR_MODEL_LATENCY'))
    summary['models']=model_summaries
    return summary
