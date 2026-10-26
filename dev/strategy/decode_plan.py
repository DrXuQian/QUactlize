"""Consume frozen resource-resolved rules; reveal original controls afterwards."""
from dataclasses import asdict
import json
from pathlib import Path

from dev.strategy.journal import planned_arms
from dev.strategy.resources import module_spec, probe_spec
from dev.strategy.space import resolved_grid, expansion_configs
from dev.strategy.vector import spec as vector_spec
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize


def original_requests(probe):
    return [[w.qtype, 2 if w.operation=='indexed' else 0, w.n, w.k, w.experts,
             w.topk, w.channels, w.m, int(w.compute=='bf16')]
            for point in probe['resolved']['points'] if (w:=Workload(**point['workload'])).m<=8]


def original_recipe(row):
    if row.get('status') != 0:
        raise ValueError('original selector miss; do not silently benchmark a fallback')
    if row['kind']==0:
        return normalize(dict(row['parent'],kind='tc',split=row['split'],grid_mode=row['grid_mode'],grid_b=row['grid_b']))
    if row['kind'] not in (1,2): raise ValueError('unknown original selector implementation')
    config = dict(row['config'],kind='simt' if row['kind']==1 else 'q4')
    if row.get('implementation', {}).get('q8_body'):
        config['q8_body'] = row['implementation']['q8_body']
    impl = row.get('implementation', {})
    if impl.get('cta_tokens'):
        config['cta_tokens'] = impl['cta_tokens']
        if row['request'][0] == 8:
            config['cta_window'] = impl['cta_window']
    return normalize(config)


def make_decode_plan(probe, originals, *, pilot=True):
    requests=original_requests(probe)
    if [r.get('request') for r in originals] != requests:
        raise ValueError('original selector requests differ from frozen small-M cohort')
    points=[]; vectors={}; modules=set(); omitted=[]; blocked=[]
    for original, point in zip(originals, [p for p in probe['resolved']['points'] if p['workload']['m']<=8]):
        w=Workload(**point['workload'])
        if w.operation not in ('dense','indexed') or w.storage!='f32':
            raise ValueError('small-M runner requires dense/indexed F32 endpoints')
        p=dict(point,original_recipe=asdict(original_recipe(original)),original_selector=original)
        arms=[];unavailable=[]
        for arm in planned_arms(p):
            r=Recipe.from_dict(arm['recipe'])
            if r.kind not in ('tc','simt','q4') or r.route not in ('fq','sf'):
                raise ValueError('small-M plan contains an unsupported producer')
            item=dict(arm)
            if r.kind=='tc':
                spec=module_spec(w,r); mid=digest(spec)
                algorithm=int(dict(r.parameters).get('grid_mode',0)>=2)
                qid=digest(probe_spec(w,r,algorithm))
                row=probe['receipts'].get(qid)
                if mid not in probe['inventory']['modules'] or row is None:
                    raise ValueError('original/control module is not in the ready resource cache: '+str(r))
                if row['status']!='READY':
                    if pilot:
                        raise ValueError('original/control module is not in the ready resource cache: '+str(r))
                    unavailable.append(dict(arm=arm['id'],roles=arm['roles'],query=qid,
                                            reason='REQUIRED_RECIPE_RESOURCE_UNAVAILABLE'))
                    continue
                launch=resolved_grid(w,r,row['device']['compute_units'],row['occupancy'])
                item.update(spec=spec,module=mid,binding=dict(query=qid,launch=launch))
                modules.add(mid)
            else:
                spec=vector_spec(w,r); mid=digest(spec)
                vectors[mid]=spec;item.update(spec=spec,module=mid)
            arms.append(item)
        if unavailable:
            # Keep the requested denominator. A missing incumbent must never
            # become a silent fallback comparison or stop unrelated weights.
            blocked.append(dict(id=p['id'],workload=p['workload'],unavailable=unavailable))
            continue
        if any(a['recipe']['route']=='sf' for a in arms) and w.qtype!=8:
            # metadata.cu's legacy F16 prepass and dequant config0 instantiate
            # the same canonical UnitPrepass body/grid/block256 (host tested).
            p['sf_configs']=sorted({max(0,c) for c in expansion_configs(w,'sf')})
            p['sf_config_aliases']={'-1':0}
        else:
            p['sf_configs']=[]
        p['arms']=arms
        points.append(p)
    for point in probe['resolved']['points']:
        if point['workload']['m']>8:
            omitted.append(dict(point=point['id'],reason='SEPARATE_LARGE_M_FULL_DEQUANT_PROVIDER_PHASE'))
    if (not points and not blocked) or pilot and len(points)>5: raise ValueError('bounded resource decode pilot requires 1..5 points')
    # Blocked points may have introduced specs before their missing incumbent.
    live_vectors={a['module'] for p in points for a in p['arms'] if a['recipe']['kind']!='tc'}
    vectors={k:v for k,v in vectors.items() if k in live_vectors}
    modules={a['module'] for p in points for a in p['arms'] if a['recipe']['kind']=='tc'}
    result=dict(schema='quactlize.rule-decode-cold.v1',resource_plan=probe['resolved']['plan_sha256'],
        points=points,vectors=vectors,tc_modules=sorted(modules),omitted=omitted,blocked=blocked,
        counts=dict(points=len(points),requested_points=len(points)+len(blocked),blocked_points=len(blocked),
            arms=sum(len(p['arms']) for p in points),
            vector_modules=len(vectors),reused_tc_modules=len(modules),
            expansion_jobs=sum(len(p['sf_configs']) for p in points)),
        precision='DENSE_F32_STORAGE_F16_COMPUTE_INDEXED_F32_STORAGE_BF16_COMPUTE',
        production_changed=False,full_campaign=False,
        full_smallm_cohort=not pilot and not bool(probe['resolved'].get('point_filter')),
        point_filter=probe['resolved'].get('point_filter',{}),
        comparison='CURRENT_SELECTED_BODY_VS_BLIND_RULES_VS_SEPARATE_CHAMPION_CONTROLS',
        cost_scope='SUM_OF_ISOLATED_KERNELS_NOT_MODEL_OR_MEASURED_E2E',
        final_admission='ORIGINAL_WIRING_MODEL_AB_STILL_REQUIRED')
    result['plan_sha256']=digest(result)
    return json.loads(json.dumps(result))


def execution_gate(probe, directories):
    from dev.strategy.resource_pilot import smoke_cases,verified_plan,sealed,validate_pass,module_record
    expected={c['id']:c for c in smoke_cases(probe)['cases']};passed={}
    for directory in directories:
        root=Path(directory).resolve(strict=True)
        plan=verified_plan(json.loads((root/'plan.json').read_text()))
        authority=json.loads((root/'authority.json').read_text())
        if authority.get('resource')!=probe['authority'] or authority.get('plan')!=plan['plan_sha256']:
            raise ValueError('execution gate uses another resource probe or plan')
        for case in plan['cases']:
            if expected.get(case['id'])!=case:raise ValueError('execution gate case differs')
            path=root/'cases'/(case['id']+'.json')
            if not path.exists():continue
            data=sealed(path,dict(authority=authority,case=case))
            if data.get('status')!='PASS':continue
            validate_pass(case,data)
            record=module_record(probe,case)
            if (data['receipt']['module_sha256']!=record['sha256'] or data['receipt']['build_key']!=record['key']):
                raise ValueError('execution gate ran another module')
            passed[case['id']]=dict(path=str(path),receipt_sha256=digest(data))
    if set(passed)!=set(expected):raise ValueError(f'execution gate coverage {len(passed)}/{len(expected)}; no cold screen started')
    return dict(scope='PRIOR_TC_MECHANISM_GATE_NOT_PERFORMANCE',passed=passed)
