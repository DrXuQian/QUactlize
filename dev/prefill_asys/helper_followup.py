"""Bounded reducer-grain/order diagnostics, with the actual production control."""
from dataclasses import asdict
import copy
import json

from .finalists import case_identity, counts, full_case
from .supplement import execution_case
from dev.strategy.prescreen import rejection
from quactlize.dispatch.dense_prefill import EVIDENCE, validate as validate_catalog
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.dense-helper-followup.v1'


def grain(case, width):
    result = copy.deepcopy(case)
    r = Recipe.from_dict(result['recipe']);p = dict(r.parameters)
    if (r.kind != 'tc' or r.route != 'sf' or p.get('dense_reducer') or
            p['split'] not in (2,4,8) or p.get('grid_mode',0) or width not in (2,4,8)):
        raise ValueError('grain comparison needs unmodified ordinary SF Split-K')
    p.update(dense_reducer=width,symbol=p['symbol']+f'_rc{width}')
    result['recipe'] = json.loads(json.dumps(asdict(Recipe.make(r.kind,r.route,**p))))
    return result


def reordered(case):
    result = copy.deepcopy(case);r = Recipe.from_dict(result['recipe'])
    if r.kind != 'full' or dict(r.parameters).get('dequant_first'):
        raise ValueError('order diagnostic needs the unchanged Full control')
    result['recipe'] = json.loads(json.dumps(asdict(Recipe.make(r.kind,r.route,**dict(r.parameters),dequant_first=True))))
    return result


def make_plan(review, catalog=None):
    catalog = catalog or json.loads(EVIDENCE.read_text());validate_catalog(catalog)
    if (review.get('review_sha256') != digest({k:v for k,v in review.items() if k != 'review_sha256'}) or
            any(review.get(k) for k in ('incomplete','pending_points','partial_points','diagnostics'))):
        raise ValueError('complete helper review required')
    updates = [u for u in catalog.get('helper_updates',[]) if u['review_sha256'] == review['review_sha256']]
    if len(updates) != 1:raise ValueError('review has not been audited into the production catalog')
    exact = {p['key']:p for p in catalog['points']}
    targets = {p['key']:{'missing-incumbent'} for p in updates[0]['retained_pending']}
    sf_controls, full_controls, small = {}, {}, []
    for key,p in review['points'].items():
        w = Workload(**p['workload'])
        if w.qtype != 8 or w.operation != 'dense' or w.compute != 'f16':raise ValueError('Q8 dense-only followup')
        rows = p['measurements']
        for r in rows:
            if r['status'] != 'PASS' or len(r.get('confirmation_captures',[])) != 2:
                raise ValueError('followup needs confirmed complete calls')
        sf = next(r for r in rows if r['roles'] == ['prior-route-winner:sf'])
        full = min((r for r in rows if r['arm'] == 'full-fp16'),key=lambda r:(r['gpu_sum_us'],r['candidate']))
        full_controls[key] = review['candidates'][full['candidate']]
        native = next(r for r in rows if r['arm'] == 'native-fp16')
        best = min((r for r in rows if r['arm'] != 'native-fp16'),key=lambda r:r['gpu_sum_us'])
        if best['arm'] == 'full-fp16' and best['gpu_sum_us'] > native['gpu_sum_us']:
            targets.setdefault(key,set()).add('full-order-residual')
        compact = next((r for r in rows if r['roles'] == ['compact-reducer:sf']),None)
        if compact:
            sf_controls[key] = review['candidates'][sf['candidate']]
            delta = compact['phases_us']['reducer']/sf['phases_us']['reducer']-1
            if delta > .2:targets.setdefault(key,set()).add('large-output-reducer')
            elif delta < 0:small.append((w.m*w.n,key))
    # Two output-size extremes are regression controls, not another geometry sweep.
    small.sort()
    for _,key in ([small[0],small[-1]] if small else []):
        targets.setdefault(key,set()).add('small-output-control')
    points=[]
    for key,reasons in sorted(targets.items()):
        old=review['points'][key];pool={};w=Workload(**old['workload'])
        def add(case,role):
            cid=digest(case);c=pool.setdefault(cid,dict(copy.deepcopy(case),id=cid,roles=[]))
            c['roles'].append(role);return cid
        current=execution_case(catalog['recipes'][exact[key]['winner']]);current_id=add(current,'production-control')
        rows=[r for r in old['measurements'] if r['arm']!='native-fp16']
        winner=min(rows,key=lambda r:(r['gpu_sum_us'],r['candidate']))
        add(review['candidates'][winner['candidate']],'review-winner')
        if reasons & {'large-output-reducer','small-output-control'}:
            control=copy.deepcopy(sf_controls[key]);control['helper_options']=dict(vector_io=4)
            add(control,'grain-control')
            for width in (2,4,8):add(grain(control,width),f'grain:{width}')
        if 'full-order-residual' in reasons:
            control=full_controls[key];add(control,'order-control');add(reordered(control),'dequant-before-input')
        cases=list(pool.values())
        points.append(dict(id=key,workload=asdict(w),consumers=old['consumers'],tensors=[],cases=cases,
            reasons=sorted(reasons),production_control=current_id,evidence_sha256=digest(old),
            candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],champion_controls=[],
            fixed_schedules=True,confirmation_captures=2))
    result=dict(schema=SCHEMA,points=points,reuse=[],confirmation_captures=2,counts=counts(points),
        inputs=dict(review_sha256=review['review_sha256'],catalog_sha256=catalog['sha256'],
                    helper_policy_sha256=digest(json.loads((EVIDENCE.parent/'dense_full_helpers_20261023.json').read_text()))),
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        scope='BOUNDED_GRAIN_AND_ORDER_DIAGNOSTICS_NOT_MODEL_TPOT',
        historical_timing_used='TARGET_SELECTION_ONLY_ALL_CONTROLS_REMEASURED',
        production_changed=False,performance_admitted=False)
    result['sha256']=digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'}) or
            plan.get('confirmation_captures')!=2 or plan.get('production_changed') is not False or
            plan.get('performance_admitted') is not False or plan.get('reuse')!=[]):
        raise ValueError('helper followup identity differs')
    seen=set();reason_counts={k:0 for k in ('large-output-reducer','small-output-control',
                                          'full-order-residual','missing-incumbent')}
    for p in plan['points']:
        w=Workload(**p['workload']);pool={c['id']:c for c in p['cases']}
        if (p['id']!=w.key or w.key in seen or w.qtype!=8 or w.operation!='dense' or
                w.compute!='f16' or w.storage!='f32' or w.m not in (128,256,512,1024,2048) or
                not 2<=len(pool)==len(p['cases'])<=8 or not p['fixed_schedules'] or p['confirmation_captures']!=2):
            raise ValueError('followup workload/budget differs')
        seen.add(w.key)
        if not p['reasons'] or len(set(p['reasons']))!=len(p['reasons']) or not set(p['reasons'])<=reason_counts.keys():
            raise ValueError('unknown followup scope')
        for reason in p['reasons']:reason_counts[reason]+=1
        role={}
        for c in p['cases']:
            if c['id']!=digest(case_identity(c)):raise ValueError('followup case identity differs')
            for tag in c['roles']:
                if tag in role:raise ValueError('duplicate followup control role')
                role[tag]=c
            r=Recipe.from_dict(c['recipe'])
            if r.kind=='tc':
                if rejection(r,w) or c['arm']!='sf' or c['helper_mode']!='cohort-current':
                    raise ValueError('invalid followup SF configuration')
            elif r.kind=='full':
                o=c['helper_options'];base=full_case(o['dequant_variant'],o['vector_io'])
                expected=reordered(base) if dict(r.parameters).get('dequant_first') else base
                if o['dequant_variant'] not in (0,22,23,36,37,38,41) or o['vector_io'] not in range(5) or case_identity(c)!=expected:
                    raise ValueError('followup Full semantics differ')
            else:raise ValueError('unexpected followup implementation')
        if role.get('production-control',{}).get('id')!=p['production_control'] or 'review-winner' not in role:
            raise ValueError('followup lost incumbent/winner control')
        required={'production-control','review-winner'}
        if set(p['reasons']) & {'large-output-reducer','small-output-control'}:
            required.update(('grain-control','grain:2','grain:4','grain:8'))
            for width in (2,4,8):
                if case_identity(role.get(f'grain:{width}',{}))!=grain(case_identity(role['grain-control']),width):
                    raise ValueError('grain candidate changes more than reducer width')
        if 'full-order-residual' in p['reasons']:
            required.update(('order-control','dequant-before-input'))
            if case_identity(role.get('dequant-before-input',{}))!=reordered(case_identity(role['order-control'])):
                raise ValueError('order diagnostic changes more than stage ordering')
        if set(role)!=required:raise ValueError('unplanned followup candidate role')
        if p['candidates']!=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc'] or p['champion_controls']:
            raise ValueError('followup resource inventory differs')
    if (not 1<=len(seen)<=16 or plan['counts']!=counts(plan['points']) or
            any(reason_counts[k]>limit for k,limit in zip(reason_counts,(6,2,4,4)))):
        raise ValueError('bounded followup exceeds budget or count differs')
    return plan
