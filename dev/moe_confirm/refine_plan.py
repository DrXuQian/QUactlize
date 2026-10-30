"""Add missing geometry strata without reading target winners or latencies."""
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import math

from dev.moe_confirm import plan as confirmation
from dev.strategy import moe_prefill_campaign as old
from dev.strategy.moe_plan import load_catalog
from dev.strategy.prescreen import Prescreen,rejection
from dev.strategy.resources import module_spec
from dev.strategy.space import prototype,rebind_template,split_variants
from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.tuning import digest

RULE='BLIND_EXACT_TM_TN_TWO_WARP_LAYOUTS_OBSERVED_DELIVERY_V2'
PROTOCOL=dict(old.PROTOCOL,grid_blocks_per_cu=[1,2,4],finalist_limit=48)
FIELDS=('tm','tn','tk','wm','wn','stages','ap','dn','persistent')


def geometry(r):
    p=dict(r.parameters)
    return (r.kind,r.route,*(p.get(k,-1) for k in FIELDS))


def parents(w,table,screen):
    """Geometry transfer only. No hard M-distance cutoff and no target N/K."""
    previous,_=old.parents(screen,table,w)
    pool={geometry(r):r for r in previous}
    donors=defaultdict(list)
    for observation in table.observations:
        ow,r=observation.workload,observation.recipe
        if (r.kind!='tc' or ow.m<=8 or (ow.n,ow.k)==(w.n,w.k) or
                ow.qtype not in range(10,15) or ow.fusion!=w.fusion or
                ow.input_stride!=ow.k or ow.output_stride!=ow.n or
                r.route=='fq' and ow.operation!='indexed'):
            continue
        r=rebind_template(prototype(r),w.qtype);p=dict(r.parameters)
        if p['tm'] not in (16,32,64,128) or p['ap'] or rejection(r,w):continue
        # Different warp ownership is not interchangeable merely because the
        # CTA tile matches. Keep up to two observed layouts per exact tile.
        stratum=(r.route,p['tm'],p['tn'],p.get('persistent',-1))
        distance=abs(math.log2(ow.n/w.n))+abs(math.log2(ow.k/w.k))+abs(math.log2(ow.m/w.m))
        rank=(ow.qtype!=w.qtype,ow.operation!=w.operation,ow.compute!=w.compute,distance)
        donors[stratum].append((rank,r))
    for rows in donors.values():
        layouts=defaultdict(list)
        for rank,r in rows:
            p=dict(r.parameters);layouts[p['wm'],p['wn']].append((rank,r))
        chosen=sorted(layouts,key=lambda k:(min(x[0] for x in layouts[k]),k))[:2]
        for warp in chosen:
            # One observed K/pipeline geometry per warp layout; keep its DN
            # variants. This is not a tile x stage x warp Cartesian sweep.
            _,seed=min(layouts[warp],key=lambda x:(x[0],geometry(x[1])))
            sp=dict(seed.parameters)
            for _,r in layouts[warp]:
                p=dict(r.parameters)
                if (p['tk'],p['stages'])==(sp['tk'],sp['stages']):pool.setdefault(geometry(r),r)
    out=sorted(pool.values(),key=geometry)
    if len(out)>96:raise ValueError('geometry pool exceeds96; inspect coverage before increasing the budget')
    return previous,out


@lru_cache(maxsize=1)
def inventory():
    table,_,sources=load_catalog();screen=Prescreen(table)
    data=confirmation.make_plan('prefill');groups={}
    for p in data['points']:groups.setdefault(confirmation.key(p['workload']),p)
    report=[];generated={}
    for key,p in sorted(groups.items()):
        w=Workload(**(p['workload']|dict(router='')))
        before,after=parents(w,table,screen)
        known={geometry(r) for r in before if r.kind=='tc'}
        added=[r for r in after if r.kind=='tc' and geometry(r) not in known]
        full_before=128<=w.m<=4096;full_after=128<=w.m<=8192
        report.append(dict(key=list(key),old_parents=len(before),expanded_parents=len(after),
            missing_geometries=[asdict(r) for r in added],new_full=full_after and not full_before,
            old_full=full_before,full=full_after,scope='HOST_GEOMETRY_COVERAGE_NOT_PERFORMANCE'))
        generated[key]=added
    return report,generated,sources


def make_plan():
    audit,missing,sources=inventory();points=[];modules={}
    # Workload scope, not candidate selection: only the two measured native
    # regressions. The audit above covers all40 host keys independently.
    priorities={(13,n,512,8,5120) for n in (2048,3072)}
    current={confirmation.key(p['workload']):p['arms'][0]['recipe']
             for p in confirmation.wired_plan()['points']}
    for point in confirmation.make_plan('prefill')['points']:
        key=confirmation.key(point['workload'])
        if key not in priorities:continue
        w=Workload(**point['workload']);arms=[];seen=set()
        def add(recipe,roles,aid=None):
            signature=digest(recipe)
            if signature in seen:
                row=next(a for a in arms if digest(a['recipe'])==signature)
                row['roles']=sorted(set(row['roles']+roles));return
            seen.add(signature)
            r=Recipe.from_dict(recipe)
            a=dict(id=aid or r.key,recipe=recipe,roles=roles,aliases=[],required=aid=='original')
            if r.kind=='tc':
                a['spec']=module_spec(w,r);a['module']=digest(a['spec']);modules[a['module']]=a['spec']
            arms.append(a)
        add(current[key],['original'],'original')
        add(point['arms'][0]['recipe'],['old-control'])
        for seed in missing[key]:
            for r in split_variants(w,seed):
                if not rejection(r,w):add(asdict(r),['rule'])
        add(asdict(Recipe.make('full','full',dequant_config=4,provider='bf16-default')),['rule'])
        points.append(dict(id=w.key,workload=asdict(w),arms=arms,sf_configs=[0,4,5],
            full_configs=[4,5,10,11],historical_references=[],search_protocol=PROTOCOL,
            selection_source='EXPLICIT_REFINEMENT_NOT_PRODUCTION',
            scope='MISSING_GEOMETRIES_PLUS_CURRENT_OLD_AND_FULL_CONTROLS'))
    return dict(schema='quactlize.moe-prefill-refine.v1',points=points,tc=modules,vectors={},
        counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),tc_modules=len(modules),
            vector_modules=0,maximum_candidates=max(len(p['arms']) for p in points)),
        coverage=dict(rule=RULE,catalog_sources=sources,host_keys=audit,
                      keys_with_gaps=sum(bool(a['missing_geometries'] or a['new_full']) for a in audit)),
        protocol=PROTOCOL,historical_times_reused=False,champion_injection=False,production_changed=False,
        baseline='CURRENT_AND_OLD_CONFIGS_WITH_CURRENT_HELPERS',
        scope='BOUNDED_MISSING_MOE_GEOMETRIES_NOT_PRODUCTION_OR_MODEL')


def finalists(points,screens):
    """Every routing profile confirms the same union, not different winners."""
    groups=defaultdict(list)
    for p in points:groups[confirmation.key(p['workload'])].append(p)
    result={}
    for group in groups.values():
        ids=set()
        for p in group:
            value=screens[p['id']]
            ids.update(old.finalists(p,value['decision']['statistics']))
            ids.update(a['id'] for a in p['arms'] if 'old-control' in a['roles'])
        common=set.intersection(*({a['id'] for a in p['arms']} for p in group))
        if not ids<=common:raise ValueError('resource availability differs across routing profiles')
        if len(ids)>48:raise ValueError('more than48 common finalists; retain screens and inspect before confirming')
        helper={}
        for route in ('sf','full'):
            costs=[]
            for p in group:
                s=screens[p['id']]
                costs.append({c:sum(s['component_records'][i]['mean_us'] for i in components)
                              for c,components in s['helper_components'][route].items()})
            common_helpers=set.intersection(*(set(c) for c in costs))
            if not common_helpers:raise ValueError('missing shared expansion choices')
            helper[route]=int(min(common_helpers,key=lambda c:
                (max(s[c]/min(s.values()) for s in costs),sum(s[c] for s in costs),c)))
        for p in group:
            arms=[]
            for a in p['arms']:
                if a['id'] not in ids:continue
                r=Recipe.from_dict(a['recipe']);params=dict(r.parameters)
                if r.route in helper and a['id']!='original' and 'old-control' not in a['roles']:
                    r=Recipe.make(r.kind,r.route,**(params|dict(dequant_config=helper[r.route])))
                    a=dict(a,id=r.key,recipe=asdict(r))
                if not any(b['id']==a['id'] for b in arms):arms.append(a)
            result[p['id']]=dict(p,arms=arms,
                sf_configs=sorted({dict(a['recipe']['parameters'])['dequant_config'] for a in arms if a['recipe']['route']=='sf'}),
                full_configs=[helper['full']],explicit_helpers=True)
    return result
