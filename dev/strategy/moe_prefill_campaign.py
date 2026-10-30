"""Bounded, blind large-M search; no historical times or production writes.

Donor strata and delivery peers select parent geometries. The resource stage
adds a logarithmic persistent-grid ladder and legal Split-K. References are
revealed only after generation and must remain covered, never appended.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import json
import math

from dev.strategy import moe_prefill_plan as audit_plan
from dev.strategy.moe_plan import load_catalog, moe_candidates, RULES
from dev.strategy.prefill_candidates import delivery_peers
from dev.strategy.prefill_plan import full_configs
from dev.strategy.prescreen import Prescreen, rejection
from dev.strategy.resources import module_spec, probe_spec, NativeProbe
from dev.strategy.space import (factored_recall, scheduled_key, split_variants,
                                grid_search, resolved_grid, expansion_configs)
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.moe-prefill-sweep.v1'
RULE = 'BLIND_DONOR_STRATA_DELIVERY_PEERS_NARROW_ROW_FLOOR_V1'
PROTOCOL = dict(screen=[4,3], confirm=[6,5], live_arms=4,
                grid_blocks_per_cu=[1,2,4,8], include_occupancy_endpoint=True,
                finalist_window_pct=5, finalist_limit=32,
                metric='MEAN_OF_SEPARATELY_PROFILED_COMPONENT_SUMS')


def seal(value):
    # Resource replay compares persisted JSON with a fresh rule resolution.
    # Canonicalize Recipe's tuple fields now, not only during file writing.
    return json.loads(json.dumps(dict(value,sha256=digest({k:v for k,v in value.items() if k!='sha256'}))))


def parents(screen, catalog, w):
    """No references or target observations may be inputs to this function."""
    narrow, report = moe_candidates(screen,w,dict(RULES,per_stratum=1,prefill_breadth=False))
    seeds = [r for r in narrow if r.kind=='full' or dict(r.parameters)['split']==1]
    peers, _ = delivery_peers(w,seeds,catalog.observations,portable_sf=True)
    pool = {scheduled_key(r):r for r in [*seeds,*peers]}
    # Token M is not per-expert M. Always retain narrow-row throughput
    # families for dispersed experts, including large total-token requests.
    # Pick the complete tied geometry family, not hash-order top-K.
    broad, _ = moe_candidates(screen,w,dict(RULES,per_stratum=1,prefill_exact_tiles=False))
    strata=defaultdict(list)
    for r in broad:
        p=dict(r.parameters)
        if r.kind=='tc' and p['split']==1 and p['tm']<=32 and not p['ap']:
            strata[r.route,p.get('persistent',-1),int(p['tn']>64)].append(r)
    for rows in strata.values():
        rank=lambda r:(dict(r.parameters)['tm'],dict(r.parameters)['stages'],
                       -min(dict(r.parameters)['tn'],w.n),-dict(r.parameters)['tk'])
        best=min(map(rank,rows))
        for r in rows:
            if rank(r)==best:pool.setdefault(scheduled_key(r),r)
    result=sorted(pool.values(),key=lambda r:r.key)
    if len(result)>32 or {r.route for r in result if r.kind=='tc'}!={'fq','sf'}:
        raise ValueError('blind MoE parent pool empty/unbounded; revise rules, not target injection')
    return result,dict(rule=RULE,parents=len(result),blind=True,champion_injection=False,
                      donor_strata=report['coverage_selected'],
                      dropped='OTHER_GEOMETRIES_DEFERRED_NOT_PROVEN_SLOWER')


def from_audit(audit):
    if audit['sha256']!=digest({k:v for k,v in audit.items() if k!='sha256'}):
        raise ValueError('large-M audit seal differs')
    catalog,_,_=load_catalog();screen=Prescreen(catalog)
    generate=lru_cache(None)(lambda w:parents(screen,catalog,w))
    points=[];modules={}
    for original in audit['points']:
        w=Workload(**original['workload'])
        seeds,why=generate(replace(w,router=''))
        full_legal=any(r['rc']==0 for r in original['full_capability'])
        candidates=[]
        for seed in seeds:
            if seed.kind=='full' and not full_legal:continue
            for r in split_variants(w,seed):
                if not rejection(r,w):candidates.append(r)
        refs=[Recipe.from_dict(r['recipe']) for r in original['historical_references']]
        recall=factored_recall(w,candidates,refs)
        if recall['missing']:
            raise ValueError('bounded blind rule misses historical recipe: '+str((asdict(w),recall)))
        if original.get('original_missing') or not original.get('original_recipe'):
            raise ValueError('missing executable current large-M control')
        arms=[];seen={}
        old=Recipe.from_dict(original['original_recipe'])
        for r,role in [(old,'original'),*[(r,'rule') for r in candidates]]:
            # An unchanged SF/Full producer may still win by selecting a
            # different freshly measured expansion. Keep the exact original
            # helper choice separate from that rule proposal.
            helper_control=role=='original' and r.route in ('sf','full')
            key=(scheduled_key(r),dict(r.parameters).get('dequant_config',-1),helper_control)
            if key in seen:
                arms[seen[key]]['roles'].append(role);continue
            item=dict(id='original' if role=='original' else r.key,recipe=asdict(r),roles=[role],
                      required=role=='original',aliases=[])
            if r.kind=='tc':
                item['spec']=module_spec(w,r);item['module']=digest(item['spec'])
                modules[item['module']]=item['spec']
            seen[key]=len(arms);arms.append(item)
        sf=sorted({max(0,c) for c in expansion_configs(w,'sf')})
        point=dict(id=w.key,workload=asdict(w),consumers=original['consumers'],arms=arms,
                   sf_configs=sf,full_configs=full_configs(w) if full_legal else [],
                   full_capability=original['full_capability'],rule_recall=recall,
                   historical_references=original['historical_references'],rule_reasons=why)
        points.append(point)
    return seal(dict(schema=SCHEMA,cohort=audit_plan.COHORT,audit=audit['sha256'],points=points,
        tc=modules,vectors={},protocol=PROTOCOL,coverage=audit['coverage'],rule=RULE,
        counts=dict(points=len(points),seed_arms=sum(len(p['arms']) for p in points),modules=len(modules),
                    historical_references=sum(p['rule_recall']['controls'] for p in points)),
        historical_times_reused=False,champion_injection=False,production_changed=False,
        scope='MODEL_LOCAL_MOE_PREFILL_BOUNDED_SCREEN_CONFIRM_NOT_NATIVE_OR_MODEL_AB',
        pending=['NATIVE_LARGE_M_AB','PUBLIC_RUNTIME_AND_MODEL_AB','FUSED_GATE_UP_SWIGLU']))


def make_plan(output,inventory):
    audit=audit_plan.make_plan(output,inventory)
    return audit,from_audit(audit)


def query_requests(point):
    w=Workload(**point['workload']);out={}
    for a in point['arms']:
        r=Recipe.from_dict(a['recipe'])
        if r.kind!='tc':continue
        domain=grid_search(w,r) if 'rule' in a['roles'] else None
        modes=set(domain['modes'] if domain else ())|{dict(r.parameters).get('grid_mode',0)}
        for algorithm in sorted({int(m>=2) for m in modes}):
            req=probe_spec(w,r,algorithm)
            out[digest(req)]=dict(request=req,module=a['module'],spec=a['spec'])
    return out


def resolve(point,receipts,images):
    w=Workload(**point['workload']);requests=query_requests(point)
    protocol=point.get('search_protocol',PROTOCOL)
    if set(receipts)!=set(requests):raise ValueError('prefill resource coverage differs')
    devices=set()
    for qid,request in requests.items():
        q=receipts[qid];image=images['tc'][request['module']]
        if (q['request']!=request['request'] or q['module_sha256']!=image['sha256'] or
                q['build_key']!=image['key'] or q['runtime_status']!=0 or
                (q['status'],q['rc']) not in (('READY',0),('UNAVAILABLE',1)) or
                q['status']=='READY' and q['occupancy']<=0 or q['device']['compute_units']<=0):
            raise ValueError('prefill resource/module identity differs')
        devices.add(digest(q['device']))
    if len(devices)>1:raise ValueError('resource queries crossed devices')
    arms=[];seen={};rejected=[];deferred=[]
    def add(a,r,role,q):
        launch=resolved_grid(w,r,q['device']['compute_units'],q['occupancy']) if q else None
        helper_control=role in ('original','old-control') and r.route in ('sf','full')
        signature=digest([a.get('module'),scheduled_key(r),launch,helper_control])
        if signature in seen:
            row=arms[seen[signature]];row['roles']=sorted(set(row['roles']+[role]))
            if r.key not in row['aliases']:row['aliases'].append(r.key)
            return
        out=dict(a,id='original' if role=='original' else r.key,recipe=asdict(r),roles=[role],aliases=[])
        if q:out['binding']=dict(launch=launch,query=digest(q['request']))
        seen[signature]=len(arms);arms.append(out)
    for a in point['arms']:
        r=Recipe.from_dict(a['recipe']);p=dict(r.parameters)
        for role in sorted(set(a['roles']),key=lambda x:x!='original'):
            if r.kind=='full':add(a,r,role,None);continue
            domain=grid_search(w,r) if role=='rule' else None
            modes=domain['modes'] if domain else [p.get('grid_mode',0)]
            for mode in modes:
                q=receipts[digest(probe_spec(w,r,int(mode>=2)))]
                if q['status']!='READY':
                    if role=='original':raise ValueError('current prefill control resource unavailable')
                    rejected.append(dict(recipe=r.key,algorithm=int(mode>=2),query=q));continue
                cu,occ=q['device']['compute_units'],q['occupancy']
                # Lower bound assumes the most concentrated legal routing:
                # top-k experts each receive all tokens. With two resident
                # waves already available, defer extra K slices/reducers.
                blocks=w.topk*math.ceil(w.m/p['tm'])*math.ceil(w.n/p['tn'])
                if role=='rule' and p['split']>1 and blocks>=2*cu*occ:
                    deferred.append(dict(recipe=r.key,algorithm=int(mode>=2),
                        reason='TWO_RESIDENT_WAVES_WITHOUT_SPLIT',minimum_blocks=blocks,capacity=cu*occ));continue
                bs=([p.get('grid_b',0)] if not domain else [0] if mode==0 else
                    sorted(set(protocol['grid_blocks_per_cu'])|{min(32,occ)}))
                for b in bs:
                    variant=Recipe.make(r.kind,r.route,**(p|dict(grid_mode=mode,grid_b=b)))
                    add(a,variant,role,q)
    # Prove each historical recipe remains an actual launch, not just a
    # parent tuple whose required grid/Split-K was subsequently pruned.
    for ref in point['historical_references']:
        r=Recipe.from_dict(ref['recipe']);p=dict(r.parameters)
        if r.kind=='full':continue
        equivalent=next((Recipe.from_dict(a['recipe']) for a in point['arms']
                         if scheduled_key(Recipe.from_dict(a['recipe']))==scheduled_key(r)),None)
        q=receipts.get(digest(probe_spec(w,equivalent,int(p.get('grid_mode',0)>=2)))) if equivalent else None
        if q is None or q['status']!='READY':raise ValueError('historical launch resource gap')
        launch=resolved_grid(w,r,q['device']['compute_units'],q['occupancy'])
        if not any('rule' in a['roles'] and scheduled_key(Recipe.from_dict(a['recipe']))==scheduled_key(r)
                   and a.get('binding',{}).get('launch')==launch for a in arms):
            raise ValueError('bounded grid/Split-K rule omitted historical launch; revise rule')
    if not arms or not any(a['id']=='original' for a in arms) or len(arms)>384:
        raise ValueError('empty/missing-control/unbounded resource-resolved prefill pool')
    return seal(dict(point,arms=arms,resource_rejected=rejected,rule_deferred=deferred,
                     resolved=True,seed_sha256=digest(point)))


def bind(point,images):
    probes={};receipts={}
    for qid,request in query_requests(point).items():
        mid=request['module']
        if mid not in probes:probes[mid]=NativeProbe(request['spec'],images['tc'][mid])
        receipts[qid]=probes[mid].query(request['request'])
    return resolve(point,receipts,images),receipts


def finalists(point,statistics):
    if set(statistics)!={a['id'] for a in point['arms']}:raise ValueError('incomplete prefill screen')
    chosen={'original'};strata=defaultdict(list)
    floor=min(r['mean_us'] for r in statistics.values())*1.05
    for a in point['arms']:
        aid=a['id'];r=statistics[aid]
        strata[a['recipe']['route'],dict(a['recipe']['parameters']).get('persistent',-1)].append(aid)
        if r['mean_us']<=floor or min(map(lambda x:sum(x)/len(x),r['samples_us']))<=floor:chosen.add(aid)
    for ids in strata.values():chosen.add(min(ids,key=lambda a:(statistics[a]['mean_us'],a)))
    if len(chosen)>point.get('search_protocol',PROTOCOL)['finalist_limit']:
        raise ValueError('too many competitive finalists; screen saved, no silent top-K truncation')
    return sorted(chosen,key=lambda a:(a!='original',statistics[a]['mean_us'],a))


def assignments(plan,count):
    """Whole point LPT; include validation volume and a per-point floor."""
    if not 1<=count<=len(plan['points']):raise ValueError('invalid prefill device count')
    def weight(p):
        w=p['workload']
        return len(p['arms'])*(1+w['m']*w['topk']*w['n']/2**20)+len(p['full_configs'])*20
    out=[[] for _ in range(count)];loads=[0.]*count
    for p in sorted(plan['points'],key=lambda p:(-weight(p),p['id'])):
        j=min(range(count),key=lambda i:(loads[i],i));out[j].append(p['id']);loads[j]+=weight(p)
    return out
