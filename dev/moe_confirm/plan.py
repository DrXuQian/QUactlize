"""A bounded confirmation of completed sweeps, not another config generator."""
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

from dev.strategy.resources import module_spec,NativeProbe,probe_spec
from dev.strategy.space import resolved_grid
from dev.strategy.vector import spec as vector_spec
from quactlize.dispatch.table import Recipe,Workload
from quactlize.execution.moe_decode_policy import FAMILIES,ROUTERS,load
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
SEED=Path(__file__).with_name('prefill_seed.json')
MS=(16,32,64,128,256,512,1024,2048,4096,5120)


def key(w):
    return tuple(w[k] for k in ('qtype','n','k','channels','m'))


def canonical(binding):
    recipe=Recipe.from_dict(binding['recipe']);p=dict(recipe.parameters)
    if recipe.route in ('sf','full'):
        config=binding.get('helper_config')
        if type(config)!=int or config<0:raise ValueError('missing measured expansion choice')
        p['dequant_config']=config
    return asdict(Recipe.make(recipe.kind,recipe.route,**p))


def seed(plan,summary):
    """Import winners plus a common confirmed representative of each family.

    A missing historical point is explicit and gets the same fresh inventory
    as its siblings. Historical latency never becomes a new confirmation.
    """
    by_id={p['id']:p for p in plan['points']};groups=defaultdict(list)
    if len(by_id)!=200 or len(summary['points'])!=200:raise ValueError('expected 200 prefill points')
    for row in summary['points']:
        point=by_id[row['point']];groups[key(point['workload'])].append((point,row))
    families=[]
    for host,group in sorted(groups.items()):
        recipes={};chosen=set();references=[];originals=set();profiles=[]
        for point,row in group:
            original=point['arms'][0]
            if original['id']!='original':raise ValueError('missing original recipe')
            # Original's explicit helper is part of its original recipe.
            binding=(row.get('bindings') or {}).get('original')
            if binding is None:
                params=dict(original['recipe']['parameters'])
                binding=dict(recipe=original['recipe'],helper_config=params.get('dequant_config'))
            r=canonical(binding);cid=digest(r);recipes[cid]=r;originals.add(cid);chosen.add(cid)
            costs={}
            if row['status']=='PASS':
                for aid,b in row['bindings'].items():
                    r=canonical(b);cid=digest(r);recipes[cid]=r
                    value=row['decision']['statistics'][aid]['mean_us']
                    if not math.isfinite(value) or value<=0:raise ValueError('nonfinite historical cost')
                    costs[cid]=min(costs.get(cid,float('inf')),value)
                winner=digest(canonical(row['bindings'][row['decision']['winner']]))
                chosen.add(winner)
                references.append(dict(point=point['id'],router=point['workload']['router'],winner=winner))
            profiles.append(costs)
        if len(originals)!=1:raise ValueError('host key has router-dependent original')
        if {p['workload']['router'] for p,_ in group}!=set(ROUTERS):raise ValueError('router coverage differs')
        # Do not drop a family just because a small helper improvement could
        # change its crossover. Use common confirmed costs, not an unconfirmed
        # screen minimum. Full remains one provider with explicit expansion.
        complete=[p for p in profiles if p]
        common=set.intersection(*(set(p) for p in complete))
        for route in ('fq','sf','full'):
            candidates=[c for c in common if recipes[c]['route']==route]
            if candidates:
                chosen.add(min(candidates,key=lambda c:(max(p[c]/min(p.values()) for p in complete),
                                                       sum(p[c] for p in complete),c)))
        families.append(dict(key=list(host),original=next(iter(originals)),
            candidates={c:recipes[c] for c in sorted(chosen)},champions=references,
            historical_missing=[p['id'] for p,r in group if r['status']!='PASS']))
    return dict(schema='quactlize.moe-confirm-seed.v1',source=summary['source_revision'],
        old_plan_sha256=plan['sha256'],families=families,historical_times_reused=False)


def make_plan(mode='all',*,native_compare=False):
    if mode=='refine-prefill':
        from dev.moe_confirm.refine_plan import make_plan as refine
        p=refine();p['native_comparison']=native_compare;return p
    if native_compare and mode!='wired-prefill':raise ValueError('native gate requires selected wiring')
    if mode=='wired-prefill':
        p=wired_plan()
        if native_compare:
            p.update(native_comparison=True,baseline='CURRENT_LINKED_NATIVE_UNMODIFIED',
                protocol=dict(order=['native','quactlize','quactlize','native'],rounds=6,samples=5,
                    timing='ASYS_SEPARATE_COMPONENT_MEANS',helper_reuse=False),
                scope='SELECTED_MOE_PROJECTION_COMPONENT_AB_NOT_FUSED_FFN_OR_MODEL')
            p['counts'].update(native_passes=400,candidate_passes=400)
        return p
    if mode not in ('all','prefill','decode'):raise ValueError('unknown confirmation mode')
    points=[];modules={};vectors={}
    def add(w,candidates,original,reference):
        w=asdict(w);arms=[]
        for cid,r in sorted(candidates.items(),key=lambda p:p[0]!=original):
            recipe=Recipe.from_dict(r)
            arm=dict(id='original' if cid==original else cid,recipe=r,
                     roles=['original' if cid==original else 'confirmed-shortlist'],aliases=[])
            if recipe.kind=='tc':
                arm['spec']=module_spec(Workload(**w),recipe);arm['module']=digest(arm['spec'])
                modules[arm['module']]=arm['spec']
            elif recipe.kind!='full':
                arm['spec']=vector_spec(Workload(**w),recipe);arm['module']=digest(arm['spec'])
                vectors[arm['module']]=arm['spec']
            arms.append(arm)
        sf=sorted({dict(r['parameters'])['dequant_config'] for r in candidates.values() if r['route']=='sf'})
        full=sorted({dict(r['parameters'])['dequant_config'] for r in candidates.values() if r['kind']=='full'})
        points.append(dict(id=digest(w),workload=w,arms=arms,sf_configs=sf,full_configs=full,
                           original_canonical=original,reference=reference,fixed_candidates=True))
    if mode!='decode':
        data=json.loads(SEED.read_text())
        expected={(q,n,k,ch,m) for q,n,k,ch in FAMILIES for m in MS}
        if ({tuple(f['key']) for f in data['families']}!=expected or
                len(data['families'])!=len(expected)):raise ValueError('prefill seed denominator differs')
        for f in data['families']:
            q,n,k,ch,m=f['key']
            for router in ROUTERS:
                w=Workload(q,n,k,m,operation='indexed',experts=256,topk=8,channels=ch,compute='bf16',router=router)
                add(w,f['candidates'],f['original'],dict(seed_sha256=sha(SEED),
                    historical_missing=f['historical_missing'],champions=f['champions']))
    if mode!='prefill':
        data=load()
        for row in data['rows']:
            q,_,n,k,_,_,ch,m,_=row['key'];profiles=row['profiles']
            first=profiles[0]
            ids={row['selected'],first['original']}
            candidates={cid:first['candidates'][cid]['recipe'] for cid in ids}
            for profile in profiles:
                w=Workload(q,n,k,m,operation='indexed',experts=256,topk=8,channels=ch,
                           compute='bf16',router=profile['router'])
                add(w,candidates,first['original'],dict(policy_selected=row['selected'],
                    source=data['source_revision'],new_decode_search=False))
    if len({p['id'] for p in points})!=len(points):raise ValueError('duplicate workloads')
    return dict(schema='quactlize.moe-confirm.v1',points=points,tc=modules,vectors=vectors,
        counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),
                    tc_modules=len(modules),vector_modules=len(vectors),
                    maximum_candidates=max(len(p['arms']) for p in points)),
        historical_times_reused=False,production_changed=False,native_comparison=False,
        baseline='OLD_CONFIG_WITH_THE_SAME_CURRENT_QUACTLIZE_HELPERS',
        scope='FIXED_SHORTLIST_F32_IO_PROJECTION_CONFIRMATION_NOT_FUSED_CHAIN_OR_MODEL_AB')


def wired_plan():
    """One actual selected recipe per point; no candidate/parameter expansion."""
    from quactlize.dispatch.moe_prefill_policy import load
    points=[];modules={}
    for row in load()['rows']:
        q,n,k,ch,m=row['key'];r=Recipe.from_dict(row['recipe'])
        for router in ROUTERS:
            w=Workload(q,n,k,m,operation='indexed',experts=256,topk=8,channels=ch,compute='bf16',router=router)
            spec=module_spec(w,r);mid=digest(spec);modules[mid]=spec
            points.append(dict(id=w.key,workload=asdict(w),cohort='prefill',
                arms=[dict(id='original',recipe=row['recipe'],roles=['original'],aliases=[],spec=spec,module=mid)],
                sf_configs=[dict(r.parameters)['dequant_config']] if r.route=='sf' else [],full_configs=[],
                fixed_candidates=True,selection_source='PRODUCTION_PREFILL_QUERY_POLICY22'))
    return dict(schema='quactlize.moe-confirm.v1',points=points,tc=modules,vectors={},
        counts=dict(points=len(points),arms=len(points),tc_modules=len(modules),vector_modules=0,maximum_candidates=1),
        historical_times_reused=False,production_changed=False,native_comparison=False,
        scope='PRODUCTION_SELECTED_PROJECTION_WIRING_NOT_FUSED_CHAIN_OR_MODEL_AB')


def bind(point,images):
    """Bind only frozen schedules; no occupancy/grid/Split-K expansion."""
    point=json.loads(json.dumps(point));queries={};probes={};w=Workload(**point['workload'])
    for arm in point['arms']:
        r=Recipe.from_dict(arm['recipe'])
        if r.kind!='tc':continue
        request=probe_spec(w,r,int(dict(r.parameters).get('grid_mode',0)>=2));qid=digest(request)
        if qid not in queries:
            mid=arm['module']
            if mid not in probes:probes[mid]=NativeProbe(arm['spec'],images['tc'][mid])
            queries[qid]=probes[mid].query(request)
        query=queries[qid]
        if query['status']!='READY':raise ValueError('confirmed recipe resource unavailable: '+str(query))
        arm['binding']=dict(query=qid,launch=resolved_grid(w,r,query['device']['compute_units'],query['occupancy']))
    return point,queries


def choose(profiles):
    """One static row per host key, no router inspection or tiny exact ties."""
    if not profiles:raise ValueError('no complete profiles')
    common=set.intersection(*(set(p['decision']['statistics']) for p in profiles))
    if any(set(p['decision']['statistics'])!=common for p in profiles) or 'original' not in common:
        raise ValueError('incomplete common confirmation; no partial policy')
    stats=[p['decision']['statistics'] for p in profiles]
    if any(not math.isfinite(v['mean_us']) or v['mean_us']<=0 for s in stats for v in s.values()):
        raise ValueError('invalid measured cost')
    eligible=[c for c in common if all(s[c]['mean_us']<=s['original']['mean_us'] for s in stats)]
    def rank(c):
        return max(s[c]['mean_us']/min(v['mean_us'] for v in s.values()) for s in stats),sum(s[c]['mean_us'] for s in stats),c
    best=min(eligible,key=rank)
    if all(s['original']['mean_us']-s[best]['mean_us']<=min(1.,.01*s[best]['mean_us']) for s in stats):
        best='original'
    return best


def preflight_points(plan):
    """Existing, reusable points covering every selected implementation family."""
    def kinds(p):
        endpoint=((p['workload']['qtype'],p['workload']['channels'])
                  if p.get('selection_source')=='PRODUCTION_PREFILL_QUERY_POLICY22' else ())
        return {(a['recipe']['kind'],a['recipe']['route'],*endpoint) for a in p['arms']}
    remaining=set.union(*(kinds(p) for p in plan['points']));out=[]
    points=sorted(plan['points'],key=lambda p:(p['workload']['m'],
        p['workload']['router']!='spread',len(p['arms']),p['id']))
    while remaining:
        p=next(p for p in points if kinds(p)&remaining)
        out.append(p['id']);remaining-=kinds(p)
    by_id={p['id']:p for p in points}
    for pid in out[:]:
        others=[kinds(by_id[i]) for i in out if i!=pid]
        if others and kinds(by_id[pid])<=set.union(*others):out.remove(pid)
    return out
