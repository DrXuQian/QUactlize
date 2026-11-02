"""Factor large launch domains without a target-winner injection or timing reuse.

One anchor per compiled geometry/Split-K/algorithm is measured first. Only
competitive groups expand their complete legal grid domain. The controls and
route/algorithm/Split-K strata remain visible. Old complete flat screens are
stronger evidence and do not need to be measured again.
"""
from collections import defaultdict
from dataclasses import asdict

from dev.strategy.space import scheduled_key, resolved_grid
from dev.strategy.resources import probe_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest


def group(arm):
    recipe=Recipe.from_dict(arm['recipe'])
    return (arm.get('module'),scheduled_key(recipe),arm.get('binding',{}).get('launch',{}).get('algorithm',-1))


def control(arm):
    return arm['id']=='original' or 'old-control' in arm['roles']


def subset(point,ids):
    selected=set(ids)
    arms=[a for a in point['arms'] if a['id'] in selected]
    if len(arms)!=len(selected) or not any(a['id']=='original' for a in arms):
        raise ValueError('staged subset lacks its declared arms/control')
    # The enclosing job/plan receives a fresh seal. Never carry a stale
    # full-domain seal on a changed candidate subset.
    return dict({k:v for k,v in point.items() if k!='sha256'},arms=arms)


def geometry_screen(domain):
    if len(domain['arms'])<=384:
        return dict(subset(domain,[a['id'] for a in domain['arms']]),screen_scope='COMPLETE_RESOURCE_DOMAIN')
    groups=defaultdict(list)
    keep={a['id'] for a in domain['arms'] if control(a)}
    for arm in domain['arms']:
        if not control(arm):groups[group(arm)].append(arm)
    for arms in groups.values():
        # Use the largest legal resident grid. Every geometry, Split-K and
        # ordinary/persistent algorithm has its OWN anchor, never just a
        # single best parent from a historical target shape.
        best=min(arms,key=lambda a:(-a.get('binding',{}).get('launch',{}).get('grid',0),a['id']))
        keep.add(best['id'])
    result=subset(domain,keep)
    if len(result['arms'])>384:
        raise ValueError(f'geometry screen itself exceeds384: {len(result["arms"])}; '
                         'no silent truncation of compiled families')
    return dict(result,screen_scope='GEOMETRY_SPLIT_ALGORITHM_ANCHORS',
                full_domain_arms=len(domain['arms']))


def groups_to_expand(point,result,window_pct=15):
    stats=result['decision']['statistics']
    if set(stats)!={a['id'] for a in point['arms']}:
        raise ValueError('geometry selection needs a complete measured screen')
    floor=min(v['mean_us'] for v in stats.values())*(1+window_pct/100)
    keep=set();strata=defaultdict(list)
    for arm in point['arms']:
        recipe=Recipe.from_dict(arm['recipe']);params=dict(recipe.parameters)
        g=group(arm)
        strata[(recipe.route,g[-1],params.get('split',1))].append(arm)
        if control(arm) or stats[arm['id']]['mean_us']<=floor:
            keep.add(g)
    for arms in strata.values():
        keep.add(group(min(arms,key=lambda a:(stats[a['id']]['mean_us'],a['id']))))
    return keep


def grid_screens(domains,anchors,results):
    """Union across router profiles; both profiles measure the same groups."""
    from dev.moe_confirm.refine_plan import routing_key
    profiles=defaultdict(list)
    for pid,p in domains.items():profiles[routing_key(p['workload'])].append(pid)
    selected={}
    for ids in profiles.values():
        if all({a['id'] for a in anchors[pid]['arms']}==
               {a['id'] for a in domains[pid]['arms']} for pid in ids):
            # Already completed a full flat screen, or the new domain was
            # small enough to measure whole. Recompose it without launching
            # anything again or discarding its already-known competitors.
            for pid in ids:selected[pid]=dict(domains[pid],screen_scope='COMPLETE_RESOURCE_DOMAIN')
            continue
        keep=set().union(*(groups_to_expand(anchors[pid],results[pid]) for pid in ids))
        for pid in ids:
            domain=domains[pid]
            selected[pid]=dict(subset(domain,[a['id'] for a in domain['arms']
                if control(a) or group(a) in keep]),screen_scope='MEASURED_GEOMETRY_GRID_REFINEMENT')
    return selected


def historical_coverage(seed,point,queries):
    """Audit AFTER blind selection. Missing champions fail, never get appended."""
    w=Workload(**seed['workload']);missing=[]
    for ref in seed['historical_references']:
        r=Recipe.from_dict(ref['recipe']);params=dict(r.parameters)
        if r.kind=='full':
            covered=any(a['recipe']['kind']=='full' for a in point['arms'])
        else:
            same=[a for a in seed['arms'] if scheduled_key(Recipe.from_dict(a['recipe']))==scheduled_key(r)]
            q=(queries.get(digest(probe_spec(w,Recipe.from_dict(same[0]['recipe']),
                int(params.get('grid_mode',0)>=2)))) if same else None)
            launch=resolved_grid(w,r,q['device']['compute_units'],q['occupancy']) if q and q['status']=='READY' else None
            covered=launch is not None and any('rule' in a['roles'] and
                scheduled_key(Recipe.from_dict(a['recipe']))==scheduled_key(r) and
                a.get('binding',{}).get('launch')==launch for a in point['arms'])
        if not covered:missing.append(asdict(r))
    if missing:
        raise ValueError(f'staged rule omitted historical launch: point={point["id"]} '
                         f'missing={missing}; retained screens can be reused with wider rules')
    return dict(references=len(seed['historical_references']),missing=[],champion_injection=False)


def available_batches(paths,point,run,engine,*,phase=None):
    """First valid capture wins; no fastest-sample selection or remeasurement."""
    wanted={a['id'] for a in point['arms']};found=set();values=[];receipts={}
    for path in sorted(paths):
        # Do not hash every other point's multi-MB raw report on every
        # lookup. Validate the small receipt first, then verify only the
        # captures whose samples are actually going to be used.
        header=engine.sealed(path)
        if (header.get('point')!=point['id'] or header.get('status')!='PASS' or
                not ((set(header.get('data',{}).get('arms',{})) & wanted)-found)):
            continue
        value=engine.saved(path,run)
        if value is None or value.get('point')!=point['id']:continue
        if phase is not None and value.get('phase')!=phase:
            raise ValueError('retained batch belongs to another timing phase')
        data=value['data'];ids=(set(data['arms']) & wanted)-found
        if not ids:continue
        # Include helper records and Full reset unchanged. compose checks
        # matching component identity, phase, source and numerical receipts.
        values.append(dict(data,arms={aid:data['arms'][aid] for aid in ids}))
        receipts[str(path)]=engine.sha(path);found.update(ids)
    return values,receipts,wanted-found
