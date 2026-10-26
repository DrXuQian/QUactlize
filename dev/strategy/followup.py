"""Read sealed screens, choose bounded finalists, never mutate old evidence."""
from collections import defaultdict
import json
import math
from pathlib import Path

from dev.strategy.components import component_key,recommendation,statistics_checked,stage_role
from dev.strategy.journal import ArmJournal,planned_arms
from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.tuning import digest


def jobs(point):
    helpers=[dict(id=digest(['sf-expansion',c]),recipe=None,roles=['expansion'],config=c)
             for c in point['sf_configs']]
    arms=planned_arms(point)
    arms.sort(key=lambda a:digest(['screen-order-v1',point['id'],a['id']]))
    return helpers+arms


def family(point):
    w=point['workload']
    return digest({k:w[k] for k in ('qtype','n','k','experts','operation','compute')})


def families(points):
    groups=defaultdict(list)
    for p in points:groups[family(p)].append(p)
    # Keep a family's fixtures resident; inspect M1/M8 first, then inner M.
    for rows in groups.values():
        rows.sort(key=lambda p:(p['workload']['m'] not in (1,8),p['workload']['m'],p['id']))
    return dict(groups)


def compose(point,data,records):
    arms={};roles={};expansions={};missing=[];receipts={}
    for j in jobs(point):
        row=data.get(j['id'])
        if not row:
            missing.append(j['id']);continue
        if not j['recipe']:expansions[j['config']]=row['components']
        else:arms[j['id']]=list(row['components']);roles[j['id']]=j['roles']
        receipts[j['id']]=row.get('receipt',row.get('scope'))
    best_sf=min(expansions,key=lambda c:sum(records[k]['median_us'] for k in expansions[c]),default=None)
    for arm in point['arms']:
        aid=arm['id'];roles.setdefault(aid,arm['roles']);arms.setdefault(aid,[])
        if arm['recipe']['route']=='sf' and point['workload']['qtype']!=8 and arms[aid]:
            config=best_sf if 'rule' in roles[aid] else max(0,dict(Recipe.from_dict(arm['recipe']).parameters).get('dequant_config',-1))
            arms[aid]=[k for k in arms[aid] if records[k]['kind']!='expansion']+expansions.get(config,[])
            if config not in expansions:arms[aid]=[]
    result=recommendation(arms,records,roles=roles)
    result['admission']='SCREEN_ONLY_ALTERNATING_FINALIST_AND_MODEL_AB_REQUIRED'
    return dict(point=point['id'],workload=point['workload'],status='FAIL' if missing else 'PASS',
                selection=result,missing=missing,best_sf=best_sf,expansion_configs=expansions,
                receipts=receipts,component_arms=arms)


def read_point(directory,point,authority,*,require_complete=False):
    from tools.run_kpack_strategy import record_load
    journal=ArmJournal(Path(directory)/'arms',dict(sha256=digest(authority)),'0'*32)
    records={};data={}
    for job in jobs(point):
        row=journal.read(point,job)
        if not row or row['status']!='PASS':
            if require_complete:raise ValueError('prior screen is incomplete: '+point['id']+'/'+job['id'])
            continue
        d=row['data'];w=Workload(**point['workload'])
        if job['recipe']:
            proofs=d.get('numerical',[])
            planes=['low','high','units'] if w.qtype in (11,13,14) else ['low','units']
            if (len(proofs)!=(6 if w.compute=='bf16' else 5) or d.get('pointer_negatives')!=planes or
                    d.get('omitted_producer')!='DETECTED' or d.get('zero_input')!='PASS'):
                raise ValueError('prior screen numerical checks incomplete')
            for proof in proofs:
                error=proof.get('relative_l1_error',float('inf'))
                if (proof.get('cells')!=w.m*w.topk*w.n or proof.get('bad')!=0 or proof.get('nonfinite')!=0 or
                        proof.get('tolerance')!=.005 or not math.isfinite(error) or not 0<=error<.005):
                    raise ValueError('invalid prior screen numerical proof')
            fixture=d['fixture']
            if (fixture['l2_bytes']!=authority['l2']['bytes'] or
                    fixture['active_ring_bytes']!=fixture['copies']*fixture['minimum_read_bytes'] or
                    fixture['active_ring_bytes']<2.25*fixture['l2_bytes']):
                raise ValueError('prior screen lacks a verified cold ring')
        else:
            proofs=d.get('numerical',[])
            if len(proofs)!=2 or any(len(pair)!=2 or any(q['bad'] for q in pair) for pair in proofs):
                raise ValueError('prior SF bit oracle incomplete')
        if d['components']!=[s['id'] for s in d['stages']]:raise ValueError('prior stage inventory differs')
        for stage in d['stages']:
            cid=stage['id'];path=Path(directory)/'components'/(cid+'.json')
            raw=json.loads(path.read_text());identity=raw['identity']
            if (digest(identity)!=cid or identity['authority']!=journal.authority or
                    identity['implementation']!=authority['body_source']):
                raise ValueError('prior component authority differs')
            value=record_load(path,identity)
            recipe=Recipe.from_dict(job['recipe']) if job['recipe'] else None
            split=dict(recipe.parameters).get('split',1) if recipe else 1
            node=identity['node']|dict(split=split)
            expected,contract=component_key(w,recipe,node,authority=journal.authority,
                                             source=authority['body_source'],cache=identity['cache'])
            if expected!=cid or contract!=identity or stage_role(node['name'],split)!=stage['role']:
                raise ValueError('prior component workload/recipe/stage semantics differ')
            if (value['kind']!=stage['role'] or value['median_us']!=stage['median_us'] or
                    value['spread_pct']!=stage['spread_pct'] or value['timed_kernel_types']!=1 or
                    not value['first_use_excluded'] or value['host_launch_cost_added']):
                raise ValueError('prior component timer scope differs')
            records[cid]=value
        data[job['id']]=d
    return compose(point,data,records),records,data


def load_screen(directory):
    """No native payload needed here; the caller verifies current DSO hashes."""
    root=Path(directory).resolve(strict=True)
    if (root/'decode/plan.json').exists():root=root/'decode'
    plan=json.loads((root/'plan.json').read_text());authority=json.loads((root/'authority.json').read_text())
    if (plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}) or
            authority['plan']!=plan['plan_sha256'] or
            authority['harness']['sha256']!=digest(authority['harness']['files'])):
        raise ValueError('prior screen plan/harness seal differs')
    if authority['protocol']!=dict(rounds=3,samples=3,replays_min=16,
                                 screen_order='HASH_SEEDED_NOT_FINALIST_CONFIRMATION'):
        raise ValueError('unsupported prior screen protocol')
    points={};records={}
    for point in plan['points']:
        result,components,data=read_point(root,point,authority,require_complete=True)
        records.update(components)
        points[point['id']]=dict(point=point,result=result,data=data)
    return dict(root=str(root),plan=plan,authority=authority,points=points,records=records,
                receipt_sha256=digest([plan,authority,{k:v['result'] for k,v in points.items()}]))


def reusable(previous,point,authority):
    """Old timings stay old; only an unchanged complete screen may be skipped.

    Cold-ring implementation changed, so never mix old/new component samples.
    Finalists are independently confirmed under the new protocol afterwards.
    """
    old=previous['points'].get(point['id'])
    if not old:return False
    a=previous['authority']
    for field in ('oracle','body_source'):
        if a[field]!=authority[field]:raise ValueError('previous screen implementation/oracle changed: '+field)
    if (a['resource']['sdk']!=authority['resource']['sdk'] or
            a['resource']['device']!=authority['resource']['device'] or a['l2']!=authority['l2']):
        raise ValueError('previous screen physical device/SDK/L2 changed')
    p=old['point']
    if p['workload']!=point['workload']:raise ValueError('prior workload identity differs')
    fingerprint=lambda p:sorted((j['id'],digest(j)) for j in jobs(p))
    if fingerprint(p)!=fingerprint(point):return False
    for arm in p['arms']:
        key='tc' if arm['recipe']['kind']=='tc' else 'vectors'
        if a[key][arm['module']]!=authority[key][arm['module']]:
            raise ValueError('previous screen device image changed')
    return True


def finalists(point,result,records,*,window_pct=5.,max_arms=16):
    """Protect incumbents/controls and every competitively noisy challenger.

    A noisy loser no longer blocks all winners. If many candidates really are
    competitive, report an incomplete shortlist rather than silently cutting
    it to an arbitrary top-K or admitting the fastest noisy outlier.
    """
    selection=result['selection'];costs=selection['costs_us'];arms=result['component_arms']
    if result['status']!='PASS' or selection['missing'] or 'original' not in costs:
        return dict(status='BLOCKED',reason='INCOMPLETE_SCREEN',arms=[])
    best=selection['best_rule']
    if best is None:return dict(status='BLOCKED',reason='NO_RULE_CANDIDATE',arms=[])
    floor=min(costs.values());threshold=floor*(1+window_pct/100)
    required={'original',best,selection['best_control']}
    for aid,keys in arms.items():
        # An arm with variable samples can still hide a fast configuration.
        lower=sum(min(records[k]['round_medians_us']) for k in keys)
        if costs[aid]<=threshold or lower<=threshold:required.add(aid)
    required.discard(None)
    # Dedup only the exact same measured stages, not a guessed TC alias.
    unique={};aliases={}
    for aid in sorted(required,key=lambda a:(a!='original',costs[a],a)):
        signature=tuple(arms[aid])
        if signature in unique:aliases[aid]=unique[signature]
        else:unique[signature]=aid
    chosen=list(unique.values())
    if len(chosen)>max_arms:
        return dict(status='BLOCKED',reason='COMPETITIVE_FINALISTS_EXCEED_BOUND',arms=chosen,
                    max_arms=max_arms,aliases=aliases)
    incumbent_only=len(chosen)==1 and chosen[0]=='original'
    return dict(status='INCUMBENT_UNCHANGED' if incumbent_only else 'READY',arms=chosen,
                aliases=aliases,window_pct=window_pct,
                discarded_noisy=[a for a in selection['noisy'] if a not in required],
                scope='SCREEN_SHORTLIST_NOT_PRODUCTION_ADMISSION')


def paired_verdict(cost_rounds,roles,*,baseline='original'):
    """No-regression decision for measured component sums, not model TPOT."""
    if baseline not in cost_rounds:raise ValueError('paired original missing')
    stats={a:statistics_checked(v) for a,v in cost_rounds.items()}
    n={len(v) for v in cost_rounds.values()}
    if len(n)!=1 or min(n)<6:raise ValueError('need six equal alternating rounds')
    rule=[a for a in stats if 'rule' in roles[a] or a==baseline]
    best=min(rule,key=lambda a:(stats[a]['median_us'],a!=baseline))
    ratios={a:[statistics_checked([x]*3)['median_us']/statistics_checked([y]*3)['median_us']
               for x,y in zip(cost_rounds[a],cost_rounds[baseline])] for a in rule}
    # The median may win while most rounds lose. Retain the incumbent then.
    admitted=(best==baseline or stats[best]['spread_pct']<=5 and stats[baseline]['spread_pct']<=5 and
              sum(r<=1 for r in ratios[best])>=len(ratios[best])*2/3)
    winner=best if admitted else baseline
    control=[a for a in stats if 'control' in roles[a] or a==baseline]
    best_control=min(control,key=lambda a:stats[a]['median_us'])
    miss=stats[best_control]['median_us']<stats[winner]['median_us']*.95
    return dict(winner=winner,statistics=stats,paired_ratios=ratios,
                status='RULE_MISS' if miss else 'PASS' if admitted else 'UNSTABLE_RETAIN_ORIGINAL',
                best_control=best_control,production_changed=False,performance_admitted=False,
                scope='ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_AB')


def validate_confirmation(data, *, decision_verifier=paired_verdict):
    """Recompute selected costs from individual raw component samples."""
    if data.get('correctness')!='PASS' or data.get('production_changed') is not False:
        raise ValueError('confirmation lacks correctness or changed production')
    if data.get('status')=='INCUMBENT_UNCHANGED':
        if data.get('winner')!='original' or data.get('scope')!='EXACT_SAME_TIMED_COMPONENTS_NO_INVENTED_AB':
            raise ValueError('unchanged incumbent scope differs')
        original=data.get('original_proof',{})
        if original.get('correctness')!='PASS' or sorted(original.get('components',[]))!=sorted(data['components']):
            raise ValueError('unchanged incumbent lacks its original numerical receipt')
        return
    if data.get('protocol')!=dict(rounds=6,samples=5,order='FORWARD_REVERSE_ALTERNATING',
                                 cache='SEPARATE_METADATA_AND_PRODUCER_RINGS_AT_LEAST_2P25_L2'):
        raise ValueError('confirmation protocol differs')
    w=Workload(**data['workload'])
    expected_planes=['low','high','units'] if w.qtype in (11,13,14) else ['low','units']
    if set(data['arms'])!=set(data['roles']) or set(data['arms'])!=set(data['arm_components']):
        raise ValueError('confirmation arm coverage differs')
    for arm in data['arms'].values():
        proofs=arm.get('numerical',[])
        if (arm.get('correctness')!='PASS' or len(proofs)!=(6 if w.compute=='bf16' else 5) or
                arm.get('zero_input')!='PASS' or arm.get('omitted_producer')!='DETECTED' or
                arm.get('pointer_negatives')!=expected_planes):
            raise ValueError('confirmation numerical/negative coverage differs')
        for p in proofs:
            error=p.get('relative_l1_error',float('inf'))
            if (p.get('cells')!=w.m*w.topk*w.n or p.get('bad')!=0 or p.get('nonfinite')!=0 or
                    p.get('tolerance')!=.005 or not math.isfinite(error) or not 0<=error<.005):
                raise ValueError('confirmation numerical proof failed')
        f=arm['fixture']
        if (f['l2_bytes']<=0 or f['active_ring_bytes']!=f['copies']*f['minimum_read_bytes'] or
                f['active_ring_bytes']<2.25*f['l2_bytes']):
            raise ValueError('confirmation producer ring is not cold')
    records=data['component_records']
    if (set(records)!=set(data['components']) or
            set(records)!={k for keys in data['arm_components'].values() for k in keys}):
        raise ValueError('confirmation component coverage differs')
    for cid,record in records.items():
        if cid!=digest(record['identity']) or len(record['samples_us'])!=6 or any(len(r)!=5 for r in record['samples_us']):
            raise ValueError('confirmation component identity/sample coverage differs')
        stats=statistics_checked(record['samples_us'])
        if (any(record.get(k)!=v for k,v in stats.items()) or record.get('timed_kernel_types')!=1 or
                not record.get('first_use_excluded') or record.get('host_launch_cost_added') or
                record.get('calls_per_graph',0)<16):
            raise ValueError('invalid confirmation component timer')
    if any(r['kind']=='expansion' for r in records.values()):
        f=data['expansion_fixture'];proofs=data['expansion_proofs']
        if (not proofs or f['domain']!='ALL_EXPERT_METADATA' or
                f['active_ring_bytes']!=f['copies']*f['minimum_read_bytes'] or
                f['active_ring_bytes']<2.25*f['l2_bytes'] or any(
                    len(v)!=2 or any(len(pair)!=2 or any(p['bad'] for p in pair) for pair in v)
                    for v in proofs.values())):
            raise ValueError('confirmation SF oracle/cold ring differs')
    costs={aid:[[sum(records[k]['samples_us'][r][i] for k in keys) for i in range(5)]
                for r in range(6)] for aid,keys in data['arm_components'].items()}
    if decision_verifier(costs,data['roles'])!=data['decision']:
        raise ValueError('confirmation decision differs from raw stages')
