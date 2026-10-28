"""Alternating finalist rounds; every timer still contains just ONE kernel."""
from contextlib import ExitStack
import time

from dev.strategy.components import component_key,stage_role,statistics_checked
from dev.strategy.followup import paired_verdict
from dev.strategy.isolated import Captured,checked
from dev.strategy.indexed import RuntimeView
from dev.strategy.cold_decode import prepared_arm
from quactlize.dispatch.table import Recipe,Workload


def confirm_point(b,metadata_b,sf,metadata_sf,point,screen,shortlist,record_for,authority,source,
                  *,rounds=6,samples=5,round_observer=None,trace_collector=None,screening=False):
    if screening and (rounds!=4 or samples!=3 or trace_collector is None):
        raise ValueError('short screen requires Asys and exactly 4 alternating rounds x 3 samples')
    if shortlist['status']!='READY' or rounds<(4 if screening else 6) or rounds%2 or samples<3:
        raise ValueError('confirmation requires a ready shortlist and >=6 even rounds x >=3 samples')
    started=time.monotonic();w=Workload(**point['workload'])
    selected={a['id']:a for a in point['arms'] if a['id'] in shortlist['arms']}
    if set(selected)!=set(shortlist['arms']):raise ValueError('finalist is not a declared screen arm')
    timers={};values={};nodes={};arm_keys={};sessions={};roles={};sf_keys={};sf_proofs={}
    observations=[]
    with ExitStack() as stack:
        def timer(capture,index,recipe,*,expansion=False):
            split=dict(recipe.parameters).get('split',1) if recipe else 1
            node=dict(capture.descriptions[index],split=split)
            kind=stage_role(node['name'],split)
            if kind=='unclassified':raise ValueError('unknown confirmation stage '+node['name'])
            if expansion:node['expert_domain']='ALL'
            rotating=kind in ('producer','expansion')
            cache='ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2' if rotating else 'REPEATED_RESIDENT_SCRATCH'
            cid,identity=component_key(w,recipe,node,authority=authority,source=source,cache=cache)
            if cid not in timers:
                graph=capture.isolate(index,weight_ring=rotating,repeats=16)
                stack.callback(graph.close);timers[cid]=graph;values[cid]=[]
                nodes[cid]=dict(identity=identity,kind=kind,calls_per_graph=graph.count,
                                timed_kernel_types=1,first_use_excluded=True,host_launch_cost_added=False)
            return cid
        required_sf=set()
        for aid,arm in selected.items():
            if arm['recipe']['route']=='sf' and w.qtype!=8:
                cfg=screen['best_sf'] if 'rule' in arm['roles'] else max(0,dict(Recipe.from_dict(arm['recipe']).parameters).get('dequant_config',-1))
                required_sf.add(cfg)
        if required_sf and (metadata_b is None or metadata_sf is None):
            raise ValueError('finalists require separate metadata-only SF timers')
        for cfg in sorted(required_sf):
            metadata_sf.poison()
            for i in range(metadata_b.ring.copies):checked(metadata_sf.launch(i,cfg),'confirm SF setup')
            sf_proofs[cfg]=[metadata_sf.check(0),metadata_sf.check(metadata_b.ring.copies-1)]
            cap=Captured(RuntimeView(metadata_b),[lambda i=i,cfg=cfg:metadata_sf.launch(i,cfg)
                                                  for i in range(metadata_b.ring.copies)])
            stack.callback(cap.close)
            if len(cap.groups)!=1 or stage_role(cap.descriptions[0]['name'],1)!='expansion':
                raise ValueError('SF confirmation contains a non-expansion kernel')
            sf_keys[cfg]=timer(cap,0,None,expansion=True)
        for aid in shortlist['arms']:
            arm=selected[aid];record,query=record_for(arm)
            session=stack.enter_context(prepared_arm(b,w,arm,record,query,sf,expansion_proven=True));sessions[aid]=session
            keys=[];r=Recipe.from_dict(arm['recipe']);split=dict(r.parameters).get('split',1)
            roles[aid]=list(arm['roles'])
            for alias,target in shortlist['aliases'].items():
                if target==aid:
                    for source_arm in point['arms']:
                        if source_arm['id']==alias:roles[aid]=sorted(set(roles[aid]+source_arm['roles']))
            for i in range(len(session.capture.groups)):
                kind=stage_role(session.capture.descriptions[i]['name'],split)
                if kind=='expansion':
                    cfg=screen['best_sf'] if 'rule' in arm['roles'] else max(0,dict(r.parameters).get('dequant_config',-1))
                    keys.append(sf_keys[cfg])
                else:keys.append(timer(session.capture,i,r))
            arm_keys[aid]=keys
        if trace_collector:trace_collector.start(next(iter(timers.values())).rt)
        try:
            for round_no in range(rounds):
                before=round_observer() if round_observer else None
                order=list(shortlist['arms'])
                if round_no%2:order.reverse()
                sampled=set()
                for aid in order:
                    for cid in arm_keys[aid]:
                        if cid in sampled:continue
                        values[cid].append([trace_collector.sample(cid,timers[cid],round_no,i)
                            if trace_collector else timers[cid].sample() for i in range(samples)])
                        sampled.add(cid)
                if sampled!=set(timers):raise ValueError('unmeasured finalist component')
                if round_observer:
                    observations.append(dict(round=round_no,before=before,after=round_observer()))
        finally:
            if trace_collector:trace_collector.stop()
        for buffer in [*(sf.buffers if sf else []),*(metadata_sf.buffers if metadata_sf else [])]:buffer.guard()
        if metadata_b:
            for buffer in metadata_b.ring.buffers:buffer.guard()
    # ExitStack performs extracted-sequence correctness and final redzone
    # checks before a single confirmation receipt may be published.
    if any(s.data['correctness']!='PASS' for s in sessions.values()):raise ValueError('finalist correctness incomplete')
    costs={} if trace_collector else {aid:[[sum(values[k][rnd][i] for k in keys) for i in range(samples)]
                for rnd in range(rounds)] for aid,keys in arm_keys.items()}
    decision=None if trace_collector else paired_verdict(costs,roles)
    result=dict(status='PASS',correctness='PASS',components=list(timers),point=point['id'],
                workload=point['workload'],
                decision=decision,arms={a:s.data for a,s in sessions.items()},roles=roles,
                arm_components=arm_keys,component_records={k:dict(nodes[k],**({} if trace_collector else statistics_checked(v))) for k,v in values.items()},
                expansion_proofs=sf_proofs,protocol=dict(rounds=rounds,samples=samples,order='FORWARD_REVERSE_ALTERNATING',
                cache='SEPARATE_METADATA_AND_PRODUCER_RINGS_AT_LEAST_2P25_L2'),
                expansion_fixture=metadata_b.ring.receipt if required_sf else None,
                wall_seconds=time.monotonic()-started,production_changed=False,performance_admitted=False)
    if round_observer:
        result['clock_observations']=observations
    if trace_collector:
        result.update(status='CAPTURED_NOT_TIMED',timing_method='ASYS_GPU_ACTIVITY',
                      trace_schedule=trace_collector.schedule)
    if screening:
        result['measurement_phase']='screen'
    return result
