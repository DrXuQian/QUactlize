"""Batched large-M correctness and alternating isolated Asys stages.

Original captured graphs and their allocations outlive every isolated timer.
Repeated helpers may reuse only sealed same-run/device/phase components.
"""
from contextlib import ExitStack
from copy import deepcopy
import ctypes as C
import math
from pathlib import Path
import statistics

from dev.strategy.asys_components import parse_component_rows
from dev.strategy.components import component_key,stage_role,statistics_checked,kernel_entry
from dev.strategy.cold_decode import replay_extracted
from dev.strategy.moe_prefill_native import AsysStages
from dev.strategy.smallm_asys import mean_decision
from dev.strategy.prefill_selection import control_helper_config
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest
from quactlize.runtime.native import checked


class AlternatingStages(AsysStages):
    def __init__(self,*,helper_source=None,**kwargs):
        super().__init__(**kwargs);self.graphs={};self.groups=[]
        self.helper_source=helper_source

    def collect(self,capture,w,recipe,authority,source,proof,**kwargs):
        if recipe is None:return super().collect(capture,w,recipe,authority,source,proof,**kwargs)
        split=dict(recipe.parameters).get('split',1)
        nodes=capture.descriptions[:len(capture.groups)]
        roles=[stage_role(n['name'],split) for n in nodes]
        if (roles.count('producer')!=1 or 'unclassified' in roles or
                not any('mm_ids_helper' in n['name'] for n in nodes) or
                split>1 and not any(r in ('reducer','reducer_finish') for r in roles)):
            raise ValueError('prefill main/helper/reducer inventory differs')
        replay_extracted(capture,proof);out=[];keys=[]
        for i,(n,role) in enumerate(zip(nodes,roles)):
            rotating=role=='producer'
            cache='ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2' if rotating else 'REPEATED_RESIDENT_SCRATCH'
            entry=kernel_entry(n['name'])
            # This exact adapter's addressing depends on the sorted row map,
            # K/strides and BF16 type, never on the downstream GEMM tile.
            key_recipe=None if entry=='quactlize::prefill::gather_bf16' else recipe
            common=(entry in ('quactlize::prefill::gather_bf16','quactlize::prefill::finish_bf16',
                               'quactlize::prefill::active_experts','mm_ids_helper'))
            implementation=self.helper_source if common and self.helper_source else source
            # These exact generic helpers do not consume the GEMM's Split-K
            # count. Tile-dependent grouped metadata is deliberately excluded.
            helper_split=1 if common else split
            cid,identity=component_key(w,key_recipe,dict(n,split=helper_split),authority=authority,source=implementation,cache=cache)
            if cid not in self.records:
                graph=capture.isolate(i,weight_ring=rotating,repeats=16)
                self.graphs[cid]=graph
                self.records[cid]=dict(component_id=cid,identity=identity,kind=role,
                    calls_per_graph=graph.count,first_use_excluded=True,timed_kernel_types=1,
                    host_launch_cost_added=False,cache=cache)
            elif self.records[cid]['identity']!=identity:
                raise ValueError('reused prefill stage identity differs')
            out.append(dict(id=cid,role=role,median_us=None,spread_pct=None));keys.append(cid)
        self.groups.append(keys)
        return out,[]

    def run(self):
        if not self.graphs:return
        self.collector.start(next(iter(self.graphs.values())).rt)
        try:
            for r in range(self.rounds):
                seen=set()
                for ids in (self.groups if r%2==0 else reversed(self.groups)):
                    for cid in ids:
                        if cid in seen or cid not in self.graphs:continue
                        for sample in range(self.samples):self.collector.sample(cid,self.graphs[cid],r,sample)
                        seen.add(cid)
                if seen!=set(self.graphs):raise ValueError('untimed prefill candidate stage')
        finally:self.collector.stop()

    def close(self):
        self.collector.stop()
        for graph in reversed(list(self.graphs.values())):graph.close()
        self.graphs.clear()


def reset_measurement(b,timer):
    """Full API's actual four-byte memset, separately profiled (not a kernel).

Keep a fixed primer prefix; correlate every remaining API with GPU memset
activity. Missing measured activity is a failure, never a zero-cost guess.
"""
    fn=b.sdk.lib.hggcMemsetAsync
    fn.argtypes=[C.c_void_p,C.c_int,C.c_size_t,C.c_void_p];fn.restype=C.c_int
    for _ in range(16):checked(fn(b.error.ptr,0,4,b.r.stream),'reset warmup')
    b.rt.sync();timer.collector.start(b.rt)
    try:
        for _ in range(16+timer.rounds*timer.samples*16):
            checked(fn(b.error.ptr,0,4,b.r.stream),'isolated prefill error reset')
    finally:timer.collector.stop()
    return dict(entry='hggcMemsetAsync',bytes=4,value=0,primer=16,repeats=16,
                rounds=timer.rounds,samples=timer.samples,
                scope='SEPARATE_GPU_MEMSET_DURATION_NOT_HOST_API_TIME')


def measure(a,point,arms,images,queries,run,phase,known):
    from dev.strategy.cold_decode import Ring,Expansion,measure_expansion
    from dev.strategy.prefill_fixture import Helpers,routing
    from dev.strategy.prefill_measure import prepared,measure_full_expansion
    from dev.strategy.moe_prefill_native import RoutedContext,RoutedTC,RoutedFull
    from tools.kpack_bf16_providers import loaded_images
    from quactlize.runtime.native import SDK
    w=Workload(**point['workload']);sdk=SDK(a.sdk)
    rounds,samples=(4,3) if phase=='screen' else (6,5)
    authority=dict(run=run['sha256'],shard=a.worker,device=run['devices'][a.worker],phase=phase)
    source=digest([run['source']['sha256'],run['images'],run['adapter_sha256']])
    timer=AlternatingStages(rounds=rounds,samples=samples,known=known,helper_source=source)
    result=dict(arms={},expansions={'sf':{},'full':{}},reset=None)
    with ExitStack() as owners:
        active=min(len(routing(w,r)['active']) for r in (0,1,2))
        ring=Ring(sdk,w,a.l2_bytes,minimum_active=active);owners.callback(ring.close)
        b=RoutedContext(sdk,ring,w,Helpers(Path(run['helpers'])/'libstrategy_prefill.so'),run['adapter'])
        owners.callback(b.close)
        sf=None
        if any(x['recipe']['route']=='sf' for x in arms):
            mr=Ring(sdk,w,a.l2_bytes,weights=ring.w,metadata_only=True);owners.callback(mr.close)
            mb=RoutedContext(sdk,mr,w,b.helpers,run['adapter']);owners.callback(mb.close)
            msf=Expansion(mb,Path(run['dequant']));owners.callback(msf.close)
            for config in point['sf_configs']:
                result['expansions']['sf'][str(config)]=measure_expansion(mb,w,msf,config,authority,source,a.attempt,
                    separate_expansion=True,stage_measure=timer.sf)
            sf=Expansion(b,Path(run['dequant']),gold=msf.gold);owners.callback(sf.close)
        try:
            with ExitStack() as sessions:
                try:
                    for arm in arms:
                        recipe=Recipe.from_dict(arm['recipe'])
                        full=recipe.kind=='full'
                        if full:
                            core=RoutedFull(b,Path(run['dequant']),a.l2_bytes,a.sdk)
                            sessions.callback(core.close)
                            for config in point['full_configs']:
                                result['expansions']['full'][str(config)]=measure_full_expansion(
                                    b,core,config,authority,source,a.attempt,stage_measure=timer.stages)
                            result['reset']=reset_measurement(b,timer)
                        else:
                            core=RoutedTC(b,arm,images['tc'][arm['module']],queries[arm['binding']['query']],sf)
                            sessions.callback(core.close)
                        config=control_helper_config(w,recipe)
                        capture,data=sessions.enter_context(prepared(b,core,recipe,config=config,full=full))
                        full_images=loaded_images() if full else None
                        if full and not full_images:raise ValueError('missing installed BF16 provider image identity')
                        implementation=digest([source,full_images]) if full else source
                        stages,_=timer.stages(capture,b,recipe,authority,implementation,a.attempt,b.check)
                        data.update(components=[s['id'] for s in stages],stages=stages)
                        data['receipt']['row_maps']='GPU_MM_IDS_HELPER_EVERY_CALL_ACTUAL_CONSUMER_POINTERS'
                        if full:data['timing_provider_images']=full_images
                        result['arms'][arm['id']]=data
                    timer.run()
                # Also on a later candidate's prepare failure, close timers
                # before ExitStack releases any earlier captured graph.
                finally:timer.close()
            if any(row['correctness']!='PASS' for row in result['arms'].values()):
                raise ValueError('prefill candidate precision incomplete')
            for arm in arms:
                data=result['arms'][arm['id']]
                if arm['recipe']['kind']=='full' and data['timing_provider_images']!=data['provider_loaded_images']:
                    raise ValueError('provider images changed during timer')
            b.validate_routes()
        finally:timer.close()
    ids=set(cid for part in [*result['arms'].values(),
        *[v for pool in result['expansions'].values() for v in pool.values()]] for cid in part['components'])
    result.update(status='CAPTURED_NOT_TIMED',point=point['id'],phase=phase,authority=authority,
        component_records={cid:timer.records[cid] for cid in ids if cid not in known},
        reused_components={cid:digest(known[cid]) for cid in ids if cid in known},
        trace_schedule=timer.collector.schedule,protocol=dict(rounds=rounds,samples=samples),
        timing_method='ASYS_GPU_ACTIVITY',performance_admitted=False,production_changed=False)
    return result


def parse_reset(contract,rows,apis):
    memset=[r for r in rows if r['kind']=='memset']
    if contract is None:
        if memset:raise ValueError('unexpected GPU memory work')
        return None
    calls=[a for a in apis if a['name'] in ('hggcMemsetAsync','cudaMemsetAsync')]
    count=contract['primer']+contract['rounds']*contract['samples']*contract['repeats']
    if len(calls)!=count or any(a['return_value']!=0 for a in calls):
        raise ValueError('isolated reset API coverage differs')
    by_id=defaultdict_list(memset,'correlation_id');durations=[]
    for i,call in enumerate(calls):
        found=by_id.pop(call['correlation_id'],[])
        if len(found)>1 or (i>=contract['primer'] and len(found)!=1):
            raise ValueError('isolated measured reset GPU activity missing/duplicated')
        for row in found:
            if row['bytes']!=4 or row['value']!=0 or row['end']<=row['start']:
                raise ValueError('isolated reset GPU activity differs')
            if i>=contract['primer']:durations.append(row['end']-row['start'])
    if by_id:raise ValueError('unassigned GPU memory activity')
    samples=[];cursor=0
    for _ in range(contract['rounds']):
        row=[]
        for _ in range(contract['samples']):
            ns=durations[cursor:cursor+contract['repeats']];cursor+=contract['repeats']
            row.append(statistics.fmean(t/1000 for t in ns))
        samples.append(row)
    return dict(contract,**statistics_checked(samples),durations_ns=durations,
                mean_us=statistics.fmean(v for r in samples for v in r),timing_method='ASYS_GPU_ACTIVITY')


def defaultdict_list(rows,key):
    out={}
    for row in rows:out.setdefault(row[key],[]).append(row)
    return out


def validate_capture(capture):
    if (capture.get('status')!='CAPTURED_NOT_TIMED' or
            capture.get('timing_method')!='ASYS_GPU_ACTIVITY' or
            capture['protocol'] not in (dict(rounds=4,samples=3),dict(rounds=6,samples=5))):
        raise ValueError('invalid prefill capture contract')
    for part in [*capture['arms'].values(),
                 *[v for pool in capture['expansions'].values() for v in pool.values()]]:
        fixture=part['fixture']
        if (part['correctness']!='PASS' or not part['components'] or
                part['components']!=[s['id'] for s in part['stages']] or
                fixture['copies']<2 or fixture['l2_bytes']<=0 or
                fixture['active_ring_bytes']<2.25*fixture['l2_bytes']):
            raise ValueError('incomplete numerical/cold/component proof')
    for part in capture['arms'].values():
        if (len(part['numerical'])!=6 or part['omitted_producer']!='DETECTED' or
                part['changed_input_and_router']!='PASS' or part['zero_a']!='PASS' or
                not part['pointer_negatives'] or part['receipt']['row_maps']!=
                'GPU_MM_IDS_HELPER_EVERY_CALL_ACTUAL_CONSUMER_POINTERS'):
            raise ValueError('missing prefill replay/negative controls')
        for proof in part['numerical']:
            error=proof['relative_l1_error']
            if (not math.isfinite(error) or not 0<=error<.005 or proof['tolerance']!=.005 or
                    proof['cells']<=0 or proof['nonfinite']!=0 or
                    proof['fixture']!='ORIGINAL_GGUF_COMMON_DYADIC_F64_FACTORIZED_DOT'):
                raise ValueError('invalid independent prefill oracle')
    for part in capture['expansions']['sf'].values():
        if (len(part['numerical'])!=2 or any(len(p)!=2 for p in part['numerical']) or
                any(p['cells']<=0 or p['bad']!=0 for pair in part['numerical'] for p in pair)):
            raise ValueError('invalid independent SF bits')
    for part in capture['expansions']['full'].values():
        p=part['numerical']
        if (p['bits']<=0 or p['bad']!=0 or not p['inactive_experts_untouched'] or
                part['mutable_count']!='PASS' or not part['pointer_negatives']):
            raise ValueError('invalid active-only Full bits/guards')


def parse(capture,known,rows,apis,provenance):
    validate_capture(capture)
    for cid,want in capture['reused_components'].items():
        if cid not in known or digest(known[cid])!=want:raise ValueError('reused stage binding differs')
    p=capture['protocol'];kernel_rows=[r for r in rows if r['kind']=='kernel']
    records=parse_component_rows(capture['component_records'],capture['trace_schedule'],kernel_rows,
                                 provenance,p['rounds'],p['samples'],runtime_apis=apis,allow_short_sample=True)
    records.update({cid:known[cid] for cid in capture['reused_components']})
    reset=parse_reset(capture['reset'],rows,apis)
    result=deepcopy(capture)
    result.update(status='PASS',component_records=records,reset=reset,latency_provenance=provenance)
    result['activity_count_exceptions']=[r['activity_count_exception'] for r in records.values()
                                        if 'activity_count_exception' in r]
    for part in [*result['arms'].values(),*[v for pool in result['expansions'].values() for v in pool.values()]]:
        for s in part['stages']:s.update({k:records[s['id']][k] for k in ('mean_us','median_us','spread_pct')})
    return result


def compose(point,results):
    """Fresh mean component sum, helper choice factored from producer config."""
    records={};data={};helpers={'sf':{},'full':{}};reset=None;authority=None
    for value in results:
        if value['status']!='PASS' or value['point']!=point['id']:raise ValueError('incomplete/mismatched prefill batch')
        if authority is not None and authority!=value['authority']:
            raise ValueError('prefill phase/run/device changed between batches')
        authority=value['authority']
        for cid,row in value['component_records'].items():
            if cid in records and records[cid]!=row:raise ValueError('conflicting shared helper timing')
            records[cid]=row
        if set(data)&set(value['arms']):raise ValueError('duplicate prefill arm')
        data.update(value['arms'])
        for role,pool in value['expansions'].items():
            for config,part in pool.items():
                ids=part['components']
                if config in helpers[role] and helpers[role][config]!=ids:raise ValueError('helper configuration changed')
                helpers[role][config]=ids
        if value['reset'] is not None:
            if reset is not None and reset!=value['reset']:raise ValueError('duplicate Full provider/reset jobs')
            reset=value['reset']
    expected={a['id'] for a in point['arms']}
    if set(data)!=expected:raise ValueError('missing prefill candidates; no partial winner')
    protocol=results[0]['protocol']
    for cid,row in records.items():
        if (row['component_id']!=cid or row['status']!='PASS' or
                row['timing_method']!='ASYS_GPU_ACTIVITY' or
                len(row['samples_us'])!=protocol['rounds'] or
                any(len(r)!=protocol['samples'] for r in row['samples_us'])):
            raise ValueError('component/phase sample inventory changed')
        statistics_checked(row['samples_us'])
        if row['mean_us']!=statistics.fmean(x for r in row['samples_us'] for x in r):
            raise ValueError('component mean differs from retained samples')
    best={role:min(pool,key=lambda c:math.fsum(records[k]['mean_us'] for k in pool[c]),default=None)
          for role,pool in helpers.items()}
    nround=next(iter(records.values()))['samples_us']
    rounds,samples=len(nround),len(nround[0]);costs={};roles={};bindings={};parts={}
    w=Workload(**point['workload'])
    for a in point['arms']:
        aid=a['id'];r=Recipe.from_dict(a['recipe']);keys=list(data[aid]['components']);roles[aid]=a['roles']
        helper='full' if r.kind=='full' else 'sf' if r.route=='sf' else None
        config=None
        if helper:
            config=str(control_helper_config(w,r)) if aid=='original' else best[helper]
            if config not in helpers[helper]:raise ValueError('missing explicit helper config')
            keys+=helpers[helper][config]
        full=r.kind=='full'
        if full and reset is None:raise ValueError('Full provider lacks isolated reset timing')
        costs[aid]=[[math.fsum(records[k]['samples_us'][i][j] for k in keys)+
                     (reset['samples_us'][i][j] if full else 0) for j in range(samples)] for i in range(rounds)]
        bindings[aid]=dict(recipe=a['recipe'],helper_config=int(config) if config is not None else None)
        parts[aid]=keys
    decision=mean_decision(costs,roles)
    return dict(status='PASS',point=point['id'],workload=point['workload'],decision=decision,
        component_records=records,arm_components=parts,bindings=bindings,best_helpers=best,reset=reset,
        authority=authority,protocol=protocol,
        precision='F32_IO_BF16_OPERANDS_FP32_ACC',timing_method='ASYS_GPU_ACTIVITY',
        production_changed=False,native_no_regression_admitted=False,model_tpot_admitted=False)
