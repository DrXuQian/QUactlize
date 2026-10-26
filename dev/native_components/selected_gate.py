"""Native is the acceptance baseline; selection success is not performance proof."""
from collections import Counter
import math
import statistics

from dev.native_components.smallm_ab import plan as selected_plan
from dev.native_components.asys import validate_capture as validate_native
from dev.strategy.smallm_asys import validate as validate_candidate, check_producers, METHOD
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.selected-native-asys.v1'
ORDER = ('native', 'quactlize', 'quactlize', 'native')
SCOPE = 'SAME_DEVICE_SELECTED_VS_NATIVE_ASYS_ISOLATED_COMPONENT_MEANS_NOT_MODEL_TPOT'


def plan(output):
    value = selected_plan(output)
    value.update(schema=SCHEMA, timing_method=METHOD, selection_metric='MEAN_COMPONENT_SUM',
        scope=SCOPE, allowed_slowdown_pct=0, config_sweep=False,
        protocol=dict(order=list(ORDER), rounds=6, samples=5, requested_clock_mhz=1700,
            clock_deviation='ADVISORY_NOT_FAILURE', helpers='SEPARATE_ASYS_KERNEL_DURATIONS',
            producer_ring='AT_LEAST_2P25_L2', warmup='EXCLUDED_BEFORE_PROFILER_START',
            native_arithmetic='UNCHANGED_LINKED_NATIVE_INCLUDING_Q8_1_IF_SELECTED',
            candidate_arithmetic='F32_IO_F16_COMPUTE_FP32_ACC_CURRENT_SELECTED_RECIPE'))
    value['sha256'] = digest({k:v for k,v in value.items() if k != 'sha256'})
    return value


def samples(components):
    if not components:raise ValueError('empty component call')
    result = []
    for component in components:
        rows = component['samples_us']
        if (component.get('timing_method') != METHOD or len(rows) != 6 or
                any(len(r) != 5 for r in rows) or
                any(not math.isfinite(t) or t <= 0 for r in rows for t in r)):
            raise ValueError('gate requires six-by-five finite Asys component samples')
        result.append(dict(role=component['role'], node=component['node'],
            mean_us=statistics.fmean(t for r in rows for t in r),
            round_means_us=[statistics.fmean(r) for r in rows]))
    return result


def validate_native_samples(data):
    """Recompute means from retained GPU intervals, including on resume."""
    intervals=[]
    for c in data['components']:
        if c.get('latency_provenance')!=data['latency_provenance']:
            raise ValueError('native component provenance differs')
        seen=set()
        for row in c.get('trace_slices',[]):
            r,s=row['round'],row['sample'];ns=row['durations_ns']
            if ((r,s) in seen or not 0<=r<6 or not 0<=s<5 or
                    row['count']!=c['calls_per_graph'] or len(ns)!=row['count'] or
                    any(type(t)is not int or t<=0 for t in ns)):
                raise ValueError('native raw interval coverage differs')
            values=[t/1000 for t in ns];mean=statistics.fmean(values)
            if (mean!=row['mean_us'] or min(values)!=row['min_us'] or max(values)!=row['max_us'] or
                    mean!=c['samples_us'][r][s]):
                raise ValueError('native samples differ from raw GPU intervals')
            seen.add((r,s));intervals.append((row['first_activity'],row['count']))
        if len(seen)!=30 or c['mean_us']!=statistics.fmean(t for r in c['samples_us'] for t in r):
            raise ValueError('native missing samples or incorrect mean')
    cursor=0
    for first,count in sorted(intervals):
        if first!=cursor:raise ValueError('native raw activity inventory has duplicate/missing indices')
        cursor+=count


def verdict(passes):
    if len(passes) != 4 or tuple(p['arm'] for p in passes) != ORDER:
        raise ValueError('gate requires fresh native/quactlize/quactlize/native passes')
    calls = [samples(p['components']) for p in passes]
    signature = lambda cs:[(c['role'],c['node']) for c in cs]
    for i,j in ((0,3),(1,2)):
        if signature(calls[i]) != signature(calls[j]):
            raise ValueError('kernel/launch inventory changed within ABBA')
    costs = [math.fsum(c['mean_us'] for c in cs) for cs in calls]
    native, candidate = statistics.fmean((costs[0],costs[3])), statistics.fmean((costs[1],costs[2]))
    roles = dict(main={'producer','provider'}, reducer={'reducer','reducer_finish'})
    breakdown = {}
    for arm,indices in (('native',(0,3)),('quactlize',(1,2))):
        for phase in ('main','reducer','helper'):
            accepts = (lambda role:role not in roles['main']|roles['reducer']) if phase=='helper' else (
                lambda role:role in roles[phase])
            breakdown[arm+'_'+phase+'_us'] = statistics.fmean(
                math.fsum(c['mean_us'] for c in calls[i] if accepts(c['role'])) for i in indices)
    rounds = [[math.fsum(c['round_means_us'][r] for c in cs) for r in range(6)] for cs in calls]
    ratios = [rounds[j][r]/rounds[i][r] for i,j in ((0,1),(3,2)) for r in range(6)]
    status = 'PASS' if candidate <= native else 'REGRESSION'
    return dict(status=status, native_us=native, quactlize_us=candidate,
        delta_pct=100*(candidate/native-1), allowed_slowdown_pct=0,
        order_drift_pct=100*max(abs(costs[i]-costs[j])/min(costs[i],costs[j]) for i,j in ((0,3),(1,2))),
        paired_round_ratios=ratios, noise_policy='DIAGNOSTIC_NOT_VETO_MEAN_NOT_MIN',
        passes=calls, component_no_regression=status=='PASS', scope=SCOPE, **breakdown)


def comparison(point, passes, run, shard):
    """Validate correctness, device/build/fixture identity BEFORE using timings."""
    if len(passes) != 4 or any(p.get('status') != 'MEASURED' for p in passes):
        raise ValueError('missing/failed arm; no performance verdict')
    parsed, hashes, traces = [], [], []
    for index, (arm,p) in enumerate(zip(ORDER,passes)):
        if p['arm'] != arm or p['block'] != index or p['point'] != point['id']:
            raise ValueError('ABBA order/point changed')
        data = p['data']; provenance = data['latency_provenance']
        required = dict(run['source_revision'], device=run['devices'][shard], sdk=run['sdk'],
            source_sha256=run['source']['sha256'], harness_sha256=run['harness']['sha256'],
            asys_sha256=run['asys_sha256'], images_sha256=digest(run['images']), timing_method=METHOD)
        if any(provenance.get(k) != v for k,v in required.items()):
            raise ValueError('paired measurement source/device/SDK/Asys changed')
        traces.append((provenance['trace_sha256'],provenance['sqlite_sha256']))
        if arm == 'quactlize':
            if p['selection'] != point['selection']:raise ValueError('production selection changed')
            validate_candidate(data);check_producers(point,data)
            if data['point'] != point['id'] or data['workload'] != point['workload']:
                raise ValueError('candidate workload changed')
            if set(data['arms']) != {'original'} or set(data['arm_components']) != {'original'}:
                raise ValueError('this gate is selected-only, not a sweep')
            authority = dict(run=run['sha256'],shard=shard,device=run['devices'][shard])
            ids = data['arm_components']['original']
            if len(ids)!=len(set(ids)) or set(ids)!=set(data['component_records']):
                raise ValueError('candidate component coverage changed')
            components=[]
            for cid in ids:
                c=data['component_records'][cid]
                if c['identity']['authority']!=authority:raise ValueError('candidate authority differs')
                components.append(dict(c,role=c['kind'],node=c['identity']['node']))
            hashes.append(data['arms']['original']['fixture']['fixture']['raw_sha256'])
        else:
            from tools.run_smallm_native_asys import authority
            auth=authority(run,shard)
            # The parser preserves every capture field; only its status changes.
            validate_native(dict(data,status='CAPTURED_NOT_TIMED'),point,auth)
            validate_native_samples(data)
            if data.get('timing_input_repeat') != 1:raise ValueError('native input differs')
            if (provenance.get('backend')!=run['backend'] or provenance.get('llama_head')!=run['llama_head'] or
                    provenance.get('native_adapter_sha256')!=run['adapter_sha256'] or
                    provenance.get('native_carrier_sha256')!=run['carrier_sha256']):
                raise ValueError('native binary provenance differs')
            components=[dict(c,node={k:c[k] for k in ('symbol','grid','block','shared_bytes')})
                        for c in data['components']]
            hashes.append(data['fixture']['raw_sha256'])
        parsed.append(dict(arm=arm,components=components))
    if any(h!=hashes[0] for h in hashes):raise ValueError('paired raw GGUF weights differ')
    if len(set(traces))!=4:raise ValueError('reused trace is not fresh ABBA')
    if passes[0]['data']['native_activation_contract']!=passes[3]['data']['native_activation_contract']:
        raise ValueError('native arithmetic changed within ABBA')
    return dict(verdict(parsed),point=point['id'],workload=point['workload'],selection=point['selection'],
        native_activation_contract=passes[0]['data']['native_activation_contract'],
        candidate_arithmetic='F32_IO_F16_COMPUTE_FP32_ACC',
        arithmetic_equivalence_claimed=False,model_tpot_admitted=False,production_changed=False)


def aggregate(plan, records):
    expected={p['id'] for p in plan['points']}
    if set(records)-expected:raise ValueError('unknown point in no-regression gate')
    counts=Counter(r['status'] for r in records.values())
    measured=counts['PASS']+counts['REGRESSION']
    missing=sorted(expected-set(records))
    status='INCOMPLETE' if measured!=len(expected) else 'REGRESSION' if counts['REGRESSION'] else 'PASS'
    return dict(status=status,expected=len(expected),measured=measured,counts=dict(counts),missing=missing,
        regressions=sorted(k for k,r in records.items() if r['status']=='REGRESSION'),
        failed=sorted(k for k,r in records.items() if r['status'] not in ('PASS','REGRESSION')),
        component_no_regression=status=='PASS',allowed_slowdown_pct=0,scope=SCOPE,
        production_changed=False,production_runtime_validated=False,model_tpot_admitted=False)
