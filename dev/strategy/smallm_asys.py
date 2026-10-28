"""Current-body small-M recovery. Historical timings never select a winner."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import statistics
import subprocess

from dev.strategy.refresh import proposed, historical, original_recipe, request
from dev.strategy.components import statistics_checked
from dev.strategy.vector import execution_contract
from dev.strategy.moe_token_pair_plan import COHORTS as MOE_PAIR_COHORTS
from dev.strategy.moe_single_row_plan import COHORT as MOE_SINGLE_COHORT
from dev.strategy.moe_model_plan import COHORT as MOE_MODEL_COHORT
from dev.strategy.moe_sweep_plan import COHORT as MOE_SWEEP_COHORT
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
METHOD='ASYS_GPU_ACTIVITY'
MODEL_TOKEN_COHORTS=('token-cta-models','token-cta-model-gaps')
TOKEN_COHORTS=('q6-token-cta','token-cta-formats',*MODEL_TOKEN_COHORTS)
MODEL_MOE_COHORTS=(MOE_MODEL_COHORT,MOE_SWEEP_COHORT)
MOE_COHORTS=(*MOE_PAIR_COHORTS,MOE_SINGLE_COHORT,*MODEL_MOE_COHORTS)
COHORTS=('regressions',*TOKEN_COHORTS,*MOE_COHORTS)


def model_coverage(cohort):
    """Actual model tuples, not N/K crossed with every supported format.

    The old N1024/K5120 measurements are coverage only; their latencies are
    never imported as new samples or counted as passes in this run.
    """
    from dev.strategy.m_coverage import families
    if cohort not in MODEL_TOKEN_COHORTS:raise ValueError('not a model token cohort')
    previous={(q,1024,5120) for q in (8,12,13,14)}
    if cohort=='token-cta-model-gaps':
        from quactlize.execution.token_cta_policy import capability
        capability()  # Do not skip old coverage after an unvalidated body change.
    all_families=families()
    covered=[f for f in all_families if f in previous]
    selected=[f for f in all_families if cohort=='token-cta-models' or f not in previous]
    return dict(registry='policies/dense_prefill_v1.json',families=all_families,
        total_families=len(all_families),total_points=8*len(all_families),selected_families=selected,
        excluded_previous_families=covered if cohort=='token-cta-model-gaps' else [],
        excluded_previous_points=8*len(covered) if cohort=='token-cta-model-gaps' else 0,
        previous_evidence={
            'q6':dict(source='724b1ed290f6f8468135bf847662c256ef20ce13',
                archive_sha256='c43f3b1a549c92a5d7ec241276519300c20c20199a404dde2d942a1a94b16224',
                reference='docs/PPU_Q6_TOKEN_CTA.md'),
            'formats':dict(source='1048a176a468e07abf30685f25344a4bae7c3410',
                archive_sha256='28a8d7dbaea294311ac1d2b61c329b39068c246f8bf3a585e0a146069a3c1631',
                reference='policies/kpack_token_cta_v1.json')},
        prior_timings_reused=False,prior_passes_added=False)


def workloads(cohort='regressions'):
    if cohort in MODEL_MOE_COHORTS:
        raise ValueError('model MoE cohort requires the actual GGUF header/TP inventory')
    if cohort==MOE_SINGLE_COHORT:
        from dev.strategy.moe_single_row_plan import workloads as single_workloads
        return single_workloads()
    if cohort in MOE_PAIR_COHORTS:
        from dev.strategy.moe_token_pair_plan import workloads as pair_workloads
        return pair_workloads(cohort)
    if cohort in MODEL_TOKEN_COHORTS:
        return [Workload(q,n,k,m) for q,n,k in model_coverage(cohort)['selected_families'] for m in range(1,9)]
    if cohort=='q6-token-cta':return [Workload(14,1024,5120,m) for m in range(1,9)]
    if cohort=='token-cta-formats':
        return [Workload(q,1024,5120,m) for q in (10,11,12,13,8) for m in range(1,9)]
    if cohort!='regressions':raise ValueError('unknown bounded small-M cohort')
    return [Workload(q,1024,5120,m) for q,ms in ((13,range(1,5)),(14,range(2,8))) for m in ms]


def make_plan(originals,cohort='regressions'):
    if cohort==MOE_SINGLE_COHORT:
        from dev.strategy.moe_single_row_plan import make_plan as single_plan
        return single_plan(originals)
    if cohort in MOE_PAIR_COHORTS:
        from dev.strategy.moe_token_pair_plan import make_plan as pair_plan
        return pair_plan(originals,cohort)
    ws=workloads(cohort)
    if [r['request'] for r in originals]!=[request(w) for w in ws]:
        raise ValueError('regression C++ selection inventory differs')
    prior={tuple(r['key']):r for r in json.loads((ROOT/'policies/kpack_component_decode_v1.json').read_bytes())['rows']}
    history=historical();points=[];tc={};vectors={}
    for w,old in zip(ws,originals):
        if old['status']:raise ValueError('current production control unavailable')
        p=proposed(dict(id=w.key,workload=asdict(w),cohort='decode',consumers=[]),
                   original_recipe(old),history,prior,allow_new_family=True)
        p.update(selection=old,execution_contract=execution_contract(w))
        if cohort=='token-cta-formats' or cohort in MODEL_TOKEN_COHORTS:
            from dev.strategy.token_cta import retain_tc_controls
            p=retain_tc_controls(p)
        if cohort in TOKEN_COHORTS:
            from dev.strategy.token_cta import augment
            p=augment(p)
        for arm in p['arms']:
            (tc if arm['recipe']['kind']=='tc' else vectors)[arm['module']]=arm['spec']
        points.append(p)
    result=dict(schema='quactlize.smallm-asys.v1',phase='regression-asys',cohort=cohort,points=points,tc=tc,vectors=vectors,
        counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),tc_modules=len(tc),
                    vector_modules=len(vectors),families=len({(w.qtype,w.n,w.k) for w in ws})),
        timing_method=METHOD,selection_metric='MEAN_COMPONENT_SUM',noise_policy='DIAGNOSTIC_NOT_VETO',
        historical_times_reused=False,production_changed=False,native_comparison=False,
        scope=('REAL_MODEL_DENSE_M1_TO_8_CTA_TOKEN_SWEEP_NOT_NATIVE_OR_MODEL_ADMISSION' if cohort in MODEL_TOKEN_COHORTS else
               'Q2_TO_Q5_Q8_DENSE_M1_TO_8_CTA_TOKEN_AB_NOT_NATIVE_OR_MODEL_ADMISSION' if cohort=='token-cta-formats' else
               'Q6_DENSE_M1_TO_8_CTA_TOKEN_AB_NOT_NATIVE_OR_MODEL_ADMISSION' if cohort=='q6-token-cta' else
               'TEN_DENSE_REGRESSION_POINTS_CURRENT_CONFIG_SWEEP_NOT_NATIVE_OR_MODEL_AB'),
        sources={p:sha(ROOT/p) for p in ('policies/kpack_component_decode_v1.json',
                  'policies/kpack_smallm_effective_v1.hpp','dev/strategy/vector.py')})
    if cohort in TOKEN_COHORTS:
        for p in ('dev/strategy/token_cta.py','quactlize/execution/simt_token_cta.cuh',
                  'quactlize/execution/simt_q8_token_cta.cuh'):
            result['sources'][p]=sha(ROOT/p)
    if cohort in MODEL_TOKEN_COHORTS:
        result['coverage']=model_coverage(cohort)
        for p in ('policies/dense_prefill_v1.json','policies/kpack_token_cta_v1.json','dev/strategy/m_coverage.py',
                  'quactlize/execution/measured_decode.inc','quactlize/execution/simt_strategy.hpp'):
            result['sources'][p]=sha(ROOT/p)
    result['sha256']=digest(result)
    return json.loads(json.dumps(result))


def assignments(plan,workers,*,balanced=False):
    """Balance whole paired points; the score is scheduling only, not a time.

    M-ordered round robin would give one card M1/M5 and another M4/M8 for
    every N/K. All comparison arms of a point must stay on one physical GPU.
    Preserve the old cohorts' assignment so their frozen resumes still work.
    """
    if not 1<=workers<=len(plan['points']):raise ValueError('invalid worker count')
    if not balanced and plan.get('cohort') not in (*MODEL_TOKEN_COHORTS,*MODEL_MOE_COHORTS):
        return [[p['id'] for j,p in enumerate(plan['points']) if j%workers==i] for i in range(workers)]
    def weight(p):
        w=p['workload']
        return (w['n']*w['k']+4*1024*1024)*len(p['arms'])*(8+w['m'])*w['topk']
    result=[[] for _ in range(workers)];loads=[0]*workers
    for p in sorted(plan['points'],key=lambda p:(-weight(p),p['id'])):
        i=min(range(workers),key=lambda j:(loads[j],j))
        result[i].append(p['id']);loads[i]+=weight(p)
    return result


def revision(root, *, archive=False):
    # HEAD is an anchor, not proof that live files equal that commit. Direct
    # runs bind source/harness hashes separately; coordinators may archive HEAD.
    root=Path(root)
    receipt=root/'.quactlize-source.json'
    if receipt.exists():
        value=json.loads(receipt.read_bytes())
        if set(value)!={'quactlize_commit','quactlize_tree','actlize_commit','snapshot'} or value['snapshot']!='COMMITTED_ARCHIVE':
            raise ValueError('invalid frozen source receipt')
    else:
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
        if Path(git('rev-parse','--show-toplevel')).resolve()!=root.resolve():
            raise ValueError('source has neither its own Git checkout nor a frozen commit receipt')
        value=dict(quactlize_commit=git('rev-parse','HEAD'),quactlize_tree=git('rev-parse','HEAD^{tree}'),
                   actlize_commit=git('rev-parse','HEAD:third_party/actlize'),
                   snapshot='COMMITTED_ARCHIVE' if archive else 'WORKTREE')
    if any(not re.fullmatch('[0-9a-f]{40}',value[k]) for k in ('quactlize_commit','quactlize_tree','actlize_commit')):
        raise ValueError('source revision is not a full commit/tree ID')
    return value


def mean_decision(costs,roles):
    if 'original' not in costs or set(costs)!=set(roles):raise ValueError('missing paired current control')
    values={}
    for aid,rounds in costs.items():
        stats=statistics_checked(rounds)
        flat=[v for r in rounds for v in r]
        stats.update(mean_us=statistics.fmean(flat),min_us=min(flat),max_us=max(flat),
                     noise_advisory=stats['spread_pct']>5)
        values[aid]=stats
    winner=min(values,key=lambda a:(values[a]['mean_us'],a!='original',a))
    return dict(status='MEASURED_MEAN_SELECTION',winner=winner,statistics=values,
        metric='ARITHMETIC_MEAN_OF_SEPARATE_COMPONENT_SUMS',noise_policy='DIAGNOSTIC_NOT_VETO',
        delta_pct=100*(values[winner]['mean_us']/values['original']['mean_us']-1),
        native_no_regression_admitted=False,model_tpot_admitted=False)


def check_producers(point,data):
    from dev.strategy.refresh_gaps import producer_entry
    arms={a['id']:a for a in point['arms']}
    unavailable=data.get('resource_unavailable',{})
    if set(data['arms'])&set(unavailable) or set(data['arms'])|set(unavailable)!=set(arms):
        raise ValueError('requested config inventory incomplete')
    for aid,query in unavailable.items():
        from dev.strategy.resources import probe_spec
        from quactlize.dispatch.table import Recipe
        arm=arms[aid];r=Recipe.from_dict(arm['recipe'])
        spec=probe_spec(Workload(**point['workload']),r,int(dict(r.parameters).get('grid_mode',0)>=2))
        if (arm['required'] or r.kind!='tc' or query.get('status')!='UNAVAILABLE' or
                query.get('rc')!=1 or query.get('runtime_status')!=0 or query.get('request')!=spec or
                not query.get('module_sha256')):
            raise ValueError('unavailable recipe lacks a resource rejection proof')
    for aid,measured in data['arms'].items():
        spec=arms[aid]['spec'];recipe=arms[aid]['recipe']
        nodes=[data['component_records'][k]['identity']['node'] for k in data['arm_components'][aid]
               if data['component_records'][k]['kind']=='producer']
        if len(nodes)!=1:raise ValueError('missing/duplicate compute kernel')
        n=producer_entry(nodes[0])
        if recipe['kind']=='tc':
            if n not in ('cutlass::device_kernel','cutlass::Kernel'):
                raise ValueError('TC carrier did not execute CUTLASS compute')
        else:
            c=spec['config']
            common=c['variant']==3 and ((c['columns']==4 and c['values']==8 and c['warps'] in (2,8)) or
                                       (c['columns']==8 and c['values']==4 and c['warps']==4) or
                                       (spec['q']==12 and (c['columns'],c['warps'],c['values'])==(4,4,4)))
            if recipe['kind']=='q4':
                compute=spec['compute']
                if (compute not in ('f16','bf16') or compute!=point['workload']['compute'] or
                        point['execution_contract']['compute_type']!=int(compute=='bf16')):
                    raise ValueError('Q4 producer compute type differs from workload/ABI')
                expected='quactlize::execution::q4_decode::'+('kernel_bf16' if compute=='bf16' else 'kernel')
            elif point.get('same_expert_pair') or point.get('indexed_single_row'):
                expected=arms[aid]['expected_producer']
            elif 'cta_tokens' in spec:
                expected='quactlize::execution::simt::'+('q8_token_cta_reader' if spec['q']==8 else 'token_cta_reader')
            elif 'q8_body' in spec:
                expected='quactlize::execution::simt::q8_address_0_'+str(spec['q8_body'])
            elif spec['q']==8 and c['variant']>=4:
                from dev.strategy.token_cta import q8_control_entry
                expected=q8_control_entry(Workload(**point['workload']),c,dict(recipe['parameters'])['split'])
            else:
                expected='quactlize::execution::simt::'+('common_reader' if common else 'register_reuse')
            if expected!=n:raise ValueError('candidate executed the wrong production reader: '+n)
            if 'moe_pair_rows' in spec or point.get('indexed_single_row') and recipe['kind']=='simt':
                w=point['workload']
                grid=[w['n']//(c['columns']*c['values']),w['m']*w['topk'],dict(recipe['parameters'])['split']]
                # Common/pair readers use 3-D ownership; retained generic,
                # measured fixed and Q8-vector controls flatten the same work.
                if n not in ('quactlize::execution::simt::common_reader',
                             'quactlize::execution::simt::moe_token_pair_reader'):
                    grid=[grid[0]*grid[1]*grid[2],1,1]
                if (nodes[0]['grid']!=grid or
                        nodes[0]['block']!=[c['warps']*32,1,1]):
                    raise ValueError('indexed reader grid/block differs from routed-row ownership')
            if 'cta_tokens' in spec:
                expected_grid=[point['workload']['n']//(c['columns']*c['values']),1,
                               dict(recipe['parameters'])['split']]
                if (nodes[0]['grid']!=expected_grid or nodes[0]['block']!=[c['warps']*32,1,1]):
                    raise ValueError('token CTA grid/block differs from the planned ownership')
            if measured['receipt'].get('execution_contract')!=point['execution_contract']:
                raise ValueError('candidate did not submit the planned dense ABI')


def validate(data):
    from dev.strategy.followup import validate_confirmation
    if data.get('timing_method')!=METHOD:raise ValueError('event/host timing is not admitted to Asys selection')
    if data.get('measurement_phase') not in (None,'screen','confirm'):raise ValueError('unknown Asys sweep phase')
    validate_confirmation(data,decision_verifier=mean_decision)
    rounds,samples=(4,3) if data.get('measurement_phase')=='screen' else (6,5)
    intervals=[]
    for r in data['component_records'].values():
        p=r.get('latency_provenance',{})
        if (p!=data.get('latency_provenance') or p.get('timing_method')!=METHOD or
                any(not re.fullmatch('[0-9a-f]{40}',p.get(k,'')) for k in
                    ('quactlize_commit','quactlize_tree','actlize_commit')) or
                any(not re.fullmatch('[0-9a-f]{64}',p.get(k,'')) for k in
                    ('trace_sha256','sqlite_sha256','images_sha256','harness_sha256','source_sha256','asys_sha256')) or
                not p.get('sdk') or not p.get('device')):
            raise ValueError('latency lacks commit/profiler provenance')
        seen=set()
        for item in r.get('trace_slices',[]):
            slot=item['round'],item['sample'];ns=item['durations_ns']
            if (slot in seen or not 0<=slot[0]<rounds or not 0<=slot[1]<samples or
                    item['count']!=r['calls_per_graph'] or len(ns)!=item['count'] or
                    any(type(x)!=int or x<=0 for x in ns)):
                raise ValueError('Asys raw component sample coverage differs')
            us=[x/1000 for x in ns];mean=statistics.fmean(us)
            if (mean!=item['mean_us'] or min(us)!=item['min_us'] or max(us)!=item['max_us'] or
                    r['samples_us'][slot[0]][slot[1]]!=mean):
                raise ValueError('component mean differs from raw GPU activity')
            seen.add(slot);intervals.append((item['first_activity'],item['count']))
        if len(seen)!=rounds*samples or r.get('mean_us')!=statistics.fmean(x for row in r['samples_us'] for x in row):
            raise ValueError('missing raw Asys samples or invalid mean')
    cursor=0
    for first,count in sorted(intervals):
        if first!=cursor:raise ValueError('Asys activity ranges overlap or have gaps')
        cursor+=count
