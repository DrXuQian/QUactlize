"""Audit known component evidence before scheduling any additional GPU work."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from dev.llama_ab.contract import inventory
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]


def plan():
    small = json.loads((ROOT/'policies/kpack_component_decode_v1.json').read_text())
    prefill = json.loads((ROOT/'policies/kpack_prefill_confirmation_v1.json').read_text())
    by_small = {p['point']:(r,p) for r in small['rows'] for p in r['profiles']}
    by_large = {p['point']:p for p in prefill['points']}
    points=[]
    for row in inventory():
        w=row['workload']; gaps=[]
        if row['cohort']=='decode':
            policy, proof=by_small[row['id']]
            recipe=policy['config']
            terms=proof.get('components')
            if terms:
                evidence=dict(status='RECORDED_SUMMARY_RAW_SAMPLES_REQUIRED',
                    components=terms, historical_sum_us=proof['selected_us'],
                    scope=small['scope'], source=policy['source'],
                    record_sha256=proof['record_sha256'], archives=small['archives'])
            else:
                # These are not missing GPU measurements. The compact policy
                # dropped unchanged-arm timings; import their screen receipts.
                evidence=dict(status='IMPORT_UNCHANGED_ARM_SCREEN',
                    historical_sum_us=None, source=policy['source'],
                    record_sha256=proof['record_sha256'], archives=small['archives'])
                gaps.append('IMPORT_RAW_SCREEN_NOT_AUTOMATIC_REMEASUREMENT')
            required=['SELECTED_CORE_WITH_ACTUAL_F32_ENDPOINTS']
            if recipe['kind']=='tc' and w['operation']=='indexed':
                required += ['INDEXED_PREPARE','INDEXED_FINISH_OR_FUSED_REDUCER_FINISH']
            if recipe.get('split',1)>1:
                required += ['ACTUAL_REDUCER_OR_FUSED_COMPLETION']
            if recipe['kind']=='tc' and recipe.get('route',0)%2 and w['qtype']!=8:
                required += ['SF_METADATA_EXPANSION']
        else:
            p=by_large[row['id']];recipe=p['recipe']
            rounds=p['component_rounds']
            if len(rounds)!=6 or any(not r for r in rounds):
                raise ValueError('prefill selected component coverage differs')
            values=[math.fsum(c['median_us'] for c in r) for r in rounds]
            kinds=[[c['kind'] for c in r] for r in rounds]
            if any(k!=kinds[0] for k in kinds):
                raise ValueError('selected component inventory changed between rounds')
            evidence=dict(status='CONFIRMED_COMPONENT_ROUND_MEDIANS_RAW_SAMPLES_REQUIRED',
                component_rounds=rounds, historical_sum_us=statistics.median(values),
                archive_sha256=prefill['archive_sha256'], record_sha256=p['result_sha256'],
                scope=prefill['scope'])
            required=['SELECTED_CORE','INPUT_ADAPTER','OUTPUT_ADAPTER_OR_FUSED_REDUCER_FINISH']
            if recipe['route'] in ('sf','full'):required += ['SELECTED_METADATA_OR_FULL_EXPANSION']
            if dict(recipe['parameters']).get('split',1)>1:required += ['ACTUAL_REDUCER_OR_FUSED_COMPLETION']
            if w['operation']=='indexed':
                required += ['GPU_IDS_TO_ROW_MAPS']
                gaps.append('MEASURE_GPU_ROW_MAP_CREATION')
            if recipe['kind']=='full' and w['operation']=='indexed':
                required += ['CURRENT_ACTIVE_EXPERT_DIRECTORY']
                # Full.launch already times active_experts. Do NOT charge it
                # again merely because the expansion also uses that directory.
            if recipe['kind']=='full':
                required += ['PER_CALL_ERROR_RESET']
                gaps.append('MEASURE_FOUR_BYTE_ERROR_RESET')
        points.append(dict(row, selected_recipe=recipe, quactlize=evidence,
            required_components=required, pending=gaps,
            native=dict(status='MEASURE_UNCHANGED_DISPATCH_COMPONENTS',
                boundary='RESIDENT_RAW_GGUF_F32_A_GPU_IDS_TO_F32_C',
                cuBLAS='ORDINARY_EVENTS_NO_CAPTURE_INTERNAL_REDUCTION_INCLUDED',
                helpers='ISOLATED_ONE_KERNEL_TIMER_EACH',
                native_precision='KEEP_ORIGINAL_INCLUDING_ACTIVATION_QUANTIZATION')))
    counts=Counter(g for p in points for g in p['pending'])
    result=dict(schema='quactlize.native-component-plan.v1',points=points,
        summary=dict(points=len(points),dense=sum(p['workload']['operation']=='dense' for p in points),
            indexed=sum(p['workload']['operation']=='indexed' for p in points),
            historical_smallm_summaries=sum('components'in p for _,p in by_small.values()),
            historical_prefill_summaries=len(by_large),pending_counts=dict(counts)),
        scope='COMPONENT_AUDIT_NOT_PERFORMANCE_ADMISSION',
        no_config_sweep=True, llama_source_patch=False, llama_branch_pin=False,
        old_six_library_bundle_required=False, production_changed=False,
        model_tpot_admitted=False,
        sum_rule='COUNT_REAL_INVOCATIONS_ONCE_NEVER_MISSING_AS_ZERO',
        reuse_rule='MATCH_SOURCE_DEVICE_SDK_PRECISION_ROUTE_DOMAIN_LAUNCH_CACHE_AND_TIMER',
        confirmation='PAIRED_SELECTED_RECIPES_ONLY_FOR_CLOSE_NOISY_OR_STALE_COMPARISONS')
    result['sha256']=digest(result)
    return result
