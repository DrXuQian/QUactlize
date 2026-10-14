#!/usr/bin/env python3
"""Audit historical selection evidence without deleting it or changing routing.

An admitted recipe and a reusable latency are different assets. Old full-call
timings cannot supply isolated producer/reducer samples by subtraction.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_kpack_strategy import load_sources
from quactlize.runtime.tuning import digest


def read(name):
    path = ROOT / name
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def audit():
    table, sources = load_sources()
    matched = read('policies/kpack_smallm_matched_v1.json')
    raw_matched = read('docs/measurements/smallm_matched_20260915.json.gz')
    if matched['evidence_sha256'] != raw_matched['evidence_sha256']:
        raise ValueError('matched policy and checked-in measurement evidence differ')
    components = read('docs/measurements/kpack_component_policy_20260914.json')
    if components['scope'] != 'SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E':
        raise ValueError('prefill component scope changed')
    if components['external_adapters_included'] or components['small_m_full_dequant']:
        raise ValueError('prefill admission contract changed')
    dequant = read('docs/measurements/kpack_cost_reuse_20260914.json')
    paired = read('policies/kpack_gate_up_selection_v2.json')
    result = dict(
        schema='quactlize.strategy-evidence-audit.v1',
        catalog_sha256=table.identity, sources=sources,
        counts=dict(observations=len(table.observations), exact_keys=len(table.exact),
                    sources=dict(Counter(r.source for r in table.observations)),
                    scopes=dict(Counter(r.scope for r in table.observations))),
        sources_review=[
            dict(source='effective-decode', records=sum(r.source == 'effective-decode' for r in table.observations),
                 reuse='KEEP_EXACT_RECIPE_AND_ORIGINAL_BASELINE',
                 latency='COMPLETE_CALL_ONLY_NOT_ISOLATED_COMPONENT',
                 caveat='F16/BF16, M, channels and router coverage remain distinct'),
            dict(source='q4-decode', records=sum(r.source == 'q4-decode' for r in table.observations),
                 reuse='KEEP_Q4_CANDIDATES', latency='COMPLETE_CALL_ONLY',
                 caveat='Not evidence for another qtype, BF16 or fused GateUp/SwiGLU'),
            dict(source='legacy-decode', records=305, reuse='CANDIDATE_ONLY',
                 latency='DO_NOT_REUSE_AS_F32_MATCHED_OR_COMPONENT_TIME',
                 caveat='275 TC_RETAINED tactics and 30 cross-cohort proposals are not 305 matched timings'),
            dict(source='historical-resident+calibrated-resident', records=3072,
                 reuse='KEEP_TACTICS_AND_ORIGINAL_CONTEXTS', latency='RESIDENT_GEMM_SCOPE_ONLY',
                 caveat='Do not add another reducer or infer a producer by subtraction; no SF prepass/F32 adapters'),
            dict(source='prefill-components', records=len(components['points']),
                 knots=len(components['knots']), components_audited=components['components_audited'],
                 reuse='KEEP_ROUTE_CANDIDATES_AND_COMPONENT_EVIDENCE',
                 latency='GEMM_MAY_INCLUDE_REDUCER; DEQUANT_SEPARATE; SUM_NOT_E2E',
                 caveat='F16-era timing is not BF16 evidence; provider/SDK/active expert domain must match'),
            dict(source='isolated-dequant', records=len(dequant['entries']),
                 reuse='CONDITIONAL_EXACT_COMPONENT_REUSE',
                 latency='REQUIRE_IMPLEMENTATION_SDK_DEVICE_PRECISION_DOMAIN_AND_CACHE_MATCH',
                 caveat='Never prorate all-expert expansion to active-only expansion'),
            dict(source='paired-gate-up', records=len(paired['rows']),
                 reuse='PRESERVE_SEPARATE_FUSED_CONTRACT', latency='NOT_ORDINARY_PROJECTION',
                 caveat='Physical N/layout, activation and output rounding belong to the key'),
        ],
        excluded_as_latency=[
            'BANDWIDTH_FORMULA_OR_ZERO_LAUNCH_COST_REDUCER',
            'NUMERIC_FAILURE_OR_NONFINITE_OR_INCOMPLETE_SAMPLES',
            'NVIDIA_TIMING_AS_PPU_TIMING',
            'WARM_CACHE_TIMING_AS_COLD_WEIGHT_TIMING',
            'CROSS_COHORT_RAW_US_RANKING',
            'FULL_CALL_MINUS_INDEPENDENT_REDUCER_AS_PRODUCER_TIME',
            'FP16_OBSERVATION_RELABELLED_BF16',
            'REPEATED_OR_ADDED_REDUCER_ALREADY_INCLUDED_IN_A_MEASUREMENT',
        ],
        open=dict(matched_exclusions=len(matched['excluded']),
                  raw_matched_missing_points=len(raw_matched['missing']),
                  router_sensitive=sum(r.get('status') == 'ROUTER_SENSITIVE_PARETO' for r in matched['exact']),
                  excluded_keys=matched['excluded']),
        policy=dict(default_selection_changed=False, raw_files_deleted=False,
                    mandatory_baseline='ACTUAL_ORIGINAL_PRODUCTION_SELECTION',
                    measurement='ONE_KERNEL_COMPONENT_PER_TIMED_GRAPH',
                    helper_reuse='EXACT_SEMANTIC_COMPONENT_KEY_AND_AUTHORITY_ONLY',
                    admission='RECOMMENDATION_ONLY_UNTIL_ORIGINAL_WIRING_MODEL_AB_PASSES',
                    regression_action='RETAIN_ORIGINAL',
                    inference_has_timing=False),
    )
    for name in ('docs/measurements/smallm_matched_20260915.json.gz',
                 'docs/measurements/kpack_cost_reuse_20260914.json',
                 'policies/kpack_gate_up_selection_v2.json'):
        result['sources'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    result['audit_sha256'] = digest(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit()
    if a.output:
        with a.output.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print('KPACK_STRATEGY_AUDIT ' + json.dumps({k: result[k] for k in ('counts', 'policy', 'audit_sha256')}))


if __name__ == '__main__':
    main()
