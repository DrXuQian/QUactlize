#!/usr/bin/env python3
"""Blind historical config replay. No PPU, JIT or champion injection.

Each query withholds ALL records of that N/K family, including other M and
precision variants. Only after generation is the champion control revealed.
This is development cross-validation, not independent new-device evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.plan import make_plan
from dev.strategy.prescreen import semantic_config
from dev.strategy.space import scheduled_key, split_variants
from dev.strategy.provenance import source_identity, verify_unchanged
from dev.strategy.neighbors import MODES
from dev.strategy.replay import replay
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest


def audit_plan(plan):
    misses=[]
    for point in plan['points']:
        report=point['pruning'];probe=report['recall']
        if probe['status']!='RULE_MISS':continue
        workload=Workload(**point['workload'])
        identity=scheduled_key if plan.get('factorized_search') else semantic_config
        remaining={identity(v) for r in report['unselected_optional']
                   for v in (split_variants(workload,Recipe.from_dict(r['config'])) if plan.get('factorized_search')
                             else (Recipe.from_dict(r['config']),))}
        for c in point['champion_controls']:
            recipe=Recipe.from_dict(c)
            if recipe.key not in probe['missing']:continue
            misses.append(dict(point=point['id'],workload=point['workload'],champion=c,
                classification='RANKING_MISS' if identity(recipe) in remaining else
                               'RULE_RANGE_OR_TEMPLATE_COVERAGE_MISS',
                champion_added_to_candidates=False))
    observations=sum(p['pruning']['recall']['controls'] for p in plan['points'])
    recalled=sum(p['pruning']['recall']['recalled'] for p in plan['points'])
    statuses=Counter(p['pruning']['recall']['status'] for p in plan['points'])
    result=dict(schema='quactlize.blind-rule-audit.v1',plan_sha256=plan['plan_sha256'],
        source='HISTORICAL_LEAVE_NK_FAMILY_OUT_DEVELOPMENT_REPLAY',
        points=len(plan['points']),statuses=dict(statuses),champions=observations,recalled=recalled,
        recall_pct=100*recalled/observations if observations else None,
        all_reference_points_recalled=bool(observations) and not misses,
        recall_scope='GENERATED_COMPONENT_SPACE' if plan.get('factorized_search') else 'FLAT_CONFIG_LIST',
        flat_list_recalled=sum(p['pruning']['literal_recall']['recalled'] for p in plan['points']),
        producer_prototype_count_distribution=dict(sorted(Counter(p['pruning']['prototype_count'] for p in plan['points']).items())),
        candidate_count_distribution=dict(sorted(Counter(len(p['candidates']) for p in plan['points']).items())),
        misses=misses,champion_injection=False,device_work=0,performance_proven=False,
        new_device_generalization_proven=False,production_changed=False,
        rule_source_sha256=plan.get('source_identity',{}).get('sha256'),
        work_counts=plan['work_counts'])
    result['audit_sha256']=digest(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=('pilot','decode','prefill','all'),default='all')
    parser.add_argument('--limit',type=int,default=8,help='optional-neighbor target, not a hard coverage cap')
    parser.add_argument('--neighbors',type=int,default=1)
    parser.add_argument('--per-stratum',type=int,default=2)
    parser.add_argument('--flat',action='store_true',help='replay the older flat-list rule for comparison')
    parser.add_argument('--k-weight',type=float,default=1.,help='development sensitivity check, not a target-specific exception')
    parser.add_argument('--reader-strata',action='store_true',help='experimental: retain each SIMT reader variant independently')
    parser.add_argument('--stencil',choices=MODES,help='default: warps for factored plans, none for --flat; not a config product')
    formats=parser.add_mutually_exclusive_group()
    formats.add_argument('--cross-format',action='store_true',help='experimental generic K-quant template transfer; disabled by default after recall regression')
    formats.add_argument('--same-format',action='store_true',help='same-format templates (the default)')
    parser.add_argument('--output',type=Path,required=True,help='new audit directory')
    args=parser.parse_args();start=time.monotonic()
    if args.output.exists():parser.error('--output must be a new directory')
    before=source_identity(ROOT)
    stencil=args.stencil if args.stencil is not None else 'none' if args.flat else 'warps'
    plan=make_plan(args.cohort,args.limit,args.neighbors,args.per_stratum,not args.flat,args.k_weight,args.cross_format,args.reader_strata,stencil)
    plan['source_identity']=verify_unchanged(before,ROOT)
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    result=audit_plan(plan)
    historical=replay(plan,ROOT)
    verify_unchanged(before,ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in (('plan',plan),('audit',result),('historical-replay',historical)):
        with (args.output/(name+'.json')).open('x') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print('STRATEGY_RULE_AUDIT '+json.dumps({k:v for k,v in result.items() if k!='misses'}),flush=True)
    print('STRATEGY_HISTORICAL_REPLAY '+json.dumps(historical.get('counts',{})),flush=True)
    print(f'STRATEGY_RULE_AUDIT_WALL seconds={time.monotonic()-start:.2f} scope=HOST_AUDIT_ONLY',flush=True)
    return 0 if result['all_reference_points_recalled'] else 2


if __name__=='__main__':raise SystemExit(main())
