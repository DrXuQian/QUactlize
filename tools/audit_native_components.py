#!/usr/bin/env python3
"""Host-only coverage/reuse audit. Does not launch kernels or change selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.plan import plan
from dev.native_components.history import unchanged_screen, import_prior, costs, format_costs
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smallm-screen',type=Path)
    parser.add_argument('--q8-followup',type=Path)
    parser.add_argument('--prior-decode',type=Path)
    parser.add_argument('--export-costs',type=Path)
    a=parser.parse_args()
    report=plan()
    wanted={r['id']:r['quactlize']for r in report['points']
            if r['quactlize']['status']=='IMPORT_UNCHANGED_ARM_SCREEN'}
    recovered={}
    for role,path in (('screen',a.smallm_screen),('q8',a.q8_followup)):
        if path:recovered.update(unchanged_screen(path,role,wanted))
    if a.prior_decode:import_prior(a.prior_decode,recovered)
    report['recovered_unchanged']=recovered
    report['summary']['unchanged_screen_recovered']=sum(r['status']=='SCREEN_COMPONENTS_RECOVERED'for r in recovered.values())
    report['summary']['unchanged_still_needs_import']=len(wanted)-report['summary']['unchanged_screen_recovered']
    report['scope']='HOST_AUDIT_ONLY_NOT_NEW_GPU_TIMINGS'
    report['source_plan_sha256']=report.pop('sha256')
    report['sha256']=digest(report)
    if a.output.exists():parser.error('--output must be a new file')
    atomic_json(a.output,report)
    if a.export_costs:
        if a.export_costs.exists():parser.error('--export-costs must be a new file')
        with a.export_costs.open('x')as out:out.write(format_costs(costs(report)))
    print('COMPONENT_AUDIT '+json.dumps(report['summary']),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
