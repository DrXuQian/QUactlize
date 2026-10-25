#!/usr/bin/env python3
"""Freeze new-M dense SF/Full comparisons from current model/TP-local families."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.m_span import make_plan, MS
from dev.strategy.journal import atomic_json
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.runtime.tuning import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='new directory for plan.json and the host control query')
    p.add_argument('--m',type=int,nargs='+',default=list(MS))
    p.add_argument('--compute-units',type=int,default=72)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result=make_plan(json.loads(EVIDENCE.read_text()),digest(json.loads(
        (ROOT/'policies/dense_full_helpers_20261023.json').read_text())),a.output,
        ms=a.m,compute_units=a.compute_units)
    atomic_json(a.output/'plan.json',result)
    print('DENSE_M_SPAN_PLAN '+json.dumps(dict(result['counts'],ms=a.m,output=str(a.output/'plan.json'),
        kernel_launches=0,production_changed=False,performance_admitted=False)))


if __name__=='__main__':main()
