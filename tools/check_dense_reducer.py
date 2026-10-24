#!/usr/bin/env python3
"""Numerical admission for reuse of the existing compact reducer in dense prefill."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.dense_helpers.reducer_gate import run
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('--output already exists')
    plan=json.loads(a.plan.read_text())
    if plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise ValueError('frozen worker plan differs')
    value=run(a.sdk,a.library,plan['points'])
    value.update(library_sha256=sha(a.library),plan_sha256=plan['plan_sha256'])
    atomic_json(a.output,value)
    print(f'DENSE_REDUCER_GATE PASS cases={value["cases"]} output={a.output}',flush=True)


if __name__=='__main__':main()
