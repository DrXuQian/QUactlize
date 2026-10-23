#!/usr/bin/env python3
"""Prune a completed dense cohort to affected/noisy full-call finalists (CPU only)."""
import argparse
import json
import lzma
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.finalists import make_plan, snapshot


def format_snapshot(value, depth=0):
    # Generated records stay one-per-line instead of thousands of scalar lines.
    if depth>=3 or not isinstance(value,(dict,list)) or not value:
        return json.dumps(value,separators=(',',':'),allow_nan=False)
    if isinstance(value,dict):
        rows=[json.dumps(k)+': '+format_snapshot(v,depth+1) for k,v in value.items()]
        brackets=('{','}')
    else:
        rows=[format_snapshot(v,depth+1) for v in value];brackets=('[',']')
    return brackets[0]+'\n'+',\n'.join('  '*(depth+1)+r for r in rows)+'\n'+'  '*depth+brackets[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--helper-policy',type=Path,default=ROOT/'policies/dense_full_helpers_20261023.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with lzma.open(a.review,'rt') as stream:review=json.load(stream)
    result=make_plan(review,json.loads(a.helper_policy.read_text()))
    with a.output.open('x') as stream:stream.write(format_snapshot(snapshot(result))+'\n')
    print('PREFILL_FINALISTS_PLAN '+json.dumps(dict(result['counts'],output=str(a.output),
        kernel_launches=0,production_changed=False,performance_admitted=False)))


if __name__=='__main__':main()
