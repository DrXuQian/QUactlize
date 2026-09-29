#!/usr/bin/env python3
"""Bounded before/after prepare check; the normal model gate also checks aliasing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

BASELINE = '39e9c6fdb82611ecd7c05a0b0511e0ae4f41f269'
CASES = [(t,k,5,1,0,bf,0) for t in (1,2,4,8) for k in (512,2048) for bf in (0,1)]


def timing_rows(text):
    rows=[]
    for line in text.splitlines():
        if line.startswith('MOE_PREPARE_TIME '):
            row=dict(re.findall(r'(\w+)=(\[[^\]]*\]|\S+)',line))
            row['samples']=json.loads(row['samples'])
            rows.append(row)
    if len(rows)!=2 or {r['arm'] for r in rows}!={'0','1'}:
        raise ValueError('missing/duplicate before-after timing rows')
    for row in rows:
        if len(row['samples'])!=60 or not all(math.isfinite(x) and x>0 for x in row['samples']):
            raise ValueError('invalid finite timing denominator')
        median=statistics.median(row['samples'])
        reported=float(row['median_us'])
        if not math.isfinite(reported) or abs(reported-median)>1e-5:
            raise ValueError('reported prepare median differs from raw samples')
        row['median_us']=median
    return sorted(rows,key=lambda r:r['arm'])


def run(bundle,output):
    bundle=bundle.resolve(strict=True)
    manifest=json.loads((bundle/'manifest.json').read_text())
    binary=bundle/'bench'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if (manifest.get('platform')!='ppu' or sha(binary)!=manifest['binary_sha256'] or
            manifest.get('prepare_baseline',{}).get('commit')!=BASELINE):
        raise ValueError('prepare integration baseline/binary identity differs')
    output.mkdir(parents=True,exist_ok=False)
    (output/'inputs.json').write_text(json.dumps(dict(binary_sha256=sha(binary),
        manifest_sha256=sha(bundle/'manifest.json'),baseline=BASELINE,cases=CASES),indent=2)+'\n')
    def launch(arguments,log):
        with log.open('w') as stream:
            result=subprocess.run([str(binary),*arguments],stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text(errors='replace')
        if result.returncode: raise ValueError(f'prepare gate rc={result.returncode}; log={log}')
        return text
    edge=launch(['--router-edge-check'],output/'router-edges.log')
    marker='MOE_ROUTER_EDGE PASS cases=56 arms=3 scope=ORDINARY_SOFTMAX_NO_BIAS'
    if edge.splitlines().count(marker)!=1: raise ValueError('router edge coverage differs')
    records=[]
    for case in CASES:
        name='-'.join(map(str,case))
        record=dict(case=case,status='FAIL')
        try:
            rows=timing_rows(launch(['--case',*map(str,case)],output/(name+'.log')))
            for row in rows:
                actual=tuple(int(row[n]) for n in ('tokens','k','mask','merged','router'))
                if actual!=case[:5] or row['compute']!=('bf16' if case[5] else 'f16') or int(row['weak'])!=case[6]:
                    raise ValueError('measured prepare case differs')
            record.update(status='PASS',rows=rows,
                delta_pct=100*(rows[1]['median_us']/rows[0]['median_us']-1))
        except (ValueError,KeyError) as exc:
            record['error']=str(exc)
        records.append(record)
        (output/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(f'MOE_PREPARE_INTEGRATION completed={len(records)}/{len(CASES)} status={record["status"]}',flush=True)
    passed=sum(r['status']=='PASS' for r in records)
    summary=dict(status='PASS' if passed==len(CASES) else 'FAIL',passed=passed,expected=len(CASES),
        router_edge_cases=56,records=records,baseline=BASELINE,
        scope='PREPARE_ONLY_NOT_GEMM_OR_MODEL',correctness='BEFORE_TIMING_FOUR_CHANGED_INPUT_REPLAYS',
        timing_idle_admission='EXTERNAL_LOAD_AUDIT_REQUIRED')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.bundle,args.output))
