#!/usr/bin/env python3
"""Fresh-process prepare comparisons. One failed arm never invalidates peers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import math


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numeric-only',action='store_true')
    a=p.parse_args();a.bundle=a.bundle.resolve()
    binary=a.bundle/'bench';manifest=json.loads((a.bundle/'manifest.json').read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=manifest['binary_sha256']:
        raise ValueError('prepare binary differs')
    a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(binary_sha256=manifest['binary_sha256'],source_hashes=manifest['source_hashes'])
    ip=a.output/'inputs.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('resume source differs')
    ip.write_text(json.dumps(identity,indent=2)+'\n')
    numeric=a.output/'candidate-correctness.json'
    if not numeric.exists():
        with (a.output/'candidate-correctness.log').open('w') as log:
            rc=subprocess.run([str(binary),'--candidate-check'],stdout=log,stderr=subprocess.STDOUT).returncode
        text=(a.output/'candidate-correctness.log').read_text()
        marker='MOE_PREPARE_CORRECTNESS PASS cases=3840 replays=4 arms=CANDIDATE_WITH_ROUTER_ORACLE'
        good=rc==0 and text.count(marker)==1
        numeric.write_text(json.dumps(dict(status='PASS' if good else 'FAIL',process_rc=rc,
            identity=identity,contexts=3840,log_sha256=hashlib.sha256(text.encode()).hexdigest(),
            scope='PREPARE_AND_ACTUAL_SIMT_SWIGLU_NOT_GEMM'),indent=2)+'\n')
    receipt=json.loads(numeric.read_text())
    if receipt['identity']!=identity or receipt['status']!='PASS':raise ValueError('candidate numerical gate did not pass')
    if a.numeric_only:return
    cases=[(t,k,mask,1,0,bf,weak) for t in (1,2,4,8) for k in (512,2048)
           for mask in (0,1,5) for bf in (0,1) for weak in (0,)]
    results=[];start=time.monotonic()
    for i,case in enumerate(cases):
        name='-'.join(map(str,case));target=a.output/(name+'.json')
        if target.exists():
            result=json.loads(target.read_text())
            if result.get('identity')!=identity:raise ValueError('case identity differs')
            results.append(result);continue
        cmd=[str(binary),'--case',*map(str,case)]
        with (a.output/(name+'.log')).open('w') as log:
            code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        text=(a.output/(name+'.log')).read_text()
        rows=[]
        for line in text.splitlines():
            if line.startswith('MOE_PREPARE_TIME '):
                fields=dict(re.findall(r'(\w+)=(\[[^\]]*\]|\S+)',line))
                fields['samples']=json.loads(fields['samples']);rows.append(fields)
        good=code==0 and len(rows)==2 and {r['arm'] for r in rows}=={'0','1'} and all(
            len(r['samples'])==60 and all(math.isfinite(x) and x>0 for x in r['samples']) for r in rows)
        result=dict(case=case,status='PASS' if good else 'FAIL',process_rc=code,rows=rows,identity=identity,
                    scope='PREPARE_ONLY_NOT_GEMM',timing_idle_admission='EXTERNAL_LOAD_AUDIT_REQUIRED')
        target.write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        print(f'MOE_PREPARE_PERF completed={i+1}/{len(cases)} failed={sum(r["status"]!="PASS" for r in results)} elapsed_s={time.monotonic()-start:.1f}',flush=True)
    summary=dict(results=results,passed=sum(r['status']=='PASS' for r in results),expected=len(cases))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('MOE_PREPARE_PERF_DONE',summary['passed'],'/',len(cases),flush=True)
    return int(summary['passed']!=len(cases))


if __name__=='__main__':raise SystemExit(main())
