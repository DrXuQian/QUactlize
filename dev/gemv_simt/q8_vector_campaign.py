#!/usr/bin/env python3
"""Bounded Q8 follow-up; per-case processes, resumable hash-bound receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def cases():
    shapes=((512,2048),(1024,2048),(1024,5120),(2048,512),(2048,4096),(4096,2048),(5120,8192),(8192,2048))
    points=[(n,k,1,0) for n,k in shapes]
    points += [(n,k,m,0) for n,k in ((512,2048),(2048,512),(2048,4096)) for m in (2,4,8)]
    points += [(n,k,m,ch) for n,k,ch in ((1024,2048,1),(2048,512,8)) for m in (1,2,4,8)]
    return [dict(n=n,k=k,tokens=m,channels=ch,compute=compute) for n,k,m,ch in points for compute in (0,1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--l2-bytes',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    identity=dict(cases=cases(),library_sha256=manifest['library_sha256'],l2_bytes=a.l2_bytes,
                  runner_sha256=sha(Path(__file__)),harness_sha256=sha(ROOT/'dev/gemv_simt/q8_vector_run.py'),
                  access_sha256=sha(ROOT/'dev/gemv_simt/q8_vector_access.py'))
    receipt=a.output/'inputs.json'
    if receipt.exists() and json.loads(receipt.read_text())!=identity:raise ValueError('resume identity differs')
    if not receipt.exists():receipt.write_text(json.dumps(identity,indent=2)+'\n')
    completed=[];failed=[];started=time.monotonic()
    env=dict(os.environ);env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    for i,c in enumerate(cases()):
        key=f"n{c['n']}-k{c['k']}-m{c['tokens']}-ch{c['channels']}-c{c['compute']}"
        output=a.output/(key+'.json');log=a.output/(key+'.log')
        result=json.loads(output.read_text()) if a.resume and output.exists() else None
        if result and (result.get('library_sha256')!=identity['library_sha256'] or result.get('status')!='PASS'):
            raise ValueError('invalid retained cell: '+key)
        if result is None:
            command=[sys.executable,str(ROOT/'dev/gemv_simt/q8_vector_run.py'),'--bundle',str(a.bundle.resolve()),
                     '--sdk',str(a.sdk.resolve()),'--phase','perf','--output',str(output.resolve()),
                     '--l2-bytes',str(a.l2_bytes)]
            for k,v in c.items():command.extend(('--'+k,str(v)))
            with log.open('w') as stream:rc=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if rc:failed.append(dict(case=key,rc=rc,log=log.name))
            else:result=json.loads(output.read_text())
        if result:
            completed.append(dict(case=key,point=c,best=result['best'],delta_pct=result['delta_pct'],
                                  result_sha256=sha(output)))
        print(f'Q8_VECTOR_PROGRESS completed={i+1}/{len(cases())} failures={len(failed)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
        summary=dict(status='PASS' if not failed and i+1==len(cases()) else 'INCOMPLETE',
                     completed=completed,failed=failed,expected=len(cases()),
                     scope='CUDA_GUIDANCE_OR_PPU_READER_EXPERIMENT_NOT_PRODUCTION_ADMISSION')
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return int(bool(failed))


if __name__=='__main__':raise SystemExit(main())
