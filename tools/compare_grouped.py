#!/usr/bin/env python3
"""Same-config grouped regression comparison between two source checkouts.

Compilation is isolated per checkout; timed calls include device scheduling
and Split-K reduction, exclude SF preparation/JIT, and use warm weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def cases():
    result=[]
    for q in (8,10,11,12,13,14):
        for compute in (0,1):
            for rows,experts in ((33,3),(1024,8)):
                for split in (1,2):
                    route=('sf' if q==8 else 'fq')+'-grouped'
                    tm=8 if q==12 else 16
                    parent=dict(symbol=f'grouped_q{q}_tm{tm}',qtype=q,route=route,
                                tm=tm,tn=64,tk=256,wm=tm,wn=16,stages=2,ap=0,dn=16,
                                persistent=-1 if q==8 else 0)
                    result.append(dict(q=q,n=512,k=2048,m=rows,experts=experts,compute=compute,
                                       parent=parent,recipe=dict(algorithm=0,split=split,grid=0)))
    return result


def compile_child(args):
    # Import the requested checkout in a fresh process. Never mutate the
    # production compiler's root or allow one checkout's includes to leak.
    sys.path.insert(0,str(args.source.resolve(strict=True)))
    from quactlize.runtime.compiler import Compiler
    from quactlize.decode.grouped_compiler import GroupedComputeCompiler
    case=json.loads(args.case.read_text())
    compiler=(GroupedComputeCompiler(args.sdk,args.cache,compute_type='bf16')
              if case['compute'] else Compiler(args.sdk,args.cache))
    record=compiler.build(case['parent'])
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    return 0


def run_child(args):
    sys.path.insert(0,str(ROOT))
    from tools.scan_configs import Fixture, tc_runner
    from quactlize.runtime.native import SDK
    case=json.loads(args.case.read_text())
    sdk=SDK(args.sdk)
    fixture=Fixture(sdk,args.bundle,case['q'],case['n'],case['k'],case['m'],case['experts'],case['compute'])
    runners={};destroy=[];errors={};samples={'before':[],'after':[]}
    records={arm:json.loads((args.modules/f'{arm}.json').read_text()) for arm in samples}
    try:
        for arm,record in records.items():
            run,close,dtype=tc_runner(fixture,args.sdk,args.cache,case['parent'],case['recipe'],args.bundle,record)
            destroy.append(close);runners[arm]=run
            sdk.fill(fixture.output,0xff,case['m']*case['n']*4)
            run();errors[arm]=fixture.check(dtype)
            for _ in range(3):run()
            sdk.synchronize(None)
        for round_id in range(4):
            order=('before','after') if round_id in (0,3) else ('after','before')
            for _ in range(args.samples):
                for arm in order:
                    value=runners[arm].sample(64)
                    if not math.isfinite(value) or value<=0:raise ValueError('invalid event duration')
                    samples[arm].append(value)
        medians={arm:statistics.median(values) for arm,values in samples.items()}
        delta=(medians['after']/medians['before']-1)*100
        result=dict(status='PASS' if delta<=5 else 'REGRESSION_REVIEW',case=case,
                    relative_l2=errors,median_us=medians,delta_pct=delta,samples_us=samples,
                    modules={a:dict(path=r['path'],sha256=r['sha256'],key=r['key']) for a,r in records.items()},
                    cache='WARM',scope='COMPLETE_GROUPED_ENTRY_EXCLUDES_SF_PREPASS_JIT_NOT_MODEL',
                    repetitions_per_sample=64)
    finally:
        sdk.synchronize(None)
        for close in reversed(destroy):close()
        fixture.close()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    return int(result['status']!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--before',type=Path)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--samples',type=int,default=11)
    for name in ('source','case','modules'):p.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    p.add_argument('--child',choices=('compile','run'),help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.child=='compile':return compile_child(args)
    if args.child=='run':return run_child(args)
    if not args.before or args.jobs<1 or not 3<=args.samples<=101:p.error('--before, positive jobs, and 3..101 samples required')
    sys.path.insert(0,str(ROOT))
    from tools.verify_kpack_dispatch import verify
    verify(args.bundle,sdk=args.sdk)
    args.output.mkdir(parents=True,exist_ok=False)
    items=cases();builds={};commands=[]
    common=[sys.executable,str(Path(__file__).resolve()),'--sdk',str(args.sdk),
            '--bundle',str(args.bundle),'--cache',str(args.cache),'--samples',str(args.samples)]
    for i,case in enumerate(items):
        case_path=args.output/f'case-{i}.json';case_path.write_text(json.dumps(case)+'\n')
        key=(case['q'],case['compute'])
        if key in builds:continue
        directory=args.output/f'modules-q{key[0]}-c{key[1]}';directory.mkdir();builds[key]=directory
        for arm,root in (('before',args.before),('after',ROOT)):
            commands.append((f'q{key[0]}-c{key[1]}-{arm}',common+['--child','compile','--case',str(case_path),
                             '--source',str(root),'--output',str(directory/f'{arm}.json')]))
    def compile(item):
        name,command=item
        with (args.output/(name+'.log')).open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        print(f'COMPARE compile={name} rc={rc}',flush=True)
        return rc
    with ThreadPoolExecutor(max_workers=min(args.jobs,len(commands))) as pool:
        failed=any(list(pool.map(compile,commands)))
    records=[]
    if not failed:
        for i,case in enumerate(items):
            output=args.output/f'result-{i}.json'
            command=common+['--child','run','--case',str(args.output/f'case-{i}.json'),
                            '--modules',str(builds[(case['q'],case['compute'])]),'--output',str(output)]
            with (args.output/f'run-{i}.log').open('w') as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            result=json.loads(output.read_text()) if output.is_file() else dict(status='FAIL',case=case,rc=rc)
            records.append(result)
            (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            print(f"COMPARE completed={i+1}/{len(items)} status={result['status']} delta_pct={result.get('delta_pct')}",flush=True)
    summary=dict(status='PASS' if len(records)==len(items) and all(r['status']=='PASS' for r in records) else 'REVIEW',
                 expected=len(items),completed=len(records),compile_failed=failed,
                 before=str(args.before.resolve()),after=str(ROOT),scope='REPRESENTATIVE_GROUPED_NOT_MODEL')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return int(summary['status']!='PASS')


if __name__=='__main__':
    raise SystemExit(main())
