#!/usr/bin/env python3
"""Build from source and run the host/device regression suite on one PPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build import read_json, source_identity, write_json
from quactlize.runtime.compiler import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--baseline-source',type=Path,help='also compare grouped kernels against this source checkout')
    args=parser.parse_args()
    device=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    if not device or ',' in device:
        parser.error('set CUDA_VISIBLE_DEVICES to exactly one device on an idle PPU box')
    output=args.output.resolve()
    if output==ROOT or ROOT.is_relative_to(output):
        parser.error('output must be a dedicated results directory')
    output.mkdir(parents=True,exist_ok=args.resume)
    results=output/'results';results.mkdir(exist_ok=True)
    sdk=args.sdk.resolve(strict=True)
    identity=source_identity()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    summary=dict(source=source,source_identity=identity,sdk=str(sdk),visible_device=device,
                 phases=[],device_validated=False,performance_validated=False)
    write_json(results/'summary.json',summary)
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    env['PATH']=str(sdk/'bin')+':'+env.get('PATH','')
    env['PYTHONPATH']=str(ROOT)+':'+env.get('PYTHONPATH','')
    def run(name,command):
        started=time.monotonic()
        print(f'VALIDATE phase={name} status=START log={results/(name+".log")}',flush=True)
        with (results/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1)
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            rc=child.wait()
        summary['phases'].append(dict(phase=name,rc=rc,seconds=time.monotonic()-started))
        write_json(results/'summary.json',summary)
        if rc:raise subprocess.CalledProcessError(rc,command)
    try:
        run('host',[sys.executable,'-m','pytest','-q','-m','not device',
                    '--junitxml='+str(results/'host.xml')])
        run('build',[sys.executable,'tools/build.py','--sdk',str(sdk),
                     '--output',str(output/'build'),'--jobs',str(args.jobs),
                     *(['--resume'] if args.resume else [])])
        receipt=read_json(output/'build/build.json')
        runtime=output/'build'/receipt['runtime']
        summary['runtime']=str(runtime)
        summary['manifest_sha256']=sha(runtime/'manifest.json')
        write_json(results/'build.json',receipt)
        write_json(results/'runtime-manifest.json',read_json(runtime/'manifest.json'))
        env.update(QUACTLIZE_SDK=str(sdk),QUACTLIZE_BUNDLE=str(runtime),
                   QUACTLIZE_JIT_CACHE=str(output/'jit'))
        run('device',[sys.executable,'-m','pytest','--device','-v','-m','device',
                      '--junitxml='+str(results/'device.xml')])
        if source_identity()!=identity:
            raise ValueError('source changed during validation')
        summary['device_validated']=True
        if args.baseline_source:
            before=args.baseline_source.resolve(strict=True)
            summary['baseline_source']=subprocess.check_output(['git','-C',str(before),'rev-parse','HEAD'],text=True).strip()
            # Keep earlier comparisons when resuming; partial timing runs are
            # not stitched into fresh measurements.
            attempt=1
            while (results/f'grouped-{attempt}').exists():attempt+=1
            run('grouped-compare',[sys.executable,'tools/compare_grouped.py','--sdk',str(sdk),
                 '--bundle',str(runtime),'--cache',str(output/'jit'),'--before',str(before),
                 '--output',str(results/f'grouped-{attempt}'),'--jobs',str(args.jobs)])
            summary['grouped_regression_validated']=True
        if source_identity()!=identity:
            raise ValueError('source changed during regression comparison')
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        summary['error']=str(error)
    write_json(results/'summary.json',summary)
    passed=summary['device_validated'] and 'error' not in summary
    print('VALIDATE status='+('PASS' if passed else 'FAIL')+
          f' results={results} scope=KERNEL_REGRESSION_NOT_MODEL_OR_PERFORMANCE',flush=True)
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
