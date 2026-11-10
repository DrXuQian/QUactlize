#!/usr/bin/env python3
"""Build and run a bounded ZW890/PPU1.5 correctness probe; no production changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import FLAGS, LIBRARIES

EXPECTED={'mma':44,'delivery':8,'pack':12,'simt':128}
FIELDS=('tools/run_zw890_bringup.py','tools/ppu15/check.py','tools/ppu15/device.py','tools/ppu15/native.cu',
        'tools/ppu15/simt.cu','quactlize/packing/ppu_pack.cu','quactlize/packing/sizes.cpp')

def plan():
    return dict(target='ppu_15',device='ZW890',expected=EXPECTED,
        scope='NATIVE_M16_AND_TRANSPORT_PACK_SIMT_CORRECTNESS_NOT_PRODUCTION_TC_OR_PERFORMANCE',
        production_changed=False,performance_admitted=False,
        unsupported=['PRODUCTION_FQ_SF_TC_NOT_PORTED','PPU10_M8_AND_PACKED_A',
                     'FULL_LIBRARY_SELECTION_AND_MODEL_INTEGRATION'])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(argv,log,env):
    with log.open('w') as stream:
        result=subprocess.run(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        raise RuntimeError(f'rc={result.returncode}; see {log}\n'+
            '\n'.join(log.read_text(errors='replace').splitlines()[-45:]))

def identity():
    def git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    # Record modifications; do not refuse a useful measurement because of them.
    return dict(commit=git('rev-parse','HEAD'),
                diff_sha256=hashlib.sha256(git('diff','HEAD','--').encode()).hexdigest(),
                sources={name:sha(ROOT/name) for name in FIELDS},
                actlize=subprocess.check_output(['git','-C',str(ROOT/'third_party/actlize'),
                    'rev-parse','HEAD'],text=True).strip())

def build(sdk,output,jobs,env):
    work=output/'build';work.mkdir()
    vendor=ROOT/'third_party/actlize'
    expected=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',
        'HEAD:third_party/actlize'],text=True).strip()
    actual=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    if actual!=expected or not (vendor/'include/cute/arch/mma_ppu0015.hpp').is_file():
        raise ValueError(f'initialize pinned actlize {expected}; this runner does not fetch submodules')
    flags=[x for x in FLAGS if not x.startswith('-arch=')]+['-arch=ppu_15']
    includes=[ROOT/'quactlize/include',ROOT/'quactlize/execution',ROOT/'quactlize/fusion',
              vendor/'include',vendor/'tools/util/include']
    sources=[ROOT/name for name in ('tools/ppu15/native.cu','tools/ppu15/simt.cu',
                                   'quactlize/packing/ppu_pack.cu','quactlize/packing/sizes.cpp')]
    def compile_one(source):
        obj=work/(source.stem+'.o')
        argv=[sdk/'bin/hgcc',*flags,*['-I'+str(p) for p in includes],'-c',source,'-o',obj]
        print('ZW890_BUILD_START '+source.name,flush=True)
        command(argv,work/(source.stem+'.log'),env)
        print('ZW890_BUILD_PASS '+source.name,flush=True)
        return dict(source=str(source.relative_to(ROOT)),command=list(map(str,argv)),
                    source_sha256=sha(source),object_sha256=sha(obj),object=str(obj))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        units=list(pool.map(compile_one,sources))
    library=work/'libqz15_probe.so'
    argv=['g++','-shared','-Wl,-Bsymbolic',*[u['object'] for u in units],'-o',library,
          '-L'+str(sdk/'lib'),*['-l'+name for name in LIBRARIES]]
    command(argv,work/'link.log',env)
    receipt=dict(target='ppu_15',units=units,link=list(map(str,argv)),library_sha256=sha(library),
        compiler_sha256=sha(sdk/'bin/hgcc'),compiler_version=subprocess.check_output(
            [str(sdk/'bin/hgcc'),'--version'],env=env,text=True))
    (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

def verdict(suite,rows,rc):
    if suite not in EXPECTED:raise ValueError('unknown suite')
    keys=[json.dumps({k:v for k,v in r.items() if k not in ('max_absdot_error','status')},sort_keys=True) for r in rows]
    passed=sum(r.get('suite')==suite and r.get('status')=='PASS' for r in rows)
    return dict(status='PASS' if rc==0 and len(set(keys))==len(rows)==passed==EXPECTED[suite] else 'FAIL',
                expected=EXPECTED[suite],passed=passed,runner_rc=rc)

def review(output):
    path=output.with_name(output.name+'.review.tgz')
    with tarfile.open(path,'x:gz') as archive:
        # Numeric receipts and errors only; exclude binaries/objects and SDK.
        for item in sorted(output.rglob('*')):
            if item.is_file() and not item.is_symlink() and item.suffix in ('.json','.jsonl','.log'):
                archive.add(item,arcname=str(item.relative_to(output)),recursive=False)
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    parser.add_argument('--device',type=int,default=int(os.environ.get('DEVICE','0')))
    parser.add_argument('--jobs',type=int,default=int(os.environ.get('JOBS','4')))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--build-only',action='store_true')
    args=parser.parse_args()
    if args.plan_only:
        print('ZW890_PLAN '+json.dumps(plan()),flush=True);return 0
    if args.jobs<1 or args.device<0:parser.error('jobs must be positive; device must be nonnegative')
    sdk=args.sdk.resolve()
    if not (sdk/'bin/hgcc').is_file():parser.error('PPU_SDK/--sdk must point at an SDK with bin/hgcc')
    if args.output:
        output=args.output.resolve()
        try:output.mkdir(parents=True,exist_ok=False)
        except FileExistsError:parser.error(f'--output must be a new directory: {output}')
    else:output=Path(tempfile.mkdtemp(prefix='qz-zw890-')).resolve()
    env=dict(os.environ,PPU_SDK=str(sdk),CUDA_VISIBLE_DEVICES=str(args.device),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    summary=dict(plan(),status='FAIL',sdk=str(sdk),requested_device=args.device,suites={})
    started=time.monotonic();stage='identity'
    print(f'ZW890_START output={output} device={args.device} target=ppu_15',flush=True)
    try:
        summary['source']=identity()
        (output/'plan.json').write_text(json.dumps(plan(),indent=2)+'\n')
        if not args.build_only:
            stage='dependencies'
            command([sys.executable,'-c','import numpy, torch, gguf'],output/'dependencies.log',env)
        stage='build';build(sdk,output,args.jobs,env)
        if args.build_only:
            summary['status']='BUILD_ONLY';return 0
        check=ROOT/'tools/ppu15/check.py'
        stage='device'
        command([sys.executable,check,'--sdk',sdk,'--output',output,'--suite','device'],output/'device.log',env)
        summary['device']=json.loads((output/'device.json').read_text())
        print('ZW890_DEVICE '+json.dumps(summary['device']),flush=True)
        for suite in EXPECTED:
            stage=suite;log=output/(suite+'.log')
            print(f'ZW890_SUITE_START suite={suite} expected={EXPECTED[suite]} log={log}',flush=True)
            argv=[sys.executable,str(check),'--sdk',str(sdk),'--output',str(output),'--suite',suite]
            with log.open('w') as stream:
                rc=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            path=output/(suite+'.jsonl')
            rows=[json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
            summary['suites'][suite]=verdict(suite,rows,rc)
            print('ZW890_SUITE_DONE '+json.dumps(dict(suite=suite,**summary['suites'][suite])),flush=True)
            if summary['suites'][suite]['status']!='PASS':
                print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]),flush=True)
        summary['status']='PASS' if all(v['status']=='PASS' for v in summary['suites'].values()) else 'FAIL'
        return 0 if summary['status']=='PASS' else 1
    except Exception as exc:
        summary.update(stage=stage,error=str(exc))
        print(f'ZW890_FAIL stage={stage} error={exc}',flush=True)
        return 1
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        path=review(output)
        print('ZW890_DONE '+json.dumps(dict(status=summary['status'],results=str(path),
            production_changed=False,performance_admitted=False)),flush=True)

if __name__=='__main__':raise SystemExit(main())
