#!/usr/bin/env python3
"""Build per-call BF16 composition and measured SF expansion off-device."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import FLAGS,LIBRARIES,sha


def build(sdk,output,jobs,*,line_info=False):
    if jobs<1:raise ValueError('jobs must be positive')
    for name in ('bin/hgcc', *[f'lib/lib{x}.so' for x in LIBRARIES]):
        if not (sdk/name).is_file():raise ValueError('missing SDK input: '+name)
    sdk=sdk.resolve(strict=True);output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    include=[ROOT/'quactlize/include',ROOT/'third_party/actlize/include',ROOT/'quactlize/execution',ROOT/'quactlize/dequant']
    dirs=include+[ROOT/'quactlize/prefill']
    inputs={str(p.relative_to(ROOT)):sha(p) for d in dirs for p in d.rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cuh','.cu','.cpp','.inc')}
    sources=['quactlize/dequant/kernels.cu','quactlize/dequant/indexed.cu','quactlize/prefill/stages.cu',
             'quactlize/prefill/runtime.cpp','quactlize/dequant/dense.cu',
             'quactlize/prefill/dense_io.cu','quactlize/prefill/dense_runtime.cpp']
    env=dict(os.environ,PATH=str(sdk/'bin')+':'+os.environ.get('PATH',''),LD_LIBRARY_PATH=str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    commands=[];start=time.monotonic()
    def compile(source):
        obj=output/(Path(source).stem+'.o')
        command=([str(sdk/'bin/hgcc'),*FLAGS,*(['-lineinfo']if line_info else [])] if source.endswith('.cu') else
                 ['g++','-std=c++17','-O2','-fPIC','-pthread',f'-I{sdk}/include'])
        command += [*[f'-I{p}' for p in include],'-c',str(ROOT/source),'-o',str(obj)]
        commands.append(command)
        with (output/(Path(source).stem+'.log')).open('w') as log:
            subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('KPACK_PREFILL_BUILD compiled='+source,flush=True)
        return obj
    with ThreadPoolExecutor(max_workers=min(jobs,4)) as pool:objects=list(pool.map(compile,sources))
    lib=output/'libquactlize_ppu_prefill.so'
    link=['g++','-shared','-pthread','-Wl,-Bsymbolic',*map(str,objects),f'-L{sdk}/lib',*[f'-l{x}' for x in LIBRARIES],'-ldl','-o',str(lib)]
    subprocess.run(link,check=True,env=env);commands.append(link)
    if any(sha(ROOT/p)!=h for p,h in inputs.items()):raise ValueError('prefill sources changed during build')
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
    for name in ('query','prepare','run','destroy','error','device_status','provider_image'):
        if f'quactlize_kpack_prefill_{name}_v1' not in symbols:raise ValueError('missing prefill ABI')
    for name in ('query','prepare','run','device_status'):
        if f'quactlize_kpack_prefill_{name}_dynamic_v1' not in symbols:raise ValueError('missing dynamic prefill ABI')
    for name in ('prepare','run','destroy','error','workspace'):
        if f'quactlize_kpack_dense_full_{name}_v1' not in symbols:raise ValueError('missing dense full ABI')
    if 'quactlize_kpack_dense_helper_policy_v1' not in symbols:raise ValueError('missing dense helper policy')
    manifest=dict(schema='quactlize.prefill-runtime.v1',library=lib.name,sha256=sha(lib),source_hashes=inputs,
        runtime={f'lib{x}.so':sha(sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},compiler_sha256=sha(sdk/'bin/hgcc'),
        commands=commands,seconds=time.monotonic()-start,device_admission='PENDING_COMPOSED_RUNTIME')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('KPACK_PREFILL_BUILD COMPLETE '+str(lib),flush=True)


def verify(sdk,output):
    """Bind an existing payload to the exact sources/SDK, not its filename."""
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['schema']!='quactlize.prefill-runtime.v1' or sha(output/manifest['library'])!=manifest['sha256']:
        raise ValueError('prefill runtime identity differs')
    if sha(sdk/'bin/hgcc')!=manifest['compiler_sha256']:raise ValueError('prefill compiler differs')
    for name,want in manifest['source_hashes'].items():
        if sha(ROOT/name)!=want:raise ValueError('prefill source differs: '+name)
    if 'quactlize/prefill/dense_helpers.hpp' not in manifest['source_hashes']:
        raise ValueError('prefill runtime predates production helper policy')
    for name,want in manifest['runtime'].items():
        if sha(sdk/'lib'/name)!=want:raise ValueError('prefill SDK runtime differs: '+name)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();build(a.sdk,a.output,a.jobs)
