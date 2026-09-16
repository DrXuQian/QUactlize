#!/usr/bin/env python3
"""Build a bounded Q8 topology experiment without changing shipping headers."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.q8_topology import SCHEMA,POINTS,CONFIGS,inventory,source
from dev.gemv_simt.q8_vector_build import link_command
from quactlize.runtime.compiler import sha


def build(args):
    sdk=args.sdk.resolve(strict=True);out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    includes=[ROOT,ROOT/'quactlize/include',ROOT/'quactlize/execution',
              ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    sources={str(p.relative_to(ROOT)):sha(p) for inc in includes[1:]
             for p in inc.rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cuh','.inc')}
    for name in ('dev/gemv_simt/q8_topology.py','dev/gemv_simt/build_q8_topology.py',
                 'dev/gemv_simt/model_followup.py','dev/gemv_simt/q8_vector_build.py',
                 'dev/gemv_simt/probe.cu','quactlize/execution/simt_codegen.py',
                 'quactlize/runtime/compiler.py'):
        sources[name]=sha(ROOT/name)
    env=dict(os.environ);started=time.monotonic()
    if args.platform=='cuda':
        compiler=sdk/'bin/nvcc'
        flags=['-std=c++17','-O3','-lineinfo','-arch=sm_120','--expt-relaxed-constexpr',
               '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1','-Xptxas=-v',
               '-Xcompiler=-fPIC','-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h')]
        includes.insert(0,ROOT/'dev/gemv_cuda/compat')
        for p in includes[0].iterdir():
            if p.is_file():sources[str(p.relative_to(ROOT))]=sha(p)
    else:
        from quactlize.runtime.compiler import FLAGS
        compiler=sdk/'bin/hgcc';flags=list(FLAGS)
        env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
        env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    (out/'q8.cu').write_text(source())
    probe=(ROOT/'dev/gemv_simt/probe.cu').read_text()
    if args.platform=='ppu':probe=re.sub(r'\bcuda(?=[A-Z_])','hggc',probe)
    (out/'probe.cu').write_text(probe)
    commands=[]
    with (out/'build.log').open('x') as log:
        for name in ('q8','probe'):
            command=[str(compiler),*flags,*[f'-I{x}' for x in includes],'-c',str(out/(name+'.cu')),'-o',str(out/(name+'.o'))]
            commands.append(command)
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands.append(link_command(args.platform,sdk,compiler,[out/'q8.o',out/'probe.o'],out/'q8.so'))
        subprocess.run(commands[-1],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.platform=='ppu':
        for option,name in (('--dump-resource-usage=all','resources.txt'),('--dump-isa','isa.txt')):
            with (out/name).open('x') as log:
                subprocess.run([sdk/'bin/hgobjdump',option,out/'q8.so'],env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(sha(ROOT/p)!=value for p,value in sources.items()):raise ValueError('source changed during compile')
    result=dict(schema=SCHEMA,platform=args.platform,library='q8.so',library_sha256=sha(out/'q8.so'),
        source_hashes=sources,source_sha256=sha(out/'q8.cu'),compiler_sha256=sha(compiler),commands=commands,
        configs=[asdict(c) for c in CONFIGS],
        points=[dict(n=n,k=k,incumbent=asdict(c),**inventory(n,k)) for n,k,c in POINTS],
        shipping_execution_sha256=sha(args.shipping/'libquactlize_ppu_execution.so') if args.shipping else None,
        build_seconds=time.monotonic()-started,device_admission='PENDING',production_selection_changed=False,
        scope='Q8_M1_F32_ENDPOINTS_F16_ACTIVATION_ROUNDING_F32_ACCUMULATION_FULL_CALL')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Q8_TOPOLOGY_BUILD PASS kernels={len(CONFIGS)} seconds={result["build_seconds"]:.1f} output={out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=('ppu','cuda'),required=True)
    for name in ('sdk','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shipping',type=Path)
    args=parser.parse_args()
    if args.platform=='ppu' and not args.shipping:parser.error('PPU build needs the immutable shipping control')
    build(args)
