#!/usr/bin/env python3
"""Rebuild the exact admitted Q4 SIMT recipe closure as an isolated control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from quactlize.execution.q4_decode_codegen import recipes
from dev.gemv_simt.build import sha


def source(n,k,configs):
    s='#include "quactlize/execution/q4_decode_kernel.cuh"\n'
    s+='extern "C" int simt_q4_control(qkg_call_v1 const* c, int index) {\n'
    s+=f'    if (!c || c->qtype!=12 || c->n!={n} || c->k!={k}) return QKG_SHAPE;\n'
    for index,(r,v,w,p,columns) in enumerate(configs):
        s+=f'    if (index=={index}) return quactlize::execution::q4_decode::launch<{r},{v},{w},{p},{columns},{n},{k}>(*c);\n'
    return s+'    return QKG_INVALID;\n}\n'


def build(a):
    a.output.mkdir(parents=True,exist_ok=False)
    plan=recipes()
    includes=[ROOT,ROOT/'quactlize/execution',ROOT/'quactlize/include',ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    if a.platform=='cuda':
        compiler=a.sdk/'bin/nvcc'
        flags=['-std=c++17','-O3','-arch=sm_120','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-Xptxas=-v',
               '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1',
               '-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h'),f'-I{ROOT}/dev/gemv_cuda/compat']
        link=['--shared','--cudart=shared','-Xlinker=-Bsymbolic']
    else:
        from quactlize.runtime.compiler import FLAGS,LIBRARIES
        compiler=a.sdk/'bin/hgcc';flags=list(FLAGS)
        link=['--shared',f'-L{a.sdk}/lib',*[f'-l{x}' for x in LIBRARIES]]
    def one(item):
        (n,k),configs=item
        file=a.output/f'q4-{n}-{k}.cu';file.write_text(source(n,k,configs))
        output=file.with_suffix('.so')
        command=[str(compiler),*flags,*[f'-I{p}' for p in includes],str(file),*link,'-o',str(output)]
        with file.with_suffix('.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'SIMT_Q4_CONTROL n={n} k={k} recipes={len(configs)}',flush=True)
        return dict(n=n,k=k,configs=configs,library=output.name,sha256=sha(output),command=command)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:records=list(pool.map(one,plan.items()))
    (a.output/'manifest.json').write_text(json.dumps(dict(platform=a.platform,
        policy_sha256=sha(ROOT/'policies/kpack_q4_decode_v1.json'),controls=records,
        scope='EXACT_Q4_PRODUCTION_READER_RECIPES_NOT_TARGET_GPU_OPTIMALITY'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',choices=('ppu','cuda'),required=True)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args()
    if a.jobs<1:p.error('jobs must be positive')
    build(a)
