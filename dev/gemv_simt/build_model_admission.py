#!/usr/bin/env python3
"""CUDA numeric-only adapter for three measured production decode recipes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha

RECIPES={8:((1,4,8,4),(5,8,4,4)),12:((3,4,4,4),),13:((3,4,2,8),)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source='#include "quactlize/execution/simt_q8_vector.cuh"\n'
    source+='#include "quactlize/execution/simt.cpp"\n'
    for q in (8,10,11,12,13,14):
        conditions=[f'f->variant=={v} && f->columns=={c} && f->warps=={w} && f->values=={n}'
                    for v,c,w,n in RECIPES.get(q,())]
        source+=f'extern "C" bool qkg_simt_supported_{q}(qkg_simt_config_v1 const* f) {{ return '+(
            ' || '.join('('+x+')' for x in conditions) or 'false')+'; }\n'
        source+=f'extern "C" int qkg_simt_launch_v2_{q}(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f) {{\n'
        for (v,c,w,n),condition in zip(RECIPES.get(q,()),conditions):
            reader='simt::q8_vector' if v>=4 else 'simt'
            source+=f'  if({condition}) return quactlize::execution::{reader}::launch_v2<{q},{v},{c},{w},{n}>(*d,f->split);\n'
        source+='  return QKG_INVALID; }\n'
        source+=f'extern "C" int qkg_simt_launch_{q}(qkg_call_v1 const* c,qkg_simt_config_v1 const* f) {{\n'
        source+=f'  qkg_simt_call_v2 d{{2,sizeof(d),*c,QKG_COMPUTE_F16}};return qkg_simt_launch_v2_{q}(&d,f); }}\n'
    unit=out/'adapter.cu';unit.write_text(source)
    includes=[ROOT/'dev/gemv_cuda/compat',ROOT,ROOT/'quactlize/include',
              ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    command=[str(a.sdk/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_120','--expt-relaxed-constexpr',
        '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1','-Xcompiler=-fPIC','-shared',
        '--cudart=shared','-Xlinker=-Bsymbolic','-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h'),
        *['-I'+str(x) for x in includes],str(unit),'-o',str(out/'simt.so')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'manifest.json').write_text(json.dumps(dict(command=command,sha256=sha(out/'simt.so'),
        scope='CUDA_NUMERIC_ONLY_PRODUCTION_LAUNCH_BODY',sources={str(p.relative_to(ROOT)):sha(p)
            for p in (ROOT/'quactlize/execution').glob('*') if p.is_file()}),indent=2)+'\n')
    print('MODEL_DECODE_ADAPTER BUILT',out/'simt.so',flush=True)


if __name__=='__main__':main()
