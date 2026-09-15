#!/usr/bin/env python3
"""Matched Q8-only reader experiment; no change to the shipping selector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quactlize.execution.simt_codegen import inventory
from dev.gemv_simt.build import sha


def link_command(platform, sdk, compiler, objects, output):
    if platform == 'cuda':
        return [str(compiler), '-shared', '--cudart=shared', '-Xlinker=-Bsymbolic',
                *map(str, objects), '-o', str(output)]
    from quactlize.runtime.compiler import LIBRARIES
    # --as-needed drops runtime libraries seen before their referencing objects.
    # Resolve all host symbols at link time, not through an accidental preload.
    return ['g++', '-shared', '-Wl,-Bsymbolic', '-Wl,-z,defs', *map(str, objects),
            f'-L{sdk}/lib', *[f'-l{name}' for name in LIBRARIES], '-o', str(output)]


def source():
    text = '''#include "dev/gemv_simt/q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
using namespace quactlize::execution::simt;
template<int Arm,int Compute,int V,int C,int W,int P>
int invoke(qkg_call_v1 c,int split) {
 auto stream=static_cast<hggcStream_t>(c.stream);
 if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
 int blocks=c.rows*split*(c.n/(C*P));
 if constexpr(Arm==0) register_reuse<8,1,V,C,W,P,Compute><<<blocks,W*32,0,stream>>>(c,split);
 else q8_vector::kernel<1,Compute,V,C,W,P><<<blocks,W*32,0,stream>>>(c,split);
 if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
 if(split>1) register_reuse_reduce<8><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,split);
 return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
extern "C" int q8_vector_run(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f,int arm) {
 if(!d || !f || d->call.qtype!=8 || d->call.input_type!=QKG_F32 || arm<0 || arm>1) return QKG_INVALID;
 auto a=q8_kpack2::arrangement();qkg_sizes_v1 sizes{};
 int rc=query_v2(*d,*f,&a,sizes);if(rc) return rc;
 rc=buffers_v2(*d,sizes);if(rc) return rc;
'''
    for c in inventory(8):
        text += f' if(f->variant=={c.variant} && f->columns=={c.columns} && f->warps=={c.warps} && f->values=={c.values}) {{\n'
        for arm in range(2):
            for compute in range(2):
                text += f'  if(arm=={arm} && d->compute_type=={compute}) return invoke<{arm},{compute},{c.variant},{c.columns},{c.warps},{c.values}>(d->call,f->split);\n'
        text += ' }\n'
    return text + ' return QKG_INVALID;\n}\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=('cuda','ppu'), required=True)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arch', default='sm_120')
    a=p.parse_args();sdk=a.sdk.resolve(strict=True);out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    inc=[ROOT, ROOT/'quactlize/include', ROOT/'quactlize/execution',
         ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    env=dict(os.environ);start=time.monotonic()
    if a.platform=='cuda':
        compiler=sdk/'bin/nvcc'
        flags=['-std=c++17','-O3','-lineinfo','-arch='+a.arch,'--expt-relaxed-constexpr',
               '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1','-Xptxas=-v',
               '-Xcompiler=-fPIC','-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h')]
        inc.insert(0,ROOT/'dev/gemv_cuda/compat')
    else:
        from quactlize.runtime.compiler import FLAGS
        compiler=sdk/'bin/hgcc';flags=list(FLAGS)
        env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
        env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    src=out/'q8.cu';src.write_text(source())
    paths={p for directory in inc if directory!=ROOT for p in directory.rglob('*')
           if p.is_file() and p.suffix in ('.h','.hpp','.cuh','.inc')}
    paths.update((ROOT/'dev/gemv_simt'/n) for n in ('q8_vector.cuh','q8_vector_build.py','q8_vector_run.py','q8_vector_access.py','probe.cu'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    command=[str(compiler),*flags,*[f'-I{x}' for x in inc],'-c',str(src),'-o',str(out/'q8.o')]
    commands=[command]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        probe=(ROOT/'dev/gemv_simt/probe.cu').read_text()
        if a.platform=='ppu':
            import re
            probe=re.sub(r'\bcuda(?=[A-Z_])','hggc',probe)
        (out/'probe.cu').write_text(probe)
        commands.append([str(compiler),*flags,*[f'-I{x}' for x in inc],'-c',str(out/'probe.cu'),'-o',str(out/'probe.o')])
        subprocess.run(commands[-1],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands.append(link_command(a.platform, sdk, compiler,
                                     [out/'q8.o', out/'probe.o'], out/'q8.so'))
        subprocess.run(commands[-1],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(sha(ROOT/p)!=v for p,v in hashes.items()): raise ValueError('source changed during compile')
    result=dict(schema='quactlize.q8-vector.v1',platform=a.platform,library='q8.so',library_sha256=sha(out/'q8.so'),
                source_hashes=hashes,commands=commands,compiler_sha256=sha(compiler),configs=[c.record() for c in inventory(8)],
                build_seconds=time.monotonic()-start,device_validated=False,production_selection_changed=False,
                scope='SAME_KPACK2_F32_IO_F16_OR_BF16_COMPUTE_COMPLETE_CALL')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Q8_VECTOR_BUILD PASS',out,flush=True)


if __name__=='__main__': main()
