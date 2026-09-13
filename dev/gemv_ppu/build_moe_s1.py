#!/usr/bin/env python3
"""Build six small, explicit Q4 S1 modules; no inference policy mutation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import moe_s1 as spec, q4_s1_port
from dev.gemv_cuda.build import digest
from quactlize.runtime.compiler import FLAGS,LIBRARIES


def code_lowering(ops,body):
    # hgcc may lower the same ((word>>s)&0x000f000f)|0x64006400
    # expression as LOP3 or as AND+OR. Both still perform packed half2
    # conversion. Do not confuse the instruction spelling with its semantics.
    if not any('f16x2' in op for op in ops): return None
    if ops['v.lop3.b32']: return 'LOP3_HALF2'
    if ops['v.and.b32'] and ops['v.or.b32'] and '0xf000f' in body and '0x64006400' in body:
        return 'AND_OR_HALF2'
    return None


def build(a):
    sdk=a.sdk.resolve(strict=True)
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    if (ROOT/'quactlize/execution/q4_s1_helpers.cuh').read_text().rstrip()!=q4_s1_port.helpers().rstrip():
        raise ValueError('helper transplant differs from frozen reader')
    if (ROOT/'quactlize/execution/q4_s1_readers.cuh').read_text().rstrip()!=q4_s1_port.bodies().rstrip():
        raise ValueError('reader transplant differs from frozen reader')
    sources=[*ROOT.glob('quactlize/execution/q4_s1*'),
             ROOT/'quactlize/execution/api.h',ROOT/'quactlize/execution/validation.hpp',
             ROOT/'quactlize/include/ppu_placed_arrangement.hpp',ROOT/'quactlize/include/q8_kpack2.hpp',
             ROOT/'quactlize/include/quactlize_ppu_config.h',ROOT/'quactlize/include/ppu_format_config.inc',
             ROOT/'dev/gemv_ppu/moe_s1.py',ROOT/'dev/gemv_ppu/build_moe_s1.py',
             ROOT/'dev/gemv_ppu/probe.cu',ROOT/'quactlize/runtime/compiler.py']
    # Follow project-local quoted includes to bind every actual header without
    # pinning unrelated development sources or CUDA compatibility code.
    import re
    pending=list(sources)
    seen=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen: continue
        seen.add(path)
        for header in re.findall(r'^\s*#\s*include\s*"([^"]+)"',path.read_text(),re.M):
            matches=[p for p in (path.parent/header,ROOT/'quactlize/include'/header,
                ROOT/'quactlize/execution'/header,ROOT/'third_party/actlize/include'/header) if p.is_file()]
            if matches: pending.append(matches[0])
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(seen)}
    env=dict(os.environ)
    env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    include=[ROOT/'quactlize/execution',ROOT/'quactlize/include',ROOT/'third_party/actlize/include']
    commands=[]

    def run(label,argv):
        argv=list(map(str,argv)); commands.append(dict(label=label,argv=argv))
        with (out/(label+'.log')).open('w') as f:
            rc=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        if rc: raise ValueError(f'{label} rc={rc}; see {out/(label+".log")}')
        print(f'Q4_MOE_S1_BUILD phase={label} elapsed_s={time.monotonic()-started:.1f}',flush=True)

    run('probe',[sdk/'bin/hgcc',*FLAGS,'-c',ROOT/'dev/gemv_ppu/probe.cu','-o',out/'probe.o'])

    def one(shape):
        n,k=shape; label=f'n{n}_k{k}'; cu=out/(label+'.cu')
        cu.write_text(spec.source(n,k))
        run(label,[sdk/'bin/hgcc',*FLAGS,*[f'-I{p}' for p in include],'-c',cu,'-o',out/(label+'.o')])
        library=out/spec.payload(n,k)
        run('link-'+label,['g++','-shared','-Wl,-Bsymbolic',out/(label+'.o'),out/'probe.o',
                          '-o',library,f'-L{sdk/"lib"}',*[f'-l{x}' for x in LIBRARIES]])
        run('isa-'+label,[sdk/'bin/hgobjdump','--dump-isa',library])
        isa=(out/('isa-'+label+'.log')).read_text()
        # Two input types for every recipe, plus the marker. This also rejects
        # empty/host-only images before box execution.
        import re
        symbols=re.findall(r'Disassembly of section \.text\.kernel\.([^\n]+):',isa)
        kernels=[s for s in symbols if 'indexed_kernel' in s]
        if len(set(kernels))!=2*len(spec.inventory()): raise ValueError('indexed native specialization denominator differs')
        if 'q4_ppu_marker' not in isa or 'v.lop3.b32' not in isa or 'v.fma.f32' not in isa:
            raise ValueError('native marker/fast dequant/FP32 dot missing')
        from collections import Counter
        blocks=list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):',isa))
        stats={}
        for i,block in enumerate(blocks):
            symbol=block[1]
            if 'indexed_kernel' not in symbol: continue
            args=tuple(map(int,re.findall(r'(?:I|E)Li(\d+)',symbol)))
            if len(args)!=7 or args[-2:]!=(n,k): raise ValueError('native recipe identity differs')
            dtype,reader,variant,warps,values,_,_=args
            recipe=spec.Recipe(reader,variant,warps,values)
            if recipe not in spec.inventory() or dtype not in (0,1): raise ValueError('unexpected native recipe')
            body=isa[block.end():blocks[i+1].start() if i+1<len(blocks) else None]
            ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
            lowering=code_lowering(ops,body)
            if not lowering or not any(op.startswith('v.fma.f32') for op in ops):
                raise ValueError('per-recipe fast dequant or FP32 dot is missing')
            stats[f'{recipe.key}:a{dtype}']=dict(symbol=symbol,code_lowering=lowering,
                operations={op:ct for op,ct in sorted(ops.items()) if op.startswith(('vmem.','tsm.','s.cbr','s.blksyn')) or
                            any(s in op for s in ('shuffle','f16x2','fma.f32','lop3')) or op in ('v.and.b32','v.or.b32')},
                scope='STATIC_NATIVE_ISA_NOT_TIMING_OR_DYNAMIC_COUNTS')
        if len(stats)!=32: raise ValueError('native input/recipe denominator differs')
        return library.name,digest(library),dict(symbols=kernels,recipes=stats,isa_sha256=digest(out/('isa-'+label+'.log')))

    with ThreadPoolExecutor(max_workers=min(a.jobs,len(spec.SHAPES))) as pool:
        results=list(pool.map(one,spec.SHAPES))
    if any(digest(ROOT/name)!=sha for name,sha in hashes.items()): raise ValueError('source changed during build')
    m=dict(schema=spec.SCHEMA,plan=spec.plan(),payloads={p:h for p,h,_ in results},
        source_hashes=hashes,compiler_sha256=digest(sdk/'bin/hgcc'),inspector_sha256=digest(sdk/'bin/hgobjdump'),
        runtime={f'lib{lib}.so':digest(sdk/'lib'/f'lib{lib}.so') for lib in LIBRARIES},
        native={p:s for p,_,s in results},commands=sorted(commands,key=lambda x:x['label']),
        seconds=time.monotonic()-started,production_changed=False,device_validated=False)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    spec.verify(out)
    print(f'Q4_MOE_S1_BUILD COMPILED shapes=6 recipes=16 input_types=2 seconds={m["seconds"]:.1f} device_validated=0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args()
    if a.jobs<1: p.error('jobs must be positive')
    build(a)
