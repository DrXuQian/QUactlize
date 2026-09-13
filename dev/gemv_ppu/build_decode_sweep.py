#!/usr/bin/env python3
"""Compile only the incremental Q4 decode closure; box never runs a compiler."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import decode_sweep as spec
from dev.gemv_ppu.build_moe_s1 import code_lowering
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES, sha
from tools.build_kpack_dispatch import plan as select_current


def build(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sdk=a.sdk.resolve(strict=True);start=time.monotonic()
    parents,families,runtime=spec.historical_tc()
    requests=sorted({spec.request(w) for w in spec.workloads()})
    selected,selections=select_current(output,requests)
    for parent in selected:parents[parent['symbol']]=parent
    for s in selections:
        if s['status']!='SELECTED':continue  # Visible policy miss; measured anchors still exist.
        _,route,_,n,k,_,_=s['request'];f=f'{"dense" if route==0 else "grouped"}:{n}x{k}'
        families[f]=sorted(set(families[f])|{s['parent']})
        runtime[f]=sorted(set(map(tuple,runtime[f]))|{(s['parent'],s['split'],s['grid_b'],s['grid_mode'])})
    if any(not names for names in families.values()):raise ValueError('missing TC family candidate pool')
    compiler=Compiler(sdk,a.cache,a.jobs)
    sources={str(p.relative_to(ROOT)):sha(p) for p in compiler.input_stats if p.is_relative_to(ROOT)}
    for path in list((ROOT/'quactlize/execution').glob('*.h*'))+[
        ROOT/'dev/gemv_ppu/decode_sweep.py',ROOT/'dev/gemv_ppu/decode_kernel.cuh',
        ROOT/'dev/gemv_ppu/decode_io.cu',ROOT/'dev/gemv_ppu/build_decode_sweep.py',
        ROOT/'dev/gemv_ppu/probe.cu',ROOT/'dev/gemv_ppu/build_moe_s1.py',
        ROOT/'benchmarks/workloads.py',ROOT/'benchmarks/moe_router_fixture.hpp',
        ROOT/'tools/kpack_native_policy.cpp',ROOT/'tools/build_kpack_dispatch.py',
        ROOT/'policies/kpack_zw810_heuristic_v1.hpp',spec.DENSE_REVIEW,spec.MOE_REVIEW,
        spec.moe_compare.TACTICS,ROOT/'prebuilt/ppu0010/q4-smallm-v1/manifest.json',
        spec.moe_compare.TC_BUNDLE/'manifest.json',spec.smallm.REVIEW]:
        sources[str(path.relative_to(ROOT))]=sha(path)
    os.environ['PATH']=str(sdk/'bin')+os.pathsep+os.environ.get('PATH','')
    os.environ['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    commands=[]
    include=[ROOT/'dev/gemv_ppu',ROOT/'quactlize/execution',ROOT/'quactlize/include',ROOT/'third_party/actlize/include']
    def run(label,argv):
        argv=list(map(str,argv));commands.append(dict(label=label,argv=argv))
        with (output/(label+'.log')).open('w') as f:
            subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(f'Q4_DECODE_BUILD phase={label} elapsed_s={time.monotonic()-start:.1f}',flush=True)
    def compile_(label,source):
        obj=output/(label+'.o')
        run(label,[sdk/'bin/hgcc',*FLAGS,*[f'-I{p}' for p in include],'-c',source,'-o',obj])
        return obj
    def link(label,objects,target):
        run(label,['g++','-shared','-Wl,-Bsymbolic',*objects,'-o',target,f'-L{sdk}/lib',*[f'-l{x}' for x in LIBRARIES]])
    def one(shape):
        n,k=shape;label=f'n{n}_k{k}';source=output/(label+'.cu');source.write_text(spec.simt_source(n,k))
        obj=compile_(label,source);so=output/f'libq4_decode_{label}.so';link('link-'+label,[obj,probe],so)
        run('isa-'+label,[sdk/'bin/hgobjdump','--dump-isa',so])
        assembly=(output/('isa-'+label+'.log')).read_text()
        sections=list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):',assembly))
        stats={}
        for i,match in enumerate(sections):
            symbol=match[1]
            if 'q4_decode_sweep' not in symbol:continue
            args=tuple(map(int,re.findall(r'(?:I|E)Li(\d+)',symbol)))
            if len(args)!=8 or args[-2:]!=(n,k):raise ValueError('SIMT symbol arguments differ')
            inp,reader,variant,warps,p,cols,_,_=args;r=spec.Simt(reader,variant,warps,p,cols)
            if r not in spec.simt_inventory(n,k) or inp not in (0,1):raise ValueError('foreign SIMT recipe')
            body=assembly[match.end():sections[i+1].start() if i+1<len(sections) else None]
            ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body));lowering=code_lowering(ops,body)
            if not lowering or not any(op.startswith('v.fma.f32') for op in ops):raise ValueError('fast dequant/FP32 dot missing')
            stats[f'{r.key}:a{inp}']=dict(symbol=symbol,code_lowering=lowering,
                operations={k:v for k,v in ops.items() if k.startswith(('vmem.','tsm.','s.blksyn')) or any(s in k for s in ('fma.f32','f16x2','lop3'))})
        if len(stats)!=2*len(spec.simt_inventory(n,k)) or 'q4_ppu_marker' not in assembly:
            raise ValueError('native SIMT inventory differs')
        return so.name,sha(so),stats
    probe=compile_('probe',ROOT/'dev/gemv_ppu/probe.cu')
    io=compile_('dense-io',ROOT/'dev/gemv_ppu/decode_io.cu');link('link-dense-io',[io],output/'libq4_dense_io.so')
    shapes=sorted(set(spec.DENSE)|set(spec.GROUPED))
    print(f'Q4_DECODE_BUILD inventory shapes={len(shapes)} parents={len(parents)} workloads=372 jobs={a.jobs}',flush=True)
    with ThreadPoolExecutor(max_workers=min(a.jobs,len(shapes))) as pool:
        simt=list(pool.map(one,shapes))
    records=compiler.compile_only(list(parents.values()),progress=lambda *x:print('Q4_DECODE_BUILD TC',*x,flush=True))
    modules=[];payloads={name:value for name,value,_ in simt}
    for r in records:
        target=output/'modules'/r['key']/'kernel.so';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['path'],target);relative=str(target.relative_to(output))
        payloads[relative]=sha(target);modules.append(r|dict(path=relative))
    shutil.copy2(spec.moe_compare.TC_BUNDLE/'libq4_moe_io.so',output/'libq4_moe_io.so')
    for name in ('libq4_dense_io.so','libq4_moe_io.so'):payloads[name]=sha(output/name)
    if any(sha(ROOT/name)!=value for name,value in sources.items()):raise ValueError('source changed during compile')
    manifest=dict(schema=spec.SCHEMA,plan=spec.plan(),modules=modules,families=families,retained_runtime=runtime,
        selection=selections,source_hashes=sources,payloads=payloads,simt_native={name:stats for name,_,stats in simt},
        simt_recipes={f'{n}x{k}':[asdict(r)|dict(key=r.key) for r in spec.simt_inventory(n,k)] for n,k in shapes},
        runtime={f'lib{x}.so':sha(sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},commands=commands,
        seconds=time.monotonic()-start,production_changed=False,device_validated=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');spec.verify(output)
    print(f'Q4_DECODE_BUILD COMPILED seconds={manifest["seconds"]:.1f} parents={len(modules)} shapes={len(shapes)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=12)
    a=p.parse_args()
    if a.jobs<1:p.error('jobs must be positive')
    build(a)
