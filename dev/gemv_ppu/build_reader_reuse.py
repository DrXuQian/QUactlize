#!/usr/bin/env python3
"""Compile the bounded reader factorial; does not rebuild shipping libraries."""
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
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.reader_reuse import SCHEMA,SHAPES,REVIEW,inventory,payload,plan,source,isa_statistics,verify
from dev.gemv_ppu.config_space import verify as verify_configs
from quactlize.runtime.compiler import FLAGS,LIBRARIES


def build(a):
    sdk=a.sdk.resolve(strict=True)
    old=verify_configs(a.config_bundle,a.previous,a.controls,a.baseline)
    if digest(sdk/'bin/hgcc')!=old['compiler_sha256']:raise ValueError('use the original control compiler')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes=dict(old['source_hashes'])
    for name in ('dev/gemv_ppu/reader_reuse.py','dev/gemv_ppu/reader_reuse.hpp','dev/gemv_ppu/build_reader_reuse.py'):
        hashes[name]=digest(ROOT/name)
    units=[];generated={}
    for n,k in SHAPES:
        name=f'n{n}_k{k}';path=out/(name+'.cu');path.write_text(source(n,k))
        generated[path.name]=digest(path);units.append((name,path))
    units.append(('probe',ROOT/'dev/gemv_ppu/probe.cu'))
    env=dict(os.environ);env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    includes=[out,ROOT/'dev/gemv_ppu',ROOT/'quactlize/execution',ROOT/'quactlize/include',ROOT/'benchmarks',
              ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    commands=[];started=time.monotonic()
    def run(label,cmd):
        cmd=list(map(str,cmd));commands.append(dict(label=label,argv=cmd))
        with (out/(label+'.log')).open('w') as log:
            rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise ValueError(f'{label}: rc={rc} log={out/(label+".log")}')
        print(f'Q4_READER_BUILD phase={label} status=PASS elapsed_s={time.monotonic()-started:.1f}',flush=True)
    def compile_one(unit):
        name,path=unit
        run(name,[sdk/'bin/hgcc',*FLAGS,'-DQKG_QTYPE=12',*[f'-I{p}' for p in includes],'-c',path,'-o',out/(name+'.o')])
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile_one,units))
    payloads={};stats={}
    for n,k in SHAPES:
        name=f'n{n}_k{k}';library=out/payload(n,k)
        run(name+'-link',['g++','-shared','-Wl,-Bsymbolic',out/(name+'.o'),out/'probe.o','-o',library,
                         f'-L{sdk/"lib"}',*[f'-l{lib}' for lib in LIBRARIES]])
        run(name+'-isa',[sdk/'bin/hgobjdump','--dump-isa',library])
        isa=(out/(name+'-isa.log')).read_text();rows=isa_statistics(isa)
        if 'q4_ppu_marker' not in isa:raise ValueError('missing native marker')
        for r in inventory(n,k):
            key=f'n{n}-k{k}-{r.key}'
            if key not in rows or not rows[key]['code_fastpath_present'] or not rows[key]['fp32_fma_present']:
                raise ValueError('reader specialization/fast dequant/FP32 FMA missing: '+key)
            stats[key]=rows[key]
        payloads[f'{n}x{k}']=dict(file=library.name,sha256=digest(library),isa_sha256=digest(out/(name+'-isa.log')))
    if any(digest(ROOT/name)!=sha for name,sha in hashes.items()):raise ValueError('source changed during build')
    (out/'isa-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    m=dict(schema=SCHEMA,plan=plan(),device_validated=False,production_changed=False,
        config_manifest_sha256=digest(a.config_bundle/'manifest.json'),review_sha256=digest(REVIEW),
        compiler_sha256=digest(sdk/'bin/hgcc'),inspector_sha256=digest(sdk/'bin/hgobjdump'),
        runtime={f'lib{lib}.so':digest(sdk/'lib'/f'lib{lib}.so') for lib in LIBRARIES},
        source_hashes=hashes,generated=generated,payloads=payloads,isa_sha256=digest(out/'isa-stats.json'),
        commands=sorted(commands,key=lambda r:r['label']),seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    verify(out,a.config_bundle,a.previous,a.controls,a.baseline)
    print(f'Q4_READER_BUILD status=COMPILED cells=32 device_validated=0 seconds={m["seconds"]:.1f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config-bundle',type=Path,default=ROOT/'prebuilt/ppu0010/q4-config-sweep-v1')
    p.add_argument('--previous',type=Path,default=ROOT/'prebuilt/ppu0010/q4-cold-shapes-v1')
    p.add_argument('--controls',type=Path,default=ROOT/'prebuilt/ppu0010/q4-h800-port-v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'prebuilt/ppu0010/q4-simt-ab-v1')
    build(p.parse_args())
