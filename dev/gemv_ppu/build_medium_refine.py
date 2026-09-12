#!/usr/bin/env python3
"""Build one small, source-bound medium-Q4 PPU experiment locally."""
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
from dev.gemv_ppu import medium_refine as spec
from quactlize.runtime.compiler import FLAGS,LIBRARIES


def build(a):
    sdk=a.sdk.resolve(strict=True)
    deps=[a.latency,a.followup,a.reuse,a.config_bundle,a.previous,a.controls,a.bundle]
    old=spec.prior.verify(*deps)
    if digest(sdk/'bin/hgcc')!=old['compiler_sha256']:raise ValueError('control compiler required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes=dict(old['source_hashes'])
    for name in ('dev/gemv_ppu/medium_refine.py','dev/gemv_ppu/medium_reduce.hpp','dev/gemv_ppu/build_medium_refine.py'):
        hashes[name]=digest(ROOT/name)
    source=out/'medium.cu';source.write_text(spec.source())
    env=dict(os.environ);env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    includes=[out,ROOT/'dev/gemv_ppu',ROOT/'quactlize/execution',ROOT/'quactlize/include',ROOT/'benchmarks',
              ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    commands=[];start=time.monotonic()
    def run(label,cmd):
        cmd=list(map(str,cmd));commands.append(dict(label=label,argv=cmd))
        with (out/(label+'.log')).open('w') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc:raise ValueError(f'{label} rc={rc} log={out/(label+".log")}')
        print(f'Q4_MEDIUM_BUILD phase={label} status=PASS elapsed_s={time.monotonic()-start:.1f}',flush=True)
    def compile(item):
        label,path=item
        run(label,[sdk/'bin/hgcc',*FLAGS,'-DQKG_QTYPE=12',*[f'-I{p}' for p in includes],'-c',path,'-o',out/(label+'.o')])
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile,[('medium',source),('probe',ROOT/'dev/gemv_ppu/probe.cu')]))
    library=out/spec.PAYLOAD
    run('link',['g++','-shared','-Wl,-Bsymbolic',out/'medium.o',out/'probe.o','-o',library,
                f'-L{sdk/"lib"}',*[f'-l{lib}' for lib in LIBRARIES]])
    run('isa',[sdk/'bin/hgobjdump','--dump-isa',library])
    raw=(out/'isa.log').read_text();stats=spec.isa(raw)
    if set(stats)!={c.key for c in spec.inventory()} or 'q4_ppu_marker' not in raw:raise ValueError('native inventory differs')
    if any(not r['code_fastpath_present'] or not r['fp32_fma_present'] for r in stats.values()):raise ValueError('fast unpack/FP32 absent')
    if any(digest(ROOT/name)!=sha for name,sha in hashes.items()):raise ValueError('source changed during build')
    (out/'isa-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    m=dict(schema=spec.SCHEMA,plan=spec.plan(),device_validated=False,production_changed=False,
        latency_manifest_sha256=digest(a.latency/'manifest.json'),review_sha256=digest(spec.REVIEW),
        compiler_sha256=digest(sdk/'bin/hgcc'),inspector_sha256=digest(sdk/'bin/hgobjdump'),
        runtime={f'lib{lib}.so':digest(sdk/'lib'/f'lib{lib}.so') for lib in LIBRARIES},
        source_hashes=hashes,generated={source.name:digest(source)},
        payloads={library.name:dict(sha256=digest(library),isa_sha256=digest(out/'isa.log'))},
        isa_sha256=digest(out/'isa-stats.json'),commands=sorted(commands,key=lambda x:x['label']),seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');spec.verify(out,*deps)
    print(f'Q4_MEDIUM_BUILD status=COMPILED contexts={len(stats)} device_validated=0 seconds={m["seconds"]:.1f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name,folder in [('latency','q4-small-latency-v1'),('followup','q4-reader-followup-v1'),('reuse','q4-reader-reuse-v1'),
                        ('config-bundle','q4-config-sweep-v1'),('previous','q4-cold-shapes-v1'),('controls','q4-h800-port-v1'),('bundle','q4-simt-ab-v1')]:
        p.add_argument('--'+name,type=Path,default=ROOT/'prebuilt/ppu0010'/folder)
    build(p.parse_args())
