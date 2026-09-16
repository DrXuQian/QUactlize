#!/usr/bin/env python3
"""Compile only the five observed recipes and two existing Q4 simplifications."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.model_followup import POINTS,SCHEMA,source
from quactlize.runtime.compiler import FLAGS,LIBRARIES,sha
from quactlize.runtime.native import sdk_identity


def build(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sdk=args.sdk.resolve(strict=True)
    includes=[ROOT,ROOT/'quactlize/execution',ROOT/'quactlize/include',
              ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    sources={str(p.relative_to(ROOT)):sha(p) for inc in includes[1:]
             for p in inc.rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cuh','.inc')}
    for name in ('model_followup.py','build_model_followup.py'):
        p=ROOT/'dev/gemv_simt'/name;sources[str(p.relative_to(ROOT))]=sha(p)
    for name in ('quactlize/runtime/compiler.py','quactlize/execution/simt_codegen.py','dev/gemv_simt/spec.py'):
        sources[name]=sha(ROOT/name)
    env=dict(os.environ)
    env['PATH']=str(sdk/'bin')+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    compiler=sdk/'bin/hgcc'
    started=time.monotonic()
    def one(point):
        src=out/(point.name+'.cu');src.write_text(source(point))
        obj=out/(point.name+'.o');lib=out/(point.name+'.so')
        commands=[[*map(str,(compiler,)),*FLAGS,*[f'-I{i}' for i in includes],'-c',str(src),'-o',str(obj)],
                  ['g++','-shared','-Wl,-Bsymbolic','-Wl,-z,defs',str(obj),f'-L{sdk}/lib',
                   *[f'-l{x}' for x in LIBRARIES],'-o',str(lib)]]
        with (out/(point.name+'.build.log')).open('x') as log:
            for command in commands:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/(point.name+'.resources.txt')).open('x') as log:
            subprocess.run([sdk/'bin/hgobjdump','--dump-resource-usage=all',lib],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'MODEL_SIMT_BUILD point={point.name} arms={point.arms} elapsed_s={time.monotonic()-started:.1f}',flush=True)
        return dict(point=asdict(point),arms=list(point.arms),library=lib.name,sha256=sha(lib),
                    source_sha256=sha(src),resources_sha256=sha(out/(point.name+'.resources.txt')),
                    commands=commands)
    with ThreadPoolExecutor(max_workers=min(args.jobs,len(POINTS))) as pool:
        records=list(pool.map(one,POINTS))
    if any(sha(ROOT/p)!=value for p,value in sources.items()):
        raise ValueError('source changed during compilation')
    data=dict(schema=SCHEMA,platform='ppu',records=records,source_hashes=sources,
              compiler_sha256=sha(compiler),sdk=sdk_identity(sdk),
              shipping_execution_sha256=sha(args.shipping/'libquactlize_ppu_execution.so'),
              build_seconds=time.monotonic()-started,device_admission='PENDING',
              selector_changed=False,scope='M1_SAME_GEOMETRY_H32_AND_UNSIGNED_FOLD')
    (out/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f'MODEL_SIMT_BUILD PASS points={len(records)} output={out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','shipping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=5)
    args=parser.parse_args()
    if args.jobs<1:parser.error('positive job count required')
    build(args)
