#!/usr/bin/env python3
"""Build only exercised production launchers and a real host selector probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import ctypes as C
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.tp2_decode.fallback_plan import SCHEMA,REFERENCE_MANIFEST,RESULT_ARCHIVE,BEST,POINTS,REDUCERS,FIELDS,record_config,kernel_names,production_config,frozen_kernel_names
from dev.tp2_decode.plan import BASE_EXECUTION
from quactlize.runtime.compiler import FLAGS,LIBRARIES,sha
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import Call,arrangement


def shape_call(p):
    return Call(version=1,size=C.sizeof(Call),qtype=p.q,mode=p.mode,n=p.n,k=p.k,
        experts=p.experts,topk=8 if p.mode else 1,channels=p.channels,rows=8 if p.mode else 1,
        input_type=1,a_row_stride=p.k,a_token_stride=p.k*p.channels,
        ids_stride=8 if p.mode else 1,out_row_stride=p.n)


def source(q,configs):
    text='#include "quactlize/execution/simt_q8_vector.cuh"\n#include "quactlize/execution/simt_validation.hpp"\n'
    text+=f'extern "C" int fallback_q{q}(qkg_simt_call_v2 const& d,qkg_simt_config_v1 const& f) {{\n'
    for v,c,w,p in sorted(configs):
        reader='simt::q8_vector' if q==8 and v>=4 else 'simt'
        text+=f'  if(f.variant=={v} && f.columns=={c} && f.warps=={w} && f.values=={p}) return quactlize::execution::{reader}::launch_v2<{q},{v},{c},{w},{p}>(d,f.split);\n'
    return text+'  return QKG_INVALID;\n}\n'


def build(args):
    sdk,ref=args.sdk.resolve(strict=True),args.reference.resolve(strict=True)
    if sha(ref/'manifest.json')!=REFERENCE_MANIFEST or sha(ref/'libquactlize_ppu_execution.so')!=BASE_EXECUTION:
        raise ValueError('frozen reference differs')
    frozen=json.loads((ref/'manifest.json').read_text())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=[p for d in ('quactlize','policies','third_party/actlize/include','dev/gemv_model','dev/tp2_decode')
            for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in ('.cu','.cuh','.h','.hpp','.inc','.cpp','.py')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    start=time.monotonic()
    (out/'catalog.inc').write_text('static std::vector<Image> const kImages{};\nstatic char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
        '-I'+str(out),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl',
        '-o',str(out/'libquactlize_kpack_dispatch.so')],check=True)
    matched=json.loads((ROOT/'policies/kpack_smallm_matched_v1.json').read_text())
    dispatcher=Dispatch(out);records=[];inventory={};required={'libquactlize_ppu_execution.so'}
    try:
        for p in POINTS:
            if p in REDUCERS:
                cfg=dict(variant=3,columns=4,warps=4,values=4,split=4)
                selection=dict(kind=1,policy=None,scope='EXPLICIT_REDUCER_CONTROL_NOT_POLICY')
                baseline_cfg=record_config(cfg,'same-producer-old-reducer',vector_reduce=False)
            else:
                pick=dispatcher.query_smallm_matched(shape_call(p),arrangement(p.q),p.compute)
                if pick is None or pick.base.kind!=1:
                    raise ValueError('expected production SIMT selection unavailable: '+p.name)
                b=pick.base;cfg={k:getattr(b.simt,k) for k in FIELDS}
                selection=dict(kind=b.kind,policy=b.policy,source_n=b.source_n,
                    source_k=b.source_k,source_tokens=b.source_tokens,scope='ACTUAL_DISPATCH_C_ABI_M1')
                donor=[p.q,p.mode,b.source_n,b.source_k,p.experts,8 if p.mode else 1,p.channels,b.source_tokens,p.compute]
                old=next(r['config'] for r in matched['exact'] if r['key']==donor)
                baseline_cfg=record_config(old,'old-bucket') if old['kind']=='simt' else None
            if p.name in BEST:
                previous=next(r for r in frozen['records'] if r['point']['name']==p.name)
                arm=BEST[p.name]
                if arm!='incumbent':required.add(previous['library'])
                authority='FROZEN_CONFIRMED_MINIMUM'
            else:
                if not baseline_cfg:raise ValueError('new point needs an explicit old SIMT control')
                previous=dict(point=asdict(p),candidates=[baseline_cfg]);arm='incumbent'
                authority='OLD_PRODUCTION_CONTROL_NO_HISTORICAL_OPTIMUM_CLAIM'
            record=dict(point=asdict(p),candidate=production_config(p,cfg),selection=selection,
                reference=dict(record=previous,arm=arm,authority=authority),
                expected_kernels=kernel_names(p,cfg))
            records.append(record)
            inventory.setdefault(p.q,set()).add(tuple(cfg[k] for k in FIELDS[:-1]))
            print('FALLBACK_BUILD_PLAN '+json.dumps(dict(point=p.name,selection=selection,config=cfg)),flush=True)
    finally:dispatcher.close()
    for name in sorted(required):
        if Path(name).name!=name or sha(ref/name)!=frozen['payloads'][name]:raise ValueError('reference payload differs: '+name)
        shutil.copy2(ref/name,out/name)
    shutil.copy2(ref/'manifest.json',out/'reference-manifest.json')
    env=dict(os.environ,PATH=str(sdk/'bin')+os.pathsep+os.environ.get('PATH',''),
             LD_LIBRARY_PATH=str(sdk/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
    inc=[ROOT,ROOT/'quactlize/include',ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    def compile_q(item):
        q,configs=item;src=out/f'q{q}.cu';src.write_text(source(q,configs))
        cmd=[str(sdk/'bin/hgcc'),*FLAGS,*['-I'+str(p) for p in inc],'-c',str(src),'-o',str(out/f'q{q}.o')]
        with (out/f'q{q}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'FALLBACK_BUILD_COMPILE q={q} PASS elapsed_s={time.monotonic()-start:.1f}',flush=True)
    with ThreadPoolExecutor(max_workers=min(args.jobs,len(inventory))) as pool:list(pool.map(compile_q,inventory.items()))
    entry='#include "quactlize/execution/simt_validation.hpp"\n'
    for q in inventory:entry+=f'extern "C" int fallback_q{q}(qkg_simt_call_v2 const&,qkg_simt_config_v1 const&);\n'
    entry+='''extern "C" int fallback_run(qkg_simt_call_v2 const* d,qkg_simt_config_v1 const* f,
quactlize_ppu_placed_arrangement_v2 const* a) {
  if(!d || !f || !a) return QKG_INVALID;
  qkg_sizes_v1 sizes{};
  int rc=quactlize::execution::simt::query_v2(*d,*f,a,sizes);if(rc)return rc;
  rc=quactlize::execution::simt::buffers_v2(*d,sizes);if(rc)return rc;
  switch(d->call.qtype) {
'''
    for q in inventory:entry+=f'    case {q}: return fallback_q{q}(*d,*f);\n'
    entry+='    default:return QKG_INVALID;\n  }\n}\n'
    (out/'entry.cpp').write_text(entry)
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-Wl,-Bsymbolic','-Wl,-z,defs',
        '-I'+str(ROOT),str(out/'entry.cpp'),*[str(out/f'q{q}.o') for q in inventory],
        '-L'+str(sdk/'lib'),*['-l'+x for x in LIBRARIES],'-o',str(out/'fallback.so')],env=env,check=True)
    for suffix,option in [('isa','--dump-isa'),('resources','--dump-resource-usage=all')]:
        with (out/(suffix+'.txt')).open('x') as f:
            subprocess.run([str(sdk/'bin/hgobjdump'),option,str(out/'fallback.so')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    # Ensure the profiler's exact producer/reducer identities really occur in
    # this image. Static instruction checks are not a device/performance pass.
    isa=(out/'isa.txt').read_text();names=re.findall(r'Disassembly of section \.text\.kernel\.([^\n]+):',isa)
    demangled=subprocess.check_output(['c++filt',*names],text=True)
    normalized=re.sub(r'\s+','',demangled)
    for record in records:
        for name in record['expected_kernels']:
            if name and name not in normalized:raise ValueError('declared native identity not emitted: '+name)
    inspection=dict(scope='STATIC_ISA_NOT_DEVICE_PERFORMANCE',kernels=len(names),
        expected={r['point']['name']:r['expected_kernels'] for r in records},
        fp32_fma=bool(re.search(r'\tv\.fma\.f32',isa)),
        paired_load=bool(re.search(r'\tvmem\.ld\.b32x2',isa)),
        paired_store=bool(re.search(r'\tvmem\.st\.b32x2',isa)))
    if not all(inspection[k] for k in ('fp32_fma','paired_load','paired_store')):raise ValueError('native decode/reducer instructions missing')
    resources=subprocess.check_output([str(sdk/'bin/hgobjdump'),'--dump-resource-usage=all',
        str(out/'libquactlize_ppu_execution.so')],env=env,text=True)
    old_names=re.findall(r'^Func \d+: (\S+)',resources,re.M)
    old_symbols=re.sub(r'\s+','',subprocess.check_output(['c++filt'],input='\n'.join(old_names),text=True))
    inspection['frozen_incumbents']={}
    for record in records:
        old=record['reference']
        if old['arm']!='incumbent':continue
        point=next(p for p in POINTS if p.name==record['point']['name'])
        expected=frozen_kernel_names(point,old['record']['candidates'][0])
        if any(name and name not in old_symbols for name in expected):
            raise ValueError('frozen incumbent identity not emitted: '+point.name)
        inspection['frozen_incumbents'][point.name]=expected
    (out/'native-inspection.json').write_text(json.dumps(inspection,indent=2)+'\n')
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise ValueError('build input changed')
    payloads={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.so','.cu','.cpp','.inc','.txt') or p.name in ('reference-manifest.json','native-inspection.json')}
    manifest=dict(schema=SCHEMA,records=records,source_hashes=hashes,payloads=payloads,
        runtime={f'lib{x}.so':sha(sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
        reference_manifest_sha256=REFERENCE_MANIFEST,result_archive_sha256=RESULT_ARCHIVE,
        compiler_sha256=sha(sdk/'bin/hgcc'),build_seconds=time.monotonic()-start,
        selector_probe_only_no_tc_catalog=True,production_replacement=False,device_validated=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'FALLBACK_BUILD PASS points={len(records)} seconds={manifest["build_seconds"]:.1f} output={out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sdk','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args()
    if a.jobs<1:p.error('positive jobs required')
    build(a)
