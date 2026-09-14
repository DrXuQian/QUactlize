#!/usr/bin/env python3
"""Freeze selected/challenger closure; reuse exact images and compile only misses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.kpack_cost_supplement import plan, request, historical
from tools.build_kpack_dispatch import plan as selected_plan, catalog
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES, sha, validate_parent
from quactlize.runtime.candidates import PARENT_FIELDS
from quactlize.runtime.tuning import digest
from tools.run_kpack_dequant_gate import save
from quactlize.dequant.native import verify as verify_dequant


def make_plan(output):
    out=plan()
    requests=sorted({request(p,r) for p in out['points'] for r in ((0,1) if p['experts']==1 else (2,3))})
    parents,selected=selected_plan(output,requests)
    parents={p['symbol']:p for p in parents}
    selected={tuple(r['request']):r for r in selected}
    for p in out['points']:
        p['routes']={}
        for route in ((0,1) if p['experts']==1 else (2,3)):
            current=selected[request(p,route)]
            old=historical(p,route)
            if old:
                parent={key:old[key] for key in PARENT_FIELDS}
                validate_parent(parent)
                parents[parent['symbol']]=parent
            p['routes'][str(route)]=dict(selected=current,historical=old)
    out['parents']=sorted(parents.values(),key=lambda p:p['symbol'])
    out['selected_requests']=list(selected.values())
    out['plan_sha256']=digest(out)
    return json.loads(json.dumps(out))


def build(a):
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    planned=make_plan(out)
    if any(r['status']!='SELECTED' for r in planned['selected_requests']):
        raise ValueError('supplement has a production heuristic miss')
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=planned:
        raise ValueError('build plan changed; use a new output directory')
    save(path,planned)
    print('COST_BUILD_PLAN '+json.dumps(dict(points=len(planned['points']),parents=len(planned['parents']),
        misses=sum(r['status']!='SELECTED' for r in planned['selected_requests']))),flush=True)
    if a.plan_only:return
    compiler=Compiler(a.sdk,a.cache,a.jobs)
    reusable={}
    for root in (ROOT/'prebuilt/ppu0010').glob('*'):
        path=root/'manifest.json'
        if not path.is_file():continue
        m=json.loads(path.read_text())
        for r in m.get('modules',[]):
            if not all(k in r for k in ('parent','identity','key','sha256','path')):continue
            p=root/r['path']
            if r['identity']!=compiler.identity or not p.is_file():continue
            if sha(p)!=r['sha256']:continue
            key=digest(dict(identity=compiler.identity,parent=r['parent'],source=compiler.source(r['parent'],'')))
            if r['key']==key:reusable[r['parent']['symbol']]=dict(r,path=str(p),cache_hit=True)
    records=[];missing=[]
    for p in planned['parents']:
        if p['symbol'] in reusable and reusable[p['symbol']]['parent']==p:
            records.append(reusable[p['symbol']])
        else:missing.append(p)
    print(f'COST_BUILD reusable={len(records)} missing={len(missing)} jobs={a.jobs}',flush=True)
    started=time.monotonic()
    records+=compiler.compile_only(missing,progress=lambda done,total:
        print(f'COST_BUILD_PROGRESS complete={done}/{total} elapsed_minutes={(time.monotonic()-started)/60:.1f}',flush=True))
    for r in records:
        r.pop('cache_hit',None)
        dst=out/'modules'/r['key']/'kernel.so'
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists() or sha(dst)!=r['sha256']:shutil.copy2(r['path'],dst)
        r['path']=str(dst.relative_to(out))
    records.sort(key=lambda r:r['parent']['symbol'])
    # Generated build input, not a second config selector.
    (out/'catalog.inc').write_text(catalog(records))
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',f'-I{out}',
        str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(out/'libquactlize_kpack_dispatch.so')],check=True)
    includes=[ROOT/'quactlize/include',ROOT/'third_party/actlize/include']
    indexed_sources=[ROOT/'quactlize/dequant'/s for s in ('indexed.cu','indexed.h','expert_selection.cuh','vector_kernels.cuh','packed_kernels.cuh','reader.hpp','api.h')]
    indexed_sources += [ROOT/'quactlize/execution'/s for s in ('validation.hpp','api.h')]
    indexed_identity={str(p.relative_to(ROOT)):sha(p) for p in indexed_sources}
    receipt=out/'indexed-build.json'
    expected=dict(sources=indexed_identity,sdk=compiler.sdk_identity,flags=FLAGS)
    lib=out/'libquactlize_dequant_indexed.so'
    if not (receipt.exists() and lib.exists() and json.loads(receipt.read_text())==expected|dict(sha256=sha(lib))):
        cmds=[
            [str(a.sdk/'bin/hgcc'),*FLAGS,*[f'-I{p}' for p in includes],'-c',str(indexed_sources[0]),'-o',str(out/'indexed.o')],
            ['g++','-shared','-Wl,-Bsymbolic',str(out/'indexed.o'),f'-L{a.sdk}/lib',*[f'-l{n}' for n in LIBRARIES],'-o',str(lib)]]
        with (out/'indexed-build.log').open('w') as log:
            for cmd in cmds:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        save(receipt,expected|dict(sha256=sha(lib)))
    dq=ROOT/'prebuilt/ppu0010/kpack-dequant-v4'
    verify_dequant(dq,a.sdk)
    shutil.copy2(dq/'libquactlize_ppu_dequant.so',out/'libquactlize_ppu_dequant.so')
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and (p.suffix=='.so' or p.name=='plan.json')}
    source_paths=[ROOT/'quactlize/dispatch/policy.hpp',ROOT/'policies/kpack_zw810_heuristic_v1.hpp',
                  ROOT/'policies/kpack_zw810_runtime_v1.hpp',ROOT/'tools/kpack_cost_supplement.py',*indexed_sources]
    save(out/'manifest.json',dict(schema='quactlize.cost-supplement-bundle.v1',files=files,modules=records,
        plan_sha256=planned['plan_sha256'],runtime={f'lib{x}.so':sha(a.sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths},indexed=expected,
        dequant_v4_manifest_sha256=sha(dq/'manifest.json'),production_changed=False))
    print(f'COST_BUILD_DONE output={out} compile_minutes={(time.monotonic()-started)/60:.1f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
    p.add_argument('--plan-only',action='store_true')
    build(p.parse_args())
