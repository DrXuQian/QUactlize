#!/usr/bin/env python3
"""Build the current heuristic's bounded prefill closure and shipping reducers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES, sha
from tools.build_kpack_dispatch import plan, catalog
from tools.compose_kpack_prefill_costs import Results
from tools.kpack_prefill_measurement import requests, families, reducer_cases, BOARD
from tools.run_kpack_dequant_gate import save
from quactlize.dequant.native import verify as verify_dequant


def build(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parents,selected=plan(out,requests())
    if any(x['status']!='SELECTED' for x in selected):raise ValueError('heuristic miss in required denominator')
    save(out/'plan.json',dict(parents=parents,requests=selected,reducers=reducer_cases()))
    print(f'PREFILL_BUILD requests={len(selected)} parents={len(parents)} reducers={len(reducer_cases())}',flush=True)
    original=Results(a.reference_dequant)
    evidence={}
    for w in families():
        fq=original.read(w['id']+'-full.json')
        sf=original.read(w['id']+'-sf.json')
        evidence[w['id']]=dict(fixture_hashes=fq['fixture_hashes'],bf16_golden_sha256=fq['golden_sha256'],
            sf_golden_sha256=sf['golden_sha256'])
    save(out/'fixture-receipts.json',dict(weights=evidence,authority=original.authority,board_sha256=sha(BOARD)))
    compiler=Compiler(a.sdk,a.cache,a.jobs)
    start=time.monotonic()
    records=compiler.compile_only(parents,progress=lambda done,total:
        print(f'PREFILL_BUILD completed={done}/{total} elapsed_s={time.monotonic()-start:.1f}',flush=True))
    for r in records:
        p=out/'modules'/r['key']/'kernel.so';p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['path'],p);r['path']=str(p.relative_to(out))
    (out/'catalog.inc').write_text(catalog(records))
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
        f'-I{out}',str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(out/'libquactlize_kpack_dispatch.so')],check=True)
    inc=[ROOT/'quactlize/include',ROOT/'third_party/actlize/include']
    commands=[
        [str(a.sdk/'bin/hgcc'),*FLAGS,*[f'-I{p}' for p in inc],'-c',
         str(ROOT/'dev/prefill_components/reducer.cu'),'-o',str(out/'reducer.o')],
        ['g++','-shared','-Wl,-Bsymbolic',str(out/'reducer.o'),'-o',str(out/'libprefill_reducer.so'),
         f'-L{a.sdk}/lib',*[f'-l{x}' for x in LIBRARIES]]]
    with (out/'reducer-build.log').open('w') as f:
        for command in commands:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
    dq=ROOT/'prebuilt/ppu0010/kpack-dequant-v4'
    verify_dequant(dq,a.sdk)
    shutil.copy2(dq/'libquactlize_ppu_dequant.so',out/'libquactlize_ppu_dequant.so')
    sources=[*sorted((ROOT/'quactlize/dispatch').glob('*')),ROOT/'tools/kpack_native_policy.cpp',
        ROOT/'policies/kpack_zw810_heuristic_v1.hpp',ROOT/'policies/kpack_zw810_runtime_v1.hpp']
    payloads=[p for p in out.rglob('*') if p.is_file() and (p.suffix=='.so' or p.name in ('plan.json','fixture-receipts.json'))]
    manifest=dict(schema='quactlize.prefill-measurement.v1',modules=records,
        policy_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file()},
        files={str(p.relative_to(out)):sha(p) for p in payloads},
        runtime={f'lib{x}.so':sha(a.sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
        reducer_source_sha256=sha(ROOT/'dev/prefill_components/reducer.cu'),
        reducer_header_sha256=sha(ROOT/'quactlize/include/actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp'),
        build_seconds=time.monotonic()-start,device_validated=False,production_changed=False,
        selection='PRODUCTION_HEURISTIC_NO_ONLINE_TUNING')
    save(out/'manifest.json',manifest)
    print(f'PREFILL_BUILD COMPLETE output={out} seconds={manifest["build_seconds"]:.1f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--reference-dequant',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args();build(a)
