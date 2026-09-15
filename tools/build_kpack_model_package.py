#!/usr/bin/env python3
"""Add only the mixed-chain gate closure to an already built small runtime."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_kpack_dispatch import plan,catalog
from tools.run_kpack_moe_gate import chain_requests
from tools.verify_kpack_dispatch import verify
from quactlize.runtime.compiler import Compiler,sha,source_contract
from quactlize.runtime.tuning import digest


def attach_bf16(root, package, sdk):
    from dev.bf16_compute.run import validate_package
    root=root.resolve(strict=True);package=package.resolve(strict=True)
    model=verify(root,sdk=sdk)
    if not model.get('compute_contract') or 'bf16_gate' in model:
        raise ValueError('requires a compute-capable package without an attached BF16 gate')
    gate=validate_package(package)
    for record in gate['modules'].values():
        if record['identity']['base_source_contract']!=model['jit_source_contract']:
            raise ValueError('BF16 gate/dispatcher kernel source contracts differ')
    if (gate.get('reused_execution') or {}).get('sha256')!=model['execution_sha256']:
        raise ValueError('BF16 gate must validate the same execution image used by the model')
    names={'manifest.json',gate['simt']['path'],gate['moe']['path']}
    names.update(r['path'] for r in gate['modules'].values())
    destination=root/'bf16';destination.mkdir()
    for name in sorted(names):
        source=(package/name).resolve(strict=True)
        relative=source.relative_to(package)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    validate_package(destination)
    model['bf16_gate']=dict(path='bf16/manifest.json',sha256=sha(destination/'manifest.json'),
        cases=len(gate['cases']),modules=len(gate['modules']),device_validated=False)
    (root/'manifest.json').write_text(json.dumps(model,indent=2)+'\n')
    print(f"MODEL_BF16_GATE_ATTACHED modules={len(gate['modules'])} cases={len(gate['cases'])} device=PENDING")


def attach_q4_bf16(root, package, sdk):
    from dev.bf16_fastpath.gate import verified
    root=root.resolve(strict=True);package=package.resolve(strict=True)
    model=verify(root,sdk=sdk);gate,library=verified(package)
    if 'q4_bf16_gate' in model or gate['sha256']!=model['execution_sha256']:
        raise ValueError('typed Q4 gate must reuse the model execution image exactly once')
    destination=root/'q4-bf16-gate';destination.mkdir()
    for source in (package/'manifest.json',library):shutil.copy2(source,destination/source.name)
    model['q4_bf16_gate']=dict(path='q4-bf16-gate/manifest.json',sha256=sha(destination/'manifest.json'),
        library='q4-bf16-gate/'+library.name,denominator=gate['plan']['denominator'],device_validated=False)
    (root/'manifest.json').write_text(json.dumps(model,indent=2)+'\n')
    verify(root,sdk=sdk)
    print(f"MODEL_TYPED_Q4_GATE_ATTACHED BF16={gate['plan']['denominator']['bf16_cells']} execution=SAME_IMAGE device=PENDING")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sdk','bundle'):p.add_argument('--'+key,type=Path,required=True)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--stage-binary',type=Path)
    action.add_argument('--bf16-gate',type=Path,help='attach the bounded gate to an already built compute package')
    action.add_argument('--q4-bf16-gate',type=Path,help='attach actual typed Q4 select/run coverage from the same execution image')
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args();root=a.bundle.resolve(strict=True)
    if a.bf16_gate:
        attach_bf16(root,a.bf16_gate,a.sdk);return
    if a.q4_bf16_gate:
        attach_q4_bf16(root,a.q4_bf16_gate,a.sdk);return
    m=verify(root,sdk=a.sdk)
    if m['modules'] or 'moe_mixed_gate' in m:raise ValueError('requires a fresh JIT-only package')
    requests={tuple(r[k] for k in ('q','route','m','n','k','experts','max_rows'))
        for t in (1,4,8) for merged in (False,True) for q in (12,13)
        for r in chain_requests(merged,t,q)}
    parents,_=plan(root,sorted(requests))
    decode,_=plan(root,sorted(r for r in requests if r[0]==12),decode=True)
    parents=list({p['symbol']:p for p in parents+decode}.values())
    compiler=Compiler(a.sdk,root/'modules',a.jobs)
    records=compiler.compile_only(parents,lambda n,total:print(f'MODEL_GATE_BUILD parents={n}/{total}',flush=True))
    for r in records:r['path']=str(Path(r['path']).relative_to(root))
    (root/'catalog.inc').write_text(catalog(records,source_contract(compiler.identity)))
    host=root/'libquactlize_kpack_dispatch.so'
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
        '-I'+str(root),str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(host)],check=True)
    shutil.copy2(a.stage_binary,root/'mixed-stages')
    m.update(modules=records,dispatch_sha256=sha(host),
        compiler_identities={digest(r['identity']):r['identity'] for r in records},
        moe_mixed_gate=dict(schema='quactlize.moe-mixed-gate.v1',stage_cases=80,chain_cases=24,
            simt_binaries=[dict(path='mixed-stages',sha256=sha(root/'mixed-stages'),
                source_sha256=sha(ROOT/'tests/kpack_moe_chain_cuda.cu'))]))
    (root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');verify(root,sdk=a.sdk)
    print(f'MODEL_GATE_BUILD COMPLETE parents={len(records)} stage_cases=80 chain_cases=24 device=PENDING')


if __name__=='__main__':main()
