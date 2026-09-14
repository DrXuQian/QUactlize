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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sdk','bundle','stage-binary'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args();root=a.bundle.resolve(strict=True);m=verify(root,sdk=a.sdk)
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
