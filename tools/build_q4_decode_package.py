#!/usr/bin/env python3
"""Package the measured decode closure; reuse TC images, rebuild host only."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_kpack_dispatch import catalog
from tools.verify_kpack_dispatch import verify
from quactlize.execution import q4_decode_codegen
from quactlize.runtime.compiler import Compiler, sha, source_contract
from quactlize.runtime.tuning import digest


def build(a):
    base=verify(a.base,sdk=a.sdk)
    execution=json.loads((a.execution/'manifest.json').read_text())
    policy=json.loads(q4_decode_codegen.POLICY.read_text())
    if (execution.get('q4_decode_policy_sha256')!=sha(q4_decode_codegen.POLICY) or
        execution['sha256']!=sha(a.execution/execution['library'])):
        raise ValueError('execution library does not contain this decode policy')
    frozen=json.loads((a.measured/'manifest.json').read_text())
    need={r['recipe'].split(':')[1] for r in policy['ranges'] if r['recipe'].startswith('tc:')}
    records={r['parent']['symbol']:(r,a.base) for r in base['modules']}
    for r in frozen['modules']:
        if r['parent']['symbol'] in need:
            if sha(a.measured/r['path'])!=r['sha256']:
                raise ValueError('measured TC image differs')
            records[r['parent']['symbol']]=(r,a.measured)
    if not need<=records.keys():
        raise ValueError('decode TC parent closure incomplete')
    compiler=Compiler(a.sdk,a.output/'unused-cache')
    for r,_ in records.values():
        if source_contract(r['identity'])!=source_contract(compiler.identity):
            raise ValueError('TC source changed; reuse is not valid')
    for name,value in execution['runtime'].items():
        if sha(a.sdk/'lib'/name)!=value:
            raise ValueError('execution runtime differs')
    a.output.mkdir(parents=True,exist_ok=False)
    modules=[]
    for r,root in records.values():
        target=a.output/r['path']; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/r['path'],target);modules.append(r)
    contract=source_contract(compiler.identity)
    (a.output/'catalog.inc').write_text(catalog(modules,contract))
    host=a.output/'libquactlize_kpack_dispatch.so'
    command=['g++','-std=c++17','-O2','-fPIC','-shared','-pthread','-Wl,-Bsymbolic',
             f'-I{a.output}',str(ROOT/'quactlize/dispatch/binding.cpp'),'-ldl','-o',str(host)]
    subprocess.run(command,check=True)
    shutil.copy2(a.execution/execution['library'],a.output/execution['library'])
    for path in (q4_decode_codegen.POLICY,q4_decode_codegen.POLICY.with_suffix('.hpp')):
        shutil.copy2(path,a.output/path.name)
    sources=[p for p in (ROOT/'quactlize/dispatch').glob('*') if p.is_file()]
    sources += [q4_decode_codegen.POLICY,q4_decode_codegen.POLICY.with_suffix('.hpp'),Path(__file__)]
    m=base | dict(modules=modules,dispatch_sha256=sha(host),execution_sha256=execution['sha256'],
        execution_receipt=execution,host_command=command,
        policy_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        compiler_identities={digest(r['identity']):r['identity'] for r in modules},
        jit_source_contract=contract,
        jit_source_identity={k:compiler.identity[k] for k in ('kernel','flags','generator')},
        jit_required=True,device_validated=False,heuristic_admitted=False,
        decode_policy=dict(path=q4_decode_codegen.POLICY.name,sha256=sha(q4_decode_codegen.POLICY),
                           measured_manifest_sha256=sha(a.measured/'manifest.json'),
                           selection_replayed=True,model_admission='PENDING'),
        reused_packages=[dict(path=str(a.base),manifest_sha256=sha(a.base/'manifest.json')),
                         dict(path=str(a.measured),manifest_sha256=sha(a.measured/'manifest.json'))])
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    verify(a.output,sdk=a.sdk)
    print(f'Q4_DECODE_PACKAGE COMPILED tc_reused={len(modules)} tc_recompiled=0 root={a.output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'prebuilt/ppu0010/kpack-fusion-v3/dispatch')
    p.add_argument('--measured',type=Path,default=ROOT/'prebuilt/ppu0010/q4-decode-sweep-v1')
    p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())
