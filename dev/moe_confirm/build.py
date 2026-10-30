"""Only the new helper wrappers; TC/vector/dequant use existing builders."""
import json
import os
from pathlib import Path
import subprocess

from dev.moe_scaffold.build import build as build_scaffold,verify as verify_scaffold
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import FLAGS,LIBRARIES,sha

ROOT=Path(__file__).resolve().parents[2]


def build(sdk,output):
    sdk=Path(sdk).resolve(strict=True);output=Path(output).resolve()
    receipt=output/'receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text());verify(value);return value
    output.mkdir(parents=True,exist_ok=False)
    scaffold=build_scaffold(sdk,output/'scaffold')
    sources={p:sha(ROOT/p) for p in ('dev/moe_confirm/bridge.cpp','dev/moe_confirm/provider_rows.cu',
        'dev/native_components/kernel_node.hpp','dev/native_components/kernel_bridge.hpp',
        'quactlize/prefill/layout.hpp',__file__.removeprefix(str(ROOT)+'/'))}
    bridge=output/'libmoe-confirm.so';cuda_lib=sdk/'CUDA_SDK/targets/x86_64-linux/lib'
    commands=[['g++','-std=c++17','-O2','-fPIC','-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(sdk/'CUDA_SDK/include'),
        '-c',str(ROOT/'dev/moe_confirm/bridge.cpp'),'-o',str(output/'bridge.o')],
        [str(sdk/'bin/hgcc'),*FLAGS,'-c',str(ROOT/'dev/moe_confirm/provider_rows.cu'),'-o',str(output/'rows.o')],
        ['g++','-shared','-Wl,-z,defs','-Wl,-Bsymbolic',str(output/'bridge.o'),str(output/'rows.o'),
         '-L'+str(cuda_lib),'-Wl,-rpath,'+str(cuda_lib),'-lcudart','-lcuda',
         '-L'+str(sdk/'lib'),'-Wl,-rpath,'+str(sdk/'lib'),*['-l'+s for s in LIBRARIES],'-o',str(bridge)]]
    env=dict(os.environ,PATH=str(sdk/'bin')+':'+os.environ.get('PATH',''))
    for i,command in enumerate(commands):
        path=output/f'bridge-{i}.log'
        with path.open('w') as log:
            rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError(f'helper bridge build rc={rc}: {path}\n'+path.read_text(errors='replace')[-8000:])
    value=dict(scaffold=scaffold,bridge=str(bridge),sha256=sha(bridge),sources=sources,commands=commands,
               cuda_libraries={str((cuda_lib/name).resolve()):sha(cuda_lib/name) for name in ('libcudart.so','libcuda.so')})
    verify(value);atomic_json(receipt,value);return value


def verify(value):
    verify_scaffold(value['scaffold'])
    if sha(value['bridge'])!=value['sha256']:raise ValueError('confirmation bridge changed')
    for p,h in value['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('confirmation helper source changed: '+p)
    for p,h in value['cuda_libraries'].items():
        if sha(p)!=h:raise ValueError('confirmation SDK bridge changed: '+p)
