#!/usr/bin/env python3
"""Compile a bounded, independent SF/full-BF16 dequant diagnostic library."""
import argparse
from collections import Counter
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha


def resource_usage(text):
    """Read inspector fields verbatim; the shared-allocation field is encoded."""
    parts=re.split(r'(?m)^Func \d+ (\S+) RESOURCE INFO:\n',text)
    result={}
    for i in range(1,len(parts),2):
        name,body=parts[i:i+2]
        if name in result:raise ValueError('duplicate kernel resource record')
        fields={}
        for source,target in [('vreg_number','registers'),('sreg_number','scalar_registers'),
                              ('shared_memory_size','shared_allocation_field'),('STACK SIZE','stack_bytes')]:
            match=re.findall(r'(?m)^'+re.escape(source)+r':\s*(\d+)\s*$',body)
            if len(match)!=1:raise ValueError('missing/duplicate native resource field: '+source)
            fields[target]=int(match[0])
        result[name]=fields
    if not result:raise ValueError('native resource records missing')
    return result


def build(sdk, output):
    sdk, output = sdk.resolve(strict=True), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [ROOT/'quactlize/dequant', ROOT/'quactlize/execution',
             ROOT/'quactlize/include', ROOT/'third_party/actlize/include']
    inputs = {str(p.relative_to(ROOT)): sha(p) for d in paths for p in d.rglob('*')
              if p.is_file() and p.suffix in ('.hpp', '.cuh', '.h', '.cu', '.inc')}
    env = dict(os.environ, PATH=str(sdk/'bin')+':'+os.environ.get('PATH',''),
               LD_LIBRARY_PATH=str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    commands = [
        [str(sdk/'bin/hgcc'), *FLAGS, *[f'-I{p}' for p in paths],
         '-c', str(ROOT/'quactlize/dequant/kernels.cu'), '-o', str(output/'dequant.o')],
        ['g++', '-shared', '-Wl,-Bsymbolic', str(output/'dequant.o'),
         '-o', str(output/'libquactlize_ppu_dequant.so'), f'-L{sdk}/lib',
         *[f'-l{x}' for x in LIBRARIES]],
    ]
    start = time.monotonic()
    with (output/'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    if any(sha(ROOT/p)!=h for p,h in inputs.items()):
        raise ValueError('dequant sources changed during compilation')
    isa=subprocess.run([str(sdk/'bin/hgobjdump'),'--dump-isa',str(output/'libquactlize_ppu_dequant.so')],
                       check=True,capture_output=True,text=True).stdout
    resources=resource_usage(subprocess.run([str(sdk/'bin/hgobjdump'),'--dump-resource-usage=all',str(output/'libquactlize_ppu_dequant.so')],
                                           check=True,capture_output=True,text=True).stdout)
    entries=list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):',isa))
    if len(entries)!=57:raise ValueError('dequant kernel specialization denominator differs')
    native={}
    for i,entry in enumerate(entries):
        body=isa[entry.end():entries[i+1].start() if i+1<len(entries) else None]
        ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
        symbol=entry[1]
        if any(x in symbol for x in ('full_transpose','sf_columns','full_wide','sf_unit16','full_shared')) and any('f64' in op for op in ops):
            raise ValueError('coordinate decomposition unexpectedly uses FP64 division lowering')
        if any(x in symbol for x in ('full_transpose','full_direct','full_wide','full_shared')) and not any('bf16.f32' in op for op in ops):
            raise ValueError('native BF16 output conversion missing')
        if 'sf_unit16' in symbol and (ops['vmem.ld.b32x4']!=1 or any(op.startswith('vmem.ld.b8') for op in ops)):
            raise ValueError('metadata uint4 load was scalarized')
        if 'full_wide' in symbol and 'ELb1EE' in symbol and not any(op.startswith('vmem.st.b32x4') for op in ops):
            raise ValueError('wide BF16 output store was scalarized')
        if 'full_shared' in symbol:
            for kind in ('vmem.ld.b32x4','tsm.ld.b32x4','vmem.st.b32x4'):
                if not any(op.startswith(kind) for op in ops):raise ValueError('shared-vector path was scalarized: '+kind)
            if resources[symbol]['stack_bytes']:raise ValueError('new shared reader has a local stack allocation')
        native[symbol]=dict(operations=dict(sorted(ops.items())),resources=resources[symbol],scope='STATIC_COUNTS_NOT_DYNAMIC_PROFILE')
    if set(native)!=set(resources):raise ValueError('ISA and resource kernel sets differ')
    (output/'native.json').write_text(json.dumps(native,indent=2)+'\n')
    manifest = dict(schema='quactlize.dequant-only.v1', source_hashes=inputs,
        library='libquactlize_ppu_dequant.so', sha256=sha(output/'libquactlize_ppu_dequant.so'),
        compiler=sha(sdk/'bin/hgcc'), runtime={f'lib{x}.so':sha(sdk/'lib'/f'lib{x}.so') for x in LIBRARIES},
        commands=commands, flags=FLAGS, seconds=time.monotonic()-start,
        formats=list(range(10,15)), config_generation=3,
        sf_configs=[0,1,2,3,4,5], sf_vector_qtypes=[12,13], full_configs=[0,1,2,3,4,5],
        full_shared_configs=[6,7,8,9], full_shared_qtypes=[12,13],
        device_validated=False, production_changed=False,
        full_output='BF16_E_N_K', full_arithmetic='RAW_GGUF_FP32_MUL_SUB_BF16_RNE',
        sf_output='FP16_E_GROUP_N_TWO_PLANES', timing='NO_GEMM')
    manifest['native_sha256']=sha(output/'native.json')
    manifest['inspector_sha256']=sha(sdk/'bin/hgobjdump')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('KPACK_DEQUANT_BUILD',json.dumps(dict(seconds=manifest['seconds'],library=str(output/manifest['library']))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.sdk,a.output)
