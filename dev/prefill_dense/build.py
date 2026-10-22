"""One small runtime/candidate library, no TC or llama rebuild."""
from pathlib import Path
import os
import re
import subprocess
from collections import Counter
from tools.build_kpack_prefill_runtime import build as runtime_build
from tools.build_kpack_dequant import resource_usage
from dev.native_components.build import run_commands
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha

ROOT=Path(__file__).resolve().parents[2]

def inspect(sdk,image):
    isa=subprocess.check_output([str(sdk/'bin/hgobjdump'),'--dump-isa',str(image)],text=True)
    resources=resource_usage(subprocess.check_output(
        [str(sdk/'bin/hgobjdump'),'--dump-resource-usage=all',str(image)],text=True))
    entries=list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):',isa));rows=[]
    for i,e in enumerate(entries):
        if not any(n in e[1]for n in ('dense_convert','dense_dequant')):continue
        body=isa[e.end():entries[i+1].start()if i+1<len(entries)else None]
        ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
        if resources[e[1]]['stack_bytes']:raise ValueError('candidate has local stack: '+e[1])
        if ('dense_convert_vec8' in e[1]or'dense_dequant_'in e[1]and'direct'not in e[1])and not any(
                op.startswith('vmem.st.b32x4')for op in ops):
            raise ValueError('candidate vector output store missing: '+e[1])
        if 'dense_convert_stripe4' in e[1]or'dense_convert_full4'in e[1]:
            input_=re.search(r'(?:stripe4|full4)ILb[01]ELb([01])',e[1])[1]=='1'
            vectors=('vmem.ld.b32x4','vmem.st.b32x2')if input_ else('vmem.ld.b32x2','vmem.st.b32x4')
            if not all(any(op.startswith(prefix)for op in ops)for prefix in vectors):
                raise ValueError('striped vector transfer missing: '+e[1])
            if resources[e[1]]['shared_allocation_field']:raise ValueError('striped IO unexpectedly stages through shared memory')
        rows.append(dict(symbol=e[1],ops=dict(ops),resources=resources[e[1]]))
        if 'dense_dequant_cooperative'in e[1]and'ELb1EEEvPKtS5_PKhPtii'in e[1]:
            if any(op.startswith('vmem.ld.b8')for op in ops):raise ValueError('cooperative metadata regressed to global byte loads: '+e[1])
            if not any(op.startswith('vmem.ld.b32x4')for op in ops):raise ValueError('cooperative vector loads missing: '+e[1])
    if len(rows)!=160:raise ValueError('typed candidate inventory differs: '+str(len(rows)))
    pairs={r['symbol']:r for r in rows if 'dense_dequant_pair' in r['symbol']}
    fast=[]
    for symbol,row in pairs.items():
        # Final template argument is the conversion mode. The rest of the
        # specialization, including output precision and geometry, is equal.
        if not symbol.endswith('ELb1EEEvPKtS5_PKhPtii'):continue
        control=pairs[symbol.replace('ELb1EEEvPKtS5_PKhPtii','ELb0EEEvPKtS5_PKhPtii')]
        count=lambda r:sum(v for op,v in r['ops'].items()if op.startswith('v.cnvt.f32.i32'))
        if count(row)>=count(control):raise ValueError('direct FP32 code conversion did not lower: '+symbol)
        fast.append(dict(symbol=symbol,control=control['symbol'],int_to_fp32=[count(control),count(row)]))
    if len(fast)!=18:raise ValueError('paired FP32 lowering inventory differs')
    return dict(scope='STATIC_CODEGEN_NOT_SPEEDUP',kernels=rows,fast_fp32_pairs=fast)

def build(sdk,output,jobs):
    runtime_build(sdk,output,jobs)
    env=dict(os.environ,PATH=str(sdk/'bin')+':'+os.environ.get('PATH',''))
    cmd=['g++','-std=c++17','-O3','-shared','-fPIC','-Wl,-z,defs',
        '-I'+str(ROOT),'-I'+str(sdk/'include'),str(ROOT/'dev/prefill_dense/timer.cpp'),
        '-L'+str(sdk/'lib'),'-L'+str(sdk/'targets/x86_64-linux/lib'),
        '-lhggc_wrapper','-lhggcrt1','-lhggc','-o',str(output/'libdense-opt-timer.so')]
    run_commands([cmd],ROOT,output/'timer-build.log',env=env,stages=['component timer'])
    native=inspect(sdk,output/'libquactlize_ppu_prefill.so');atomic_json(output/'native.json',native)
    atomic_json(output/'timer.json',dict(command=cmd,source=sha(ROOT/'dev/prefill_dense/timer.cpp'),
        sha256=sha(output/'libdense-opt-timer.so')))
