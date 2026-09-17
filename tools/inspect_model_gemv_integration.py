#!/usr/bin/env python3
"""Inspect only the seven integrated reader specializations, not a full ISA dump."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from tools.build_kpack_dequant import resource_usage

SYMBOLS={
    'q4-paired-routed':('gate_up','quactlize::fusion::simt_gate_up_model<12,1,1,8>'),
    'q5-routed-down':('execution','quactlize::execution::simt::register_reuse_model<13,1,3,4,2,8,1,3>'),
    'q8-paired-shared':('gate_up','quactlize::fusion::simt_gate_up_model<8,1,0,8>'),
    'q8-shared-down':('execution','quactlize::execution::simt::q8_vector::kernel_s1<1,0,1,4,2,4,true>'),
    'q8-ssm-out':('execution','quactlize::execution::simt::q8_vector::kernel_model<1,0,1,8,4,4,false,2048,4096,8>'),
    'q8-qkv':('execution','quactlize::execution::simt::q8_vector::kernel_model<1,0,1,8,4,4,true,8192,2048,1>'),
    'q8-attn-gate':('execution','quactlize::execution::simt::q8_vector::kernel_s1<1,0,1,4,8,4,false>'),
}


def operations(text):
    return Counter(re.findall(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}[ \t]+)+([a-z][\w.]+)\b',text,re.MULTILINE))


def inspect(args):
    args.output.mkdir(parents=True,exist_ok=False)
    names={};rows=[]
    for lib in ('execution','gate_up'):
        path=args.bundle/f'libquactlize_ppu_{lib}.so'
        text=subprocess.check_output([args.sdk/'bin/hgobjdump','--dump-resource-usage=all',path],text=True)
        mangled=sorted(set(re.findall(r'^Func \d+ (\S+) RESOURCE INFO:',text,re.MULTILINE)))
        demangled=subprocess.check_output(['c++filt'],input='\n'.join(mangled),text=True).splitlines()
        names[lib]=dict(zip(mangled,demangled))
    old=json.loads((args.reference/'native-inspection.json').read_text())
    evidence=json.loads((ROOT/'docs/measurements/model_gemv_20260917.json').read_text())
    for point,(lib,pattern) in SYMBOLS.items():
        matches=[s for s,n in names[lib].items() if pattern+'(' in re.sub(r'\s+','',n)]
        if len(matches)!=1:raise ValueError('missing/duplicate integrated symbol: '+point)
        symbol=matches[0];path=args.bundle/f'libquactlize_ppu_{lib}.so'
        resource_text=subprocess.check_output([args.sdk/'bin/hgobjdump','--dump-resource-usage='+symbol,path],text=True)
        resource=resource_usage(resource_text)[symbol]
        isa=subprocess.check_output([args.sdk/'bin/hgobjdump','--dump-isa','--dump-function='+symbol,path],text=True)
        ops=operations(isa)
        if not any(op.startswith('v.fma.f32') for op in ops) or not any(op.startswith('vmem.ld.b32x') for op in ops):
            raise ValueError('integrated fast dequant/F32 FMA/vector load missing: '+point)
        measured=next(r for r in evidence['records'] if r['point']==point)
        baseline=old[point][measured['arm']]
        reference_isa=subprocess.check_output([args.sdk/'bin/hgobjdump','--dump-isa','--dump-function='+baseline['symbol'],args.reference/(point+'.so')],text=True)
        reference_ops=operations(reference_isa)
        row=dict(point=point,library=path.name,library_sha256=sha(path),symbol=symbol,
                 demangled=names[lib][symbol],resources=resource,
                 admitted_resources=baseline['resources'],operations=dict(sorted(ops.items())),
                 admitted_operations=dict(sorted(reference_ops.items())),
                 opcode_counts_match=ops==reference_ops,
                 resource_match=resource==baseline['resources'],
                 isa_sha256=hashlib.sha256(isa.encode()).hexdigest())
        (args.output/(point+'.isa.txt')).write_text(isa)
        rows.append(row)
        print('MODEL_GEMV_INTEGRATED_ISA '+json.dumps({k:row[k] for k in ('point','resources','admitted_resources','opcode_counts_match')}),flush=True)
    result=dict(status='PASS',rows=rows,scope='STATIC_ISA_NOT_DEVICE_ADMISSION')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    inspect(parser.parse_args())
