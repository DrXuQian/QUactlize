#!/usr/bin/env python3
"""Native ISA identity of the promoted reader and retained paired controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.inspect_selected_decode import instruction_bytes
from tools.build_kpack_dequant import resource_usage
from quactlize.runtime.compiler import sha


def symbols(inspector,library):
    text=subprocess.check_output([inspector,'--list-elf',library],text=True)
    names=re.findall(r'^Func \d+:\s+(\S+)\s*$',text,re.MULTILINE)
    decoded=subprocess.check_output(['c++filt'],input='\n'.join(names),text=True).splitlines()
    return dict(zip(names,[re.sub(r'\s+','',n) for n in decoded]))


def inspect(args):
    inspector=args.sdk/'bin/hgobjdump'
    current=args.bundle/'libquactlize_ppu_gate_up.so'
    control=args.bundle/'paired-control/libquactlize_ppu_gate_up.so'
    reference=args.reference/'tp2-chain.so'
    frozen=json.loads((args.reference/'manifest.json').read_text())
    if sha(reference)!=frozen['payloads']['tp2-chain.so']:raise ValueError('measured candidate image differs')
    tests=[('q4-paired16',reference,'point_12_1024_3072_arm_1(',
            'simt_gate_up_paired16<4,4,3,1024,3072>('),
           ('q4-paired32',control,'simt_gate_up_model<12,1,1,8,1024,2048>(',
            'simt_gate_up_model<12,1,1,8,1024,2048>('),
           ('q8-paired16',control,'simt_gate_up_q8_tile16(','simt_gate_up_q8_tile16(')]
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name,before,old_pattern,new_pattern in tests:
        observations=[]
        for label,path,pattern in (('reference',before,old_pattern),('candidate',current,new_pattern)):
            matches=[s for s,n in symbols(inspector,path).items() if pattern in n]
            if len(matches)!=1:raise ValueError('missing/duplicate recipe: '+name+'/'+label)
            symbol=matches[0]
            isa=subprocess.check_output([inspector,'--dump-isa','--dump-function='+symbol,path],text=True)
            raw=instruction_bytes(isa,symbol)
            resource=resource_usage(subprocess.check_output([inspector,'--dump-resource-usage='+symbol,path],text=True))[symbol]
            (args.output/(name+'-'+label+'.isa.txt')).write_text(isa)
            observations.append(dict(symbol=symbol,library_sha256=sha(path),resources=resource,
                instruction_sha256=hashlib.sha256(raw).hexdigest(),instruction_bytes=len(raw)))
        a,b=observations
        record=dict(name=name,reference=a,candidate=b,
            instructions_identical=a['instruction_sha256']==b['instruction_sha256'],
            resources_identical=a['resources']==b['resources'])
        records.append(record)
        print('PAIRED_RECIPE_ISA '+json.dumps(record),flush=True)
    result=dict(scope='STATIC_ISA_NOT_DEVICE_ADMISSION',records=records,
        status='PASS' if all(r['instructions_identical'] and r['resources_identical'] for r in records) else 'REVIEW_REQUIRED')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='PASS':raise ValueError('native implementation changed; review before handoff')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('sdk','bundle','reference','output'):parser.add_argument('--'+field,type=Path,required=True)
    inspect(parser.parse_args())
