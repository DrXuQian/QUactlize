"""Build only Q8, and prove both prior bodies unchanged under this compiler."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.build_general import inputs
from dev.gemv_simt.build_q6_metadata import build as build_pair
from dev.gemv_simt.general_reader import source as previous_source
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.q6_schedule_audit import instructions
from dev.gemv_simt.q8_dense_source import ARMS,source,symbol_name
from dev.strategy.journal import atomic_json


def native_audit(candidate,reference):
    rows=[]
    for compute in ('f16','bf16'):
        bodies=instructions((candidate/f'{compute}.isa.txt').read_text())
        old=instructions((reference/f'{compute}.isa.txt').read_text())
        resources=(candidate/f'{compute}.resources.txt').read_text()
        for config in range(len(configs(8))):
            for arm in range(len(ARMS)):
                name=symbol_name(config,arm)
                symbol,=[s for s in bodies if f'{len(name)}{name}E' in s]
                code=bodies[symbol]
                counts=Counter(r[2].split()[0] for r in code if r[2]!='s.nop')
                if not any(k.startswith('v.fma.f32') for k in counts):raise ValueError('missing Q8 FP32 body')
                resource=resources.split(symbol+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
                fields={k:int(re.search(k+r':(\d+)',resource)[1]) for k in ('vreg_number','sreg_number','shared_memory_size','STACK SIZE')}
                if fields['STACK SIZE']:raise ValueError('candidate stack; review before Box')
                raw=b''.join(row[1] for row in code)
                preserved=None
                if arm in (0,5):
                    prior,=[s for s in old if f'{len(name)}{name}E' in s]
                    preserved=raw==b''.join(row[1] for row in old[prior])
                    if not preserved:raise ValueError('prior control/candidate native bytes changed: '+name+'/'+compute)
                rows.append(dict(compute=compute,config=config,arm=arm,symbol=symbol,
                                 prior_native_bytes_unchanged=preserved,resources=fields,
                                 instruction_sha256=hashlib.sha256(raw).hexdigest(),
                                 non_nop_sites=sum(counts.values()),opcode_sites=dict(counts)))
    return dict(records=rows,scope='NATIVE_CODEGEN_NOT_DEVICE_NUMERICS_OR_SPEED',production_changed=False)


def build(sdk,output,object_only=False):
    output.mkdir(parents=True,exist_ok=False)
    def one(name,generator):
        return build_pair(sdk,output/name,object_only,
                          generator=lambda compute:generator(8,compute,configs(8)),source_inputs=inputs,
                          schema='q8-dense-ablation-v1',record_contract=dict(qtype=8,configs=configs(8)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        candidate=pool.submit(one,'candidate',source)
        reference=pool.submit(one,'reference',previous_source)
        result=dict(candidate=candidate.result(),reference=reference.result(),
                    object_only=object_only,production_changed=False)
    result['audit']=native_audit(output/'candidate',output/'reference')
    atomic_json(output/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-only',action='store_true');a=p.parse_args()
    build(a.sdk,a.output.resolve(),a.object_only)
