"""Native gate for signed-address candidates and unchanged prior Q8 controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.build_q6_metadata import build as pair
from dev.gemv_simt.build_general import inputs
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.q6_schedule_audit import instructions
from dev.gemv_simt.q8_dense_source import source as old_source, symbol_name as old_symbol
from dev.gemv_simt.q8_address_source import source, symbol_name, ARMS
from dev.strategy.journal import atomic_json


def audit(candidate,reference):
    rows=[]
    for compute in ('f16','bf16'):
        old=instructions((reference/f'{compute}.isa.txt').read_text())
        new=instructions((candidate/f'{compute}.isa.txt').read_text())
        resources=(candidate/f'{compute}.resources.txt').read_text()
        for config in range(2):
            for arm in range(len(ARMS)):
                name=symbol_name(config,arm)
                symbol,=[s for s in new if f'{len(name)}{name}E' in s]
                code=new[symbol];raw=b''.join(r[1] for r in code)
                fields=resources.split(symbol+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
                usage={k:int(re.search(k+r':(\d+)',fields)[1]) for k in
                       ('vreg_number','sreg_number','shared_memory_size','STACK SIZE')}
                if usage['STACK SIZE']:raise ValueError('Q8 candidate stack/spill requires review')
                if not any(r[2].startswith('v.fma.f32') for r in code):raise ValueError('missing FP32 math')
                preserved=None
                if arm in (0,1,2):
                    before=old_symbol(config,{0:0,1:4,2:5}[arm])
                    previous,=[s for s in old if f'{len(before)}{before}E' in s]
                    preserved=raw==b''.join(r[1] for r in old[previous])
                    if not preserved:raise ValueError('prior Q8 native control changed: '+name)
                rows.append(dict(compute=compute,config=config,arm=arm,symbol=symbol,
                    resources=usage,prior_native_bytes_unchanged=preserved,
                    instruction_sha256=hashlib.sha256(raw).hexdigest()))
    return rows


def build(sdk,output,object_only=False):
    output.mkdir(parents=True,exist_ok=False)
    def one(name,generator):
        return pair(sdk,output/name,object_only,generator=lambda c:generator(8,c,configs(8)),
            source_inputs=inputs,schema='q8-address-window-v1',record_contract=dict(qtype=8,configs=configs(8)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        new=pool.submit(one,'candidate',source);old=pool.submit(one,'reference',old_source)
        result=dict(candidate=new.result(),reference=old.result(),object_only=object_only,
                    production_changed=False)
    result['audit']=audit(output/'candidate',output/'reference')
    atomic_json(output/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-only',action='store_true');a=p.parse_args()
    build(a.sdk,a.output,a.object_only)
