"""Compile only the previously measured geometries; current production baseline."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import re
from dev.gemv_simt.build_q6_metadata import build as pair
from dev.gemv_simt.build_general import inputs
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.split_plan import QTYPES
from dev.gemv_simt.split_source import source
from dev.gemv_simt.q6_schedule_audit import instructions
from dev.strategy.journal import atomic_json


def build(sdk,output,object_only=False):
    output.mkdir(parents=True,exist_ok=False)
    def one(q):
        path=output/str(q)
        value=pair(sdk,path,object_only,generator=lambda c:source(q,c),source_inputs=inputs,
                   schema='simt-common-split-v1',record_contract=dict(qtype=q,configs=configs(q)))
        audit=[]
        for compute in ('f16','bf16'):
            bodies=instructions((path/f'{compute}.isa.txt').read_text())
            resources=(path/f'{compute}.resources.txt').read_text()
            for i in range(len(configs(q))):
                for mode in range(3):
                    name=f'split_candidate_{i}_{mode}'
                    symbol,=[s for s in bodies if f'{len(name)}{name}E' in s]
                    if not any(r[2].startswith('v.fma.f32') for r in bodies[symbol]):
                        raise ValueError('missing FP32 candidate arithmetic')
                    resource=resources.split(symbol+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
                    usage={k:int(re.search(k+r':(\d+)',resource)[1]) for k in
                           ('vreg_number','sreg_number','shared_memory_size','STACK SIZE')}
                    if usage['STACK SIZE']:raise ValueError('candidate stack/spill requires separate review')
                    audit.append(dict(compute=compute,config=i,mode=mode,symbol=symbol,resources=usage,
                        instruction_bytes_sha256=hashlib.sha256(b''.join(r[1] for r in bodies[symbol])).hexdigest(),
                        opcode_sites=dict(Counter(r[2].split()[0] for r in bodies[symbol]))))
        atomic_json(path/'native-audit.json',dict(kernels=audit,no_stack=True,
                    scope='STATIC_CODEGEN_NOT_DYNAMIC_COUNTERS_OR_SPEED'))
        return value
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(one,QTYPES))
    result=dict(schema='simt-common-split-v1',builds=values,object_only=object_only,production_changed=False)
    atomic_json(output/'manifest.json',result)
    return result
