"""Compile the production common reader and compare its native body to the measured producer."""
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
from dev.gemv_simt.split_source import source as measured_source
from dev.gemv_simt.q6_schedule_audit import instructions
from dev.strategy.journal import atomic_json


def production_source(q,compute):
    c=int(compute=='bf16')
    result='#include "quactlize/execution/simt_common.cuh"\n'
    result+='extern "C" int common_admission_run(qkg_simt_call_v2 const* d,int config,int split) {\n'
    result+=f'    if(d->compute_type!={c})return QKG_INVALID;\n    switch(config) {{\n'
    for i,f in enumerate(configs(q)):
        result+=f'    case {i}:return quactlize::execution::simt::common_reader_launch_typed<{q},{f["columns"]},{f["warps"]},{f["values"]},{c}>(d->call,split);\n'
    return result+'    default:return QKG_INVALID;\n    }\n}\n'


def audit(old,new,q):
    rows=[]
    for compute in ('f16','bf16'):
        before=instructions((old/f'{compute}.isa.txt').read_text())
        after=instructions((new/f'{compute}.isa.txt').read_text())
        resource=(new/f'{compute}.resources.txt').read_text()
        for i,f in enumerate(configs(q)):
            for mode in range(3):
                name=f'split_candidate_{i}_{mode}'
                original,=[s for s in before if f'{len(name)}{name}E' in s]
                suffix=f'common_readerILi{mode}ELi{q}ELi{f["columns"]}ELi{f["warps"]}ELi{f["values"]}ELi{int(compute=="bf16")}EE'
                target,=[s for s in after if suffix in s]
                raw=lambda code:b''.join(r[1] for r in code)
                if raw(before[original])!=raw(after[target]):
                    raise ValueError(f'production native body differs: q{q}/{compute}/c{i}/mode{mode}')
                fields=resource.split(target+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
                stack=int(re.search(r'STACK SIZE:(\d+)',fields)[1])
                if stack:raise ValueError('unexpected production spill')
                rows.append(dict(q=q,compute=compute,config=i,mode=mode,symbol=target,
                    sha256=hashlib.sha256(raw(after[target])).hexdigest(),stack=stack,
                    measured_native_bytes_equal=True))
    return rows


def build(sdk,output,object_only=False):
    output.mkdir(parents=True,exist_ok=False)
    def one(q):
        root=output/str(q);root.mkdir()
        old=pair(sdk,root/'measured',object_only,generator=lambda c:measured_source(q,c),source_inputs=inputs)
        new=pair(sdk,root/'production',object_only,generator=lambda c:production_source(q,c),source_inputs=inputs)
        return dict(q=q,measured=old,production=new,audit=audit(root/'measured',root/'production',q))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(one,range(10,15)))
    result=dict(schema='simt-common-native-port-v1',builds=rows,object_only=object_only,
                scope='NATIVE_CODE_IDENTITY_NOT_NEW_DEVICE_OR_MODEL_TEST')
    atomic_json(output/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--object-only',action='store_true');a=p.parse_args()
    build(a.sdk,a.output,a.object_only)
