#!/usr/bin/env python3
"""Compile bounded reader pairs for each qtype/precision, no TC rebuild."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt import build_q6_metadata as base
from dev.gemv_simt.general_reader import QTYPES,source
from dev.gemv_simt.general_plan import configs
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def inputs():
    result=base.inputs()
    for p in sorted((ROOT/'dev/gemv_simt').glob('*')):
        if p.suffix in ('.py','.cuh','.hpp'):
            result[str(p.relative_to(ROOT))]=sha(p)
    return result


def build(sdk,output,object_only=False,qtypes=QTYPES):
    output.mkdir(parents=True,exist_ok=False)
    def one(q):
        result=base.build(sdk,output/str(q),object_only,
                          generator=lambda compute:source(q,compute,configs(q)),source_inputs=inputs,
                          schema='simt-general-readers-v1',record_contract=dict(qtype=q,configs=configs(q)))
        from dev.gemv_simt.general_audit import audit
        atomic_json(output/str(q)/'native-audit.json',[audit(output/str(q),c,configs(q)) for c in ('f16','bf16')])
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        result=list(pool.map(one,qtypes))
    manifest=dict(schema='simt-general-readers-v1',builds=result,object_only=object_only,production_changed=False)
    atomic_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-only',action='store_true');p.add_argument('--qtypes',default=','.join(map(str,QTYPES)))
    a=p.parse_args();qs=tuple(map(int,a.qtypes.split(',')))
    if not qs or len(set(qs))!=len(qs) or any(q not in QTYPES for q in qs):p.error('invalid qtypes')
    build(a.sdk,a.output.resolve(),a.object_only,qs)
