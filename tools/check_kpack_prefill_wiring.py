#!/usr/bin/env python3
"""Check the real dispatcher tickets against all 430 confirmed compositions.

Resource queries only: no arithmetic/timing pass is inferred from this check.
The subsequent actual llama A/B exercises compute, adapters and dynamic inputs.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.journal import atomic_json
from quactlize.dispatch.native import Dispatch,Request
from quactlize.runtime.native import SDK
from quactlize.runtime.compiler import sha
from tools.fit_prefill_confirmation import EVIDENCE
from tools.verify_kpack_dispatch import verify


def check(dispatch,row):
    w=row['workload'];recipe=row['recipe'];p=dict(recipe['parameters'])
    grouped=w['operation']=='indexed';compute=int(w['compute']=='bf16')
    mapping=0x51344b5034540001 if w['qtype']==12 else 0x514b504b54000001
    request=Request(1,C.sizeof(Request),w['qtype'],2 if grouped else 0,w['m']*w['topk'],
                    w['n'],w['k'],w['experts'],w['m'],mapping)
    selected=dispatch.prefill(request,compute,w['channels'])
    expected={'fq':0,'sf':1,'full':2}[recipe['route']]
    if (selected is None or selected.policy!=19 or selected.compute_type!=compute or
            selected.base.predicted or selected.base.route!=expected or
            selected.base.dequant_config!=p['dequant_config'] or selected.base.measured_tokens!=w['m']):
        raise ValueError('pure query differs from confirmed composition')
    result=dict(point=row['point'],status='PASS',route=recipe['route'],dequant=p['dequant_config'],policy=19)
    if recipe['kind']=='tc':
        request.route=(2 if grouped else 0)+expected
        ticket=dispatch.query_prefill(request,compute,w['channels'])
        parent=ticket.parent.decode()
        if (parent!=p['symbol'] or ticket.policy!=19 or ticket.split!=p['split'] or
                ticket.compute_units!=72 or ticket.device!=0 or not ticket.ticket or
                (ticket.grid!=0 if p['grid_mode']==0 else ticket.grid<=0)):
            raise ValueError('materialized TC ticket differs from confirmed recipe')
        result.update(parent=parent,build=ticket.build_key.decode(),split=ticket.split,
                      algorithm=ticket.algorithm,grid=ticket.grid,workspace_bytes=ticket.workspace_bytes,
                      shared_bytes=ticket.shared_bytes,compute_type=compute)
    else:
        result['provider_execution']='PENDING_LLAMA_AB'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','sdk','jit-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=verify(a.bundle,sdk=a.sdk)
    if m.get('prefill_confirmed',{}).get('sha256')!=sha(EVIDENCE):
        raise ValueError('dispatcher lacks the current confirmed evidence')
    sdk=SDK(a.sdk)  # Keep SDK handles alive through module destruction.
    d=Dispatch(a.bundle,jit=dict(python=sys.executable,helper=ROOT/'tools/kpack_jit.py',sdk=a.sdk,cache=a.jit_cache))
    results=[]
    try:
        for row in json.loads(EVIDENCE.read_text())['points']:
            try:result=check(d,row)
            except Exception as error:result=dict(point=row['point'],status='FAIL',error=str(error))
            results.append(result)
            atomic_json(a.output/(row['point']+'.json'),result)
            if len(results)%25==0 or result['status']!='PASS':
                print(f'PREFILL_WIRING_PROGRESS completed={len(results)}/430 failed={sum(r["status"]!="PASS" for r in results)}',flush=True)
    finally:d.close()
    failed=sum(r['status']!='PASS' for r in results)
    summary=dict(status='FAIL' if failed else 'PASS',points=len(results),failed=failed,
                 scope='ACTUAL_SELECTED_TICKETS_RESOURCE_ONLY',kernel_launches=0,
                 performance_admitted=False,device_arithmetic_admitted=False)
    atomic_json(a.output/'summary.json',summary)
    print('PREFILL_WIRING_DONE '+json.dumps(summary),flush=True)
    return int(bool(failed))


if __name__=='__main__':raise SystemExit(main())
