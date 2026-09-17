#!/usr/bin/env python3
"""Exact canonical repack and compact-output oracle before model execution."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_kpack_gate_up import Weights, Bench, raw_weight, planes, device_identity
from dev.gemv_simt.native import Runtime, Graph, checked
from quactlize.fusion.native import Library, integration_entries, Repack, MappedCall, Config
from tools.verify_kpack_dispatch import verify


def repack(rt,library,entries,w,merged):
    sources=[np.stack([raw_weight(w.q,w.n,w.k,17+side*419+e*73)[0]
                      for e in range(w.experts)]) for side in (0,1)]
    if merged:sources=[np.concatenate(sources,axis=1)]
    canonical=[]
    for source in sources:
        packed=[planes(raw,w.q) for raw in source]
        canonical.append({name:rt.upload(np.stack([p[name] for p in packed])) for name in ('low','units')})
    output={}
    for name,bits in (('low',8 if w.q==8 else 4),('units',0)):
        size=w.experts*2*w.n*w.k*bits//8 if bits else w.experts*2*w.n*(w.k//(32 if w.q==8 else 256))*(2 if w.q==8 else 16)
        pointer=rt.allocate(size+32);rt.fill(pointer,size+32);output[name]=pointer+16
    call=Repack(1,C.sizeof(Repack),w.q,w.n,w.k,w.experts,int(merged),
        canonical[0]['low'],canonical[0]['units'],None if merged else canonical[1]['low'],
        None if merged else canonical[1]['units'],output['low'],output['units'])
    checked(entries['repack'](C.byref(call),C.byref(w.layout),rt.stream),'canonical repack')
    rt.sync()
    for (allocation,expected),name in zip(w.guards,('low','units')):
        actual=rt.download(output[name]-16,expected.size+32)
        if not np.array_equal(actual[16:-16],expected) or not np.all(actual[:16]==0xa5) or not np.all(actual[-16:]==0xa5):
            raise ValueError('canonical-to-paired byte/guard mismatch: '+name)
    wrong=Repack.from_buffer_copy(call);wrong.low=wrong.gate_low
    if entries['repack'](C.byref(wrong),C.byref(w.layout),rt.stream)==0:
        raise ValueError('repack alias negative was accepted')
    return output|{'high':None}


def main(args):
    manifest=verify(args.bundle,sdk=args.sdk)
    if 'paired_gate_up' not in manifest:raise ValueError('missing paired package')
    args.output.mkdir(parents=True,exist_ok=False)
    rt=Runtime(args.sdk,'ppu');library=Library(args.bundle/'libquactlize_ppu_gate_up.so')
    entries=integration_entries(library)
    result=dict(status='INCOMPLETE',device=device_identity(rt),records=[],repack_cases=0,replays=0,negatives=0)
    try:
        for q in (8,12):
            base=len(rt.allocations);w=Weights(rt,library,q,256,512,1 if q==8 else 8)
            for merged in (False,True):
                paired=repack(rt,library,entries,w,merged);result['repack_cases']+=1;result['negatives']+=1
                for tokens in range(1,9):
                    for output in ((1,) if q==8 else (1,2)):
                        bench=Bench(rt,library,w,0 if q==8 else 2,tokens,int(q==12),1,output,int(q==12))
                        mapping=rt.upload(np.arange(bench.rows,dtype='i4')) if q==12 else None
                        status=rt.upload(np.array([0],dtype='i4'))
                        for backend,split in ((0,1),(1,1),(1,2)):
                            config=Config(backend,split,8 if backend else 0,0 if backend else 8)
                            typed=MappedCall(bench.call,mapping,status)
                            typed.call.input.call.low=paired['low'];typed.call.input.call.units=paired['units']
                            invoke=lambda:entries['run'](C.byref(typed),C.byref(config),C.byref(w.layout))
                            graph=None
                            try:
                                for repeat in range(2):
                                    bench.update(repeat)
                                    if mapping:
                                        perm=np.argsort(bench.owners,kind='stable').astype('i4')
                                        if repeat:perm=perm[::-1].copy()
                                        rt.copy(mapping,perm);bench.gold=bench.gold[perm]
                                    bench.poison();checked(invoke(),'mapped eager fusion');rt.sync()
                                    error=bench.check(split)
                                    if graph is None:graph=Graph(rt,[invoke])
                                    bench.poison();checked(rt.GraphLaunch(graph.instance,rt.stream),'mapped replay');rt.sync()
                                    error=max(error,bench.check(split));result['replays']+=1
                                rt.copy(status,np.array([1],dtype='i4'));checked(invoke(),'invalid routing poison');rt.sync()
                                image=rt.download(bench.output+16,bench.rows*(w.n+8)*(4 if output==1 else 2))
                                values=image.view('u4' if output==1 else 'u2').reshape(bench.rows,w.n+8)[:,:w.n]
                                # Quiet NaNs, not stale finite values; independent of payload bits.
                                mask=0x7f800000 if output==1 else 0x7f80
                                mantissa=0x007fffff if output==1 else 0x007f
                                if not np.all((values&mask)==mask) or not np.all((values&mantissa)!=0):
                                    raise ValueError('invalid routing status was not propagated')
                                result['negatives']+=1
                                rt.copy(status,np.array([0],dtype='i4'))
                                result['records'].append(dict(q=q,merged=merged,tokens=tokens,output=output,backend=backend,split=split,error=error))
                            finally:
                                if graph:graph.close()
                        rt.release_after(bench.start)
                print(f'GATE_UP_INTEGRATION_PROGRESS q={q} merged={int(merged)} records={len(result["records"])}',flush=True)
            rt.release_after(base)
        if len(result['records'])!=144 or result['replays']!=288 or result['negatives']!=148:
            raise ValueError('incomplete integration denominator')
        result['status']='PASS'
    finally:
        rt.close();(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GATE_UP_INTEGRATION PASS repack=4 cells=144 replays=288 negatives=148',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','sdk','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
