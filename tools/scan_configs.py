#!/usr/bin/env python3
"""Benchmark an explicit, bounded list of resident K-pack TC/SIMT configurations.

Each candidate runs in a separate process. Only numerically valid candidates
receive timings. Results never modify the production selection tables.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.native import SDK, Call, Module, Recipe, Resources, checked
from quactlize.runtime.compiler import Compiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.dispatch.native import DeviceCall, GroupedMetadata
from quactlize.execution.native import Call as SimtCall, SimtCallV2, SimtConfig, Sizes as SimtSizes, bind_simt_compute
from quactlize.packing.native import Sizes, canonical_arrangement, bind as bind_pack


class Fixture:
    """Finite GGUF blocks, independently decoded by the official gguf package."""
    def __init__(self, sdk, bundle, q, n, k, m, experts, compute, input_scale=1):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        self.sdk, self.allocations = sdk, []
        self.q,self.n,self.k,self.m,self.experts,self.compute=q,n,k,m,experts,compute
        self.arr = canonical_arrangement(q)
        block, width, d_offset, min_offset = {
            8:(32,34,0,None),10:(256,84,80,82),11:(256,110,108,None),
            12:(256,144,0,2),13:(256,176,0,2),14:(256,210,208,None)}[q]
        if min(n,k,m,experts)<=0 or n%256 or k%512:
            raise ValueError('scan fixture needs positive dimensions, N%256=0 and K%512=0')
        rng=np.random.default_rng(89173+q)
        raw=rng.integers(0,256,(experts,n,k//block,width),dtype=np.uint8)
        d=np.asarray([1/256],dtype='<f2').view('u1')
        raw[...,d_offset:d_offset+2]=d
        if min_offset is not None:raw[...,min_offset:min_offset+2]=d
        self.raw=raw
        a=(rng.normal(0,.125,(m,k))*input_scale).astype('f4')
        # Match the public activation compute contract before the oracle dot.
        if compute:
            bits=a.view('u4'); rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
            a=rounded.view('f4')
        else:a=a.astype('f2').astype('f4')
        self.a_float=a
        self.rows=np.bincount(np.arange(m)%experts,minlength=experts).astype('i4')
        self.offsets=np.concatenate(([0],np.cumsum(self.rows))).astype('i4')
        self.gold=np.empty((m,n),dtype='f4')
        for e in range(experts):
            begin,end=self.offsets[e:e+2]
            if begin==end:continue
            w=dequantize(raw[e].reshape(n,-1),GGMLQuantizationType(q)).reshape(n,k)
            self.gold[begin:end]=a[begin:end]@w.T
        lib=C.CDLL(str(Path(bundle)/'pack/libquactlize_ppu_pack.so'),mode=C.RTLD_LOCAL)
        _,query,pack=bind_pack(lib)
        sizes=Sizes()
        checked(query(n,k,experts,q,C.byref(self.arr),C.byref(sizes)),'packing sizes')
        self.low=self.allocate(sizes.low_bytes);self.high=self.allocate(sizes.high_bytes)
        self.units=self.allocate(sizes.units_bytes)
        raw_ptr=self.upload(raw)
        checked(pack(raw_ptr,self.low,self.high,self.units,n,k,experts,q,C.byref(self.arr),None),'pack')
        sdk.synchronize(None)
        self.sizes=sizes
        self.a=self.upload((a.view('u4')>>16).astype('u2') if compute else a.astype('f2'))
        self.a_f32=self.upload(a)
        self.offsets_device=self.upload(self.offsets)
        self.output=self.allocate(m*n*4)

    def allocate(self, size):
        if not size:return None
        pointer=self.sdk.allocate(int(size));self.allocations.append(pointer);return pointer

    def upload(self, array):
        pointer=self.sdk.upload(array.tobytes());self.allocations.append(pointer);return pointer

    def close(self):
        self.sdk.synchronize(None)
        for p in reversed(self.allocations):self.sdk.free(p)
        self.allocations=[]

    def check(self, dtype):
        import numpy as np
        self.sdk.synchronize(None)
        raw=self.sdk.download(self.output,self.m*self.n*(4 if dtype=='f32' else 2))
        got=np.frombuffer(raw,dtype='<f4' if dtype=='f32' else '<u2' if dtype=='bf16' else '<f2')
        if dtype=='bf16':got=(got.astype('u4')<<16).view('f4')
        got=got.astype('f4').reshape(self.gold.shape)
        if not np.isfinite(got).all():raise ValueError('nonfinite output')
        error=float(np.linalg.norm(got-self.gold)/max(np.linalg.norm(self.gold),1e-30))
        limit=.02 if self.compute else .005
        if error>=limit:raise ValueError(f'independent GGUF relative L2 {error} >= {limit}')
        return error


def simt_runner(f, bundle, config):
    lib=C.CDLL(str(Path(bundle)/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
    query,run=bind_simt_compute(lib)
    c=SimtCall(1,C.sizeof(SimtCall),f.q,f.n,f.k,f.experts,f.m,int(f.experts>1),1,1,1,
               f.k,f.k,1,f.n,f.a_f32,f.low,f.high,f.units,
               f.offsets_device if f.experts>1 else None,None,f.output,None,0,None)
    typed=SimtCallV2(c,f.compute);cfg=SimtConfig(**config);sizes=SimtSizes()
    checked(query(C.byref(typed),C.byref(cfg),C.byref(f.arr),C.byref(sizes)),'SIMT query')
    typed.call.workspace=f.allocate(sizes.workspace_bytes)
    typed.call.workspace_bytes=sizes.workspace_bytes
    return lambda:checked(run(C.byref(typed),C.byref(cfg),C.byref(f.arr)),'SIMT run'),lambda:None,'f32'


class PreparedTC:
    def __init__(self,module,handle):
        self.module,self.handle=module,handle

    def __call__(self):
        checked(self.module.run(self.handle,None),'TC run')

    def sample(self,repeats):
        value=C.c_double()
        checked(self.module.measure(self.handle,None,repeats,C.byref(value)),'TC measure')
        return value.value


def tc_runner(f, sdk_path, cache, parent, recipe, execution, module_record=None):
    if parent['qtype']!=f.q or parent['route'].endswith('grouped')!=(f.experts>1):
        raise ValueError('parent format/operator differs from fixture')
    if f.compute and f.experts==1:
        raise ValueError('this resident TC scan accepts F16 dense or typed F16/BF16 grouped; dense BF16 uses decode/api.h')
    if module_record is None:
        compiler=(GroupedComputeCompiler(sdk_path,cache,compute_type='bf16') if f.compute else Compiler(sdk_path,cache))
        module_record=compiler.build(parent)
    if module_record['parent']!=parent or module_record['identity'].get('compute_type','f16')!=('bf16' if f.compute else 'f16'):
        raise ValueError('compiled parent/compute differs from the scan request')
    module=Module(module_record);device=module.device_identity()
    scale,zero=f.units,None
    if parent['route'].startswith('sf') and f.q!=8:
        plane=f.experts*(f.k//f.arr.group_size)*f.n*2
        scale,zero=f.allocate(plane),f.allocate(plane)
        lib=C.CDLL(str(Path(execution)/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
        prepass=lib.quactlize_kpack_sf_prepare_v2
        prepass.argtypes=[C.c_int]*4+[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p,C.c_uint64,C.POINTER(type(f.arr)),C.c_int,C.c_void_p]
        prepass.restype=C.c_int
        checked(prepass(f.q,f.n,f.k,f.experts,f.units,f.sizes.units_bytes,scale,zero,plane,C.byref(f.arr),f.compute,None),'SF prepass')
        f.sdk.synchronize(None)
    c=Call(1,C.sizeof(Call),f.m,f.n,f.k,f.experts,f.arr.group_size,device['ordinal'],device['compute_units'],
           f.arr.mapping_id,f.a,f.low,f.high,scale,zero,f.output,None,None,
           f.offsets_device if f.experts>1 else None,None,0,None)
    r=Recipe(1,C.sizeof(Recipe),**recipe);resources=Resources();handle=C.c_void_p()
    if f.experts>1:
        dc=DeviceCall(2,C.sizeof(DeviceCall),c,int(f.rows.max()),0)
        typed=GroupedMetadata(4,C.sizeof(GroupedMetadata),dc,f.compute,f.compute if f.q!=8 else 0)
        query=module.lib.quactlize_kpack_grouped_query_v4
        prepare=module.lib.quactlize_kpack_grouped_prepare_v4
        query.argtypes=[C.POINTER(GroupedMetadata),C.POINTER(Recipe),C.POINTER(Resources)];query.restype=C.c_int
        prepare.argtypes=[C.POINTER(GroupedMetadata),C.POINTER(Recipe),C.POINTER(C.c_void_p)];prepare.restype=C.c_int
        checked(query(C.byref(typed),C.byref(r),C.byref(resources)),'TC query')
        typed.device_call.call.workspace=f.allocate(resources.workspace_bytes)
        typed.device_call.call.workspace_bytes=resources.workspace_bytes
        checked(prepare(C.byref(typed),C.byref(r),C.byref(handle)),'TC prepare')
    else:
        checked(module.query(C.byref(c),C.byref(r),C.byref(resources)),'TC query')
        c.workspace=f.allocate(resources.workspace_bytes);c.workspace_bytes=resources.workspace_bytes
        checked(module.prepare(C.byref(c),C.byref(r),C.byref(handle)),'TC prepare')
    return PreparedTC(module,handle),lambda:module.destroy(handle),'bf16' if f.compute else 'f16'


def time_entries(sdk, run, samples):
    bindings={
        'hggcEventCreate':[C.POINTER(C.c_void_p)],'hggcEventRecord':[C.c_void_p,C.c_void_p],
        'hggcEventSynchronize':[C.c_void_p],'hggcEventDestroy':[C.c_void_p],
        'hggcEventElapsedTime':[C.POINTER(C.c_float),C.c_void_p,C.c_void_p]}
    # Calls use the default stream, so time each complete C entry directly.
    # No graph capture is attempted on that stream; this includes launch gaps.
    for name,args in bindings.items():
        fn=getattr(sdk.lib,name);fn.argtypes=args;fn.restype=C.c_int
    start,end=C.c_void_p(),C.c_void_p()
    checked(sdk.lib.hggcEventCreate(C.byref(start)),'event')
    checked(sdk.lib.hggcEventCreate(C.byref(end)),'event')
    try:
        for _ in range(3):run()
        sdk.synchronize(None)
        values=[]
        for _ in range(samples):
            checked(sdk.lib.hggcEventRecord(start,None),'event record')
            run()
            checked(sdk.lib.hggcEventRecord(end,None),'event record')
            checked(sdk.lib.hggcEventSynchronize(end),'event wait')
            elapsed=C.c_float()
            checked(sdk.lib.hggcEventElapsedTime(C.byref(elapsed),start,end),'elapsed')
            values.append(elapsed.value*1000)
        return values
    finally:
        checked(sdk.lib.hggcEventDestroy(start),'event destroy')
        checked(sdk.lib.hggcEventDestroy(end),'event destroy')


def child(args, candidate):
    from tools.verify_kpack_dispatch import verify
    verify(args.bundle,sdk=args.sdk)
    sdk=SDK(args.sdk);f=None;destroy=lambda:None
    try:
        f=Fixture(sdk,args.bundle,args.qtype,args.n,args.k,args.m,args.experts,int(args.compute=='bf16'))
        if candidate['kind']=='simt':run,destroy,dtype=simt_runner(f,args.bundle,candidate['config'])
        elif candidate['kind']=='tc':run,destroy,dtype=tc_runner(f,args.sdk,args.cache,candidate['parent'],candidate['recipe'],args.bundle)
        else:raise ValueError('candidate kind must be tc or simt')
        sdk.fill(f.output,0xff,args.m*args.n*4)
        run();error=f.check(dtype)
        values=time_entries(sdk,run,args.samples)
        error=max(error,f.check(dtype))
        return dict(status='PASS',relative_l2=error,samples_us=values,median_us=statistics.median(values),
                    scope='RESIDENT_FULL_ENTRY_EVENT_TIME_INCLUDES_LAUNCH_GAPS_EXCLUDES_SF_PREPASS',cache='WARM')
    finally:
        if f is not None:
            sdk.synchronize(None);destroy();f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','cache','candidates','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--qtype',type=int,choices=(8,10,11,12,13,14),required=True)
    for name in ('n','k','m'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--experts',type=int,default=1)
    p.add_argument('--compute',choices=('f16','bf16'),default='f16')
    p.add_argument('--samples',type=int,default=11)
    p.add_argument('--candidate-index',type=int,help=argparse.SUPPRESS)
    a=p.parse_args();items=json.loads(a.candidates.read_text())
    if not isinstance(items,list) or not 1<=len(items)<=64 or not 3<=a.samples<=101:
        p.error('provide 1..64 explicit candidates and 3..101 samples')
    if a.candidate_index is not None:
        try:result=child(a,items[a.candidate_index])
        except Exception as error:result=dict(status='FAIL',error=str(error))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        return int(result['status']!='PASS')
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for i,candidate in enumerate(items):
        destination=a.output/f'candidate-{i}.json'
        command=[sys.executable,str(Path(__file__).resolve())]
        for name in ('sdk','bundle','cache','candidates','qtype','n','k','m','experts','compute','samples'):
            command += ['--'+name,str(getattr(a,name))]
        command += ['--output',str(destination),'--candidate-index',str(i)]
        with (a.output/f'candidate-{i}.log').open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        result=json.loads(destination.read_text()) if destination.is_file() else dict(status='FAIL',error=f'child rc={rc}')
        results.append(dict(index=i,candidate=candidate,**result))
        print(f"SCAN completed={i+1}/{len(items)} status={result['status']}",flush=True)
        (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    winners=[r for r in results if r['status']=='PASS']
    if winners:
        winner=min(winners,key=lambda r:r['median_us'])
        print(f"SCAN winner={winner['index']} median_us={winner['median_us']:.3f} cache=WARM production_policy=UNCHANGED")
    return int(len(winners)!=len(items))


if __name__=='__main__':
    raise SystemExit(main())
