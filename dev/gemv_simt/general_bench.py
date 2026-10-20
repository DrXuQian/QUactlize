"""Fresh small-M contexts, fixed-format weights and isolated producer timing."""
import ctypes as C
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from dev.gemv_simt.general_fixture import Weights,grouped
from dev.smallm_closure.bench import Buffer
from dev.bf16_compute.fixture import compare
from dev.strategy.indexed import point,RuntimeView
from dev.strategy.cold_decode import Ring
from dev.strategy.isolated import Captured
from dev.strategy.components import statistics_checked
from quactlize.execution.native import Call,SimtCallV2,Arrangement,arrangement
from quactlize.runtime.native import checked
from tools.run_kpack_gemv_gate import Resources
from tools.run_q6_metadata import raw_equal


class Context:
    def __init__(self,sdk,ring,w):
        self.sdk,self.ring,self.w=sdk,ring,w;self.r=Resources(sdk)
        try:
            self.p=point(w);self.p['mode']={'dense':0,'grouped':1,'indexed':2}[w.operation]
            self.rows=w.m*(1 if w.operation=='dense' else w.topk)
            data=self.make_data(0)
            self.a=Buffer(self.r,data['a'].nbytes)
            self.ids=Buffer(self.r,data['ids'].nbytes) if data['ids'] is not None else None
            self.offsets=Buffer(self.r,data['offsets'].nbytes) if 'offsets' in data else None
            self.out=Buffer(self.r,self.rows*w.n*4)
            self.arr=arrangement(w.qtype);self.update(0)
        except BaseException:
            self.close();raise

    def make_data(self,repeat):
        data=self.ring.w.inputs(self.p,repeat)
        if self.w.operation!='dense' and len(np.unique(data['owners']))<self.w.topk:
            raise ValueError('fixture violates active-expert cold-ring lower bound')
        return grouped(data,self.w.experts,self.w.channels,self.w.topk) if self.w.operation=='grouped' else data

    def update(self,repeat,large=False):
        self.data=self.make_data(repeat)
        if large:
            from dev.strategy.pilot_native import scale_activation_data
            scale_activation_data(self.data,2**20)
            if np.max(np.abs(self.data['a']))<=65504:raise ValueError('missing BF16 range coverage')
        self.a.upload(self.data['a'])
        if self.ids:self.ids.upload(self.data['ids'])
        if self.offsets:self.offsets.upload(self.data['offsets'])

    def check(self):
        self.sdk.synchronize(self.r.stream)
        for b in (self.a,self.ids,self.offsets):
            if b:b.guard()
        return compare(self.out.read((self.rows,self.w.n)),self.data['gold'],self.data['denom'])

    def close(self):self.r.close()


class Kernel:
    def __init__(self,b,library,*,entry='general_reader_run'):
        self.b=b;self.lib=C.CDLL(str(library),mode=C.RTLD_LOCAL)
        self.fn=getattr(self.lib,entry)
        self.fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.c_int];self.fn.restype=C.c_int
        w=b.w
        call=Call(version=1,size=C.sizeof(Call),qtype=w.qtype,n=w.n,k=w.k,experts=w.experts,
                  rows=b.rows,mode=b.p['mode'],input_type=1,
                  channels=w.channels if w.operation=='indexed' else 1,topk=w.topk if w.operation=='indexed' else 1,
                  a_row_stride=w.k,a_token_stride=w.channels*w.k,ids_stride=w.topk,out_row_stride=w.n,
                  a=b.a.ptr,ids=b.ids.ptr if b.ids else None,offsets=b.offsets.ptr if b.offsets else None,
                  output=b.out.ptr,stream=b.r.stream.value)
        self.calls=[]
        for plane in b.ring.planes:
            d=SimtCallV2(Call.from_buffer_copy(call),int(w.compute=='bf16'))
            for name,ptr in plane.items():setattr(d.call,name,ptr)
            self.calls.append(d)

    def launch(self,config,arm,copy=0):
        return self.fn(C.byref(self.calls[copy]),C.byref(self.b.arr),config,arm)


def single_copy(sdk,weights,cleanup):
    r=Resources(sdk);cleanup.callback(r.close)
    buffers={name:Buffer(r,v.nbytes) for name,v in weights.planes.items() if v.nbytes}
    for name,b in buffers.items():b.upload(weights.planes[name])
    planes={name:buffers[name].ptr if name in buffers else None for name in weights.planes}
    objects={name:buffers[name] for name in buffers}
    return SimpleNamespace(w=weights,r=r,planes=[planes],objects=[objects],buffers=list(buffers.values()),
                           copies=1,receipt=dict(copies=1,cache='CORRECTNESS_OR_DIRECT_ACU_NOT_COLD_TIMING',fixture=weights.record))


def numerical(core,config):
    b=core.b;proofs=[]
    for repeat in (0,1,2):
        b.update(repeat,large=repeat==2 and b.w.compute=='bf16')
        for copy in sorted({0,b.ring.copies-1}):
            outputs=[]
            for arm in (0,1):
                b.out.poison();checked(core.launch(config,arm,copy),'general numerical')
                proofs.append(b.check());outputs.append(b.out.read((b.rows,b.w.n)).copy())
            raw_equal(*outputs)
    b.update(0);b.out.poison()
    try:b.check()
    except ValueError as e:
        if not str(e).startswith('typed oracle mismatch:'):raise
    else:raise ValueError('omitted producer accepted')
    for arm in (0,1):
        b.a.upload(np.zeros_like(b.data['a']));b.out.poison()
        checked(core.launch(config,arm),'zero A');b.sdk.synchronize(b.r.stream)
        if np.any(b.out.read((b.rows,b.w.n))):raise ValueError('zero A is not zero')
    b.update(0)
    if b.ids:
        ids=b.data['ids'].copy();ids[0,0]=-1;b.ids.upload(ids)
        for arm in (0,1):
            checked(core.launch(config,arm),'invalid GPU expert ID');b.sdk.synchronize(b.r.stream)
            out=b.out.read((b.rows,b.w.n))
            if not np.isnan(out[0]).all() or not np.isfinite(out[1:]).all():raise ValueError('invalid ID scope differs')
        b.update(0)
    # Distinct address binding and all required planes, not just output poison.
    for name,buf in b.ring.objects[-1].items():
        try:
            b.ring.r.fill(buf.ptr,0,buf.size);b.sdk.synchronize(b.ring.r.stream)
            for arm in (0,1):
                checked(core.launch(config,arm,b.ring.copies-1),'plane negative')
                try:b.check()
                except ValueError as e:
                    if not str(e).startswith('typed oracle mismatch:'):raise
                else:raise ValueError('plane negative accepted: '+name)
        finally:
            b.sdk.synchronize(b.r.stream);buf.upload(b.ring.w.planes[name])
    b.update(0)
    return dict(checks=proofs,raw_equal=True,zero_a='PASS',omitted='DETECTED',
                invalid_ids='PASS' if b.ids else 'NOT_APPLICABLE',plane_negatives=list(b.ring.objects[-1]))


def timing(core,config,recipe):
    b=core.b;rt=RuntimeView(b);nodes=[];graphs=[]
    with ExitStack() as cleanup:
        for arm in (0,1):
            for copy in range(b.ring.copies):checked(core.launch(config,arm,copy),'excluded ring warmup')
            b.check()
            cap=Captured(rt,[lambda i=i,a=arm:core.launch(config,a,i) for i in range(b.ring.copies)])
            cleanup.callback(cap.close)
            expected=f'general_control_{config}(' if arm==0 else f'general_candidate_{config}_{b.p["mode"]}('
            grid=[b.rows*(b.w.n//(recipe['columns']*recipe['values'])),1,1] if arm==0 else [b.w.n//(recipe['columns']*recipe['values']),b.rows,1]
            if (len(cap.groups)!=1 or expected not in cap.descriptions[0]['name'] or
                    cap.descriptions[0]['grid']!=grid or cap.descriptions[0]['block']!=[recipe['warps']*32,1,1]):
                raise ValueError('general reader capture inventory differs')
            nodes.append(cap.descriptions[0])
            graph=cap.isolate(0,weight_ring=True,repeats=16);cleanup.callback(graph.close);graphs.append(graph)
        b.update(1);outputs=[];proofs=[]
        for g in graphs:
            b.out.poison();g.run();rt.sync();proofs.append(b.check());outputs.append(b.out.read((b.rows,b.w.n)).copy())
        raw_equal(*outputs);b.update(0)
        samples=[[],[]]
        for rnd in range(6):
            for arm in ((0,1) if rnd%2==0 else (1,0)):
                samples[arm].append([graphs[arm].sample() for _ in range(5)]);b.check()
        stats=[statistics_checked(s) for s in samples]
        active=len(np.unique(b.data['owners']))
        weight_bytes=sum(v.nbytes for v in b.ring.w.planes.values())//b.w.experts*active
        return dict(status='PASS',nodes=nodes,timings=stats,graph_checks=proofs,
                    calls_per_graph=[g.count for g in graphs],raw_equal=True,
                    active_experts=active,compulsory_weight_bytes=weight_bytes,
                    modeled_weight_mbu_pct=[weight_bytes/s['median_us']/27000 for s in stats],
                    assumed_peak_gbps=2700,mbu_scope='WEIGHT_BYTES_PER_EVENT_TIME_NOT_ACU_DRAM_BANDWIDTH',
                    delta_pct=100*(stats[1]['median_us']/stats[0]['median_us']-1),
                    scope='S1_SINGLE_KERNEL_COLD_ROTATING_SECONDARY_TO_CE_CYCLES')


def profile(core,config,arm):
    b=core.b;proofs=[]
    for repeat in (0,1,0):
        b.update(repeat);checked(core.launch(config,arm),'direct profile warmup');proofs.append(b.check())
    start,stop=b.sdk.lib.hggcProfilerStart,b.sdk.lib.hggcProfilerStop
    for fn in (start,stop):fn.argtypes,fn.restype=[],C.c_int
    print('GENERAL_PROFILE_BEGIN copies=1 graphs=0 calls=1',flush=True)
    checked(start(),'ACU start')
    try:checked(core.launch(config,arm),'one direct kernel');b.sdk.synchronize(b.r.stream)
    finally:checked(stop(),'ACU stop')
    proofs.append(b.check());print('GENERAL_PROFILE_END',flush=True)
    return dict(status='PASS',checks=proofs,weight_copies=1,graphs=0,timed_calls=1)
