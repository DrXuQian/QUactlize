#!/usr/bin/env python3
"""Minimal synthetic fixed-capacity serving graph + optional inspectable trace.

Does not download a model, allocate inside run, or claim end-to-end model speed.
Use the platform's existing GPU Torch; no pip installation/build extension.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quactlize.serving import Serving, Matmul
from quactlize.compat.native import problem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--m',type=int,default=4)
    p.add_argument('--experts',type=int,default=512)
    p.add_argument('--topk',type=int,default=10)
    p.add_argument('--k',type=int,default=64)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--trace',type=Path,help='optional Chrome trace from platform Torch profiler')
    a=p.parse_args()
    if min(a.m,a.experts,a.topk,a.k,a.hidden,a.iterations)<1 or a.k%32 or a.hidden%32:
        p.error('positive geometry/iterations and K/H multiples of 32 required')
    if a.trace and a.trace.exists():p.error('--trace must be a new file')
    stream=torch.cuda.Stream();device=torch.cuda.current_device();dtype=torch.bfloat16
    def zeros(shape,dt=dtype):return torch.zeros(shape,device=device,dtype=dt)
    def raw_q8(shape):
        e,n,k=shape;rng=np.random.default_rng(n+k)
        raw=np.zeros((e,n,k//32,34),dtype='u1')
        raw[...,:2]=np.array([.015625],dtype='<f2').view('u1')
        raw[...,2:]=rng.integers(-16,17,raw[...,2:].shape,dtype='i1').view('u1')
        return torch.tensor(raw,device=device,dtype=torch.uint8)
    with torch.cuda.stream(stream):
        ops=Serving(a.library,device,stream)
        m,e,t,k,h=a.m,a.experts,a.topk,a.k,a.hidden
        logits=zeros((m,e));selected=zeros((m,t),torch.float32)
        ids=zeros((m,t),torch.int32);weights=zeros((m,t),torch.float32)
        live=torch.full((1,),m,device=device,dtype=torch.int32);error=zeros((1,),torch.int32)
        lookup_ids=torch.arange(m,device=device,dtype=torch.int64)%7
        embedding=raw_q8((1,7,k));x=zeros((m,1,k))
        # Already-ready CPU PLE staging, 5 IQ4_NL blocks = 90 bytes per row.
        ple_np=np.zeros((7,5,18),dtype='u1')
        ple_np[...,:2]=np.array([.03125],dtype='<f2').view('u1');ple_np[...,2:]=0xf0
        ple=torch.tensor(ple_np,device=device,dtype=torch.uint8);ple_out=zeros((m,160))
        gu=zeros((m,t,2*h));mid=zeros((m,t,h));down=zeros((m,t,k));output=zeros((m,k),torch.float32)
        wup=raw_q8((e,2*h,k));wdown=raw_q8((e,k,h))
        pu=problem(8,m,2*h,k,experts=e,topk=t,device=device)
        pd=problem(8,m,k,h,experts=e,topk=t,channels=t,device=device)
        nbytes=max(Matmul.query(ops,pu).workspace_bytes,Matmul.query(ops,pd).workspace_bytes)
        scratch=torch.empty(nbytes,device=device,dtype=torch.uint8)
        up=Matmul(ops,pu,input=x,output=gu,workspace=scratch,error=error,ids=ids,raw=wup)
        dn=Matmul(ops,pd,input=mid,output=down,workspace=scratch,error=error,ids=ids,raw=wdown)
        steps={
            'embedding':lambda:ops.gather_rows(embedding,lookup_ids,x.view(m,k),error,qtype=8,table_rows=7,count=live),
            'ple':lambda:ops.gather_rows(ple,lookup_ids,ple_out,error,qtype=20,table_rows=7,count=live),
            'router':lambda:ops.router(logits,selected,ids,weights,error,live),
            'gate_up':lambda:up(x,gu,scratch,error,ids=ids,raw=wup),
            'swiglu':lambda:ops.swiglu(gu.view(m*t,2*h),mid.view(m*t,h),round_silu=True),
            'down':lambda:dn(mid,down,scratch,error,ids=ids,raw=wdown),
            'merge':lambda:ops.merge(down,weights,output),
        }
        def chain():
            error.zero_() # exactly one caller-owned reset at graph entry
            for call in steps.values():call()
        for _ in range(3):chain()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):chain()
        for active in (m,m//2,0,m):
            live.fill_(active);logits.normal_();lookup_ids.add_(1).remainder_(7)
            graph.replay();stream.synchronize()
            if error.item():raise RuntimeError(f'device status={error.item()}')
            if active<m and torch.count_nonzero(output[active:]):raise RuntimeError('inactive output not zero')
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();end.record();end.synchronize()
        def time_call(call):
            for _ in range(3):call()
            start.record()
            for _ in range(a.iterations):call()
            end.record();end.synchronize()
            return start.elapsed_time(end)*1000/a.iterations
        report=dict(device=torch.cuda.get_device_name(),library=str(a.library.resolve()),
            m=m,experts=e,topk=t,workspace_bytes=nbytes,
            component_us={name:time_call(call) for name,call in steps.items()},
            graph_replay_us=time_call(graph.replay),
            metric='STREAM_EVENT_ELAPSED_INCLUDES_SUBMISSION_GAPS_NOT_ISOLATED_KERNEL_SUMS',
            scope='WARM_COMPONENTS_AND_SYNTHETIC_GRAPH_NOT_MODEL_PERFORMANCE')
        if a.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as trace:
                graph.replay()
                # This synchronization is measurement infrastructure, outside
                # the submitted graph, and should appear as such in the trace.
            stream.synchronize();trace.export_chrome_trace(str(a.trace))
            events=[ev for ev in trace.events() if ev.device_type==torch.autograd.DeviceType.CUDA]
            if not events:raise RuntimeError('profiler returned no device activity; no kernel evidence available')
            kernels=[ev for ev in events if not ev.name.lower().startswith(('memcpy','memset'))]
            report['device_activity_count']=len(events)
            report['kernel_count']=len(kernels)
            report['device_activity_names']=sorted({ev.name for ev in events})
            report['trace']=str(a.trace)
        print(json.dumps(report,indent=2),flush=True)
        graph.reset();stream.synchronize();up.close();dn.close();ops.close()


if __name__=='__main__':main()
