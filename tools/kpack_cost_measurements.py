"""GPU measurements for the finite cost supplement, without online selection."""
import ctypes as C
import hashlib
import math
import statistics
import time

import numpy as np

from quactlize.runtime.native import SDK, Call as GemmCall, Recipe, Module, Resources as Query, checked
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import arrangement
from quactlize.dequant.native import Call as DequantCall, bind, traffic
from tools.kpack_prefill_measurement import Weights
from tools.kpack_bf16_fixture import Oracle
from tools.kpack_bf16_providers import Cublas, DeepGemm, loaded_images
from tools.kpack_dequant_fixture import compare
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind, DeviceCall
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_dequant_gate import device
from tools.kpack_cost_supplement import domain, configs

DQ_SCOPE = 'ISOLATED_DEQUANT_ACTUAL_EXPERT_DOMAIN'
GEMM_SCOPE = 'SELECTED_OR_HISTORICAL_COMPLETE_GEMM_NO_DEQUANT_NO_EXTERNAL_ADAPTERS'
BF16_SCOPE = 'ISOLATED_BF16_PROVIDER_NO_DEQUANT_NO_EXTERNAL_ADAPTERS'


class Context:
    def __init__(self, sdk, bundle, dev=None):
        self.sdk=SDK(sdk);graph_bind(self.sdk)
        self.r=Resources(self.sdk)
        self.lib,self.dequant,probe=bind(bundle)
        self.device=device(self.sdk,probe)
        if dev is not None and self.device!=dev:raise ValueError('component PPU changed')
        self.indexed_lib=C.CDLL(str(bundle/'libquactlize_dequant_indexed.so'),mode=C.RTLD_LOCAL)
        self.indexed=self.indexed_lib.quactlize_kpack_dequant_indexed_v1
        self.indexed.argtypes=[C.POINTER(DequantCall),C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p]
        self.indexed.restype=C.c_int
        fn=self.sdk.lib.hggcMemcpyAsync
        fn.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p];fn.restype=C.c_int

    def upload(self, array, copies=1):
        if array.size==0:return [None]*copies
        base=self.r.alloc(array.nbytes*copies)
        self.copy(base,array)
        for i in range(1,copies):
            checked(self.sdk.lib.hggcMemcpyAsync(base+i*array.nbytes,base,array.nbytes,3,self.r.stream),'D2D ring setup')
        return [base+i*array.nbytes for i in range(copies)]

    def copy(self, ptr, array):
        self.sdk.synchronize(self.r.stream)
        array=np.ascontiguousarray(array)
        checked(self.sdk.lib.hggcMemcpy(ptr,array.ctypes.data,array.nbytes,1),'untimed H2D')
        self.sdk.synchronize(None)

    def outputs(self, size, copies):
        stride=size+256;base=self.r.alloc(stride*copies)
        self.r.fill(base,0xa5,stride*copies)
        return [base+i*stride+128 for i in range(copies)]

    def read(self, pointer, size):
        self.sdk.synchronize(self.r.stream)
        raw=self.sdk.download(pointer-128,size+256)
        if raw[:128]!=b'\xa5'*128 or raw[-128:]!=b'\xa5'*128:
            raise ValueError('output guard changed')
        return np.frombuffer(raw[128:-128],dtype='<u2').copy()

    def close(self):self.r.close()


def samples(ctx, sequence, copies, proof):
    graph=Replay(ctx.sdk,ctx.r.stream,sequence,2)
    try:
        checked(graph(),'first replay excluded');checked(graph(),'second replay excluded');proof()
        values=[]
        for _ in range(3):
            values.extend(v/(2*copies) for v in ctx.r.samples(graph,5));proof()
        return timing(values,copies)
    finally:graph.close()


def timing(values,copies):
    return dict(samples_us=values,median_us=statistics.median(values),
        round_medians_us=[statistics.median(values[i:i+5]) for i in (0,5,10)],
        calls_per_graph=2*copies,copies=copies,first_use_excluded=True,
        timing='CAPTURED_COMPLETE_RING_EVENTS',production_changed=False)


def active_inputs(ctx,p):
    active=np.asarray(p['active_ids'],dtype='<i4')
    ids=np.full(p['experts'],-1,dtype='<i4');ids[:len(active)]=active
    return ctx.upload(ids)[0],ctx.upload(np.array([len(active)],dtype='<i4'))[0],ctx.upload(np.zeros(1,dtype='<i4'))[0]


def dq_call(ctx,p,w,operation,config,planes,output,zero,active):
    arr=arrangement(p['q']);sizes=traffic(p['q'],p['n'],p['k'],p['experts'],operation)
    c=DequantCall(version=1,size=C.sizeof(DequantCall),qtype=p['q'],n=p['n'],k=p['k'],experts=p['experts'],
        operation=operation,config=config,output=output,zero=zero,output_bytes=sizes['output'],
        low_bytes=sizes['low'],high_bytes=sizes['high'],unit_bytes=sizes['units'],
        stream=ctx.r.stream.value,**planes)
    if operation and p['full_indexed']:
        return ctx.indexed(C.byref(c),C.byref(arr),*active)
    return ctx.dequant(C.byref(c),C.byref(arr))


def dequant_measure(a,p,w,operation):
    ctx=Context(a.sdk,a.bundle,a.device)
    try:
        sizes=traffic(p['q'],p['n'],p['k'],p['experts'],operation)
        used=p['active_ids'] if operation else list(range(p['experts']))
        reads=sizes['reads']//p['experts']*len(used)
        writes=sizes['writes']//p['experts']*len(used)
        copies=max(2,math.ceil(2.25*ctx.device['l2_bytes']/reads))
        names=('low','high','units') if operation else ('units',)
        planes={name:ctx.upload(w.planes[name],copies) for name in names}
        output=ctx.outputs(sizes['output'],copies)
        zero=ctx.outputs(sizes['output'],copies) if not operation else [None]*copies
        active=active_inputs(ctx,p) if operation and p['full_indexed'] else None
        gold=w.gold if operation else np.stack([w.planes[k].view('<u2') for k in ('scale','zero')])
        allowed=configs(p['q'],operation,p['full_indexed'] if operation else False)
        rows=[]
        unused=sorted(set(range(p['experts']))-set(used))
        def run(config,i=0):
            return dq_call(ctx,p,w,operation,config,{k:v[i] for k,v in planes.items()},output[i],zero[i],active)
        def read(i):
            if operation:return ctx.read(output[i],sizes['output']).reshape(gold.shape)
            return np.stack([ctx.read(v[i],sizes['output']) for v in (output,zero)]).reshape(gold.shape)
        def proof(i=0):
            got=read(i)
            if operation:
                result=compare(got[used],gold[used])
                if unused and np.any(got[unused]!=0x7f7f):raise ValueError('inactive expert was expanded')
            else:
                result=compare(got,gold)
                if result['signed_zero_differences']:raise ValueError('SF signed zeros differ')
            if active and np.frombuffer(ctx.sdk.download(active[2],4),dtype='<i4')[0]:raise ValueError('indexed device error')
            return result
        for config in allowed:
            for ptr in output:ctx.r.fill(ptr,0x7f,sizes['output'])
            for ptr in zero:
                if ptr:ctx.r.fill(ptr,0x7f,sizes['output'])
            checked(run(config),'dequant correctness');numeric=proof()
            key='low' if operation else 'units';original=w.planes[key]
            ctx.r.fill(planes[key][0],0,original.nbytes)
            checked(run(config),'blank dequant input')
            observed=read(0)
            negative=int(np.count_nonzero(observed[used]!=gold[used] if operation else observed!=gold))
            if negative<=0:raise ValueError('dequant negative not detected')
            ctx.copy(planes[key][0],original)
            # The same captured kernel must consume mutable GPU IDs/count.
            if active:
                control=Replay(ctx.sdk,ctx.r.stream,lambda:run(config),1)
                try:
                    ctx.copy(active[1],np.array([0],dtype='<i4'))
                    ctx.r.fill(output[0],0x7f,sizes['output'])
                    checked(control(),'empty active list graph')
                    if np.any(read(0)!=0x7f7f):raise ValueError('count=0 wrote weights')
                    selected=used[-1]
                    ids=np.full(p['experts'],-1,dtype='<i4');ids[0]=selected
                    ctx.copy(active[0],ids);ctx.copy(active[1],np.array([1],dtype='<i4'))
                    checked(control(),'changed expert ID graph')
                    got=read(0);compare(got[selected],gold[selected])
                    if np.any(np.delete(got,selected,axis=0)!=0x7f7f):raise ValueError('ID mapped to compact rather than original expert')
                    ids[0]=p['experts'];ctx.copy(active[0],ids)
                    ctx.r.fill(output[0],0x7f,sizes['output'])
                    checked(control(),'invalid expert ID graph')
                    if np.any(read(0)!=0x7f7f) or np.frombuffer(ctx.sdk.download(active[2],4),dtype='<i4')[0]!=2:
                        raise ValueError('invalid expert ID not rejected on device')
                    ids[:len(used)]=used
                    ctx.copy(active[0],ids);ctx.copy(active[1],np.array([len(used)],dtype='<i4'))
                    ctx.copy(active[2],np.zeros(1,dtype='<i4'))
                finally:control.close()
            def sequence():
                for i in range(copies):checked(run(config,i),'isolated dequant')
                return 0
            sequence()
            for i in range(copies):proof(i)
            result=samples(ctx,sequence,copies,lambda:(proof(0),proof(copies-1)))
            rows.append(result|dict(config=config,proof=numeric,negative_bad=negative,
                inactive_experts_untouched=True,guard='PASS',device_count_control='PASS' if active else 'NOT_INDEXED',
                changed_id_graph='PASS' if active else 'NOT_INDEXED'))
            print(f'COST_DEQUANT config={config} operation={operation} case={p["weight_id"]} us={result["median_us"]:.3f}',flush=True)
        return dict(status='PASS',kind='dequant',scope=DQ_SCOPE,operation=operation,weight_id=p['weight_id'],
            indexed=bool(operation and p['full_indexed']),active_ids=used,expanded_experts=len(used),device=ctx.device,
            rows=rows,fixture=w.identity,input_ring_bytes=reads*copies,output_ring_bytes=writes*copies,
            logical_read_bytes=reads,logical_write_bytes=writes,cache='ROTATING_ACTIVE_INPUTS_AND_OUTPUTS',
            gemm_timed=False,production_changed=False)
    finally:ctx.close()


def recipe(config,p,occupancy):
    mode=config['grid_mode'];persistent=mode in (2,3,'capacity','balanced')
    if not persistent:return Recipe(1,C.sizeof(Recipe),0,config['split'],0)
    tm,tn=config['tm'],config['tn'];m=sum(p['rows']);e=p['experts'];bound=p['tokens']
    mt=(m+tm-1)//tm if e==1 else min(e*((bound+tm-1)//tm),(m+min(m,e)*(tm-1))//tm)
    tiles=mt*((p['n']+tn-1)//tn)*(config['split'] if e>1 else 1)
    cap=72*max(1,min(config['grid_b'],occupancy))
    grid=(tiles+(tiles+cap-1)//cap-1)//((tiles+cap-1)//cap) if mode in (3,'balanced') else min(tiles,cap)
    return Recipe(1,C.sizeof(Recipe),1,config['split'],grid)


def gemm_measure(a,p,w,route,historical=False):
    ctx=Context(a.sdk,a.bundle,a.device);dispatch=Dispatch(a.bundle);graph=None;handles=[]
    module=None
    try:
        current=dispatch.query(p['q'],route,sum(p['rows']),p['n'],p['k'],p['experts'],p['tokens'],arrangement(p['q']).mapping_id)
        if not historical and current is None:raise ValueError('current heuristic unavailable: '+str(dispatch.last_miss))
        old=p['routes'][str(route)]['historical'] if historical else None
        if historical and old is None:return dict(status='UNAVAILABLE',reason='NO_HISTORICAL_FAMILY')
        if historical:
            record=next(r for r in a.manifest['modules'] if r['parent']['symbol']==old['symbol'])
            module=Module(record|dict(path=str(a.bundle/record['path'])))
            choice=dict(parent=old['symbol'],build_key=record['key'],split=old['split'],policy='HISTORICAL_CHALLENGER')
        else:
            expected=p['routes'][str(route)]['selected']
            if current.parent.decode()!=expected['parent'] or current.split!=expected['split']:raise ValueError('production selection changed')
            choice=receipt(current)
        sf=route%2==1;names=('low','high','scale','zero') if sf else ('low','high','units')
        size_per_expert=sum(w.planes[n].nbytes for n in names)//p['experts']
        reads=size_per_expert*len(p['active_ids']);copies=max(2,math.ceil(2.25*ctx.device['l2_bytes']/reads))
        inputs={name:ctx.upload(w.planes[name],copies) for name in names}
        rows,indices,_=domain(p['tokens'],p['experts'],p['profile']);m=len(indices)
        activation,coeff=w.activation(m);ap=ctx.upload(activation)[0]
        offsets=ctx.upload(np.r_[0,rows.cumsum()].astype('<i4'))[0] if p['experts']>1 else None
        size=m*p['n']*2;outputs=ctx.outputs(size,copies);calls=[]
        for i in range(copies):
            c=GemmCall(version=1,size=C.sizeof(GemmCall),m=m,n=p['n'],k=p['k'],experts=p['experts'],
                group_size=arrangement(p['q']).group_size,device=ctx.device['ordinal'],compute_units=ctx.device['compute_units'],
                mapping_id=arrangement(p['q']).mapping_id,a=ap,low=inputs['low'][i],high=inputs['high'][i],
                metadata=inputs['scale' if sf else 'units'][i],zero=inputs['zero'][i] if sf else None,
                output=outputs[i],offsets_device=offsets,stream=ctx.r.stream.value)
            if not historical:
                c.workspace_bytes=current.workspace_bytes;c.workspace=ctx.r.alloc(max(16,c.workspace_bytes))
                calls.append(dispatch.prepare(current,c))
            else:
                rec=recipe(old,p,1);res=Query()
                if p['experts']>1:
                    query=module.lib.quactlize_kpack_grouped_query_v2;prepare=module.lib.quactlize_kpack_grouped_prepare_v2
                    query.argtypes=[C.POINTER(DeviceCall),C.POINTER(Recipe),C.POINTER(Query)];query.restype=C.c_int
                    prepare.argtypes=[C.POINTER(DeviceCall),C.POINTER(Recipe),C.POINTER(C.c_void_p)];prepare.restype=C.c_int
                    d=DeviceCall(2,C.sizeof(DeviceCall),c,p['tokens'],0)
                    checked(query(C.byref(d),C.byref(rec),C.byref(res)),'historical grouped query')
                else:checked(module.query(C.byref(c),C.byref(rec),C.byref(res)),'historical dense query')
                rec=recipe(old,p,res.occupancy)
                if p['experts']>1:checked(query(C.byref(d),C.byref(rec),C.byref(res)),'historical grouped recipe')
                else:checked(module.query(C.byref(c),C.byref(rec),C.byref(res)),'historical dense recipe')
                c.workspace_bytes=res.workspace_bytes;c.workspace=ctx.r.alloc(max(16,res.workspace_bytes))
                h=C.c_void_p()
                if p['experts']>1:
                    d.call=c;checked(prepare(C.byref(d),C.byref(rec),C.byref(h)),'historical grouped prepare')
                else:checked(module.prepare(C.byref(c),C.byref(rec),C.byref(h)),'historical dense prepare')
                handles.append(h);calls.append(lambda h=h:module.run(h,ctx.r.stream))
                choice.update(algorithm=rec.algorithm,grid=rec.grid,workspace_bytes=res.workspace_bytes,shared_bytes=res.shared_bytes)
        def sequence():
            for fn in calls:checked(fn(),'complete GEMM')
            return 0
        errors=[]
        def proof(i=0,zero=False):
            out=ctx.read(outputs[i],size).view('<f2').reshape(m,p['n'])
            if zero:
                if not np.isfinite(out).all() or np.any(out):raise ValueError('zero A output differs')
            else:
                err=w.error(out,coeff,indices)
                if err>=.005:raise ValueError(f'GGUF dot error {err}')
                errors.append(err)
        ctx.sdk.synchronize(ctx.r.stream);sequence()
        for i in range(copies):proof(i)
        if w.error(np.zeros((m,p['n']),dtype='<f2'),coeff,indices)<=.005:raise ValueError('oracle missed zero output')
        ctx.copy(ap,np.zeros_like(activation));sequence();proof(0,True)
        ctx.copy(ap,activation);sequence();proof()
        graph=Replay(ctx.sdk,ctx.r.stream,sequence,2)
        ctx.copy(ap,-activation);coeff*=-1;checked(graph(),'changed input graph');proof(0);proof(copies-1)
        ctx.copy(ap,activation);coeff*=-1;checked(graph(),'restored graph');proof()
        result=samples(ctx,sequence,copies,lambda:(proof(0),proof(copies-1)))
        return result|dict(status='PASS',kind='gemm',route=route,historical=historical,selection=choice,
            scope=GEMM_SCOPE,device=ctx.device,point=p['id'],rows=p['rows'],routes_sha256=p['routes_sha256'],
            error=max(errors),guards='PASS',zero_a='PASS',changed_input_graph='PASS',fixture=w.identity,
            active_input_ring_bytes=reads*copies,reducer_included=True,sf_prepass_timed=False,max_rows_bound=p['tokens'])
    finally:
        ctx.sdk.synchronize(ctx.r.stream)
        if graph:graph.close()
        for h in handles:module.destroy(h)
        dispatch.close();ctx.close()


def bf16_measure(a,p,w,best):
    import torch
    ctx=Context(a.sdk,a.bundle,a.device);provider=None;weights=[];graph=None
    try:
        stream=torch.cuda.ExternalStream(ctx.r.stream.value)
        provider=Cublas(a.sdk,ctx.r.stream) if p['experts']==1 else DeepGemm()
        if not hasattr(w,'bf16_oracle'):w.bf16_oracle=Oracle(w.gold)
        oracle=w.bf16_oracle
        rows,indices,_=domain(p['tokens'],p['experts'],p['profile']);m=len(indices)
        bits,coeff=oracle.activations(m)
        if oracle.error(np.zeros((m,p['n']),dtype='<u2'),coeff,indices)<=.005:
            raise ValueError('BF16 oracle missed zero output')
        read_bytes=p['n']*p['k']*2*len(p['active_ids']);copies=max(2,math.ceil(2.25*ctx.device['l2_bytes']/read_bytes))
        planes={name:ctx.upload(w.planes[name])[0] for name in ('low','high','units')}
        active=active_inputs(ctx,p) if p['full_indexed'] else None
        guard=64
        with torch.cuda.stream(stream):
            at=torch.from_numpy(bits.view('<i2')).view(torch.bfloat16).to('cuda')
            ids=torch.from_numpy(indices).to('cuda');counts=torch.from_numpy(rows).to('cuda')
            for _ in range(copies):
                b=torch.full(w.gold.shape,float('nan'),dtype=torch.bfloat16,device='cuda')
                checked(dq_call(ctx,p,w,1,best['config'],planes,b.data_ptr(),None,active),'untimed BF16 weight creation')
                ctx.sdk.synchronize(ctx.r.stream)
                for e in p['active_ids']:
                    got=np.frombuffer(ctx.sdk.download(b[e].data_ptr(),p['n']*p['k']*2),dtype='<u2').reshape(p['n'],p['k'])
                    compare(got,w.gold[e])
                weights.append(b)
            backs=[torch.full((m*p['n']+2*guard,),-123.,dtype=torch.bfloat16,device='cuda') for _ in weights]
            outputs=[x[guard:-guard].reshape(m,p['n']) for x in backs]
            def launch(i=0):provider(at,weights[i][0] if p['experts']==1 else weights[i],outputs[i],ids,counts)
            errors=[]
            def proof(i=0,zero=False):
                ctx.sdk.synchronize(ctx.r.stream)
                if not bool((backs[i][:guard]==-123).all()) or not bool((backs[i][-guard:]==-123).all()):raise ValueError('BF16 output guard')
                got=np.frombuffer(ctx.sdk.download(outputs[i].data_ptr(),m*p['n']*2),dtype='<u2').reshape(m,p['n'])
                if zero:
                    if np.any(got&0x7fff):raise ValueError('BF16 zero A differs')
                else:
                    error=oracle.error(got,coeff,indices)
                    if error>=.005:raise ValueError(f'BF16 dot error {error}')
                    errors.append(error)
            start=time.monotonic();launch();ctx.sdk.synchronize(ctx.r.stream);first=time.monotonic()-start
            proof()
            at.zero_();outputs[0].fill_(float('nan'));launch();proof(0,True)
            at.copy_(torch.from_numpy(bits.view('<i2')).view(torch.bfloat16));launch();proof()
            for i in range(copies):launch(i);proof(i)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(2):
                    for i in range(copies):launch(i)
            graph.replay();graph.replay();proof(0);proof(copies-1)
            at.neg_();coeff*=-1;graph.replay();proof(0);proof(copies-1)
            at.neg_();coeff*=-1;graph.replay();proof()
            values=[]
            for _ in range(3):
                values.extend(v/(2*copies) for v in ctx.r.samples(lambda:graph.replay() or 0,5));proof(0);proof(copies-1)
        return timing(values,copies)|dict(status='PASS',kind='bf16',scope=BF16_SCOPE,point=p['id'],device=ctx.device,
            provider=provider.identity,loaded_images=loaded_images(),error=max(errors),guards='PASS',zero_a='PASS',
            changed_input_graph='PASS',fixture=w.identity,rows=p['rows'],routes_sha256=p['routes_sha256'],
            active_input_ring_bytes=read_bytes*copies,first_use_seconds=first,dequant_timed=False,
            weight_source='UNTIMED_VALIDATED_DEVICE_DEQUANT',dequant_config=best['config'])
    finally:
        ctx.sdk.synchronize(ctx.r.stream)
        if graph is not None:del graph
        if provider:provider.close()
        weights.clear();ctx.close()
