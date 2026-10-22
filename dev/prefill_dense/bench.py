"""Isolated components and full calls; profiler mode never creates a cold ring."""
import ctypes as C
import math
import statistics
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from quactlize.prefill.native import Call
from quactlize.dequant.native import Call as Weight
from quactlize.execution.native import Arrangement,arrangement
from quactlize.runtime.native import checked
from dev.smallm_closure.fixture import Weights
from dev.bf16_compute.fixture import raw_weight,bf16_bits,round_compute
from tools.run_kpack_gemv_gate import Resources
from .spec import arms,legacy_config,STAGES


class Options(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('sdk',C.c_char_p)]+[
        (x,C.c_int32)for x in ('storage','vector_io','dequant_variant','math_mode','algorithm')]


def bind(lib,name,args,restype=C.c_int):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=restype;return fn


class Guard:
    def __init__(self,r,size):
        self.r,self.size=r,int(size);self.base=r.alloc(self.size+512);self.ptr=self.base+256
        if self.ptr%256:raise ValueError('workspace alignment differs')
        r.fill(self.base,0xa5,self.size+512);r.sdk.synchronize(r.stream)
    def upload(self,x):
        x=np.ascontiguousarray(x)
        if x.nbytes!=self.size:raise ValueError('upload byte extent differs')
        checked(self.r.sdk.lib.hggcMemcpy(self.ptr,x.ctypes.data,x.nbytes,1),'fixture upload')
        self.r.sdk.synchronize(None)
    def guard(self):
        for ptr in (self.base,self.ptr+self.size):
            if self.r.sdk.download(ptr,256)!=b'\xa5'*256:raise ValueError('redzone overwritten')


class Bench:
    def __init__(self,sdk,sdk_path,library,timer,w,l2,*,profile=False):
        self.sdk,self.w,self.r=sdk,w,Resources(sdk);self.buffers=[];self.handles=[]
        self.lib=C.CDLL(str(library),mode=C.RTLD_LOCAL);self.timer=C.CDLL(str(timer),mode=C.RTLD_LOCAL)
        self.run=bind(self.lib,'quactlize_kpack_dense_full_run_v1',[C.c_void_p,C.POINTER(Call),C.c_int])
        self.prepare=bind(self.lib,'quactlize_kpack_dense_full_prepare_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(Options),C.POINTER(C.c_void_p)])
        self.destroy=bind(self.lib,'quactlize_kpack_dense_full_destroy_v1',[C.c_void_p],None)
        self.error=bind(self.lib,'quactlize_kpack_dense_full_error_v1',[],C.c_char_p)
        self.io=bind(self.lib,'quactlize_dense_io_v1',[C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p])
        self.measure=bind(self.timer,'dense_opt_samples',[C.c_void_p,C.c_void_p,C.POINTER(Call),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_double)])
        self.arr=arrangement(w.qtype);self.sdk_path=str(sdk_path).encode()
        self.weights=Weights(w.qtype,w.n,w.k,1)
        packed=sum(x.nbytes for x in self.weights.planes.values())
        self.copies=1 if profile else max(2,math.ceil(2.25*l2/packed))
        self.shape=Call();s=self.shape;s.version=1;s.size=C.sizeof(Call)
        s.weight=Weight(1,C.sizeof(Weight),w.qtype,w.n,w.k,1,1,legacy_config(w))
        s.m=s.a_rows=w.m;s.a_stride=w.k;s.output_stride=w.n;s.device=0
        query=bind(self.lib,'quactlize_kpack_prefill_query_dynamic_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(C.c_uint64)])
        size=C.c_uint64();checked(query(C.byref(s),C.byref(self.arr),C.byref(size)),'workspace query')
        self.a=self.alloc(w.m*w.k*4);self.out=self.alloc(w.m*w.n*4)
        self.calls=(Call*self.copies)();self.workspaces=[]
        for i in range(self.copies):
            c=self.calls[i];C.memmove(C.byref(c),C.byref(s),C.sizeof(s))
            for name in ('low','high','units'):
                v=self.weights.planes[name]
                if v.nbytes:
                    b=self.alloc(v.nbytes);b.upload(v);setattr(c.weight,name,b.ptr)
                setattr(c.weight,{'units':'unit_bytes'}.get(name,name+'_bytes'),v.nbytes)
            ws=self.alloc(size.value);self.workspaces.append(ws)
            c.workspace=ws.ptr;c.workspace_bytes=ws.size;c.a=self.a.ptr;c.output=self.out.ptr
            c.weight.stream=self.r.stream.value
        self.update(0)
        self.receipt=dict(workload=asdict(w),copies=self.copies,l2_bytes=l2,packed_bytes=packed,
            rotating_bytes=self.copies*packed,profile=profile,fixture=self.weights.record,
            precision='F32_ENDPOINTS_FP32_ACCUMULATOR_EXPLICIT_F16_OR_BF16_INTERMEDIATES',
            workspace_bytes=size.value,cublas_workspace_bytes=4<<20)

    def alloc(self,size):
        b=Guard(self.r,size);self.buffers.append(b);return b
    def update(self,repeat,zero=False):
        coeff=round_compute(np.random.default_rng(931+repeat).uniform(-.5,.5,(self.w.m,4)).astype('f4'),'bf16')
        self.coeff=np.zeros_like(coeff)if zero else coeff
        self.a.upload(self.coeff[:,self.weights.categories].astype('f4'))
    def handle(self,arm):
        opts=Options(1,C.sizeof(Options),self.sdk_path,**{k:v for k,v in arm.items()if k!='name'})
        h=C.c_void_p();rc=self.prepare(C.byref(self.shape),C.byref(self.arr),C.byref(opts),C.byref(h))
        if rc:raise ValueError('dense prepare: '+self.error().decode())
        self.handles.append(h);return h
    def launch(self,h,copy=0,stage=-1):
        rc=self.run(h,C.byref(self.calls[copy]),stage)
        if rc:raise ValueError('dense run: '+self.error().decode()+f' rc={rc}')
    def check(self,zero=False):
        self.sdk.synchronize(self.r.stream);maximum=0.
        for first in range(0,self.w.m,64):
            stop=min(first+64,self.w.m)
            got=np.frombuffer(self.sdk.download(self.out.ptr+first*self.w.n*4,(stop-first)*self.w.n*4),dtype='f4').reshape(stop-first,self.w.n)
            if not np.isfinite(got).all():raise ValueError('nonfinite complete output')
            if zero:
                if np.any(got):raise ValueError('zero input produced nonzero output')
                continue
            coeff=self.coeff[first:stop].astype('f8')
            want=coeff@self.weights.sums[0];denom=np.abs(coeff)@self.weights.absolute[0]
            maximum=max(maximum,float((np.abs(got-want)/np.maximum(denom,1e-30)).max()))
        if maximum>=.005:raise ValueError(f'independent GGUF dot error={maximum}')
        for b in self.buffers:b.guard()
        return dict(relative_l1_error=maximum,tolerance=.005,nonfinite=0)
    def check_weights(self,storage,copy=0):
        # Layout.weights is offset zero; exactly one final F16/BF16 rounding
        # of the independent raw GGUF decoder, never our K-pack reader.
        self.sdk.synchronize(self.r.stream);checked_cells=0
        for first in range(0,self.w.n,256):
            n=min(256,self.w.n-first)
            _,gold=raw_weight(self.w.qtype,n,self.w.k,710+self.w.qtype*1009+first)
            want=bf16_bits(gold)if storage else gold.astype('<f2').view('<u2')
            got=np.frombuffer(self.sdk.download(self.workspaces[copy].ptr+first*self.w.k*2,n*self.w.k*2),dtype='<u2').reshape(n,self.w.k)
            if np.any((got!=want)&(((got|want)&0x7fff)!=0)):raise ValueError('full dequant independent bits differ')
            checked_cells+=got.size
        return dict(bits=checked_cells,bad=0,storage=storage)
    def samples(self,h,stage,samples=7):
        out=(C.c_double*samples)()
        checked(self.measure(C.cast(self.run,C.c_void_p),h,self.calls,self.copies,stage,samples,out),'ordinary timer')
        values=list(out)
        if any(not math.isfinite(x)or x<=0 for x in values):raise ValueError('invalid timing')
        return values
    def close(self):
        self.sdk.synchronize(self.r.stream)
        for h in reversed(self.handles):self.destroy(h)
        self.r.close()


def io_gate(sdk,sdk_path,library,*,vectors=(0,1)):
    """RNE bits, tails, misalignment, row strides, output status and guards."""
    lib=C.CDLL(str(library));fn=bind(lib,'quactlize_dense_io_v1',
        [C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p])
    owner=Resources(sdk);cases=[]
    try:
        shapes=[(3,2048,0,0),(3,1031,0,0),(3,1024,4,0),(3,1027,5,0),(2,1024,0,4)]
        if 2 in vectors:shapes.extend([(3,1028,4,0),(1,3,0,0),(1,2049,0,0)])
        if 3 in vectors:shapes.extend([(4,1024,0,0),(2,4096,0,0),(1,4095,0,0),(1,4097,0,0)])
        for rows,width,pad,offset in shapes:
            stride=width+pad;values=np.random.default_rng(width).uniform(-3,3,(rows,stride)).astype('f4')
            edge=np.array([0.,-0.,65504.,65520.,1e-8,-1e-8,243383.484,-243383.484],dtype='f4')
            values.flat[:min(8,values.size)]=edge[:min(8,values.size)]
            raw=Guard(owner,values.nbytes+offset);src=raw.ptr+offset
            checked(sdk.lib.hggcMemcpy(src,values.ctypes.data,values.nbytes,1),'io fixture');sdk.synchronize(None)
            for storage in (0,1):
                low=Guard(owner,rows*width*2+offset);dst=Guard(owner,values.nbytes+offset);error=Guard(owner,4)
                error.upload(np.zeros(1,dtype='i4'))
                with np.errstate(over='ignore'):
                    want=bf16_bits(values[:,:width])if storage else values[:,:width].astype('<f2').view('<u2')
                for vec in vectors:
                    checked(fn(storage,1,vec,src,low.ptr+offset,rows,width,stride,error.ptr,owner.stream),'input conversion')
                    sdk.synchronize(owner.stream)
                    got=np.frombuffer(sdk.download(low.ptr+offset,rows*width*2),dtype='u2').reshape(rows,width)
                    if not np.array_equal(got,want):raise ValueError('typed input RNE bits differ')
                    checked(fn(storage,0,vec,low.ptr+offset,dst.ptr+offset,rows,width,stride,error.ptr,owner.stream),'output conversion')
                    sdk.synchronize(owner.stream)
                    out=np.frombuffer(sdk.download(dst.ptr+offset,values.nbytes),dtype='u4').reshape(rows,stride)
                    expected=(want.astype('u4')<<16)if storage else want.view('<f2').astype('f4').view('u4')
                    if not np.array_equal(out[:,:width],expected):raise ValueError('typed output bits differ')
                    if pad and np.any(out[:,width:]!=0xa5a5a5a5):raise ValueError('stride padding touched')
                    for b in (raw,low,dst,error):b.guard()
                    cases.append(dict(rows=rows,width=width,pad=pad,offset=offset,storage=storage,vector_io=vec,status='PASS'))
                error.upload(np.ones(1,dtype='i4'))
                checked(fn(storage,0,max(vectors),low.ptr+offset,dst.ptr+offset,rows,width,stride,error.ptr,owner.stream),'status finish')
                sdk.synchronize(owner.stream)
                out=np.frombuffer(sdk.download(dst.ptr+offset,values.nbytes),dtype='f4').reshape(rows,stride)
                if not np.isnan(out[:,:width]).all():raise ValueError('nonzero device status not propagated')
    finally:owner.close()
    return dict(status='PASS',cases=cases,fp16_overflow='IEEE_INF_NOT_CLIPPED',bf16_large_finite='PRESERVED')


def dequant_gate(sdk,library,*,variants=(0,1,2,3),formats=range(10,15)):
    from reference import gguf_kpack as ref
    from tools.kpack_warmup_fixture import prepare_expert
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    lib=C.CDLL(str(library));fn=bind(lib,'quactlize_kpack_dequant_dense_v1',
        [C.POINTER(Weight),C.POINTER(Arrangement),C.c_int,C.c_int])
    old=bind(lib,'quactlize_kpack_dequant_v1',[C.POINTER(Weight),C.POINTER(Arrangement)])
    r=Resources(sdk);rows=[]
    try:
        for q in formats:
            n,k=256,1024;s=ref.SPECS[q];rng=np.random.default_rng(4721+q)
            raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
            for offset in (s.d_offset,s.dmin_offset):
                if offset>=0:raw[:,offset:offset+2]=rng.uniform(.0001,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
            p=prepare_expert(raw,q,n,k);gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
            buffers={}
            for name in ('low','high','units'):
                if p[name].size:buffers[name]=Guard(r,p[name].nbytes);buffers[name].upload(p[name])
            out=Guard(r,n*k*2);arr=arrangement(q)
            call=Weight(1,C.sizeof(Weight),q,n,k,1,1,5,
                *[buffers[x].ptr if x in buffers else None for x in ('low','high','units')],
                *[p[x].nbytes for x in ('low','high','units')],out.ptr,None,out.size,r.stream.value)
            for storage in (0,1):
                want=bf16_bits(gold)if storage else gold.astype('<f2').view('<u2')
                for variant in ([0]if q<12 else variants):
                    # A missing writer must not inherit the preceding arm's
                    # correct result and pass this independent raw-bit gate.
                    out.upload(np.full((n,k),0x7fff,dtype='<u2'))
                    checked(fn(C.byref(call),C.byref(arr),storage,variant),'non-dyadic typed expansion')
                    sdk.synchronize(r.stream)
                    got=np.frombuffer(sdk.download(out.ptr,out.size),dtype='u2').reshape(n,k)
                    if np.any((got!=want)&(((got|want)&0x7fff)!=0)):raise ValueError(f'dequant random bits differ q={q} storage={storage} variant={variant}')
                    rows.append(dict(q=q,storage=storage,variant=variant,bits=got.size,bad=0))
                if storage:
                    out.upload(np.full((n,k),0x7fff,dtype='<u2'))
                    checked(old(C.byref(call),C.byref(arr)),'legacy expansion control');sdk.synchronize(r.stream)
                    got=np.frombuffer(sdk.download(out.ptr,out.size),dtype='u2').reshape(n,k)
                    if np.any((got!=want)&(((got|want)&0x7fff)!=0)):raise ValueError('legacy expansion bits differ')
            for b in [out,*buffers.values()]:b.guard()
    finally:r.close()
    return dict(status='PASS',cases=rows,oracle='INDEPENDENT_GGUF_NON_DYADIC_RAW_BITS')


def dynamic_gate(sdk,sdk_path,library,timer,l2,*,round_two=False,variants=None,vector_io=None,formats=range(10,15)):
    from quactlize.dispatch.table import Workload
    results=[]
    for q in formats:
        b=None;other=None
        try:
            b=Bench(sdk,sdk_path,library,timer,Workload(q,256,512,128,storage='f32',compute='f16'),l2,profile=True)
            other=Resources(sdk);alternate=Call.from_buffer_copy(b.calls[0])
            ws=b.alloc(alternate.workspace_bytes);out=b.alloc(128*256*4)
            alternate.workspace=ws.ptr;alternate.output=out.ptr;alternate.weight.stream=other.stream.value
            for storage in (0,1):
                for variant in ([0]if q<12 else variants if variants is not None else [4,5,6,7]if round_two else[1]):
                    arm=dict(name='contract',storage=storage,vector_io=vector_io if vector_io is not None else 2 if round_two else 1,
                             dequant_variant=variant,math_mode=16,algorithm=-1)
                    h=b.handle(arm)
                    for repeat in (0,1):
                        b.update(repeat);b.launch(h)
                        checked(b.run(h,C.byref(alternate),-1),'second live stream/dynamic addresses')
                        sdk.synchronize(other.stream);proof=b.check()
                        if sdk.download(out.ptr,out.size)!=sdk.download(b.out.ptr,b.out.size):
                            raise ValueError('disjoint-workspace concurrent output differs')
                        results.append(dict(q=q,storage=storage,variant=variant,repeat=repeat,status='PASS',proof=proof))
        finally:
            if other:other.close()
            if b:b.close()
    return dict(status='PASS',cases=results,streams=2,allocation_timed=False)
