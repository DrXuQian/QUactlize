"""Indexed/compact Q4 S1 numerical gate and rotating-weight measurement."""
import ctypes as C
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np

from dev.gemv_ppu import moe_s1 as spec
from dev.gemv_ppu.run import query_l2_attribute, resolve_l2
from dev.gemv_ppu.small_latency import footprint
from quactlize.execution.native import Call, Sizes, Arrangement, arrangement
from quactlize.runtime.native import SDK, checked
from tools.kpack_execution_fixture import IndexedWeights
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity


class Config(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[
        (s,C.c_int32) for s in ('reader','variant','warps','values')]

    def __init__(self,r):
        super().__init__(1,C.sizeof(type(self)),r.reader,r.variant,r.warps,r.values)


def bind(library):
    query=library.quactlize_q4_s1_query_v1
    query.argtypes=[C.POINTER(Call),C.POINTER(Config),C.POINTER(Arrangement),C.POINTER(Sizes)]
    query.restype=C.c_int
    run=library.quactlize_q4_s1_run_v1
    run.argtypes=query.argtypes[:-1]; run.restype=C.c_int
    return query,run


def fixture(w,tokens,channels,clustered=False):
    # Padding is deliberately different on all three public strides.
    topk=8; k=w.k; n=w.n
    ids=np.full((tokens,topk+3),-99,dtype='<i4')
    for t in range(tokens):
        ids[t,:topk]=(np.arange(topk)*17+3+(0 if clustered else t*13))%w.experts
    values=np.random.default_rng(811+101*tokens+7*channels).normal(0,.2,(tokens*channels,4)).astype('<f4')
    ah=values[:,w.categories[0]].astype('<f2')
    a=np.full((tokens,channels,k+8),np.nan,dtype='<f4')
    a[:,:,:k]=values[:,w.categories[0]].reshape(tokens,channels,k)
    expert=ids[:,:topk].reshape(-1)
    arows=(np.arange(tokens*topk)//topk)*channels+np.arange(tokens*topk)%topk%channels
    rounded=values.astype('<f2').astype('f8')
    golden=np.stack([rounded[arows[r]]@w.sums[expert[r]] for r in range(tokens*topk)])
    denom=np.stack([np.abs(rounded[arows[r]])@w.abs_sums[expert[r]] for r in range(tokens*topk)])
    return dict(a=a,ah=ah,ids=ids,expert=expert,arows=arows,golden=golden,denom=denom,
                tokens=tokens,channels=channels,rows=tokens*topk,n=n,k=k)


def error(got,data):
    if got.shape!=data['golden'].shape or not np.isfinite(got).all():
        raise ValueError('nonfinite/unwritten/wrong-shape output')
    return float(np.max(np.abs(got.astype('f8')-data['golden'])/np.maximum(data['denom'],1e-30)))


def a_model(r,input_type,base):
    size=2 if input_type==0 else 4
    if r.reader==0:
        offsets=[(lane//8)*32+lane%8 for lane in range(32)]; width=size
    elif r.reader==2 and r.variant&1:
        offsets=[(lane//4)*32+(lane%4)*8 for lane in range(32)]
        width=16  # F32 load8 is TWO float4 instructions.
    else:
        offsets=[(lane//4)*32 for lane in range(32)]
        width=4 if input_type==0 else 16  # F16 values4 is TWO half2 loads.
    addresses=[base+size*x for x in offsets]
    return dict(element_bytes=size,lane_byte_addresses=addresses,width_bytes=width,
        extra_instruction_offset_bytes=16 if input_type==1 and r.reader==2 and r.variant&1 else
            4 if input_type==0 and r.reader!=0 and not(r.reader==2 and r.variant&1) else None,
        granules={str(s):footprint(addresses,width,s) for s in (32,64,128)},
        scope='SOURCE_FIRST_A_INSTRUCTION_NOT_MEASURED_TRANSACTIONS')


class Bench:
    def __init__(self,args,n,k):
        self.args=args; self.n=n; self.k=k
        self.sdk=SDK(args.sdk); graph_bind(self.sdk)
        self.r=Resources(self.sdk); self.case_r=None
        self.library=C.CDLL(str(args.bundle/spec.payload(n,k)),mode=C.RTLD_LOCAL)
        self.query,self.run=bind(self.library)
        probe=self.library.q4_ppu_probe
        probe.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p]; probe.restype=C.c_int
        l2,sm,warp,name=C.c_int(),C.c_int(),C.c_int(),C.create_string_buffer(256)
        checked(probe(C.byref(l2),C.byref(sm),C.byref(warp),name),'image/marker')
        self.device=device_identity(self.sdk)|resolve_l2(l2.value,args.l2_bytes,query_l2_attribute(self.sdk.lib))
        self.device.update(name=name.value.decode(),properties_sm=sm.value,warp=warp.value)
        if warp.value!=32: raise ValueError('32-lane reader on incompatible device')
        self.w=IndexedWeights(12,n,k,256,progress=lambda done,total:print(
            f'Q4_MOE_S1_FIXTURE shape={n}x{k} experts={done}/{total}',flush=True))
        self.expert_bytes=n*k*9//16
        # Even the clustered/token1 cases touch eight distinct experts. The
        # ring bound counts those bytes, NOT all 256 allocated expert slots.
        self.copies=max(2,math.ceil(2.25*self.device['l2_bytes']/(8*self.expert_bytes)))
        self.calls_per_graph=max(2,math.ceil(32/self.copies))*self.copies
        self.planes=[]; self.plane_sizes={x:self.w.planes[x].nbytes for x in ('low','units')}
        bases={x:self.r.alloc(v*self.copies) for x,v in self.plane_sizes.items()}
        for i in range(self.copies):
            record={}
            for x,size in self.plane_sizes.items():
                record[x]=bases[x]+i*size
                checked(self.sdk.lib.hggcMemcpy(record[x],self.w.planes[x].ctypes.data,size,1),'ring upload')
            self.planes.append(record)
        self.weight_sha256={x:hashlib.sha256(self.w.planes[x]).hexdigest() for x in self.plane_sizes}
        self.zero_low=self.r.alloc(self.plane_sizes['low'])
        self.r.fill(self.zero_low,0,self.plane_sizes['low'])
        self.arr=arrangement(12)
        self.graphs={}
        self.immutable=None
        if (n,k)==(512,2048):
            import json
            from dev.gemv_cuda.build import digest
            bundle=spec.ROOT/'prebuilt/ppu0010/q4-smallm-v1'
            file=bundle/'libq4_smallm_n512_k2048.so'
            row=json.loads((bundle/'manifest.json').read_text())['payloads'][file.name]
            if digest(file)!=row['sha256']: raise ValueError('immutable dense control differs')
            self.immutable_sha256=row['sha256']
            self.immutable_library=C.CDLL(str(file),mode=C.RTLD_LOCAL)
            self.immutable=self.immutable_library.q4_smallm_run
            self.immutable.argtypes=[C.c_int]*3+[C.c_void_p]*5; self.immutable.restype=C.c_int

    def setup(self,tokens,channels,clustered=False):
        self.release_case()
        self.case_r=Resources(self.sdk); r=self.case_r
        self.data=fixture(self.w,tokens,channels,clustered)
        d=self.data; m=d['rows']; self.output_stride=self.n+8
        self.output_bytes=m*self.output_stride*4
        self.out_base=r.alloc(self.output_bytes+32); self.out=self.out_base+16
        self.a=r.upload(d['a']); self.ids=r.upload(d['ids'])
        self.base=Call(version=1,size=C.sizeof(Call),qtype=12,n=self.n,k=self.k,experts=256,rows=m,
            mode=2,input_type=1,channels=channels,topk=8,a_row_stride=self.k+8,
            a_token_stride=channels*(self.k+8),ids_stride=11,out_row_stride=self.output_stride,
            a=self.a,ids=self.ids,output=self.out,stream=r.stream.value)
        self.base.low,self.base.units=self.planes[0]['low'],self.planes[0]['units']
        self.sdk.synchronize(None)

    def invoke(self,key,call=None):
        f=Config(spec.lookup(key))
        return self.run(C.byref(self.base if call is None else call),C.byref(f),C.byref(self.arr))

    def read(self):
        self.sdk.synchronize(self.case_r.stream)
        raw=np.frombuffer(self.sdk.download(self.out_base,self.output_bytes+32),dtype='u1')
        got=raw[16:-16].view('<f4').reshape(self.data['rows'],self.output_stride)
        if not np.all(raw[:16]==0xa5) or not np.all(raw[-16:]==0xa5) or not np.all(got[:,self.n:].view('u1')==0xa5):
            raise ValueError('output guard or padded stride overwritten')
        return got[:,:self.n].copy()

    def checked_output(self,key,call=None):
        self.case_r.fill(self.out_base,0xa5,self.output_bytes+32)
        checked(self.invoke(key,call),'S1 numerical launch')
        return self.read()

    def correctness(self,key):
        f=Config(spec.lookup(key)); sizes=Sizes()
        checked(self.query(C.byref(self.base),C.byref(f),C.byref(self.arr),C.byref(sizes)),'S1 query')
        if sizes.workspace_bytes: raise ValueError('S1 unexpectedly needs workspace')
        got=self.checked_output(key); err=error(got,self.data)
        if err>=.005: raise ValueError(f'independent GGUF error {err:.6g}')
        r=self.case_r; c=Call.from_buffer_copy(self.base)
        # Numerically DIFFERENT float inputs that round to the same F16 data
        # must exactly match the F16 input specialization.
        ah=np.full(self.data['a'].shape,np.nan,dtype='<f2'); ah[:,:,:self.k]=self.data['a'][:,:,:self.k]
        c.a=r.upload(ah); c.input_type=0
        half=self.checked_output(key,c)
        if not np.array_equal(got.view('u4'),half.view('u4')):
            raise ValueError('F32 register-rounding and F16 reader disagree bitwise')
        # Sorted compact rows use the same per-expert planes, with empty
        # experts. The actual grouped locate helper is exercised, not emulated.
        order=np.argsort(self.data['expert'],kind='stable')
        counts=np.bincount(self.data['expert'],minlength=256)
        offsets=np.r_[0,counts.cumsum()].astype('<i4')
        compact=np.full((self.data['rows'],self.k+8),np.nan,dtype='<f2')
        compact[:,:self.k]=self.data['ah'][self.data['arows'][order]]
        c.mode=1; c.input_type=0; c.channels=c.topk=1; c.ids=None
        c.offsets=r.upload(offsets); c.a=r.upload(compact)
        grouped=self.checked_output(key,c)[np.argsort(order)]
        if not np.array_equal(got.view('u4'),grouped.view('u4')):
            raise ValueError('indexed versus compact grouped row/expert mismatch')
        # Exercise DENSE with the same rebased expert slices and padded output
        # pointers. Blue host per-row launch is untimed, never a MoE route.
        c=Call.from_buffer_copy(self.base); c.mode=0; c.experts=c.rows=1; c.channels=c.topk=1
        c.ids=None; c.input_type=0
        dense_a=r.upload(self.data['ah'])
        r.fill(self.out_base,0xa5,self.output_bytes+32)
        for row,e in enumerate(self.data['expert']):
            c.a=dense_a+int(self.data['arows'][row])*self.k*2
            c.low=self.base.low+int(e)*self.n*self.k//2; c.units=self.base.units+int(e)*self.n*self.k//16
            c.output=self.out+row*self.output_stride*4
            checked(self.invoke(key,c),'untimed dense row control')
        if not np.array_equal(self.read().view('u4'),got.view('u4')):
            raise ValueError('indexed versus dense expert-slice mismatch')
        immutable='OUT_OF_SCOPE'
        recipe=spec.lookup(key)
        if self.immutable and recipe.reader==0:
            r.fill(self.out_base,0xa5,self.output_bytes+32)
            for row,e in enumerate(self.data['expert']):
                checked(self.immutable((4,8,16).index(recipe.warps),1,0,
                    dense_a+int(self.data['arows'][row])*self.k*2,
                    self.base.low+int(e)*self.n*self.k//2,self.base.units+int(e)*self.n*self.k//16,
                    self.out+row*self.output_stride*4,r.stream),'frozen dense M1 body control')
            if not np.array_equal(self.read().view('u4'),got.view('u4')):
                raise ValueError('transplant differs from immutable dense reader')
            immutable=self.immutable_sha256
        c=Call.from_buffer_copy(self.base)
        zero_a=np.zeros_like(self.data['a']); c.a=r.upload(zero_a)
        if np.any(self.checked_output(key,c)!=0): raise ValueError('zero A does not produce zero output')
        c=Call.from_buffer_copy(self.base); c.low=self.zero_low
        if error(self.checked_output(key,c),self.data)<=.005: raise ValueError('zeroed code plane escaped oracle')
        c=Call.from_buffer_copy(self.base)
        bad_ids=self.data['ids'].copy(); bad_ids[:,:8]=256; c.ids=r.upload(bad_ids)
        if not np.isnan(self.checked_output(key,c)).all(): raise ValueError('invalid ID did not return NaN')
        # Actual stale-IDs graph replay: update the device IDs without changing
        # its pointer. A captured kernel must consume the new expert IDs.
        replay=Replay(self.sdk,r.stream,lambda:self.invoke(key),1)
        try:
            checked(replay(),'first graph launch'); self.sdk.synchronize(r.stream)
            changed=self.data['ids'].copy(); changed[:,:8]=np.roll(changed[:,:8],1,axis=1)
            checked(self.sdk.lib.hggcMemcpy(self.ids,changed.ctypes.data,changed.nbytes,1),'IDs replay upload')
            r.fill(self.out_base,0xa5,self.output_bytes+32)
            checked(replay(),'replay new IDs'); changed_got=self.read()
            if error(changed_got,self.data)<=.005: raise ValueError('fixture cannot reject stale expert IDs')
            changed_experts=changed[:,:8].reshape(-1)
            # Compute the independent answer for the altered routing too.
            values=self.data['ah'][:,[int(np.flatnonzero(self.w.categories[0]==g)[0]) for g in range(4)]].astype('f8')
            new_data=dict(golden=np.stack([values[self.data['arows'][i]]@self.w.sums[e] for i,e in enumerate(changed_experts)]),
                denom=np.stack([np.abs(values[self.data['arows'][i]])@self.w.abs_sums[e] for i,e in enumerate(changed_experts)]))
            if error(changed_got,new_data)>=.005: raise ValueError('updated device IDs fail independent oracle')
            # Restore and require exact output to distinguish stale pointer
            # capture from a timing-only repeated launch.
            checked(self.sdk.lib.hggcMemcpy(self.ids,self.data['ids'].ctypes.data,self.data['ids'].nbytes,1),'restore IDs')
            r.fill(self.out_base,0xa5,self.output_bytes+32); checked(replay(),'restored IDs replay')
            if not np.array_equal(self.read().view('u4'),got.view('u4')): raise ValueError('graph replay differs')
        finally: replay.close()
        return dict(error=err,fp16_fp32_bits='PASS',indexed_grouped_bits='PASS',indexed_dense_bits='PASS',
                    immutable_dense=immutable,zero_a='PASS',zero_codes='PASS',invalid_ids='PASS',
                    output_guard='PASS',gpu_ids_replay='PASS',workspace_bytes=0,
                    output_sha256=hashlib.sha256(got).hexdigest())

    def graph(self,key):
        if key not in self.graphs:
            index=0
            def issue():
                nonlocal index
                c=Call.from_buffer_copy(self.base)
                c.low,c.units=self.planes[index%self.copies]['low'],self.planes[index%self.copies]['units']
                index+=1
                return self.invoke(key,c)
            self.graphs[key]=Replay(self.sdk,self.case_r.stream,issue,self.calls_per_graph)
            checked(self.graphs[key](),'graph upload excluded'); self.sdk.synchronize(self.case_r.stream)
        return self.graphs[key]

    def measure(self,key,samples):
        replay=self.graph(key)
        values=[t/self.calls_per_graph for t in self.case_r.samples(replay,samples)]
        # The last graph call uses the last ring member; every member carries
        # the same frozen bytes. Check actual replay output, not setup only.
        if error(self.read(),self.data)>=.005: raise ValueError('timed ring replay numerical failure')
        return values

    def profile(self,key):
        from tools.profile_kpack_gpu_compact import AcuRange
        checked(self.invoke(key),'profile warmup excluded'); self.sdk.synchronize(self.case_r.stream)
        with AcuRange(self.sdk):
            checked(self.invoke(key),'profile S1'); self.sdk.synchronize(self.case_r.stream)
        if error(self.read(),self.data)>=.005: raise ValueError('profile numerical failure')

    def release_case(self):
        if self.case_r:
            self.sdk.synchronize(self.case_r.stream)
            for graph in self.graphs.values(): graph.close()
            self.graphs.clear(); self.case_r.close(); self.case_r=None

    def close(self):
        self.release_case(); self.r.close()
